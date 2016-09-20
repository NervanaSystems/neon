# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Define a layer that takes Region Proposal Network's output and generate
region proposals in the format of bounding boxes, compares with ground truth boxes
and generates bounding box target labels and regression targets
"""
from __future__ import division
import numpy as np
from neon.layers.layer import Layer
from generate_anchors import generate_all_anchors
from util import compute_targets, calculate_bb_overlap

# Thresholds for the IoU overlap to consider a proposed ROI as
# foreground or background. Less than BG_THRESH_LO is ignored.
FG_THRESH = 0.5
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

# The percentage of ROIs are from foreground within a minibatch
FG_FRACTION = 0.25

BBOX_NORMALIZE_MEANS = [0.0, 0.0, 0.0, 0.0]
BBOX_NORMALIZE_STDS = [0.1, 0.1, 0.2, 0.2]

# precomputed empirically...
# means: [-0.00624775 -0.01916088 -0.11854464  0.16050844]
# std: [ 0.27227189  0.42585105  0.48241054  0.49931084]


class ProposalLayer(Layer):
    """
    Proposal layer takes as input:
    (1) output of RPN cls_score
    (2) output of RPN bbox regression

    Converts that to an output ROIs
    (5, num_ROIs) <- [image_idx, x_min, y_min, x_max, y_max]

    Steps:
    1. For each anchor, generate anchor boxes and apply bbox deltas to get
       bbox proposals
    2. Clip bboxes to image
    3. remove bbox with H < threshold or W < threshold
    4. set the scores to be -1 for the padded area
    5. Take top pre_nms_N scores
    6. apply NMS with a threshold
    7. return the top num_ROIs proposals
    8. provide ROIs
    9. compute bbox targets and store in target_buffers


    """

    def __init__(self, rpn_layers, dataloader, inference=False, num_rois=128, pre_nms_N=12000,
                 post_nms_N=2000, nms_thresh=0.7, min_bbox_size=16,
                 fg_fraction=None, fg_thresh=None, bg_thresh_hi=None, bg_thresh_lo=None,
                 deterministic=False, name=None, debug=False):
        """
        Arguments:
            rpn_layers (list): References to the RPN layers: [RPN_1x1_obj, RPN_1x1_bbox]
            target_buffers (tuple): Target buffers for training fast-rcnn: (class, bbox regression)
            num_rois (int, optional): Number of ROIs to sample from proposed (default: 128)
            pre_nms_N (int, optional): Number of ROIs to retain before using NMS (default: 12000)
            post_nms_N (int, optional): Number of ROIs to retain after using NMS (default: 2000)
            nms_thresh (float, optional): Threshold for non-maximum supression (default: 0.7)
            min_bbox_size (integer, optional): Minimize bboxes side length (default: 16)
            name (string, optional): Name of layer (default: None)
        """
        super(ProposalLayer, self).__init__(name)

        self.rpn_obj, self.rpn_bbox = rpn_layers
        self.num_rois = num_rois
        self.pre_nms_N = pre_nms_N
        self.post_nms_N = post_nms_N
        self.nms_thresh = nms_thresh
        self.min_bbox_size = min_bbox_size
        self.num_classes = dataloader.num_classes
        self.fg_fraction = fg_fraction if fg_fraction else FG_FRACTION
        self.fg_thresh = fg_thresh if fg_thresh else FG_THRESH
        self.bg_thresh_hi = bg_thresh_hi if bg_thresh_hi else BG_THRESH_HI
        self.bg_thresh_lo = bg_thresh_lo if bg_thresh_lo else BG_THRESH_LO
        self.deterministic = deterministic
        self.debug = debug

        # the output shape of this layer depends on whether the network
        # will be used for inference. In inference mode, the output shape is
        # (5, post_nms_N). For training, a smaller set of rois are sampled, yielding
        # an output shape of (5, num_rois)
        self.inference = inference

        # set references to dataset object buffers
        self.dataloader = dataloader
        self._conv_height = dataloader.conv_height
        self._conv_width = dataloader.conv_width
        self._scale = dataloader.conv_scale

        # generate anchors and load onto device
        # self._anchors has shape (KHW, 4)
        self._anchors = generate_all_anchors(self._conv_height, self._conv_width, self._scale)
        self._dev_anchors = self.be.array(self._anchors)
        self._num_anchors = self._anchors.shape[0]

    def configure(self, in_obj):
        super(ProposalLayer, self).configure(in_obj)
        # set out_shape as the ROI shape
        if(self.inference):
            self.out_shape = ((5, self.post_nms_N))
        else:
            self.out_shape = ((5, self.num_rois))

        self.in_shape = in_obj.out_shape

        return (in_obj, self)

    def get_description(self, **kwargs):
        skip = ['rpn_layers', 'global_buffers', 'dataloader']
        if 'skip' in kwargs:
            kwargs['skip'].append(skip)
        else:
            kwargs['skip'] = skip
        return super(ProposalLayer, self).get_description(**kwargs)

    def allocate(self):
        super(ProposalLayer, self).allocate()

        # internal buffer to store transformed proposals
        self._proposals = self.be.zeros((self._num_anchors, 4))

        # to store the detections (scores + proposals)
        self.dets = self.be.zeros((self.pre_nms_N, 5))

        # buffer to store proposals after they have sorted
        # and filtered with NMS.
        # Note: The buffer has shape (num_ROIs, 5), where each column
        # is (0, x_min, y_min, x_max, y_max)
        # This format is designed to be compatible with the
        # roi-pooling layer fprop code from Fast-RCNN.
        self.dev_proposals = self.be.zeros((self.post_nms_N, 5))

        # buffer to store proposals after they have been sampled.
        # this is passed forward during training.
        self.dev_proposals_filtered = self.be.zeros((self.num_rois, 5))

        # class member as a view to get scores from RPN outputs
        self.rpn_scores_v = None
        self.bbox_deltas_v = None

        # create a local buffer for the rpn scores, otherwise, any in-place
        # memory change will affect the final cost and training
        self._scores = self.be.zeros((self._num_anchors, 1))

    def fprop(self, inputs, inference=False):
        """
        fprop function that does no proposal filtering
        """
        assert self.inference == inference, \
            "Model was configured for inference={}".format(self.inference)

        # get needed metadata from the dataloader
        (self.im_shape, self.im_scale, self.gt_boxes,
            self.gt_classes, self.num_gt_boxes, _) = self.dataloader.get_metadata_buffers()

        # real H and W need to get in fprop, as every image is different
        real_H = int(np.round(self.im_shape.get()[1] * self._scale))
        real_W = int(np.round(self.im_shape.get()[0] * self._scale))

        if self.rpn_scores_v is None:
            # get output from the RPN network
            # transform the scores and slice the score for the label=1 class
            # shape: (KWH, 1)
            self.rpn_scores_v = self.rpn_obj[0].outputs.reshape((2, -1))[1].T

        if self.bbox_deltas_v is None:
            # transform the bbox deltas, reshape to (4, KHW) then transpose to
            # (KHW, 4) to match the shape of anchors.
            self.bbox_deltas_v = self.rpn_bbox[0].outputs.reshape((4, -1)).T

        # 1. Convert anchors into proposals via bbox transformations
        # store output in proposals buffer
        self._bbox_transform_inv(self._dev_anchors, self.bbox_deltas_v, output=self._proposals)

        # 2. clip predicted boxes to image
        self._clip_boxes(self._proposals, self.im_shape)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_scale)
        keep = self._filter_boxes(self._proposals,
                                  self.min_bbox_size * float(self.im_scale.get()))

        # we set the score of those we discard to -1
        self._scores[:] = (self.rpn_scores_v * keep) - (1 - keep)

        # 4. set the scores to be -1 for the padded area
        # set the scores of all the proposals from the padded area to be -1
        # in order to ignore them after sorting
        scores_np = self._scores.get()
        scores_np.reshape(-1, self._conv_height, self._conv_width)[:, real_H:] = -1
        scores_np.reshape(-1, self._conv_height, self._conv_width)[:, :, real_W:] = -1
        self._scores[:] = self.be.array(scores_np)

        # 5. sort the scores from highest to lowest and take top pre_nms_topN
        top_N_ind = self.get_top_N_index(self._scores, self.pre_nms_N)

        # take top pre_nms_topN (e.g. 12000)
        # (make scores & proposals attributes of layer for unit testing)
        self.dets.fill(0)
        self.dets[:len(top_N_ind), :4] = self._proposals[top_N_ind]
        self.dets[:len(top_N_ind), 4] = self._scores[top_N_ind]

        # 6. apply nms (e.g. threshold = 0.7)
        keep = self.be.nms(self.dets, self.nms_thresh)

        # 7. take post_nms_N (e.g. 2000)
        keep = keep[:self.post_nms_N]
        self.num_proposals = len(keep)

        # for training or debugging, we need to copy these detections to host.
        if self.debug or not inference:
            # make scores & proposals attributes of layer for unit testing
            self.proposals = self.dets[keep, :4].get()
            self.scores = self.dets[keep, -1].get()

        # 8. provide ROIs in the format of [0, x1, y1, x2, y2]
        self.dev_proposals.fill(0)
        self.dev_proposals[:self.num_proposals, 1:] = self.dets[keep, :4]

        # If training, sample the proposals and only propagate those forward
        if not inference:
            # Next, we need to set the target buffers with the class labels,
            # and bbox targets for each roi.
            ((frcn_labels, frcn_labels_mask),
             (frcn_bbtargets, frcn_bbmask)) = self.dataloader.get_target_buffers()

            non_zero_gt_boxes = self.gt_boxes.get()
            num_gt_boxes = self.num_gt_boxes.get()[0][0]
            non_zero_gt_boxes = non_zero_gt_boxes[:num_gt_boxes]

            # Include ground-truth boxes in the set of candidate rois
            all_rois = np.vstack((self.proposals, non_zero_gt_boxes))

            # 1. Compute the overlap of each proposal roi with each ground truth roi
            overlaps = calculate_bb_overlap(all_rois, non_zero_gt_boxes)

            # 2. Use overlaps to compute the gt box each proposal is 'closest' to
            gt_assignment = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)
            labels = self.gt_classes.get()[:num_gt_boxes]
            labels = labels[gt_assignment]

            # Sample num_rois fg and bg indicies based on overlaps with gt bocxes
            keep_inds, fg_rois_this_img = self._sample_fg_bg(max_overlaps)
            # Select sampled values from various arrays:
            labels = labels[keep_inds]
            # Clamp labels for the background RoIs to 0
            labels[fg_rois_this_img:] = 0

            rois = all_rois[keep_inds]
            targets = compute_targets(non_zero_gt_boxes[gt_assignment[keep_inds]], rois)

            targets = (targets - np.array(BBOX_NORMALIZE_MEANS)) / np.array(BBOX_NORMALIZE_STDS)

            num_proposals = rois.shape[0]
            bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels(targets, labels)

            # Load fast-rcnn training labels and bbox targets back to global buffers
            labels_full = self._onehot_labels(labels.ravel())
            frcn_labels[:] = labels_full
            labels_mask = np.zeros((self.num_rois, self.num_classes))
            labels_mask[:num_proposals, :] = 1.0
            frcn_labels_mask[:] = np.ascontiguousarray(labels_mask.T)

            # fcrn_*.shape = (num_classes*4 , 256), so transpose first
            frcn_bbtargets[:] = np.ascontiguousarray(bbox_targets.T)
            frcn_bbmask[:] = np.ascontiguousarray(bbox_inside_weights.T)

            # load the sampled proposals back to device
            rois = np.hstack([np.zeros((num_proposals, 1)), rois])
            rois = np.ascontiguousarray(rois, dtype=np.float32)

            self.dev_proposals_filtered.fill(0)
            self.dev_proposals_filtered[:num_proposals, :] = rois
            self.num_proposals = num_proposals

            # During training, only propagate sampled proposals
            return (inputs, self.dev_proposals_filtered.T)

        # If inference=True, we're testing, so propagate all proposals
        else:
            return (inputs, self.dev_proposals.T)

    def get_proposals(self):
        return (self.dev_proposals, self.num_proposals)

    def get_top_N_index(self, scores, N):
        # this function handles scores still being device tensors
        count = len(np.where(scores.get() > -1)[0])
        order = scores.get().ravel().argsort()[::-1].tolist()
        order = order[:count]
        if N > 0:
            order = order[:N]

        return order

    def bprop(self, errors, alpha=1.0, beta=0.0):
        """This layer propagate gradients from ROIs back to lower VGG layers"""
        self.deltas = errors
        self.prev_layer.deltas[:] = errors
        return errors

    def _clip_boxes(self, boxes, im_shape):
        boxes[:, 0] = self.be.clip(boxes[:, 0], 0, im_shape[0] - 1)
        boxes[:, 1] = self.be.clip(boxes[:, 1], 0, im_shape[1] - 1)
        boxes[:, 2] = self.be.clip(boxes[:, 2], 0, im_shape[0] - 1)
        boxes[:, 3] = self.be.clip(boxes[:, 3], 0, im_shape[1] - 1)

        return boxes

    def _bbox_transform_inv(self, boxes, deltas, output):
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = self.be.exp(dw) * widths
        pred_h = self.be.exp(dh) * heights

        # pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        output[:, 0] = pred_ctr_x - 0.5 * pred_w
        # y1
        output[:, 1] = pred_ctr_y - 0.5 * pred_h
        # x2
        output[:, 2] = pred_ctr_x + 0.5 * pred_w
        # y2
        output[:, 3] = pred_ctr_y + 0.5 * pred_h

        return output

    def _filter_boxes(self, boxes, min_size):
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = (ws >= min_size) * (hs >= min_size)

        return keep

    def _onehot_labels(self, labels):
        """Converts the roi labels from compressed (1, num_rois) shape
        to the one-hot format required for the global buffers of shape
        (num_classes, num_rois)"""
        labels_full = np.zeros((self.num_classes, self.num_rois))
        for idx, l in enumerate(labels):
            labels_full[int(l), idx] = 1
        return labels_full

    def _get_bbox_regression_labels(self, bbox_target_data, labels):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form N x (tx, ty, tw, th)
        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).
        Returns:
            bbox_targets (ndarray): N x 4K blob of regression targets
            bbox_inside_weights (ndarray): N x 4K blob of loss weights
        """
        bbox_targets = np.zeros((self.num_rois, 4 * self.num_classes), dtype=np.float32)
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(labels > 0)[0]
        for ind in inds:
            l = labels[ind]
            start = int(4 * l)
            end = int(start + 4)
            bbox_targets[ind, start:end] = bbox_target_data[ind]
            bbox_inside_weights[ind, start:end] = 1.0
        return bbox_targets, bbox_inside_weights

    def _sample_fg_bg(self, max_overlaps):
        """Return sample of at most fg_fraction * num_rois foreground indicies, padding
        the remaining num_rois with background indicies. Foreground and background labels
        are determined based on max_overlaps and the thresholds fg_thresh, bg_thresh_hi,
        bg_thresh_lo.
        Returns:
            keep_inds (array): (num_rois,) sampled indicies of bboxes.
            fg_rois_per_this_image (int): number of fg rois sampled from the image.
        """
        # Split proposals into foreground and background based on overlap
        fg_inds = np.where(max_overlaps >= self.fg_thresh)[0]
        fg_rois_per_image = np.round(self.fg_fraction * self.num_rois)

        # Guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)

        # Sample foreground regions without replacement
        if fg_inds.size > 0 and not self.deterministic:
            fg_inds = self.be.rng.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)
        elif fg_inds.size > 0:
            fg_inds = fg_inds[:fg_rois_per_this_image]

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < self.bg_thresh_hi) &
                           (max_overlaps >= self.bg_thresh_lo))[0]

        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = self.num_rois - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)

        # Sample background regions without replacement
        if bg_inds.size > 0 and not self.deterministic:
            bg_inds = self.be.rng.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)
        elif bg_inds.size > 0:
            bg_inds = bg_inds[:bg_rois_per_this_image]

        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)
        return keep_inds, int(fg_rois_per_this_image)
