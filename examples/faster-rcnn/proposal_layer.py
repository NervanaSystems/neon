from neon.layers.layer import Layer
from generate_anchors import generate_all_anchors
from py_cpu_nms import py_cpu_nms as nms
from pycaffe_proposal_layer import PyCaffeProposalLayer
from pycaffe_proposal_target_layer import PyCaffeProposalTargetLayer
from objectlocalization import _compute_targets, calculate_bb_overlap
import numpy as np

# Thresholds for the IoU overlap to consider a proposed ROI as
# foreground or background. Less than BG_THRESH_LO is ignored.
FG_THRESH = 0.5
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

FG_FRACTION = 0.25

REFERENCE_TEST = False

BBOX_NORMALIZE_MEANS = [0.0, 0.0, 0.0, 0.0]
BBOX_NORMALIZE_STDS = [0.1, 0.1, 0.2, 0.2]

# precomputed empirically...
# means: [-0.00624775 -0.01916088 -0.11854464  0.16050844]
# std: [ 0.27227189  0.42585105  0.48241054  0.49931084]


class ProposalLayer(Layer):
    """
    Proposal layer takes as input:
    (1) output of cls_score
    (2) output of bbox regression

    Converts that to an output ROIs
    (5, num_ROIs) <- [batch_size, x_min, y_min, x_max, y_max]

    Steps:
    1. For each anchor, generate anchor boxes and apply bbox deltas to get
       bbox proposals
    2. Clip bboxes to image
    3. remove bbox with H < threshold or W < threshold
    4. Take top pre_nms_N scores
    5. apply NMS with threshold 0.7
    6. return the top num_ROIs proposals
    7. compute bbox targets and store in target_buffers


    """

    def __init__(self, rpn_layers, global_buffers, inference=False, num_rois=128, pre_nms_N=12000,
                 post_nms_N=2000, nms_thresh=0.7, min_bbox_size=16, num_classes=21,
                 fg_fraction=None, fg_thresh=None, bg_thresh_hi=None, bg_thresh_lo=None,
                 deterministic=False, name=None):
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
        self.num_classes = num_classes
        self.fg_fraction = fg_fraction if fg_fraction else FG_FRACTION
        self.fg_thresh = fg_thresh if fg_thresh else FG_THRESH
        self.bg_thresh_hi = bg_thresh_hi if bg_thresh_hi else BG_THRESH_HI
        self.bg_thresh_lo = bg_thresh_lo if bg_thresh_lo else BG_THRESH_LO
        self.deterministic = deterministic

        # the output shape of this layer depends on whether the network
        # will be used for inference. In inference mode, the output shape is
        # (5, post_nms_N). For training, a smaller set of rois are sampled, yielding
        # an output shape of (5, num_rois)
        self.inference = inference

        # set references to dataset object buffers
        self.target_buffers = global_buffers['target_buffers']
        self.im_shape, self.im_scale = global_buffers['img_info']
        self.gt_boxes, self.gt_classes, self.num_gt_boxes = global_buffers['gt_boxes']
        self._conv_size, self._scale = global_buffers['conv_config']

        # generate anchors and load onto device
        # self._anchors has shape (KHW, 4)
        self._anchors = generate_all_anchors(self._conv_size, self._conv_size, self._scale)
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
        # from neon.layers import BranchNode
        # if type(in_obj) is BranchNode:
        #     in_obj = in_obj.prev_layer
        return (in_obj, self)

    def get_description(self, **kwargs):
        skip = ['rpn_layers', 'global_buffers']
        if 'skip' in kwargs:
            kwargs['skip'].append(skip)
        else:
            kwargs['skip'] = skip
        return super(ProposalLayer, self).get_description(**kwargs)

    def allocate(self):
        super(ProposalLayer, self).allocate()

        # internal buffer to store transformed proposals
        self._proposals = self.be.zeros((self._num_anchors, 4))

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

        # buffer to store scores
        self.dev_scores = self.be.zeros((self._num_anchors, 1))

    def fprop(self, inputs, inference=False):
        """
        fprop function that does no proposal filtering
        """
        assert self.inference == inference, \
            "Model was configured for inference={}".format(self.inference)
 
        # get output from the RPN network
        scores = self.rpn_obj[0].outputs  # shape: (2KHW, 1)
        bbox_deltas = self.rpn_bbox[0].outputs  # shape (4KHW, 1)

        # reshape to (4, KHW) then transpose to (KHW, 4) to
        # match the shape of anchors.
        bbox_deltas = bbox_deltas.reshape((4, -1)).T

        # same transform for scores, except for we slice the
        # score for the label=1 class.
        scores = scores.reshape((2, -1))[1, :].T

        # 1. Convert anchors into proposals via bbox transformations
        # store output in proposals buffer
        self._bbox_transform_inv(self._dev_anchors, bbox_deltas, output=self._proposals)

        # 2. clip predicted boxes to image
        self._clip_boxes(self._proposals, self.im_shape)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_scale)
        keep = self._filter_boxes(self._proposals,
                                  self.min_bbox_size * float(self.im_scale.get()))
        # we set the score of those we discard to -1
        # the result is stored in this layers' own buffer.
        self.dev_scores[:] = (scores * keep) - (1 - keep)

        # -- TODO: move compute to device -- #
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 12000)
        # (make scores & proposals attributes of layer for unit testing)
        self.scores = self.dev_scores.get()
        self.proposals = self._proposals.get()

        real_H = np.round(self.im_shape.get()[1] * self._scale).astype(int)
        real_W = np.round(self.im_shape.get()[0] * self._scale).astype(int)
        self.scores = self.scores.reshape(-1, self._conv_size, self._conv_size)[:, :real_H, :real_W].reshape(-1, 1)
        self.proposals = self.proposals.reshape(-1, self._conv_size, self._conv_size, 4)[:, :real_H, :real_W].reshape(-1, 4)

        keeps = list(np.where(self.scores != -1)[0])
        # Combine the filtered index list with the original anchor index list (from global buffer)
        # to get final list of unique proposals to keep
        # keeps = list(set(all_anchor_inds).union(filt_idx))
        self.scores = self.scores[keeps]
        self.proposals = self.proposals[keeps]

        (self.scores, self.proposals) = self.get_top_N(self.scores, self.proposals, self.pre_nms_N)

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take post_nms_N (e.g. 2000)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((self.proposals, self.scores)), self.nms_thresh)
        keep = keep[:self.post_nms_N]
        self.proposals = self.proposals[keep]
        self.scores = self.scores[keep]
        # -- END TODO -- #

        # load proposals back to device
        num_proposals = self.proposals.shape[0]
        _proposals = np.hstack([np.zeros((num_proposals, 1)), self.proposals])
        _proposals = np.ascontiguousarray(_proposals, dtype=np.float32)
        self.dev_proposals[:num_proposals, :] = _proposals

        self.num_proposals = num_proposals

        # If training, sample the proposals and only propagate those forward
        if not inference:
            # Next, we need to set the target buffers with the class labels,
            # and bbox targets for each roi.
            ((frcn_labels, frcn_labels_mask), (frcn_bbtargets, frcn_bbmask)) = self.target_buffers

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
            targets = _compute_targets(non_zero_gt_boxes[gt_assignment[keep_inds]], rois)

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

            self.dev_proposals_filtered[:num_proposals, :] = rois
            self.num_proposals = num_proposals

            if REFERENCE_TEST:
                self._reference_test(_proposals, self.scores, labels.ravel(),
                                     bbox_targets, bbox_inside_weights, keep_inds)

            # During training, only propagate sampled proposals
            return (inputs, self.dev_proposals_filtered.T)

        # If inference=True, we're testing, so propagate all proposals
        else:
            return (inputs, self.dev_proposals.T)

    def get_proposals(self):
        return (self.dev_proposals, self.num_proposals)

    def _reference_test(self, target_proposals, target_scores,
                        frcn_labels, frcn_bbtargets, frcn_bbmask, keep_inds):
        prop_layer = PyCaffeProposalLayer()
        (H, W) = (62, 62)

        # prepare top and bottom data buffers
        scores = self.rpn_obj[0].outputs.get()  # shape: (2KHW, 1)
        bbox_deltas = self.rpn_bbox[0].outputs.get()  # shape (4KHW, 1)

        # reshape from (4KHW, 1) -> (1, K4, H, W) format for pycaffe
        # NB: pycaffe uses A where we use K
        bbox_deltas = bbox_deltas.reshape((4, -1, H, W)).transpose((1, 0, 2, 3))
        bbox_deltas = bbox_deltas.reshape((1, -1, H, W))

        # reshape from (2KHW, 1) -> (1, K2, H, W)
        scores = scores.reshape((2, -1, H, W)).transpose((0, 1, 2, 3))
        scores = scores.reshape((1, -1, H, W))

        bottom = [None, None, None]
        bottom[0] = scores
        bottom[1] = bbox_deltas
        bottom[2] = [self.im_shape[1], self.im_shape[0], self.im_scale]

        top = [None, None]

        prop_layer.setup(bottom, top)
        prop_layer.forward(bottom, top)

        # Test proposed bounding boxes against PyCaffeProposalLayer
        # Because the two implementations handle ties in different ways,
        # a direct comparison is currently not possible.
        # assert len(np.setdiff1d(top[0], target_proposals)) / float(target_proposals.size) < 0.2
        # assert len(np.setdiff1d(top[1], target_scores)) / float(target_scores.size) < 0.02

        # Test the proposal target generation
        target_layer = PyCaffeProposalTargetLayer()

        t_bottom = [0, 1]
        # use target proposals from neon RPN
        t_bottom[0] = target_proposals
        # convert format of gt_boxes from (num_classes, 4) to (num_gt_boxes, 5)
        # concat the boxes and the classes and clip to num_gt_boxes and pass it in 
        t_bottom[1] = np.hstack((self.gt_boxes.get(),
                                 self.gt_classes.get()))[:self.num_gt_boxes.get()[0][0]]

        t_top = [None, None, None, None, None]

        target_layer.setup(t_bottom, t_top, keep_inds)
        target_layer.forward(t_bottom, t_top)

        frcn_bbtargets_reference = np.zeros(frcn_bbtargets.shape, dtype=np.float32)
        frcn_bbmask_reference = np.zeros(frcn_bbmask.shape, dtype=np.float32)

        frcn_bbtargets_reference[:t_top[2].shape[0]] = t_top[2]
        frcn_bbmask_reference[:t_top[3].shape[0]] = t_top[3]

        # Test proposal layer targets against pycaffe layer
        assert (np.alltrue(t_top[1] == frcn_labels))  # target labels
        assert (np.allclose(frcn_bbtargets_reference, frcn_bbtargets, atol=0.01))  # target bbox
        assert (np.alltrue(frcn_bbmask_reference == frcn_bbmask))   # target bbox mask

    def get_top_N(self, scores, proposals, N):
        order = scores.ravel().argsort()[::-1]
        if N > 0:
            order = order[:N]
        proposals = proposals[order, :]
        scores = scores[order]

        return (scores, proposals)

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
        bbox_targets = np.zeros((self.num_rois, 4 * self.num_classes), 
                                dtype=np.float32)
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(labels > 0)[0]
        for ind in inds:
            l = labels[ind]
            start = 4 * l
            end = start + 4
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
            fg_inds = self.be.rng.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
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
            bg_inds = self.be.rng.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
        elif bg_inds.size > 0:
            bg_inds = bg_inds[:bg_rois_per_this_image]

        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)
        return keep_inds, fg_rois_per_this_image
