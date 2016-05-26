from neon.layers.layer import Layer
from generate_anchors import generate_all_anchors
from py_cpu_nms import py_cpu_nms as nms
from pycaffe_proposal_layer import PyCaffeProposalLayer
import numpy as np


class ProposalLayer(Layer):
    """
    Proposal layer takes as input:
    (1) output of cls_score
    (2) output of bbox regression

    Converts that to an output ROIs
    (5, um_ROIs) <- [batch_size, x_min, y_min, x_max, y_max]

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

    def __init__(self, rpn_layers, global_buffers, num_rois=2000,
                 pre_nms_N=12000, nms_thresh=0.7, min_bbox_size=16, name=None):
        """
        Arguments:
            rpn_layers (list): References to the RPN layers: [RPN_1x1_obj, RPN_1x1_bbox]
            target_buffers (tuple): Target buffers for training fast-rcnn: (class, bbox regression)
            num_rois (int, optional): Number of ROIs to propose (default: 2000)
            pre_nms_N (int, optional): Number of ROIs to retain before using NMS (default: 12000)
            nms_thresh (float, optional): Threshold for non-maximum supression (default: 0.7)
            min_bbox_size (integer, optional): Minimize bboxes side length (default: 16)
            name (string, optional): Name of layer (default: None)
        """
        super(ProposalLayer, self).__init__(name)

        self.rpn_obj, self.rpn_bbox = rpn_layers
        self.num_rois = num_rois
        self.pre_nms_N = pre_nms_N
        self.nms_thresh = nms_thresh
        self.min_bbox_size = min_bbox_size

        # set references to dataset object buffers
        self.target_buffers = global_buffers['target_buffers']
        self.im_shape, self.im_scale = global_buffers['img_info']
        self.gt_boxes, self.num_gt_boxes = global_buffers['gt_boxes']
        self._conv_size, self._scale = global_buffers['conv_config']

        # generate anchors and load onto device
        # self._anchors has shape (KHW, 4)
        self._anchors = generate_all_anchors(self._conv_size, self._scale)
        self._dev_anchors = self.be.array(self._anchors)
        self._num_anchors = self._anchors.shape[0]

    def configure(self, in_obj):

        # set out_shape as the ROI shape
        self.out_shape = ((5, self.num_rois))
        self.in_shape = in_obj.out_shape
        return (in_obj, self)

    def allocate(self):
        super(ProposalLayer, self).allocate()

        # buffer to store transformed proposals
        self.dev_proposals = self.be.zeros((self._num_anchors, 4))
        self._proposals = self.be.zeros((5, self.num_rois))

        # buffer to store scores
        self.dev_scores = self.be.zeros((self._num_anchors, 1))

        # buffer to store proposals after they have sorted
        # and filtered with NMS.
        # Note: The buffer has shape (num_ROIs, 5), where each column
        # is (0, x_min, y_min, x_max, y_max)
        # This format is designed to be compatible with the
        # roi-pooling layer fprop code from Fast-RCNN.
        self.dev_proposals_filtered = self.be.zeros((self.num_rois, 5))

    def fprop(self, inputs, inference=False):
        """
        fprop function that does no proposal filtering
        """
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
        self._bbox_transform_inv(self._dev_anchors, bbox_deltas, output=self.dev_proposals)

        # 2. clip predicted boxes to image
        self._clip_boxes(self.dev_proposals, self.im_shape)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_scale)
        keep = self._filter_boxes(self.dev_proposals,
                                  self.min_bbox_size * float(self.im_scale.get()))
        # we set the score of those we discard to -1
        # the result is stored in this layers' own buffer.
        self.dev_scores[:] = (scores * keep) - (1 - keep)

        # -- TODO: move compute to device -- #
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        scores = self.dev_scores.get()
        proposals = self.dev_proposals.get()
        (scores, proposals) = self.get_top_N(scores, proposals, self.pre_nms_N)

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), self.nms_thresh)
        keep = keep[:self.num_rois]
        proposals = proposals[keep, :]
        scores = scores[keep]
        # -- END TODO -- #

        # load proposals back to device
        num_proposals = proposals.shape[0]
        proposals = np.hstack([np.zeros((num_proposals, 1)), proposals])
        proposals = np.ascontiguousarray(proposals, dtype=np.float32)
        self.dev_proposals_filtered[:num_proposals, :] = proposals

        if True:
            self._reference_test(proposals, scores)

        return (inputs, self.dev_proposals_filtered.T)

    def _reference_test(self, target_proposals, target_scores):
        layer = PyCaffeProposalLayer()
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

        bottom = [0, 1, 2]
        bottom[0] = scores
        bottom[1] = bbox_deltas
        bottom[2] = [self.im_shape[1], self.im_shape[0], self.im_scale]

        top = [0, 1]
        top[0] = None
        top[1] = None

        layer.setup(bottom, top)
        layer.forward(bottom, top)

    def get_top_N(self, scores, proposals, N):
        order = scores.ravel().argsort()[::-1]
        if N > 0:
            order = order[:N]
        proposals = proposals[order, :]
        scores = scores[order]

        return (scores, proposals)

    def bprop(self, errors):
        pass

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
