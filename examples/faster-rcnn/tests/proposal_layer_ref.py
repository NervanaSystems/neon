# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import print_function
from builtins import object

import numpy as np

from generate_anchors import generate_anchors
from util import bbox_transform_inv, clip_boxes, nms

DEBUG = False


class PyCaffeProposalLayer(object):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top, pre_nms_topN=12000, post_nms_topN=2000,
              nms_thresh=0.7, min_size=16):

        self._feat_stride = 16
        anchor_scales = (8, 16, 32)
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        self.pre_nms_topN = pre_nms_topN
        self.post_nms_topN = post_nms_topN
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if DEBUG:
            print('feat_stride: {}'.format(self._feat_stride))
            print('anchors:')
            print(self._anchors)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #   top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].shape[0] == 1, \
            'Only single item batches are supported'

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0][:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1]
        im_info = [float(x.get()) for x in bottom[2]]

        if DEBUG:
            print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print('scale: {}'.format(im_info[2]))

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print('score map size: {}'.format(scores.shape))

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]

        # Generate anchors in same order as we do in neon for unit testing
        # anchors = self._anchors.reshape((1, A, 4)) + \
        #          shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = self._anchors.reshape((1, A, 4)).transpose((1, 0, 2)) + \
            shifts.reshape((1, K, 4))

        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        # bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Re-order proposals to match neon for unit testing
        # bbox_deltas = bbox_deltas.reshape((38, 50, 9, 4)).transpose((2, 0, 1, 3)).reshape((-1,4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        # scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Also re-order scores
        # scores = scores.reshape((38, 50, 9, 1)).transpose((2, 0, 1, 3)).reshape((-1, 1))

        scores = scores.reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, self.min_size * im_info[2])

        proposals = proposals[keep, :]
        scores = scores[keep]

        if DEBUG:
            print("(CAFFE) len(keep) before nms: {}".format(len(keep)))

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if self.pre_nms_topN > 0:
            order = order[:self.pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        if DEBUG:
            print("(CAFFE) len(proposals) after get_top_N: {}".format(len(proposals)))

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), self.nms_thresh)

        if DEBUG:
            print("(CAFFE) len(keep) before clipping: {}".format(len(keep)))

        if self.post_nms_topN > 0:
            keep = keep[:self.post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        if DEBUG:
            print("(CAFFE) len(keep) after nms: {}".format(len(keep)))

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0] = blob
        top[1] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
