# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object

import numpy as np
import numpy.random as npr

from generate_anchors import generate_anchors
from util import bbox_transform

DEBUG = False


class AnchorTargetLayer(object):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        # layer_params = yaml.load(self.param_str_)

        layer_params = dict()
        layer_params['feat_stride'] = 16
        layer_params['scales'] = (8, 16, 32)
        layer_params['allowed_border'] = 0
        self.RPN_NEGATIVE_OVERLAP = 0.3
        self.RPN_POSITIVE_OVERLAP = 0.7
        self.RPN_FG_FRACTION = 0.5
        self.RPN_BATCHSIZE = 256
        self.EPS = 1e-14
        self.RPN_BBOX_INSIDE_WEIGHTS = 1
        self.RPN_POSITIVE_WEIGHT = -1

        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        if DEBUG:
            # print 'anchors:'
            # print self._anchors
            # print 'anchor shapes:'
            # print np.hstack((
            #     self._anchors[:, 2::4] - self._anchors[:, 0::4],
            #     self._anchors[:, 3::4] - self._anchors[:, 1::4],
            # ))
            self._counts = self.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)
        # height, width = bottom[0].data.shape[-2:]
        # if DEBUG:
        #     print 'AnchorTargetLayer: height', height, 'width', width

        # labels
        # top[0].reshape(1, 1, A * height, width)
        # # bbox_targets
        # top[1].reshape(1, A * 4, height, width)
        # # bbox_inside_weights
        # top[2].reshape(1, A * 4, height, width)
        # # bbox_outside_weights
        # top[3].reshape(1, A * 4, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        # assert bottom[0].data.shape[0] == 1, \
        #     'Only single item batches are supported'
        # map of shape (..., H, W)
        height = bottom[0].shape[0]
        width = bottom[0].shape[1]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1]
        # im_info
        im_info = bottom[2]

        if DEBUG:
            print('')
            print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print('scale: {}'.format(im_info[2]))
            print('height, width: ({}, {})'.format(height, width))
            print('rpn: gt_boxes.shape', gt_boxes.shape)
            print('rpn: gt_boxes', gt_boxes)

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[0] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[1] + self._allowed_border)    # height
        )[0]

        if DEBUG:
            print('total_anchors', total_anchors)
            print('inds_inside', len(inds_inside))

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print('anchors.shape', anchors.shape)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = calculate_bb_overlap(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.RPN_POSITIVE_OVERLAP] = 1

        if DEBUG:
            print('rpn: max max_overlap', np.max(max_overlaps))
            print('rpn: num_positive', np.sum(labels == 1))
            print('rpn: num_negative', np.sum(labels == 0))

        # if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        #     # assign bg labels last so that negative labels can clobber positives
        #     labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(self.RPN_FG_FRACTION * self.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(self.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if self.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((self.RPN_POSITIVE_WEIGHT > 0) &
                    (self.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = self.RPN_POSITIVE_WEIGHT / np.sum(labels == 1)
            negative_weights = (1.0 - self.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0)
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print('means:')
            print(means)
            print('stdevs:')
            print(stds)

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print('rpn: max max_overlap', np.max(max_overlaps))
            print('rpn: num_positive', np.sum(labels == 1))
            print('rpn: num_negative', np.sum(labels == 0))
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print('rpn: num_positive avg', self._fg_sum / self._count)
            print('rpn: num_negative avg', self._bg_sum / self._count)

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1] = bbox_targets
        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    # assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def calculate_bb_overlap(rp, gt):
    """
    Returns a matrix of overlaps between every possible pair of the two provided
    bounding box lists.

    Arguments:
        rp (list): an array of region proposals, shape (R, 4)
        gt (list): an array of ground truth ROIs, shape (G, 4)

    Outputs:
        overlaps: a matrix of overlaps between 2 list, shape (R, G)
    """
    R = rp.shape[0]
    G = gt.shape[0]
    overlaps = np.zeros((R, G), dtype=np.float32)

    for g in range(G):
        gt_box_area = float(
            (gt[g, 2] - gt[g, 0] + 1) *
            (gt[g, 3] - gt[g, 1] + 1)
        )
        for r in range(R):
            iw = float(
                min(rp[r, 2], gt[g, 2]) -
                max(rp[r, 0], gt[g, 0]) + 1
            )
            if iw > 0:
                ih = float(
                    min(rp[r, 3], gt[g, 3]) -
                    max(rp[r, 1], gt[g, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (rp[r, 2] - rp[r, 0] + 1) *
                        (rp[r, 3] - rp[r, 1] + 1) +
                        gt_box_area - iw * ih
                    )
                    overlaps[r, g] = iw * ih / ua
    return overlaps
