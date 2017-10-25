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
Utility functions for Faster-RCNN model. Includes methods for:
- Defining and loading VGG weights
- Non-max suppression
- Bounding box calculations

Reference:
    "Faster R-CNN"
    https://arxiv.org/pdf/1506.01497
    https://github.com/rbgirshick/py-faster-rcnn

"""
from __future__ import division
from __future__ import print_function
from builtins import zip
import numpy as np
import os

from neon.util.persist import load_obj
from neon.data.datasets import Dataset
from neon.initializers import Constant, Xavier
from neon.transforms import Rectlin
from neon.layers import Conv, Pooling

BB_XMIN_IDX = 0
BB_YMIN_IDX = 1
BB_XMAX_IDX = 2
BB_YMAX_IDX = 3
FRCN_EPS = 1.0

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
FRCN_PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717])


def add_vgg_layers():

    # setup layers
    init1_vgg = Xavier(local=True)
    relu = Rectlin()

    conv_params = {'strides': 1,
                   'padding': 1,
                   'init': init1_vgg,
                   'bias': Constant(0),
                   'activation': relu}

    # Set up the model layers
    vgg_layers = []

    # set up 3x3 conv stacks with different feature map sizes
    vgg_layers.append(Conv((3, 3, 64), name="skip", **conv_params))
    vgg_layers.append(Conv((3, 3, 64), name="skip", **conv_params))
    vgg_layers.append(Pooling(2, strides=2))
    vgg_layers.append(Conv((3, 3, 128), name="skip", **conv_params))
    vgg_layers.append(Conv((3, 3, 128), name="skip", **conv_params))
    vgg_layers.append(Pooling(2, strides=2))
    vgg_layers.append(Conv((3, 3, 256), **conv_params))
    vgg_layers.append(Conv((3, 3, 256), **conv_params))
    vgg_layers.append(Conv((3, 3, 256), **conv_params))
    vgg_layers.append(Pooling(2, strides=2))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Pooling(2, strides=2))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    # not used after this layer
    # vgg_layers.append(Pooling(2, strides=2))
    # vgg_layers.append(Affine(nout=4096, init=initfc, bias=Constant(0), activation=relu))
    # vgg_layers.append(Dropout(keep=0.5))
    # vgg_layers.append(Affine(nout=4096, init=initfc, bias=Constant(0), activation=relu))
    # vgg_layers.append(Dropout(keep=0.5))
    # vgg_layers.append(Affine(nout=1000, init=initfc, bias=Constant(0), activation=Softmax()))

    return vgg_layers


def scale_bbreg_weights(model, means, stds, num_classes):
    means = np.array(num_classes * means)
    stds = np.array(num_classes * stds)

    means_be = model.be.array(means)
    stds_be = model.be.array(stds)
    model.layers.layers[2].layers[1].layers[1].layers[-3].W[:] = \
        model.layers.layers[2].layers[1].layers[1].layers[-3].W * stds_be
    model.layers.layers[2].layers[1].layers[1].layers[-2].W[:] = \
        model.layers.layers[2].layers[1].layers[1].layers[-2].W * stds_be + means_be
    return model


def load_vgg_all_weights(model, path):
    # load a pre-trained VGG16 from Neon model zoo to the local
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/'
    filename = 'VGG_D_fused_conv_bias.p'
    size = 553440655

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    print('De-serializing the pre-trained VGG16 model...')
    pdict = load_obj(filepath)

    param_layers = [l for l in model.layers.layers[0].layers]
    param_dict_list = pdict['model']['config']['layers']

    i = 0
    for layer, ps in zip(param_layers, param_dict_list):
        # finished loading param_dict_list[00 - 29] and param_layers[00-29]
        if i == 30:
            break
        layer.load_weights(ps, load_states=False)
        i += 1
        print(layer.name + " <-- " + ps['config']['name'])

    # to load the fc6 and fc7 from caffe into neon fc layers after ROI pooling
    neon_fc_layers = model.layers.layers[2].layers[1].layers[0].layers[2:5] +\
        model.layers.layers[2].layers[1].layers[0].layers[6:9]
    vgg_fc_layers = param_dict_list[31:34] + param_dict_list[35:38]

    for layer, ps in zip(neon_fc_layers, vgg_fc_layers):
        layer.load_weights(ps, load_states=True)
        print(layer.name + " <-- " + ps['config']['name'])


def load_vgg_weights(model, path):
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/'
    filename = 'VGG_D_Conv_fused_conv_bias.p'
    size = 58867537

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    print('De-serializing the pre-trained VGG16 model...')
    pdict = load_obj(filepath)

    param_layers = [l for l in model.layers.layers[0].layers]
    param_dict_list = pdict['model']['config']['layers']

    i = 0
    for layer, ps in zip(param_layers, param_dict_list):
        # finished loading param_dict_list[00 - 29] and param_layers[00-29]
        if i == 30:
            break
        layer.load_weights(ps, load_states=False)
        i += 1
        print(layer.name + " <-- " + ps['config']['name'])


# Below utility functions were modified from:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def compute_targets(gt_bb, rp_bb):
    """
    Given ground truth bounding boxes and proposed boxes, compute the regresssion
    targets according to:

    t_x = (x_gt - x) / w
    t_y = (y_gt - y) / h
    t_w = log(w_gt / w)
    t_h = log(h_gt / h)

    where (x,y) are bounding box centers and (w,h) are the box dimensions
    """
    # calculate the region proposal centers and width/height
    (x, y, w, h) = _get_xywh(rp_bb)
    (x_gt, y_gt, w_gt, h_gt) = _get_xywh(gt_bb)

    # the target will be how to adjust the bbox's center and width/height
    # note that the targets are generated based on the original RP, which has not
    # been scaled by the image resizing
    targets_dx = (x_gt - x) / w
    targets_dy = (y_gt - y) / h
    targets_dw = np.log(w_gt / w)
    targets_dh = np.log(h_gt / h)

    targets = np.concatenate((targets_dx[:, np.newaxis],
                              targets_dy[:, np.newaxis],
                              targets_dw[:, np.newaxis],
                              targets_dh[:, np.newaxis],
                              ), axis=1)
    return targets


def _get_xywh(bb):
    """
    Given bounding boxes with coordinates (x_min, y_min, x_max, y_max), transform to
    (x_center, y_center, width, height)
    """
    w = bb[:, BB_XMAX_IDX] - bb[:, BB_XMIN_IDX] + FRCN_EPS
    h = bb[:, BB_YMAX_IDX] - bb[:, BB_YMIN_IDX] + FRCN_EPS
    x = bb[:, BB_XMIN_IDX] + 0.5 * w
    y = bb[:, BB_YMIN_IDX] + 0.5 * h

    return (x, y, w, h)


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
