# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
Utility functions for Fast-RCNN example and demo.
"""
import os
import numpy as np
from neon.data.datasets import Dataset
from neon.initializers import Gaussian, Constant, Xavier, GlorotUniform
from neon.transforms import Rectlin, Softmax, Identity
from neon.models import Model
from neon.layers import Conv, Pooling, Affine, Dropout, RoiPooling, BranchNode, Tree
from neon.util.persist import load_obj


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
    vgg_layers.append(Conv((3, 3, 64), **conv_params))
    vgg_layers.append(Conv((3, 3, 64), **conv_params))
    vgg_layers.append(Pooling(2, strides=2))
    vgg_layers.append(Conv((3, 3, 128), **conv_params))
    vgg_layers.append(Conv((3, 3, 128), **conv_params))
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

def load_vgg_weights(model, weight_path):
    pdict = load_obj(weight_path)

    param_layers = [l for l in model.layers_to_optimize]
    param_dict_list = pdict['layer_params_states']
    i = 0
    for layer, ps in zip(param_layers, param_dict_list):
        i = i+1
        print i, layer.name
        layer.set_params(ps)
        if i == 26:
            break

def nonmaximum_suppression(detections, scores, thre):
    """
    apply non-maximum suppression (for each class indepdently) that rejects
    a region if it has an intersection-over-union (IoU) overlap with a higher
    scoring selected region larger than a learned threshold
    """
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]

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
        iou = w * h
        overlap = iou / (areas[i] + areas[order[1:]] - iou)

        inds = np.where(overlap <= thre)[0]
        order = order[inds + 1]

    return keep


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in range(1, num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nonmaximum_suppression(dets[:, :4], dets[:, -1], thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes
