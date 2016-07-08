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

from neon.initializers import Constant, Xavier
from neon.transforms import Rectlin
from neon.layers import Conv, Pooling
from neon.util.persist import load_obj
from neon.data.datasets import Dataset
from voc_eval import voc_eval
import numpy as np
import os
import cPickle

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


def load_vgg_weights(model, path):
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/'
    filename = 'VGG_D_Conv.p'
    size = 169645138

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    print 'De-serializing the pre-trained VGG16 model...'
    pdict = load_obj(filepath)

    param_layers = [l for l in model.layers.layers[0].layers]
    param_dict_list = pdict['model']['config']['layers']

    for layer, ps in zip(param_layers, param_dict_list):
        layer.load_weights(ps, load_states=True)
        print layer.name + " <-- " + ps['config']['name']

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


def run_voc_eval(annopath, imagesetfile, year, image_set, classes, output_dir):
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = 'voc_{}_{}_{}.txt'.format(
            year, image_set, cls)
        filepath = os.path.join(output_dir, filename)
        rec, prec, ap = voc_eval(filepath, annopath, imagesetfile, cls,
                                 output_dir, ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))


