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
Utility functions for Fast-RCNN model and demo.
"""
import os
import cPickle
import numpy as np

from neon.data.datasets import Dataset
from neon.initializers import Gaussian, Constant, Xavier
from neon.transforms import Rectlin, Softmax, Identity
from neon.models import Model
from neon.layers import Conv, Pooling, Affine, Dropout, RoiPooling, BranchNode, Tree
from neon.util.persist import load_obj

from voc_eval import voc_eval

# how the indices in the bounding box array map to cooridnates
BB_XMIN_IDX = 0
BB_YMIN_IDX = 1
BB_XMAX_IDX = 2
BB_YMAX_IDX = 3
FRCN_EPS = 1e-14


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


def create_frcn_model(frcn_fine_tune=False):

    b1 = BranchNode(name="b1")

    imagenet_layers = add_vgg_layers()
    HW = (7, 7)

    frcn_layers = [
        RoiPooling(
            layers=imagenet_layers, HW=HW, bprop_enabled=frcn_fine_tune),
        Affine(nout=4096, init=Gaussian(scale=0.005),
               bias=Constant(.1), activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=4096, init=Gaussian(scale=0.005),
               bias=Constant(.1), activation=Rectlin()),
        Dropout(keep=0.5),
        b1,
        Affine(nout=21, init=Gaussian(scale=0.01),
               bias=Constant(0), activation=Softmax())
    ]
    bb_layers = [
        b1,
        Affine(nout=84, init=Gaussian(scale=0.001),
               bias=Constant(0), activation=Identity())
    ]

    return Model(layers=Tree([frcn_layers, bb_layers]))


def scale_bbreg_weights(model, means, stds):
    means_be = model.be.array(means)
    stds_be = model.be.array(stds)
    model.layers.layers[1].layers[1].W[:] = \
        model.layers.layers[1].layers[1].W * stds_be
    model.layers.layers[1].layers[2].W[:] = \
        model.layers.layers[1].layers[2].W * stds_be + means_be
    return model


def load_vgg_weights(model, path):
    # load a pre-trained VGG16 from Neon model zoo to the local
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/'
    filename = 'VGG_D.p'
    size = 554227541

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    print 'De-serializing the pre-trained VGG16 model...'
    pdict = load_obj(filepath)

    param_layers = [l for l in model.layers.layers[0].layers[0].layers]
    param_dict_list = pdict['model']['config']['layers']
    i = 0
    for layer, ps in zip(param_layers, param_dict_list):
        i += 1
        print layer.name, ps['config']['name']
        layer.load_weights(ps, load_states=True)
        if i == 43:
            break


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
