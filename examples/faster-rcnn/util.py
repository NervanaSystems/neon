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
