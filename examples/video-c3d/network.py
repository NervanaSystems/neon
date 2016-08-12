# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, Affine, GeneralizedCost
from neon.models import Model
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti


def create_network():
    # weight initialization
    g1 = Gaussian(scale=0.01)
    g5 = Gaussian(scale=0.005)
    c0 = Constant(0)
    c1 = Constant(1)

    # model initialization
    padding = {'pad_d': 1, 'pad_h': 1, 'pad_w': 1}
    strides = {'str_d': 2, 'str_h': 2, 'str_w': 2}
    layers = [
        Conv((3, 3, 3, 64), padding=padding, init=g1, bias=c0, activation=Rectlin()),
        Pooling((1, 2, 2), strides={'str_d': 1, 'str_h': 2, 'str_w': 2}),
        Conv((3, 3, 3, 128), padding=padding, init=g1, bias=c1, activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Conv((3, 3, 3, 256), padding=padding, init=g1, bias=c1, activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Conv((3, 3, 3, 256), padding=padding, init=g1, bias=c1, activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Conv((3, 3, 3, 256), padding=padding, init=g1, bias=c1, activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Affine(nout=2048, init=g5, bias=c1, activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=2048, init=g5, bias=c1, activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=101, init=g1, bias=c0, activation=Softmax())
    ]
    return Model(layers=layers), GeneralizedCost(costfunc=CrossEntropyMulti())
