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
from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, GeneralizedCost, Activation, Affine
from neon.layers import MergeSum, SkipNode, BatchNorm
from neon.models import Model
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti


def conv_params(fsize, nfm, stride=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=batch_norm)


def module_s1(nfm, first=False):
    '''
    non-strided
    '''
    sidepath = Conv(**conv_params(1, nfm * 4, 1, False, False)) if first else SkipNode()
    mainpath = [] if first else [BatchNorm(), Activation(Rectlin())]
    mainpath.append(Conv(**conv_params(1, nfm)))
    mainpath.append(Conv(**conv_params(3, nfm)))
    mainpath.append(Conv(**conv_params(1, nfm * 4, relu=False, batch_norm=False)))

    return MergeSum([sidepath, mainpath])


def module_s2(nfm):
    '''
    strided
    '''
    module = [BatchNorm(), Activation(Rectlin())]
    mainpath = [Conv(**conv_params(1, nfm, stride=2)),
                Conv(**conv_params(3, nfm)),
                Conv(**conv_params(1, nfm * 4, relu=False, batch_norm=False))]
    sidepath = [Conv(**conv_params(1, nfm * 4, stride=2, relu=False, batch_norm=False))]
    module.append(MergeSum([sidepath, mainpath]))
    return module


def create_network(stage_depth):
    # Structure of the deep residual part of the network:
    # stage_depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
    nfms = [2**(stage + 4) for stage in sorted(list(range(3)) * stage_depth)]
    strides = [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

    # Now construct the network
    layers = [Conv(**conv_params(3, 16))]
    layers.append(module_s1(nfms[0], True))

    for nfm, stride in zip(nfms[1:], strides):
        res_module = module_s1(nfm) if stride == 1 else module_s2(nfm)
        layers.append(res_module)
    layers.append(BatchNorm())
    layers.append(Activation(Rectlin()))
    layers.append(Pooling('all', op='avg'))
    layers.append(Affine(10, init=Kaiming(local=False), batch_norm=True, activation=Softmax()))

    return Model(layers=layers), GeneralizedCost(costfunc=CrossEntropyMulti())
