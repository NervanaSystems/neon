#!/usr/bin/env python
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
"""
GoogLeNet v1 Benchmark
https://github.com/soumith/convnet-benchmarks

./googlenet_v1.py
./googlenet_v1.py -d f16

Derived from full model found here:
https://github.com/NervanaSystems/ModelZoo/tree/master/ImageClassification/ILSVRC2012/Googlenet
"""

from neon import NervanaObject
from neon.util.argparser import NeonArgparser
from neon.initializers import Xavier
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, MergeBroadcast
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, CrossEntropyMulti
from neon.models import Model
from neon.data import ArrayIterator
import numpy as np

parser = NeonArgparser(__doc__)
args = parser.parse_args()

NervanaObject.be.enable_winograd = 4

# setup data provider
X_train = np.random.uniform(-1, 1, (128, 3 * 224 * 224))
y_train = np.random.randint(0, 999, (128, 1000))
train = ArrayIterator(X_train, y_train, nclass=1000, lshape=(3, 224, 224))

init1 = Xavier(local=False)
initx = Xavier(local=True)
relu = Rectlin()

common = dict(activation=relu, init=initx)
commonp1 = dict(activation=relu, init=initx, padding=1)
commonp2 = dict(activation=relu, init=initx, padding=2)
pool3s1p1 = dict(fshape=3, padding=1, strides=1)
pool3s2p1 = dict(fshape=3, padding=1, strides=2, op='max')


def inception(kvals):
    (p1, p2, p3, p4) = kvals

    branch1 = [Conv((1, 1, p1[0]), **common)]
    branch2 = [Conv((1, 1, p2[0]), **common), Conv((3, 3, p2[1]), **commonp1)]
    branch3 = [Conv((1, 1, p3[0]), **common), Conv((5, 5, p3[1]), **commonp2)]
    branch4 = [Pooling(op="max", **pool3s1p1), Conv((1, 1, p4[0]), **common)]
    return MergeBroadcast(layers=[branch1, branch2, branch3, branch4], merge="depth")

model = Model(layers=[
    Conv((7, 7, 64), padding=3, strides=2, **common),
    Pooling(**pool3s2p1),
    Conv((1, 1, 64), **common),
    Conv((3, 3, 192), **commonp1),
    Pooling(**pool3s2p1),
    inception([(64,), (96, 128), (16, 32), (32,)]),
    inception([(128,), (128, 192), (32, 96), (64,)]),
    Pooling(**pool3s2p1),
    inception([(192,), (96, 208), (16, 48), (64,)]),
    inception([(160,), (112, 224), (24, 64), (64,)]),
    inception([(128,), (128, 256), (24, 64), (64,)]),
    inception([(112,), (144, 288), (32, 64), (64,)]),
    inception([(256,), (160, 320), (32, 128), (128,)]),
    Pooling(**pool3s2p1),
    inception([(256,), (160, 320), (32, 128), (128,)]),
    inception([(384,), (192, 384), (48, 128), (128,)]),
    Pooling(fshape=7, strides=1, op="avg"),
    Affine(nout=1000, init=init1)])

weight_sched = Schedule([22, 44, 65], (1 / 250.)**(1 / 3.))
opt_gdm = GradientDescentMomentum(0.01, 0.0, wdecay=0.0005, schedule=weight_sched)
opt = MultiOptimizer({'default': opt_gdm})
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.benchmark(train, cost=cost, optimizer=opt, niterations=10, nskip=1)
