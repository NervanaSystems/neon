#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2017 Nervana Systems Inc.
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
Alexnet Benchmark
https://github.com/soumith/convnet-benchmarks

./alexnet.py
./alexnet.py -d f16
"""

import numpy as np

from neon import NervanaObject
from neon.benchmark import Benchmark
from neon.data import ArrayIterator
from neon.initializers import Gaussian
from neon.layers import Conv, Pooling, GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti
from neon.util.argparser import NeonArgparser

parser = NeonArgparser(__doc__)
args = parser.parse_args()

NervanaObject.be.enable_winograd = 4

# setup data provider
X_train = np.random.uniform(-1, 1, (128, 3 * 224 * 224))
y_train = np.random.randint(0, 999, (128, 1000))
train = ArrayIterator(X_train, y_train, nclass=1000, lshape=(3, 224, 224))

layers = [Conv((11, 11, 64), init=Gaussian(scale=0.01),
               activation=Rectlin(), padding=3, strides=4),
          Pooling(3, strides=2),
          Conv((5, 5, 192), init=Gaussian(scale=0.01), activation=Rectlin(), padding=2),
          Pooling(3, strides=2),
          Conv((3, 3, 384), init=Gaussian(scale=0.03), activation=Rectlin(), padding=1),
          Conv((3, 3, 256), init=Gaussian(scale=0.03), activation=Rectlin(), padding=1),
          Conv((3, 3, 256), init=Gaussian(scale=0.03), activation=Rectlin(), padding=1),
          Pooling(3, strides=2),
          Affine(nout=4096, init=Gaussian(scale=0.01), activation=Rectlin()),
          Affine(nout=4096, init=Gaussian(scale=0.01), activation=Rectlin()),
          Affine(nout=1000, init=Gaussian(scale=0.01), activation=Softmax())]

weight_sched = Schedule([22, 44, 65], (1 / 250.)**(1 / 3.))
opt_gdm = GradientDescentMomentum(0.01, 0.0, wdecay=0.0005, schedule=weight_sched)
opt = MultiOptimizer({'default': opt_gdm})

model = Model(layers=layers, optimizer=opt)

cost = GeneralizedCost(costfunc=CrossEntropyMulti())
model.initialize(train, cost=cost)

b = Benchmark(model)
print("Forward and backward")
res = b.time(train, niterations=5)
b.print_stats(res, nskip=2)
