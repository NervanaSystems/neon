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
Example highlighting the ability to mix different optimizers in different
layers, or different components of the same layer.

Usage:

    python examples/multi_optimizer.py

"""

from neon.data import MNIST
from neon.initializers import Gaussian, Constant
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, RMSProp
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

dataset = MNIST(path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

# weight initialization
init_norm = Gaussian(loc=0.0, scale=0.01)

# initialize model
layers = []
layers.append(Affine(nout=100, init=init_norm, bias=Constant(0),
                     activation=Rectlin()))
layers.append(Affine(nout=10, init=init_norm, bias=Constant(0),
                     activation=Logistic(shortcut=True),
                     name='special_linear'))

cost = GeneralizedCost(costfunc=CrossEntropyBinary())
mlp = Model(layers=layers)

# fit and validate
optimizer_one = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)
optimizer_two = RMSProp()

# all bias layers and the last linear layer will use
# optimizer_two. all other layers will use optimizer_one.
opt = MultiOptimizer({'default': optimizer_one,
                      'Bias': optimizer_two,
                      'special_linear': optimizer_two})

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)

mlp.fit(train_set, optimizer=opt, num_epochs=args.epochs,
        cost=cost, callbacks=callbacks)
