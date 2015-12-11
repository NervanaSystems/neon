#!/usr/bin/env python
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
Simplified version of VGG model B, D, or E:
 - adds batch normalization
 - removes scale jittering
 - removes convolutional inference
"""

from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.backends import gen_backend
from neon.initializers import Constant, GlorotUniform
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

# hyperparameters
args.batch_size = 64
cost_scale = 1.
VGG = 'B'
use_batch_norm = True
biases = None if use_batch_norm else Constant(0)

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# initialize the data provider
img_set_options = dict(repo_dir=args.data_dir, inner_size=224, dtype=args.datatype, subset_pct=100)
train = ImageLoader(set_name='train', **img_set_options)
test = ImageLoader(set_name='validation', do_transforms=False, **img_set_options)

init1 = GlorotUniform()
relu = Rectlin()
common_params = dict(init=init1, activation=Rectlin(), batch_norm=use_batch_norm, bias=biases)
conv_params = dict(padding=1, **common_params)

# Set up the model layers, using 3x3 conv stacks with different feature map sizes
layers = []

for nofm in [64, 128, 256, 512, 512]:
    layers.append(Conv((3, 3, nofm), **conv_params))
    layers.append(Conv((3, 3, nofm), **conv_params))
    if nofm > 128:
        if VGG in ('D', 'E'):
            layers.append(Conv((3, 3, nofm), **conv_params))
        if VGG == 'E':
            layers.append(Conv((3, 3, nofm), **conv_params))
    layers.append(Pooling(3, strides=2))

layers.append(Affine(nout=4096, **common_params))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=4096, **common_params))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=1000, init=init1, bias=Constant(0), activation=Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti(scale=cost_scale))

mlp = Model(layers=layers)

# configure callbacks
valmetric = TopKMisclassification(k=5)
callbacks = Callbacks(mlp, train, eval_set=test, metric=valmetric, **args.callback_args)

# create learning rate schedules and optimizers
weight_sched = Schedule(range(14, 75, 15), 0.1)
opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched)
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=weight_sched)
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

mlp.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
