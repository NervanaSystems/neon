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
Simplified version of VGG model B:
 - adds batch normalization
 - removes scale jittering
 - removes convolutional inference
"""

from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Constant, GlorotUniform
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImgMaster
from neon.callbacks.callbacks import Callbacks

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
batch_size = 64
cost_scale = 1.

# setup backend
be = gen_backend(backend=args.backend,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 batch_size=batch_size,
                 default_dtype=args.datatype)

train = ImgMaster(repo_dir=args.data_dir,
                  inner_size=224,
                  set_name='train',
                  subset_pct=1,
                  dtype=args.datatype)
test = ImgMaster(repo_dir=args.data_dir,
                 inner_size=224,
                 set_name='validation',
                 subset_pct=50,
                 dtype=args.datatype,
                 do_transforms=False)

train.init_batch_provider()
test.init_batch_provider()

init1 = GlorotUniform()
relu = Rectlin()

# drop LR by 1/250**(1/3) at beginning of epochs 23, 45, 66
weight_sched = Schedule([22, 44, 65], (1/250.)**(1/3.))
opt_gdm = GradientDescentMomentum(1*0.01/cost_scale, 0.9, wdecay=0.0005,
                                  schedule=weight_sched,
                                  stochastic_round=args.rounding)

# drop bias weights by 1/10 at the beginning of epoch 45.
opt_biases = GradientDescentMomentum(1*0.02/cost_scale, 0.9,
                                     schedule=Schedule([44], 0.1),
                                     stochastic_round=args.rounding)

conv_params = {'strides': 1,
               'padding': 1,
               'init': init1,
               'batch_norm': True,
               'activation': relu}

conv_params_nobn = {'strides': 1,
                    'padding': 1,
                    'init': init1,
                    'bias': Constant(0),
                    'batch_norm': False,
                    'activation': relu}


# Set up the model layers
layers = []

# set up 3x3 conv stacks with different feature map sizes
VGG = 'B'

if VGG == 'B':
    for nofm in [64, 128, 256, 512, 512]:
        layers.append(Conv((3, 3, nofm), **conv_params))
        layers.append(Conv((3, 3, nofm), **conv_params))
        layers.append(Pooling(3, strides=2))
elif VGG == 'D':
    for nofm in [64, 128]:
        layers.append(Conv((3, 3, nofm), **conv_params))
        layers.append(Conv((3, 3, nofm), **conv_params))
        layers.append(Pooling(3, strides=2))
    for nofm in [256, 512, 512]:
        layers.append(Conv((3, 3, nofm), **conv_params))
        layers.append(Conv((3, 3, nofm), **conv_params))
        layers.append(Conv((3, 3, nofm), **conv_params))
        layers.append(Pooling(3, strides=2))
else:
    raise ValueError("Invalid specification for VGG model")

layers.append(Affine(nout=4096, init=init1, batch_norm=True, activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=4096, init=init1, batch_norm=True, activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=1000, init=init1, bias=Constant(0), activation=Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti(scale=cost_scale))

opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, train, eval_set=test, metric=TopKMisclassification(k=5),
                      **args.callback_args)

mlp.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

test.exit_batch_provider()
train.exit_batch_provider()
