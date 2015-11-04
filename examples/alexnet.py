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
Runs one epoch of Alexnet on imagenet data.
"""

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, DropoutBinary, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImgMaster
from neon.callbacks.callbacks import Callbacks

# For running complete alexnet
# alexnet.py -e 90 -val 1 -s <save-path> -w <path-to-saved-batches>
# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

img_master_options = dict(repo_dir=args.data_dir,
                          inner_size=224,
                          dtype=args.datatype,
                          subset_pct=100)

train = ImgMaster(set_name='train', **img_master_options)
test = ImgMaster(set_name='validation', do_transforms=False, **img_master_options)

train.init_batch_provider()
test.init_batch_provider()

init1 = Gaussian(scale=0.01)
init1b = Gaussian(scale=0.03)
relu = Rectlin()

# drop LR by 1/250**(1/3) at beginning of epochs 23, 45, 66
weight_sched = Schedule([22, 44, 65], (1/250.)**(1/3.))
opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched)

# drop bias weights by 1/10 at the beginning of epoch 45.
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=Schedule([44], 0.1))

# Set up the model layers
layers = [Conv((11, 11, 64), padding=3, strides=4, init=init1, bias=Constant(0), activation=relu),
          Pooling(3, strides=2),
          Conv((5, 5, 192), padding=2, init=init1, bias=Constant(1), activation=relu),
          Pooling(3, strides=2),
          Conv((3, 3, 384), padding=1, init=init1b, bias=Constant(0), activation=relu),
          Conv((3, 3, 256), padding=1, init=init1b, bias=Constant(1), activation=relu),
          Conv((3, 3, 256), padding=1, init=init1b, bias=Constant(1), activation=relu),
          Pooling(3, strides=2),
          Affine(nout=4096, init=init1, bias=Constant(1), activation=relu),
          DropoutBinary(keep=0.5),
          Affine(nout=4096, init=init1, bias=Constant(1), activation=relu),
          DropoutBinary(keep=0.5),
          Affine(nout=1000, init=init1, bias=Constant(-7), activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

model = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(model, train, args, eval_set=test, metric=TopKMisclassification(k=5))

try:
    model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
finally:
    test.exit_batch_provider()
    train.exit_batch_provider()
