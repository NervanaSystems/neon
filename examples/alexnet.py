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

import sys
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImgMaster
from neon.callbacks.callbacks import Callbacks, Callback

# For running complete alexnet
# alexnet.py -e 90 -val 1 -s <save-path> -w <path-to-saved-batches>
# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--model_file', help='load model from pkl file')
args = parser.parse_args()

# hyperparameters
batch_size = 128

# setup backend
be = gen_backend(backend=args.backend, rng_seed=args.rng_seed, device_id=args.device_id,
                 batch_size=batch_size, default_dtype=args.datatype)

try:
    train = ImgMaster(repo_dir=args.data_dir, inner_size=224, set_name='train')
    test = ImgMaster(repo_dir=args.data_dir, inner_size=224, set_name='validation',
                     do_transforms=False)
except (OSError, IOError, ValueError) as err:
    print err
    sys.exit(0)

train.init_batch_provider()
test.init_batch_provider()

init1 = Gaussian(scale=0.01)
init2 = Gaussian(scale=0.03)
relu = Rectlin()

# drop LR by 1/250**(1/3) at beginning of epochs 23, 45, 66
weight_sched = Schedule([22, 44, 65], (1/250.)**(1/3.))
opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched)

# drop bias weights by 1/10 at the beginning of epoch 45.
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=Schedule([44], 0.1))

# Set up the model layers
layers = []
layers.append(Conv((11, 11, 64), strides=4, pad=3, init=init1, bias=Constant(0), activation=relu))
layers.append(Pooling(3, strides=2))
layers.append(Conv((5, 5, 192), pad=2, init=init1, bias=Constant(1), activation=relu))
layers.append(Pooling(3, strides=2))
layers.append(Conv((3, 3, 384), pad=1, init=init2, bias=Constant(0), activation=relu))
layers.append(Conv((3, 3, 256), pad=1, init=init2, bias=Constant(1), activation=relu))
layers.append(Conv((3, 3, 256), pad=1, init=init2, bias=Constant(1), activation=relu))
layers.append(Pooling(3, strides=2))
layers.append(Affine(nout=4096, init=init1, bias=Constant(1), activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=4096, init=init1, bias=Constant(1), activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=1000, init=init1, bias=Constant(-7), activation=Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

mlp = Model(layers=layers)

if args.model_file:
    import os
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    mlp.load_weights(args.model_file)

# configure callbacks
callbacks = Callbacks(mlp, train, output_file=args.output_file)

if args.validation_freq:
    class TopKMetrics(Callback):
        def __init__(self, valid_set, epoch_freq=args.validation_freq):
            super(TopKMetrics, self).__init__(epoch_freq=epoch_freq)
            self.valid_set = valid_set

        def on_epoch_end(self, epoch):
            self.valid_set.reset()
            allmetrics = TopKMisclassification(k=5)
            stats = mlp.eval(self.valid_set, metric=allmetrics)
            print ", ".join(allmetrics.metric_names) + ": " + ", ".join(map(str, stats.flatten()))

    callbacks.add_callback(TopKMetrics(test))

if args.save_path:
    checkpoint_schedule = range(1, args.epochs)
    callbacks.add_serialize_callback(checkpoint_schedule, args.save_path, history=2)

mlp.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

test.exit_batch_provider()
train.exit_batch_provider()
