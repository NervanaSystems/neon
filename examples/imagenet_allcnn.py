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
AllCNN style convnet on imagenet data.
"""

import sys
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.data import ImgMaster

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--deconv', action='store_true', help='save visualization data from deconvolution')
parser.add_argument('--model_file', help='load model from pkl file')
args = parser.parse_args()


# they used 64 samples batch
# hyperparameters
batch_size = 64

#450 000 iterations, 1.2 mil / 64 batchsize
num_epochs = 25

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype)

try:
    train = ImgMaster(repo_dir=args.data_dir, inner_size=224, set_name='train')
    valid_set = ImgMaster(repo_dir=args.data_dir, inner_size=224, set_name='validation',
                    do_transforms=False)
except (OSError, IOError, ValueError) as err:
    print err
    sys.exit(0)

train.init_batch_provider()
valid_set.init_batch_provider()

relu = Rectlin()

init_uni = GlorotUniform()

# These parameters below are straight out of the paper
opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  schedule=Schedule(step_config=[10],
                                                    change=0.1),
                                  momentum_coef=0.9, wdecay=.0005)


# set up model layers
layers = []
layers.append(Conv((11,11,96), init=init_uni, activation=relu, strides=4, pad=5))
layers.append(Conv((1,1,96), init=init_uni, activation=relu, strides=1))
layers.append(Conv((3,3,96), init=init_uni, activation=relu, strides=2, pad=1))
layers.append(Conv((5,5,256), init=init_uni, activation=relu, strides=1))
layers.append(Conv((1,1,256), init=init_uni, activation=relu, strides=1))
layers.append(Conv((3,3,256), init=init_uni, activation=relu, strides=2, pad=1))
layers.append(Conv((3,3,384), init=init_uni, activation=relu, strides=1, pad=1))
layers.append(Conv((1,1,384), init=init_uni, activation=relu, strides=1))
layers.append(Conv((3,3,384), init=init_uni, activation=relu, strides=2, pad=1))
layers.append(Dropout(keep=0.5))
layers.append(Conv((3,3,1024), init=init_uni, activation=relu, strides=1, pad=1))
layers.append(Conv((1,1,1024), init=init_uni, activation=relu, strides=1))
layers.append(Conv((1,1,1000), init=init_uni, activation=relu, strides=1))
layers.append(Pooling(6, op='avg'))

layers.append(Activation(Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

if args.model_file:
    import os
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    mlp.load_weights(args.model_file)

# configure callbacks
callbacks = Callbacks(mlp, train, output_file=args.output_file)

if args.deconv:
    callbacks.add_deconv_callback(train, valid_set, args.epochs)

if args.save_path:
    checkpoint_schedule = range(1, args.epochs)
    callbacks.add_serialize_callback(checkpoint_schedule, args.save_path, history=25)

callbacks.add_serialize_callback(1, './IMAGENET_cnn.pkl', history=35)
callbacks.add_guided_callback(train, valid_set, 1)

mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

valid_set.exit_batch_provider()
train.exit_batch_provider()
