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

Reference:
    Striving for Simplicity: the All Convolutional Net `[Springenberg2014]`_
..  _[Springenberg2014]: http://arxiv.org/pdf/1412.6806.pdf
"""

from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost, DataTransform
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Normalizer
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.data import ImgMaster, ImageLoader

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--deconv', action='store_true',
                    help='save visualization data from deconvolution')
parser.add_argument('--loader_version', default='old', choices=['old', 'new'],
                    help='whether to use old dataloader (ImgMaster) or new (ImageLoader)')
args = parser.parse_args()


# hyperparameters
batch_size = 64

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 datatype=args.datatype)

# setup data provider
img_provider = ImgMaster if args.loader_version == 'old' else ImageLoader
img_set_options = dict(repo_dir=args.data_dir,
                       inner_size=224,
                       dtype=args.datatype,
                       subset_pct=100)
train = img_provider(set_name='train', **img_set_options)
test = img_provider(set_name='validation', do_transforms=False, **img_set_options)
train.init_batch_provider()
test.init_batch_provider()

relu = Rectlin()

init_uni = GlorotUniform()

# The parameters below are straight out of [Springenberg2014]
opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  schedule=Schedule(step_config=[10],
                                                    change=0.1),
                                  momentum_coef=0.9, wdecay=.0005)


# set up model layers
layers = []
layers.append(DataTransform(transform=Normalizer(divisor=128.)))

layers.append(Conv((11, 11, 96), init=init_uni, activation=relu, strides=4, padding=1))
layers.append(Conv((1, 1, 96),   init=init_uni, activation=relu, strides=1))
layers.append(Conv((3, 3, 96),   init=init_uni, activation=relu, strides=2,  padding=1))  # 54->27

layers.append(Conv((5, 5, 256),  init=init_uni, activation=relu, strides=1))              # 27->23
layers.append(Conv((1, 1, 256),  init=init_uni, activation=relu, strides=1))
layers.append(Conv((3, 3, 256),  init=init_uni, activation=relu, strides=2,  padding=1))  # 23->12

layers.append(Conv((3, 3, 384),  init=init_uni, activation=relu, strides=1,  padding=1))
layers.append(Conv((1, 1, 384),  init=init_uni, activation=relu, strides=1))
layers.append(Conv((3, 3, 384),  init=init_uni, activation=relu, strides=2,  padding=1))  # 12->6

layers.append(Dropout(keep=0.5))
layers.append(Conv((3, 3, 1024), init=init_uni, activation=relu, strides=1, padding=1))
layers.append(Conv((1, 1, 1024), init=init_uni, activation=relu, strides=1))
layers.append(Conv((1, 1, 1000), init=init_uni, activation=relu, strides=1))
layers.append(Pooling(6, op='avg'))

layers.append(Activation(Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

if args.model_file:
    import os
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    mlp.load_weights(args.model_file)

# configure callbacks
callbacks = Callbacks(mlp, train, eval_set=test, **args.callback_args)
if args.deconv:
    callbacks.add_deconv_callback(train, test)

mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

test.exit_batch_provider()
train.exit_batch_provider()
