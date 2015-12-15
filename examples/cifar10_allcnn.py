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
AllCNN style convnet on CIFAR10 data.

Reference:
    Striving for Simplicity: the All Convolutional Net `[Springenberg2015]`_
..  _[Springenber2015]: http://arxiv.org/pdf/1412.6806.pdf
"""

from neon.initializers import Gaussian
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import DataIterator, load_cifar10
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument("--learning_rate", default=0.05, help="initial learning rate")
parser.add_argument("--weight_decay", default=0.001, help="weight decay")
parser.add_argument('--deconv', action='store_true',
                    help='save visualization data from deconvolution')
args = parser.parse_args()

# hyperparameters
num_epochs = args.epochs

(X_train, y_train), (X_test, y_test), nclass = load_cifar10(path=args.data_dir,
                                                            normalize=False,
                                                            contrast_normalize=True,
                                                            whiten=True)

# really 10 classes, pad to nearest power of 2 to match conv output
train_set = DataIterator(X_train, y_train, nclass=16, lshape=(3, 32, 32))
valid_set = DataIterator(X_test, y_test, nclass=16, lshape=(3, 32, 32))

init_uni = Gaussian(scale=0.05)
opt_gdm = GradientDescentMomentum(learning_rate=float(args.learning_rate), momentum_coef=0.9,
                                  wdecay=float(args.weight_decay),
                                  schedule=Schedule(step_config=[200, 250, 300], change=0.1))

relu = Rectlin()
conv = dict(init=init_uni, batch_norm=False, activation=relu)
convp1 = dict(init=init_uni, batch_norm=False, activation=relu, padding=1)
convp1s2 = dict(init=init_uni, batch_norm=False, activation=relu, padding=1, strides=2)

layers = [Dropout(keep=.8),
          Conv((3, 3, 96), **convp1),
          Conv((3, 3, 96), **convp1),
          Conv((3, 3, 96), **convp1s2),
          Dropout(keep=.5),
          Conv((3, 3, 192), **convp1),
          Conv((3, 3, 192), **convp1),
          Conv((3, 3, 192), **convp1s2),
          Dropout(keep=.5),
          Conv((3, 3, 192), **convp1),
          Conv((1, 1, 192), **conv),
          Conv((1, 1, 16), **conv),
          Pooling(8, op="avg"),
          Activation(Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

if args.model_file:
    import os
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    mlp.load_params(args.model_file)

# configure callbacks
callbacks = Callbacks(mlp, train_set, eval_set=valid_set, **args.callback_args)

if args.deconv:
    callbacks.add_deconv_callback(train_set, valid_set)


mlp.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
print('Misclassification error = %.1f%%' % (mlp.eval(valid_set, metric=Misclassification())*100))
