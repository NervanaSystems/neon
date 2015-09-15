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
"""

from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import DataIterator, load_cifar10
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
batch_size = 128
num_epochs = args.epochs

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype)

(X_train, y_train), (X_test, y_test), nclass = load_cifar10(path=args.data_dir)

# really 10 classes, pad to nearest power of 2 to match conv output
train_set = DataIterator(X_train, y_train, nclass=16, lshape=(3, 32, 32))
valid_set = DataIterator(X_test, y_test, nclass=16, lshape=(3, 32, 32))

init_uni = GlorotUniform()
opt_gdm = GradientDescentMomentum(learning_rate=0.5,
                                  schedule=Schedule(step_config=[200, 250, 300],
                                                    change=0.1),
                                  momentum_coef=0.9, wdecay=.0001)
relu = Rectlin()
layers = []
layers.append(Dropout(keep=.8))
layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=relu))
layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=relu, pad=1))
layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=relu, pad=1, strides=2))
layers.append(Dropout(keep=.5))

layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=relu, pad=1))
layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=relu, pad=1))
layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=relu, pad=1, strides=2))
layers.append(Dropout(keep=.5))

layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=relu))
layers.append(Conv((1, 1, 192), init=init_uni, batch_norm=True, activation=relu))
layers.append(Conv((1, 1, 16), init=init_uni, activation=relu))

layers.append(Pooling(6, op="avg"))
layers.append(Activation(Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, train_set, output_file=args.output_file, valid_set=valid_set,
                      valid_freq=args.validation_freq, progress_bar=args.progress_bar)

mlp.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
print mlp.eval(valid_set, metric=Misclassification())
