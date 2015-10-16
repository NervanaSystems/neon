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
MNIST example demonstrating the use of merge layers.
"""

from neon.data import DataIterator, load_mnist
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, MergeSum
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
num_epochs = args.epochs

(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
train_set = DataIterator([X_train, X_train], y_train, nclass=nclass)
valid_set = DataIterator([X_test, X_test], y_test, nclass=nclass)

# weight initialization
init_norm = Gaussian(loc=0.0, scale=0.01)

# initialize model
path1 = Model(layers=[Affine(nout=100, init=init_norm, activation=Rectlin()),
                      Affine(nout=100, init=init_norm, activation=Rectlin())])

path2 = Model(layers=[Affine(nout=100, init=init_norm, activation=Rectlin()),
                      Affine(nout=100, init=init_norm, activation=Rectlin())])

layers = [
    MergeSum([path1.layers, path2.layers]),
    Affine(nout=10, init=init_norm, activation=Logistic(shortcut=True))
]

model = Model(layers=layers)
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

# fit and validate
optimizer = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)

# configure callbacks
callbacks = Callbacks(model, train_set, output_file=args.output_file,
                      valid_set=valid_set, valid_freq=args.validation_freq,
                      progress_bar=args.progress_bar)

model.fit(train_set, cost=cost, optimizer=optimizer, num_epochs=num_epochs, callbacks=callbacks)
