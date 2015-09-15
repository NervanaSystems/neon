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
Small CIFAR10 based MLP with fully connected layers.
"""

from neon.backends import gen_backend
from neon.data import DataIterator, load_cifar10
from neon.initializers import Uniform
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Misclassification, CrossEntropyBinary, Logistic, Rectlin
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

train = DataIterator(X_train, y_train, nclass=nclass)
test = DataIterator(X_test, y_test, nclass=nclass)

init_uni = Uniform(low=-0.1, high=0.1)
opt_gdm = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9)

# set up the model layers
layers = []
layers.append(Affine(nout=200, init=init_uni, activation=Rectlin()))
layers.append(Affine(nout=10, init=init_uni, activation=Logistic(shortcut=True)))

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, train, output_file=args.output_file,
                      valid_set=test, valid_freq=args.validation_freq,
                      progress_bar=args.progress_bar)

mlp.fit(train, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

print mlp.eval(test, metric=Misclassification())
