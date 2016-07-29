#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
Example for using the HDF5 data iterator on a MLP with the MNIST data.

See mnist_mlp.py for more information on the model and data used in this example.

Usage:

    python examples/mnist_hdf5.py

"""

from neon.callbacks.callbacks import Callbacks
from neon.data import HDF5IteratorOneHot, MNIST
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, Misclassification
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger
import h5py
import numpy as np

# parse the command line arguments
parser = NeonArgparser(__doc__)

args = parser.parse_args()

# load up the mnist data set
dataset = MNIST(path=args.data_dir)
# split into train and tests sets
(X_train, y_train), (X_test, y_test), nclass = dataset.load_data()

# generate the HDF5 file
datsets = {'train': (X_train, y_train),
           'test': (X_test, y_test)}

for ky in ['train', 'test']:
    df = h5py.File('mnist_%s.h5' % ky, 'w')

    # input images
    in_dat = datsets[ky][0]
    df.create_dataset('input', data=in_dat)
    df['input'].attrs['lshape'] = (1, 28, 28)  # (C, H, W)

    # can also add in a mean image or channel by channel mean for color image
    # for mean subtraction during data iteration
    # e.g.
    if ky == 'train':
        mean_image = np.mean(X_train, axis=0)
    # use training set mean for both train and val data sets
    df.create_dataset('mean', data=mean_image)

    target = datsets[ky][1].reshape((-1, 1))  # make it a 2D array
    df.create_dataset('output', data=target)
    df['output'].attrs['nclass'] = 10
    df.close()

# setup a training set iterator
# use the iterator that generates 1-hot output. other HDF5Iterator (sub) classes are
# available for different data layouts
train_set = HDF5IteratorOneHot('mnist_train.h5')
valid_set = HDF5IteratorOneHot('mnist_test.h5')

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.01)

# setup model layers
layers = [Affine(nout=100, init=init_norm, activation=Rectlin()),
          Affine(nout=10, init=init_norm, activation=Logistic(shortcut=True))]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

# setup optimizer
optimizer = GradientDescentMomentum(
    0.1, momentum_coef=0.9, stochastic_round=args.rounding)

# initialize model object
mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)

# run fit
mlp.fit(train_set, optimizer=optimizer,
        num_epochs=args.epochs, cost=cost, callbacks=callbacks)
error_rate = mlp.eval(valid_set, metric=Misclassification())
neon_logger.display('Misclassification error = %.1f%%' % (error_rate * 100))
