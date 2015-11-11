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
Example that demonstrates the ability to periodically save the latest and best
weights to disk, then later load them back in.
"""

import numpy as np
import os
import pickle
import sys

from neon.backends import gen_backend
from neon.data import DataIterator, load_mnist
from neon.initializers import Uniform, Constant
from neon.layers import GeneralizedCost, Affine, Conv, Pooling
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser


def compare_model_pickles(fn1, fn2):
    print 'comparing pickle files %s and %s' % (fn1, fn2)
    with open(fn1, 'r') as fn:
        model1 = pickle.load(fn)

    with open(fn2, 'r') as fn:
        model2 = pickle.load(fn)
    compare_objs(model1, model2)
    return


# recursively compare the pickle objects
def compare_objs(x, y):
    assert type(x) is type(y)
    if type(x) is dict:
        assert x.keys().sort() == y.keys().sort()
        for ky in x:
            compare_objs(x[ky], y[ky])
    elif type(x) is list:
        assert len(x) == len(y)
        for ind in range(len(x)):
            compare_objs(x[ind], y[ind])
    elif type(x) is np.ndarray:
        assert x.shape == y.shape
        if not np.allclose(x, y, atol=1.0e-5, rtol=0.0):
            x = x.reshape(x.size)
            y = y.reshape(y.size)
            dd = x-y
            worst_case = np.max(np.abs(dd))
            print 'worst case abs diff = %e' % worst_case
            ind = np.where((x != 0) | (y != 0))
            rel_err = np.abs(np.divide(dd[ind], np.abs(x[ind]) + np.abs(y[ind])))
            worst_case = np.max(rel_err)
            print 'worst case rel diff = %e' % worst_case
            assert False
    else:
        assert x == y


# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
batch_size = 128
num_epochs = args.epochs


def gen_model(backend_type):
    # setup backend
    gen_backend(backend=backend_type,
                batch_size=batch_size,
                rng_seed=2,
                device_id=args.device_id,
                datatype=args.datatype)

    init_uni = Uniform(low=-0.1, high=0.1)

    # Set up the model layers
    layers = []
    layers.append(Conv((5, 5, 16), init=init_uni, bias=Constant(0), activation=Rectlin()))
    layers.append(Pooling(2))
    layers.append(Conv((5, 5, 32), init=init_uni, activation=Rectlin()))
    layers.append(Pooling(2))
    layers.append(Affine(nout=500, init=init_uni, activation=Rectlin()))
    layers.append(Affine(nout=10, init=init_uni, activation=Logistic(shortcut=True)))

    mlp = Model(layers=layers)
    return mlp


def main():
    # setup the model and run for num_epochs saving the last state only
    # this is at the top so that the be is generated
    mlp = gen_model(args.backend)

    # setup data iterators
    (X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
    if args.backend == 'nervanacpu' or args.backend == 'cpu':
        # limit data since cpu backend runs slower
        train = DataIterator(X_train[:1000], y_train[:1000], nclass=nclass, lshape=(1, 28, 28))
        valid = DataIterator(X_test[:1000], y_test[:1000], nclass=nclass, lshape=(1, 28, 28))
    else:
        train = DataIterator(X_train, y_train, nclass=nclass, lshape=(1, 28, 28))
        valid = DataIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

    # serialization related
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())
    opt_gdm = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)

    checkpoint_model_path = os.path.join('./', 'test_oneshot.pkl')
    checkpoint_schedule = 1  # save at every step

    callbacks = Callbacks(mlp, train)
    callbacks.add_serialize_callback(checkpoint_schedule, checkpoint_model_path, history=2)

    # run the fit all the way through saving a checkpoint e
    mlp.fit(train, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

    # setup model with same random seed run epoch by epoch
    # serializing and deserializing at each step
    mlp = gen_model(args.backend)
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())
    opt_gdm = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)

    # reset data iterators
    train.reset()
    valid.reset()

    checkpoint_model_path = os.path.join('./', 'test_manyshot.pkl')
    checkpoint_schedule = 1  # save at evey step
    callbacks = Callbacks(mlp, train)
    callbacks.add_serialize_callback(checkpoint_schedule,
                                     checkpoint_model_path,
                                     history=num_epochs)
    for epoch in range(num_epochs):
        # _0 points to state at end of epoch 0
        mlp.fit(train, optimizer=opt_gdm, num_epochs=epoch+1, cost=cost, callbacks=callbacks)

        # load saved file
        prts = os.path.splitext(checkpoint_model_path)
        fn = prts[0] + '_%d' % epoch + prts[1]
        mlp.load_weights(fn)  # load the saved weights

    # compare test_oneshot_<num_epochs>.pkl to test_manyshot_<num_epochs>.pkl
    try:
        compare_model_pickles('test_oneshot_%d.pkl' % (num_epochs-1),
                              'test_manyshot_%d.pkl' % (num_epochs-1))
    except:
        print 'test failed....'
        sys.exit(1)

if __name__ == '__main__':
    main()
