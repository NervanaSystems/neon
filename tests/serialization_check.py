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

import os
import sys

from neon.backends import gen_backend
from neon.data import ArrayIterator, load_mnist
from neon.initializers import Uniform, Constant
from neon.layers import GeneralizedCost, Affine, Conv, Pooling
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from neon.callbacks.callbacks import Callbacks, SerializeModelCallback
from neon.util.argparser import NeonArgparser
from neon.util.modeldesc import ModelDescription


# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
batch_size = 128
num_epochs = args.epochs


def compare_model_pickles(fn1, fn2):
    print 'comparing pickle files %s and %s' % (fn1, fn2)
    model1 = ModelDescription(fn1)
    model2 = ModelDescription(fn2)
    return ModelDescription.match(model1, model2)


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

    model = Model(layers=layers)
    return model


def main():
    # setup the model and run for num_epochs saving the last state only
    # this is at the top so that the be is generated
    model = gen_model(args.backend)

    # setup data iterators
    (X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
    NN = batch_size*5  # avoid partial mini batches
    if args.backend == 'nervanacpu' or args.backend == 'cpu':
        # limit data since cpu backend runs slower
        train = ArrayIterator(X_train[:NN], y_train[:NN],
                              nclass=nclass, lshape=(1, 28, 28))
        valid = ArrayIterator(X_test[:NN], y_test[:NN],
                              nclass=nclass, lshape=(1, 28, 28))
    else:
        train = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(1, 28, 28))
        valid = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

    # serialization related
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())
    opt_gdm = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)

    checkpoint_model_path = os.path.join('./', 'test_oneshot.pkl')
    checkpoint_schedule = 1  # save at every step

    callbacks = Callbacks(model)
    callbacks.add_callback(SerializeModelCallback(checkpoint_model_path,
                                                  checkpoint_schedule,
                                                  history=2))

    # run the fit all the way through saving a checkpoint e
    model.fit(train,
              optimizer=opt_gdm,
              num_epochs=num_epochs,
              cost=cost,
              callbacks=callbacks)

    # setup model with same random seed run epoch by epoch
    # serializing and deserializing at each step
    model = gen_model(args.backend)
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())
    opt_gdm = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)

    # reset data iterators
    train.reset()
    valid.reset()

    checkpoint_model_path = os.path.join('./', 'test_manyshot.pkl')
    checkpoint_schedule = 1  # save at evey step
    for epoch in range(num_epochs):
        # _0 points to state at end of epoch 0
        callbacks = Callbacks(model)
        callbacks.add_callback(SerializeModelCallback(checkpoint_model_path,
                                                      checkpoint_schedule,
                                                      history=num_epochs))
        model.fit(train,
                  optimizer=opt_gdm,
                  num_epochs=epoch+1,
                  cost=cost,
                  callbacks=callbacks)

        # load saved file
        prts = os.path.splitext(checkpoint_model_path)
        fn = prts[0] + '_%d' % epoch + prts[1]
        model.load_params(fn)  # load the saved weights

    # compare test_oneshot_<num_epochs>.pkl to test_manyshot_<num_epochs>.pkl
    if not compare_model_pickles('test_oneshot_%d.pkl' % (num_epochs-1),
                                 'test_manyshot_%d.pkl' % (num_epochs-1)):
        print 'No Match'
        sys.exit(1)
    else:
        print 'Match'

if __name__ == '__main__':
    main()
