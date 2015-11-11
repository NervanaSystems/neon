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
Example that trains a small multi-layer perceptron with multiple branches

Branch nodes are used to indicate points at which different layer sequences diverge

The topology of the network is:

 cost1      cost3
  |          /
 m_l4      b2_l2
  |        /
  | ___b2_l1
  |/
 m_l3       cost2
  |          /
 m_l2      b1_l2
  |        /
  | ___b1_l1
  |/
  |
 m_l1
  |
  |
 data

"""

import logging

from neon.callbacks.callbacks import Callbacks
from neon.data import DataIterator, load_mnist
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, BranchNode, Multicost, Tree
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, Misclassification, Softmax
from neon.transforms import CrossEntropyBinary, CrossEntropyMulti
from neon.util.argparser import NeonArgparser


# parse the command line arguments
parser = NeonArgparser(__doc__)

args = parser.parse_args()

# load up the mnist data set
# split into train and tests sets
(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)

# setup a training set iterator
train_set = DataIterator(X_train, y_train, nclass=nclass)
# setup a validation data set iterator
valid_set = DataIterator(X_test, y_test, nclass=nclass)

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.01)

normrelu = dict(init=init_norm, activation=Rectlin())
normsigm = dict(init=init_norm, activation=Logistic(shortcut=True))
normsoft = dict(init=init_norm, activation=Softmax())

# setup model layers
b1 = BranchNode(name="b1")
b2 = BranchNode(name="b2")


p1 = [Affine(nout=100, linear_name="m_l1", **normrelu),
      b1,
      Affine(nout=32, linear_name="m_l2", **normrelu),
      Affine(nout=16, linear_name="m_l3", **normrelu),
      b2,
      Affine(nout=10, linear_name="m_l4", **normsoft)]

p2 = [b1,
      Affine(nout=16, linear_name="b1_l1", **normrelu),
      Affine(nout=10, linear_name="b1_l2", **normsigm)]

p3 = [b2,
      Affine(nout=16, linear_name="b2_l1", **normrelu),
      Affine(nout=10, linear_name="b2_l2", **normsigm)]


# setup cost function as CrossEntropy
cost = Multicost(costs=[GeneralizedCost(costfunc=CrossEntropyMulti()),
                        GeneralizedCost(costfunc=CrossEntropyBinary()),
                        GeneralizedCost(costfunc=CrossEntropyBinary())],
                 weights=[1, 0., 0.])

# setup optimizer
optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9, stochastic_round=args.rounding)

# initialize model object
alphas = [1, 0.25, 0.25]
mlp = Model(layers=Tree([p1, p2, p3], alphas=alphas))

# setup standard fit callbacks
callbacks = Callbacks(mlp, train_set, eval_set=valid_set, **args.callback_args)

# run fit
mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

logging.getLogger('neon').info("Misclassification error = %.1f%%",
                               (mlp.eval(valid_set, metric=Misclassification())*100))
print('Misclassification error = %.1f%%' % (mlp.eval(valid_set, metric=Misclassification())*100))
