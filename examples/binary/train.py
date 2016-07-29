#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
Trains BinaryNet on MNIST dataset.

Reference:
    "Binarized Neural Networks: Training Neural Networks with Weights and
         Activations Constrained to +1 or -1"
    http://arxiv.org/pdf/1602.02830v3.pdf

Usage:
    python binary/train.py -e 20
"""

from neon.callbacks.callbacks import Callbacks
from neon.data import MNIST
from neon.initializers import Uniform
from neon.layers import BinaryAffine, GeneralizedCost
from neon.models import Model
from neon.optimizers import MultiOptimizer, ShiftAdaMax, ShiftSchedule
from neon.transforms import Identity, Misclassification, Sign, SquareHingeLoss
from neon.util.argparser import NeonArgparser


# parse the command line arguments
parser = NeonArgparser(__doc__)

args = parser.parse_args()

# load up the mnist data set
dataset = MNIST(path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

# setup weight initialization function
init = Uniform(-1, 1)

# setup layers
layers = [
    BinaryAffine(nout=4096, init=init, batch_norm=True, activation=Sign()),
    BinaryAffine(nout=4096, init=init, batch_norm=True, activation=Sign()),
    BinaryAffine(nout=4096, init=init, batch_norm=True, activation=Sign()),
    BinaryAffine(nout=10, init=init, batch_norm=True, activation=Identity())
]

# setup cost function as Square Hinge Loss
cost = GeneralizedCost(costfunc=SquareHingeLoss())

# setup optimizer
LR_start = 1.65e-2


def ShiftAdaMax_with_Scale(LR=1):
    return ShiftAdaMax(learning_rate=LR_start * LR, schedule=ShiftSchedule(2, shift_size=1))


optimizer = MultiOptimizer({
    'default': ShiftAdaMax_with_Scale(),
    'BinaryLinear_0': ShiftAdaMax_with_Scale(57.038),
    'BinaryLinear_1': ShiftAdaMax_with_Scale(73.9008),
    'BinaryLinear_2': ShiftAdaMax_with_Scale(73.9008),
    'BinaryLinear_3': ShiftAdaMax_with_Scale(52.3195)
})

# initialize model object
bnn = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(bnn, eval_set=valid_set, **args.callback_args)

# run fit
bnn.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
print('Misclassification error = %.1f%%' % (bnn.eval(valid_set, metric=Misclassification())*100))
