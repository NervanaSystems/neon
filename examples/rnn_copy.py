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
Train a network with one GRU layer to perform the vector copy task.

Reference:

    Neural Turing Machines `[Graves2014]`_
.. _[Graves2014]: https://arxiv.org/pdf/1410.5401.pdf

Usage:

    python examples/rnn_copy.py

"""

from neon.initializers import Uniform
from neon.layers import GeneralizedCostMask, Affine, GRU
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Tanh, CrossEntropyBinary, Logistic
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.data import Ticker, CopyTask

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

batch_size = 128
seq_len_max = 5
repeat_count_max = 2
vec_size = 8

# these hyperparameters are from the paper
hidden_size = 100
gradient_clip_value = 5

# load data and parse on character-level
ticker_task = CopyTask(seq_len_max, vec_size)
train_set = Ticker(ticker_task)

# weight initialization
init = Uniform(low=-0.08, high=0.08)

output_size = 8
N = 120  # number of memory locations
M = 8  # size of a memory location

# model initialization
layers = [
    GRU(hidden_size, init, activation=Tanh(), gate_activation=Logistic()),
    Affine(train_set.nout, init, bias=init, activation=Logistic())
]

cost = GeneralizedCostMask(costfunc=CrossEntropyBinary())

model = Model(layers=layers)

optimizer = RMSProp(gradient_clip_value=gradient_clip_value,
                    stochastic_round=args.rounding)

# configure callbacks
callbacks = Callbacks(model, **args.callback_args)

# we can use the training set as the validation set,
# since the data is tickerally generated
callbacks.add_watch_ticker_callback(train_set)

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=args.epochs,
          cost=cost,
          callbacks=callbacks)
