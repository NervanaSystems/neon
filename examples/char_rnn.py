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
Train a network with one recurrent layer of tanh units on the Penn
treebank data parsing on character-level.

Reference:

  Advances in optimizing recurrent networks `[Pascanu2012]`_
.. _[Pascanu2012]: http://arxiv.org/pdf/1212.0901.pdf

Usage:

    python examples/char_rnn.py

"""

from neon.backends import gen_backend
from neon.data import PTB
from neon.initializers import Uniform
from neon.layers import GeneralizedCost, Affine, Recurrent
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Tanh, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

# these hyperparameters are from the paper
args.batch_size = 50
time_steps = 150
hidden_size = 500
gradient_clip_value = None

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# download penn treebank
dataset = PTB(time_steps, path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

# weight initialization
init = Uniform(low=-0.08, high=0.08)

# model initialization
layers = [Recurrent(hidden_size, init, activation=Tanh()),
          Affine(len(train_set.vocab), init, bias=init, activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))

model = Model(layers=layers)

optimizer = RMSProp(gradient_clip_value=gradient_clip_value, stochastic_round=args.rounding)

# configure callbacks
callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=args.epochs,
          cost=cost, callbacks=callbacks)
