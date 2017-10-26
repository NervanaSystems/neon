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
Train a LSTM or GRU network for sentiment analysis on IMDB review data.

Reference:

    When are Tree Structures Necessary for Deep Learning of Representations? `[Li2015]_
..  _[Li2015]: http://arxiv.org/pdf/1503.00185v5.pdf

Usage:

    python examples/imdb_lstm.py -e 2 -eval 1 --rlayer_type lstm

"""

from neon import logger as neon_logger
from neon.backends import gen_backend
from neon.data import IMDB
from neon.initializers import Uniform, GlorotUniform
from neon.layers import (GeneralizedCost, LSTM, Affine, Dropout, LookupTable,
                         RecurrentSum, Recurrent, DeepBiLSTM, DeepBiRNN)
from neon.models import Model
from neon.optimizers import Adagrad
from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyMulti, Accuracy
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--rlayer_type', default='lstm',
                    choices=['bilstm', 'lstm', 'birnn', 'bibnrnn', 'rnn'],
                    help='type of recurrent layer to use (lstm, bilstm, rnn, birnn, bibnrnn)')

args = parser.parse_args(gen_be=False)

# hyperparameters from the reference
args.batch_size = 128
gradient_clip_value = 15
vocab_size = 20000
sentence_length = 128
embedding_dim = 128
hidden_size = 128
reset_cells = True

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# make dataset
dataset = IMDB(vocab_size, sentence_length, path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.test_iter

neon_logger.display("Vocab size - {}".format(vocab_size))
neon_logger.display("Sentence Length - {}".format(sentence_length))
neon_logger.display("# of train sentences {}".format(train_set.Xdev[0].shape[0]))
neon_logger.display("# of test sentence {}".format(valid_set.Xdev[0].shape[0]))


# weight initialization
uni = Uniform(low=-0.1 / embedding_dim, high=0.1 / embedding_dim)
g_uni = GlorotUniform()

if args.rlayer_type == 'lstm':
    rlayer = LSTM(hidden_size, g_uni, activation=Tanh(),
                  gate_activation=Logistic(), reset_cells=reset_cells)
elif args.rlayer_type == 'bilstm':
    rlayer = DeepBiLSTM(hidden_size, g_uni, activation=Tanh(), depth=1,
                        gate_activation=Logistic(), reset_cells=reset_cells)
elif args.rlayer_type == 'rnn':
    rlayer = Recurrent(hidden_size, g_uni, activation=Tanh(), reset_cells=reset_cells)
elif args.rlayer_type == 'birnn':
    rlayer = DeepBiRNN(hidden_size, g_uni, activation=Tanh(),
                       depth=1, reset_cells=reset_cells, batch_norm=False)
elif args.rlayer_type == 'bibnrnn':
    rlayer = DeepBiRNN(hidden_size, g_uni, activation=Tanh(),
                       depth=1, reset_cells=reset_cells, batch_norm=True)

layers = [
    LookupTable(vocab_size=vocab_size, embedding_dim=embedding_dim, init=uni),
    rlayer,
    RecurrentSum(),
    Dropout(keep=0.5),
    Affine(2, g_uni, bias=g_uni, activation=Softmax())
]

model = Model(layers=layers)

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
optimizer = Adagrad(learning_rate=0.01,
                    gradient_clip_value=gradient_clip_value)

# configure callbacks
callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

# train model
model.fit(train_set, optimizer=optimizer,
          num_epochs=args.epochs, cost=cost, callbacks=callbacks)

# eval model
neon_logger.display("Train Accuracy - {}".format(100 * model.eval(train_set, metric=Accuracy())))
neon_logger.display("Test  Accuracy - {}".format(100 * model.eval(valid_set, metric=Accuracy())))
