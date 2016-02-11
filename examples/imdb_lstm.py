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
Example that trains an LSTM or GRU networks for sentiment analysis

Reference:
   See J.Li et al, EMNLP2015 - http://arxiv.org/pdf/1503.00185v5.pdf

$ python examples/imdb_lstm.py -e 2 -eval 1 --rlayer_type lstm


"""

from neon.backends import gen_backend
from neon.data.dataloaders import load_imdb
from neon.data.dataiterator import ArrayIterator
from neon.data.text_preprocessing import pad_data
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
path = load_imdb(path=args.data_dir)
(X_train, y_train), (X_test, y_test), nclass = pad_data(path,
                                                        vocab_size=vocab_size,
                                                        sentence_length=sentence_length)

print "Vocab size - ", vocab_size
print "Sentence Length - ", sentence_length
print "# of train sentences", X_train.shape[0]
print "# of test sentence", X_test.shape[0]

train_set = ArrayIterator(X_train, y_train, nclass=2)
valid_set = ArrayIterator(X_test, y_test, nclass=2)

# weight initialization
uni = Uniform(low=-0.1/embedding_dim, high=0.1/embedding_dim)
g_uni = GlorotUniform()

if args.rlayer_type == 'lstm':
    rlayer = LSTM(hidden_size, g_uni, activation=Tanh(),
                  gate_activation=Logistic(), reset_cells=True)
elif args.rlayer_type == 'bilstm':
    rlayer = DeepBiLSTM(hidden_size, g_uni, activation=Tanh(), depth=1,
                        gate_activation=Logistic(), reset_cells=True)
elif args.rlayer_type == 'rnn':
    rlayer = Recurrent(hidden_size, g_uni, activation=Tanh(), reset_cells=True)
elif args.rlayer_type == 'birnn':
    rlayer = DeepBiRNN(hidden_size, g_uni, activation=Tanh(), depth=1,
                       reset_cells=True, batch_norm=False, bi_sum=False)
elif args.rlayer_type == 'bibnrnn':
    rlayer = DeepBiRNN(hidden_size, g_uni, activation=Tanh(), depth=1,
                       reset_cells=True, batch_norm=True)


layers = [
    LookupTable(vocab_size=vocab_size, embedding_dim=embedding_dim, init=uni),
    rlayer,
    RecurrentSum(),
    Dropout(keep=0.5),
    Affine(2, g_uni, bias=g_uni, activation=Softmax())
]

model = Model(layers=layers)

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
optimizer = Adagrad(learning_rate=0.01, gradient_clip_value=gradient_clip_value)

# configure callbacks
callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

# train model
model.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

# eval model
print "Train Accuracy - ", 100 * model.eval(train_set, metric=Accuracy())
print "Test  Accuracy - ", 100 * model.eval(valid_set, metric=Accuracy())
