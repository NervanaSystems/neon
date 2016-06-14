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
Example that does inference on an LSTM networks for amazon review analysis

$ python examples/imdb/inference.py --model_weights imdb.p --vocab_file imdb.vocab

"""

from __future__ import print_function
from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below
from builtins import input  # noqa
import numpy as np  # noqa
from neon.backends import gen_backend  # noqa
from neon.initializers import Uniform, GlorotUniform  # noqa
from neon.layers import LSTM, Affine, Dropout, LookupTable, RecurrentSum  # noqa
from neon.models import Model  # noqa
from neon.transforms import Logistic, Tanh, Softmax  # noqa
from neon.util.argparser import NeonArgparser, extract_valid_args  # noqa
from neon.util.compat import pickle  # noqa
from neon.data.text_preprocessing import clean_string  # noqa


# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--model_weights', required=True,
                    help='pickle file of trained weights')
parser.add_argument('--vocab_file', required=True,
                    help='vocabulary file')
args = parser.parse_args()


# hyperparameters from the reference
batch_size = 1
clip_gradients = True
gradient_limit = 5
vocab_size = 20000
sentence_length = 128
embedding_dim = 128
hidden_size = 128
reset_cells = True
num_epochs = args.epochs

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))
be.bsz = 1


# define same model as in train
init_glorot = GlorotUniform()
init_emb = Uniform(low=-0.1 / embedding_dim, high=0.1 / embedding_dim)
nclass = 2
layers = [
    LookupTable(vocab_size=vocab_size, embedding_dim=embedding_dim, init=init_emb,
                pad_idx=0, update=True),
    LSTM(hidden_size, init_glorot, activation=Tanh(),
         gate_activation=Logistic(), reset_cells=True),
    RecurrentSum(),
    Dropout(keep=0.5),
    Affine(nclass, init_glorot, bias=init_glorot, activation=Softmax())
]


# load the weights
print("Initialized the models - ")
model_new = Model(layers=layers)
print("Loading the weights from {0}".format(args.model_weights))

model_new.load_params(args.model_weights)
model_new.initialize(dataset=(sentence_length, batch_size))

# setup buffers before accepting reviews
xdev = be.zeros((sentence_length, 1), dtype=np.int32)  # bsz is 1, feature size
xbuf = np.zeros((1, sentence_length), dtype=np.int32)
oov = 2
start = 1
index_from = 3
pad_char = 0
vocab, rev_vocab = pickle.load(open(args.vocab_file, 'rb'))


while True:
    line = input('Enter a Review from testData.tsv file \n')

    # clean the input
    tokens = clean_string(line).strip().split()

    # check for oov and add start
    sent = [len(vocab) + 1 if t not in vocab else vocab[t] for t in tokens]
    sent = [start] + [w + index_from for w in sent]
    sent = [oov if w >= vocab_size else w for w in sent]

    # pad sentences
    xbuf[:] = 0
    trunc = sent[-sentence_length:]
    xbuf[0, -len(trunc):] = trunc
    xdev[:] = xbuf.T.copy()
    y_pred = model_new.fprop(xdev, inference=True)  # inference flag dropout

    print("Sent - {0}".format(xbuf))
    print("Pred - {0} ".format(y_pred.get().T))
    print('-' * 100)
