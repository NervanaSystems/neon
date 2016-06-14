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
Train a LSTM network for sentiment analysis on IMDB review data.

Get the data from Kaggle: https://www.kaggle.com/c/word2vec-nlp-tutorial/data

If choose to initialize the word embedding layer using Word2Vec, please make sure
to get the data GoogleNews-vectors-negative300.bin from:

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

Usage:

    python examples/imdb/train.py -f labeledTrainData.tsv -e 2 -eval 1 -s imdb.p \
        --vocab_file imdb.vocab [--use_w2v]

"""

from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below
from prepare import build_data_train  # noqa
from neon import logger as neon_logger  # noqa
from neon.backends import gen_backend  # noqa
from neon.data import ArrayIterator  # noqa
from neon.initializers import Uniform, GlorotUniform, Array  # noqa
from neon.layers import GeneralizedCost, Affine, Dropout, LookupTable, LSTM, RecurrentSum  # noqa
from neon.models import Model  # noqa
from neon.optimizers import Adagrad  # noqa
from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyMulti, Accuracy  # noqa
from neon.util.argparser import NeonArgparser, extract_valid_args  # noqa
from neon.util.compat import pickle  # noqa
from neon.callbacks.callbacks import Callbacks  # noqa
from neon.data.text_preprocessing import get_paddedXY, get_google_word2vec_W  # noqa
import h5py  # noqa

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('-f', '--review_file',
                    default='labeledTrainData.tsv',
                    help='input movie review file')
parser.add_argument('--vocab_file',
                    default='labeledTrainData.tsv.vocab',
                    help='output file to save the processed vocabulary')
parser.add_argument('--use_w2v', action='store_true',
                    help='use downloaded Google Word2Vec')
parser.add_argument('--w2v',
                    default='GoogleNews-vectors-negative300.bin',
                    help='the pre-built Word2Vec')
args = parser.parse_args()


# hyperparameters
hidden_size = 128
embedding_dim = 128
vocab_size = 20000
sentence_length = 128
batch_size = 32
gradient_limit = 5
clip_gradients = True
num_epochs = args.epochs
embedding_update = True

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# get the preprocessed and tokenized data
fname_h5, fname_vocab = build_data_train(filepath=args.review_file,
                                         vocab_file=args.vocab_file, skip_headers=True)


# play around with google-news word vectors for init
if args.use_w2v:
    w2v_file = args.w2v
    vocab, rev_vocab = pickle.load(open(fname_vocab, 'rb'))
    init_emb_np, embedding_dim, _ = get_google_word2vec_W(w2v_file, vocab,
                                                          vocab_size=vocab_size, index_from=3)
    neon_logger.display(
        "Done loading the Word2Vec vectors: embedding size - {}".format(embedding_dim))
    embedding_update = True
    init_emb = Array(val=be.array(init_emb_np))
else:
    init_emb = Uniform(-0.1 / embedding_dim, 0.1 / embedding_dim)


h5f = h5py.File(fname_h5, 'r')
reviews, h5train, h5valid = h5f['reviews'], h5f['train'], h5f['valid']
ntrain, nvalid, nclass = reviews.attrs[
    'ntrain'], reviews.attrs['nvalid'], reviews.attrs['nclass']


# make train dataset
Xy = h5train[:ntrain]
X = [xy[1:] for xy in Xy]
y = [xy[0] for xy in Xy]
X_train, y_train = get_paddedXY(
    X, y, vocab_size=vocab_size, sentence_length=sentence_length)
train_set = ArrayIterator(X_train, y_train, nclass=nclass)

# make valid dataset
Xy = h5valid[:nvalid]
X = [xy[1:] for xy in Xy]
y = [xy[0] for xy in Xy]
X_valid, y_valid = get_paddedXY(
    X, y, vocab_size=vocab_size, sentence_length=sentence_length)
valid_set = ArrayIterator(X_valid, y_valid, nclass=nclass)


# initialization
init_glorot = GlorotUniform()


# define layers
layers = [
    LookupTable(vocab_size=vocab_size, embedding_dim=embedding_dim, init=init_emb,
                pad_idx=0, update=embedding_update),
    LSTM(hidden_size, init_glorot, activation=Tanh(), gate_activation=Logistic(),
         reset_cells=True),
    RecurrentSum(),
    Dropout(keep=0.5),
    Affine(nclass, init_glorot, bias=init_glorot, activation=Softmax())
]

# set the cost, metrics, optimizer
cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
metric = Accuracy()
model = Model(layers=layers)
optimizer = Adagrad(learning_rate=0.01)

# configure callbacks
callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=num_epochs,
          cost=cost,
          callbacks=callbacks)

# eval model
neon_logger.display("Train Accuracy - {}".format(100 * model.eval(train_set, metric=metric)))
neon_logger.display("Test Accuracy - {}".format(100 * model.eval(valid_set, metric=metric)))
