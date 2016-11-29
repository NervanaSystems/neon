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
Train a sentence embedding model on the BookCorpus dataset.

Reference:
    "Skip-thought vectors"
    http://arxiv.org/abs/1506.06726

"""
from __future__ import division
from __future__ import print_function
import os

from neon.backends import gen_backend
from neon.util.persist import load_obj
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.initializers import Uniform, Orthonormal, Constant
from neon.layers import GeneralizedCostMask, Multicost, GRU, SkipThought
from neon.models import Model
from neon.transforms import CrossEntropyMulti, Logistic, Tanh
from neon.callbacks.callbacks import Callbacks, MetricCallback
from neon.optimizers import Adam
from neon import logger as neon_logger

from data_loader import load_data
from data_iterator import SentenceHomogenous

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--output_dir', default='output/',
                    help='choose the directory to save the files')
parser.add_argument('--max_vocab_size', default=20000,
                    help='number of (most frequent) words to use from vocabulary')
parser.add_argument('--max_len_w', default=30,
                    help='maximum sentence length for training')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args(gen_be=False)

# hyperparameters from the reference
args.batch_size = 64
embed_dim = 620

valid_split = None
# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# load the documents by giving the path and what extension the files are
data_file, vocab_file = load_data(args.data_dir, valid_split=valid_split,
                                  max_vocab_size=args.max_vocab_size,
                                  max_len_w=args.max_len_w,
                                  output_path=args.output_dir,
                                  file_ext=['txt'],
                                  subset_pct=args.subset_pct)
vocab, rev_vocab, word_count = load_obj(vocab_file)

vocab_size = len(vocab)
neon_logger.display("\nData loading complete.")
neon_logger.display("\nVocab size from the dataset is: {}".format(vocab_size))

index_from = 2  # 0: padding 1: oov
vocab_size_layer = vocab_size + index_from

init_embed_dev = Uniform(low=-0.1, high=0.1)

# sent2vec network
nhidden = 2400
gradient_clip_norm = 5.0

train_set = SentenceHomogenous(data_file=data_file, sent_name='train', text_name='report_train',
                               nwords=vocab_size_layer, max_len=args.max_len_w,
                               index_from=index_from)


if valid_split and valid_split > 0.0:
    valid_set = SentenceHomogenous(data_file=data_file, sent_name='valid',
                                   text_name='report_valid', nwords=vocab_size_layer,
                                   max_len=args.max_len_w, index_from=index_from)

skip = SkipThought(vocab_size_layer, embed_dim, init_embed_dev, nhidden, rec_layer=GRU,
                   init_rec=Orthonormal(), activ_rec=Tanh(), activ_rec_gate=Logistic(),
                   init_ff=Uniform(low=-0.1, high=0.1), init_const=Constant(0.0))
model = Model(skip)

if args.model_file and os.path.isfile(args.model_file):
    neon_logger.display("Loading saved weights from: {}".format(args.model_file))
    model_dict = load_obj(args.model_file)
    model.deserialize(model_dict, load_states=True)
elif args.model_file:
    neon_logger.display("Unable to find model file {}, restarting training.".
                        format(args.model_file))

cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti(usebits=True)),
                        GeneralizedCostMask(costfunc=CrossEntropyMulti(usebits=True))],
                 weights=[1, 1])

optimizer = Adam(gradient_clip_norm=gradient_clip_norm)

# metric
valmetric = None
# configure callbacks
if valid_split and valid_split > 0.0:
    callbacks = MetricCallback(eval_set=valid_set, metric=valmetric, epoch_freq=args.eval_freq)
else:
    callbacks = Callbacks(model, metric=valmetric, **args.callback_args)

# train model
model.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
