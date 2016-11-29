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

from __future__ import print_function
import numpy as np
import h5py
import os

from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.util.persist import load_obj, save_obj
from neon import logger as neon_logger

from data_loader import load_data
from data_iterator import SentenceEncode
from util import SentenceVector, prep_data, load_sent_encoder


# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--vector_name', required=True,
                    help='the cached data file for all the sentence vectors')
parser.add_argument('--output_dir', required=True,
                    help='directory to save/load the saved datasets')
args = parser.parse_args(gen_be=False)

# hyperparameters from the reference
args.batch_size = 1
valid_split = None

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# load the documents by giving the path and what extension the files are
data_file, vocab_file = load_data(args.data_dir,
                                  valid_split=valid_split,
                                  output_path=args.output_dir)
vocab, rev_vocab, word_count = load_obj(vocab_file)

vocab_size = len(vocab)
neon_logger.display("\nVocab size from the dataset is: {}".format(vocab_size))

index_from = 2  # 0: padding 1: oov
oov = 1
vocab_size_layer = vocab_size + index_from
max_len = 30

# load trained model
model_dict = load_obj(args.model_file)
model = load_sent_encoder(model_dict)

h5f = h5py.File(data_file, 'r')

if valid_split:
    h5train, h5valid = h5f['train'], h5f['valid']
    h5train_text, h5valid_text = h5f['report_train'], h5f['report_valid']
else:
    h5train = h5f['train']
    h5train_text = h5f['report_train']

if os.path.exists(args.vector_name):
    neon_logger.display("cached encoded vectors exists: {}".format(args.vector_name))
    (train_vec, sentences) = load_obj(args.vector_name)
    model.initialize(dataset=(max_len, 1))
else:
    neon_logger.display("Encoding the entire training set....")
    # encode all the training sentences
    model.initialize(dataset=(max_len, 1))
    train_set = SentenceEncode(h5train, h5train_text, h5train.attrs['nsample'], vocab_size_layer,
                               max_len=max_len, index_from=index_from)
    sentences = h5train_text[:train_set.ndata].reshape(-1, 1)
    train_vec = model.get_outputs(train_set)
    neon_logger.display("Encoding complete. Saving to {}".format(args.vector_name))
    save_obj((train_vec, sentences), args.vector_name)

s2v = SentenceVector(train_vec, sentences)

xdev = be.zeros((max_len, 1), dtype=np.int32)  # bsz is 1, feature size
xbuf = np.zeros((max_len, 1), dtype=np.int32)

# Add python 2 & 3 support for raw_input
try:
    input = raw_input
except NameError:
    pass

while True:
    line = input('\nEnter a new sentence for inference: \n')
    xbuf = prep_data(line, 'text', max_len, vocab)
    xdev[:] = xbuf
    query_vec = model.fprop(xdev, inference=True).get().T

    sim_sent, _ = s2v.find_similar(query_vec, n=8)

    print("\nSimilar sentences:\n")
    for sent in sim_sent:
        print(sent[0][0] + "\n")
