#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# ----------------------------------------------------------------------------
"""
    $python inference_sent2vec.py --model_file output/model.prm
                                   --data_dir book_corpus/
                                   --output_dir output/
                                   --vector_name output/book_vectors.pkl
"""

import cPickle
import numpy as np
import h5py
import os

from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.initializers import Uniform, GlorotUniform, Array, Orthonormal
from neon.layers import Sequential, RecurrentLast, RecurrentSum
from neon.models import Model
from neon.transforms import Softmax, CrossEntropyMulti, Logistic, Tanh
from neon.callbacks.callbacks import Callbacks
from neon.util.persist import save_obj

from data_loader import load_data, clean_string, load_sent_encoder, prep_data
from sent_vectors import SentenceVector
from data_iterator import SentenceEncode

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

# to run it on the cloud, can use ncloud train resume-training <id>, then it
# will look for the file defaulted named to be model.prm
if args.model_file is None:
    args.model_file = 'model.prm'

args.callback_args['model_file'] = None

# load the documents by giving the path and what extension the files are
data_file, vocab_file = load_data(args.data_dir,
                                  valid_split=valid_split,
                                  output_path=args.output_dir)
vocab, rev_vocab, word_count = cPickle.load(open(vocab_file, 'rb'))

vocab_size = len(vocab)
print "\nVocab size from the dataset is: {}".format(vocab_size)

index_from = 2  # 0: padding 1: oov
oov = 1
vocab_size_layer = vocab_size + index_from
max_len = 30

model = load_sent_encoder(args.model_file)

h5f = h5py.File(data_file, 'r')

if valid_split:
    h5train, h5valid = h5f['train'], h5f['valid']
    h5train_text, h5valid_text = h5f['report_train'], h5f['report_valid']
else:
    h5train = h5f['train']
    h5train_text = h5f['report_train']

if os.path.exists(args.vector_name):
    print "cached encoded vectors exists: {}".format(args.vector_name)
    f = open(args.vector_name, 'rb')
    (train_vec, sentences) = cPickle.load(f)
    model.be.bsz = 1
    model.initialize(dataset=(max_len, 1))
else:
    print "Encoding the entire training set...."
    # encode all the training sentences
    # set the backend bsz to be 1 for inference
    model.be.bsz = args.batch_size
    model.initialize(dataset=(max_len, 1))
    train_set = SentenceEncode(h5train, h5train_text, h5train.attrs['nsample'], vocab_size_layer,
                               max_len=max_len, index_from=index_from)
    sentences = h5train_text[:train_set.ndata].reshape(-1, 1)
    train_vec = model.get_outputs(train_set)
    print "Encoding complete. Saving to {}".format(args.vector_name)
    f = open(args.vector_name, 'wb')
    cPickle.dump((train_vec, sentences), f)

s2v = SentenceVector(train_vec, sentences)

xdev = be.zeros((max_len, 1), dtype=np.int32)  # bsz is 1, feature size
xbuf = np.zeros((max_len, 1), dtype=np.int32)

while True:
    line = raw_input('\nEnter a new sentence for inference: \n')
    xbuf = prep_data(line, 'text', max_len, vocab)
    xdev[:] = xbuf
    query_vec = model.fprop(xdev, inference=True).get().T

    print "Query vec: {}".format(query_vec)

    sim_sent, _ = s2v.find_similar(query_vec, n=8)

    print "\nSimilar sentences:\n"
    for sent in sim_sent:
        print sent
        print "\n"
