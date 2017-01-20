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
# --------------------------------------------------------
# Skip-Thoughts
# Licensed under Apache License 2.0 [see LICENSE for details]
# Written by Ryan Kiros
# --------------------------------------------------------
'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)

Modified from: https://github.com/ryankiros/skip-thoughts/blob/master/eval_sick.py
'''
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from neon.util.argparser import NeonArgparser
from neon.util.persist import load_obj
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.data import ArrayIterator
from neon.models import Model
from neon.optimizers import Adam
from neon.transforms import Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.data import SICK
from neon import logger as neon_logger

from data_iterator import SentenceEncode
from data_loader import load_data
from util import load_sent_encoder, tokenize, get_w2v_vocab


def main():
    # parse the command line arguments
    parser = NeonArgparser(__doc__)
    parser.add_argument('--output_path', required=True,
                        help='Output path used when training model')
    parser.add_argument('--w2v_path', required=False, default=None,
                        help='Path to GoogleNews w2v file for voab expansion.')
    parser.add_argument('--eval_data_path', required=False, default='./SICK_data',
                        help='Path to the SICK dataset for evaluating semantic relateness')
    parser.add_argument('--max_vocab_size', required=False, default=1000000,
                        help='Limit the vocabulary expansion to fit in GPU memory')
    parser.add_argument('--subset_pct', required=False, default=100,
                        help='subset of training dataset to use (use to retreive \
                        preprocessed data from training)')
    args = parser.parse_args(gen_be=True)

    # load vocab file from training
    _, vocab_file = load_data(args.data_dir, output_path=args.output_path,
                              subset_pct=float(args.subset_pct))
    vocab, _, _ = load_obj(vocab_file)

    vocab_size = len(vocab)
    neon_logger.display("\nVocab size from the dataset is: {}".format(vocab_size))

    index_from = 2  # 0: padding 1: oov
    vocab_size_layer = vocab_size + index_from
    max_len = 30

    # load trained model
    model_dict = load_obj(args.model_file)

    # Vocabulary expansion trick needs to pass the correct vocab set to evaluate (for tokenization)
    if args.w2v_path:
        neon_logger.display("Performing Vocabulary Expansion... Loading W2V...")
        w2v_vocab, w2v_vocab_size = get_w2v_vocab(args.w2v_path,
                                                  int(args.max_vocab_size), cache=True)

        vocab_size_layer = w2v_vocab_size + index_from
        model = load_sent_encoder(model_dict, expand_vocab=True, orig_vocab=vocab,
                                  w2v_vocab=w2v_vocab, w2v_path=args.w2v_path, use_recur_last=True)
        vocab = w2v_vocab
    else:
        # otherwise stick with original vocab size used to train the model
        model = load_sent_encoder(model_dict, use_recur_last=True)

    model.initialize(dataset=(max_len, 1))

    evaluate(model, vocab=vocab, data_path=args.eval_data_path, evaltest=True,
             vocab_size_layer=vocab_size_layer)


def evaluate(model, vocab, data_path, seed=1234, evaltest=False, vocab_size_layer=20002):
    """
    Run experiment
    """
    neon_logger.display('Preparing SICK evaluation data...')
    # Check if SICK data exists in specified directory, otherwise download
    sick_data = SICK(path=data_path)
    train, dev, test, scores = sick_data.load_eval_data()

    np.random.seed(seed)
    shuf_idxs = np.random.permutation(range(len(train[0])))
    train_A_shuf = train[0][shuf_idxs]
    train_B_shuf = train[1][shuf_idxs]
    scores_shuf = scores[0][shuf_idxs]

    train_A_tok = tokenize_input(train_A_shuf, vocab=vocab)
    train_B_tok = tokenize_input(train_B_shuf, vocab=vocab)
    dev_A_tok = tokenize_input(dev[0], vocab=vocab)
    dev_B_tok = tokenize_input(dev[1], vocab=vocab)

    # Get iterator from tokenized data
    train_set_A = SentenceEncode(train_A_tok, [], len(train_A_tok), vocab_size_layer,
                                 max_len=30, index_from=2)
    train_set_B = SentenceEncode(train_B_tok, [], len(train_B_tok), vocab_size_layer,
                                 max_len=30, index_from=2)

    # Compute embeddings using iterator
    trainA = model.get_outputs(train_set_A)
    trainB = model.get_outputs(train_set_B)

    dev_set_A = SentenceEncode(dev_A_tok, [], len(dev_A_tok), vocab_size_layer,
                               max_len=30, index_from=2)
    dev_set_B = SentenceEncode(dev_B_tok, [], len(dev_B_tok), vocab_size_layer,
                               max_len=30, index_from=2)

    devA = model.get_outputs(dev_set_A)
    devB = model.get_outputs(dev_set_B)

    trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
    devF = np.c_[np.abs(devA - devB), devA * devB]

    trainY = encode_labels(scores_shuf, ndata=len(trainF))
    devY = encode_labels(scores[1], ndata=len(devF))

    lrmodel, opt, cost = prepare_model(ninputs=trainF.shape[1])

    neon_logger.display('Training the regression model...')
    bestlrmodel = train_model(lrmodel, opt, cost, trainF,
                              trainY, devF, devY, scores[1][:len(devF)])

    if evaltest:
        test_A_tok = tokenize_input(test[0], vocab=vocab)
        test_B_tok = tokenize_input(test[1], vocab=vocab)

        test_set_A = SentenceEncode(test_A_tok, [], len(test_A_tok), vocab_size_layer,
                                    max_len=30, index_from=2)
        test_set_B = SentenceEncode(test_B_tok, [], len(test_B_tok), vocab_size_layer,
                                    max_len=30, index_from=2)

        testA = model.get_outputs(test_set_A)
        testB = model.get_outputs(test_set_B)

        testF = np.c_[np.abs(testA - testB), testA * testB]

        neon_logger.display('Evaluating using vectors and linear regression model')
        r = np.arange(1, 6)
        yhat = np.dot(bestlrmodel.get_outputs(ArrayIterator(testF)), r)
        pr = pearsonr(yhat, scores[2][:len(yhat)])[0]
        sr = spearmanr(yhat, scores[2][:len(yhat)])[0]
        se = np.mean((yhat - scores[2][:len(yhat)]) ** 2)
        neon_logger.display('Test Pearson: ' + str(pr))
        neon_logger.display('Test Spearman: ' + str(sr))
        neon_logger.display('Test MSE: ' + str(se))

        return yhat


def prepare_model(ninputs=9600, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    layers = [Affine(nout=nclass, init=Gaussian(loc=0.0, scale=0.01), activation=Softmax())]

    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    opt = Adam()
    lrmodel = Model(layers=layers)

    return lrmodel, opt, cost


def train_model(lrmodel, opt, cost, X, Y, devX, devY, devscores):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    r = np.arange(1, 6)

    train_set = ArrayIterator(X=X, y=Y, make_onehot=False)
    valid_set = ArrayIterator(X=devX, y=devY, make_onehot=False)

    eval_epoch = 10

    while not done:
        callbacks = Callbacks(lrmodel, eval_set=valid_set)

        lrmodel.fit(train_set, optimizer=opt, num_epochs=eval_epoch,
                    cost=cost, callbacks=callbacks)

        # Every 10 epochs, check Pearson on development set
        yhat = np.dot(lrmodel.get_outputs(valid_set), r)
        score = pearsonr(yhat, devscores)[0]
        if score > best:
            neon_logger.display('Dev Pearson: {}'.format(score))
            best = score
            bestlrmodel = copy.copy(lrmodel)
        else:
            done = True

        eval_epoch += 10

    yhat = np.dot(bestlrmodel.get_outputs(valid_set), r)
    score = pearsonr(yhat, devscores)[0]
    neon_logger.display('Dev Pearson: {}'.format(score))
    return bestlrmodel


def encode_labels(labels, nclass=5, ndata=None):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j, i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j, i] = np.floor(y) - y + 1
    if ndata:
        Y = Y[:ndata]

    return Y


def tokenize_input(input_sent, vocab):
    """
    Return a numpy array where each row is the word-indexes for each sentence
    """
    input_tok = []

    # map text to integers
    for sent in input_sent:
        text_int = [-1 if t not in vocab else vocab[t] for t in tokenize(sent)]

        input_tok.append(np.array(text_int))

    return np.array(input_tok)


if __name__ == "__main__":
    main()
