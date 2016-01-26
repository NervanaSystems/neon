# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
Defines text datatset preprocessing routines
"""

import cPickle
import numpy as np


def pad_sentences(sentences, sentence_length=None, dtype=np.int32, pad_val=0.):
    lengths = [len(sent) for sent in sentences]

    nsamples = len(sentences)
    if sentence_length is None:
        sentence_length = np.max(lengths)

    X = (np.ones((nsamples, sentence_length)) * pad_val).astype(dtype=np.int32)
    for i, sent in enumerate(sentences):
        trunc = sent[-sentence_length:]
        X[i, -len(trunc):] = trunc
    return X


def pad_data(path, vocab_size=20000, sentence_length=100, oov=2,
             start=1, index_from=3, seed=113, test_split=0.2):

    f = open(path, 'rb')
    X, y = cPickle.load(f)
    f.close()

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    if start is not None:
        X = [[start] + [w + index_from for w in x] for x in X]
    else:
        X = [[w + index_from for w in x] for x in X]

    if not vocab_size:
        vocab_size = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1
    # (start), 2 (OOV)
    if oov is not None:
        X = [[oov if w >= vocab_size else w for w in x] for x in X]

    X_train = X[:int(len(X)*(1-test_split))]
    y_train = y[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = y[int(len(X)*(1-test_split)):]

    X_train = pad_sentences(X_train, sentence_length=sentence_length)
    y_train = np.array(y_train).reshape((len(y_train), 1))

    X_test = pad_sentences(X_test, sentence_length=sentence_length)
    y_test = np.array(y_test).reshape((len(y_test), 1))

    nclass = 1 + max(np.max(y_train), np.max(y_test))

    return (X_train, y_train), (X_test, y_test), nclass
