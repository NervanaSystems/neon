# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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
from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below
from builtins import map  # noqa
import numpy as np  # noqa
import re  # noqa
from neon.util.compat import pickle  # noqa


def clean_string(base):
    """
    Tokenization/string cleaning.
    Original from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    base = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", base)
    base = re.sub(r"\'re", " \'re", base)
    base = re.sub(r"\'d", " \'d", base)
    base = re.sub(r"\'ll", " \'ll", base)
    base = re.sub(r"\'s", " \'s", base)
    base = re.sub(r"\'ve", " \'ve", base)
    base = re.sub(r"n\'t", " n\'t", base)
    base = re.sub(r"!", " ! ", base)
    base = re.sub(r",", " , ", base)
    base = re.sub(r"\)", " \) ", base)
    base = re.sub(r"\(", " \( ", base)
    base = re.sub(r"\?", " \? ", base)
    base = re.sub(r"\s{2,}", " ", base)
    return base.strip().lower()


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
    X, y = pickle.load(f)
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

    X_train = X[:int(len(X) * (1 - test_split))]
    y_train = y[:int(len(X) * (1 - test_split))]

    X_test = X[int(len(X) * (1 - test_split)):]
    y_test = y[int(len(X) * (1 - test_split)):]

    X_train = pad_sentences(X_train, sentence_length=sentence_length)
    y_train = np.array(y_train).reshape((len(y_train), 1))

    X_test = pad_sentences(X_test, sentence_length=sentence_length)
    y_test = np.array(y_test).reshape((len(y_test), 1))

    nclass = 1 + max(np.max(y_train), np.max(y_test))

    return (X_train, y_train), (X_test, y_test), nclass


def get_paddedXY(X, y, vocab_size=20000, sentence_length=100, oov=2,
                 start=1, index_from=3, seed=113, shuffle=True):

    if shuffle:
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

    # word ids - pad (0), start (1), oov (2)
    if oov is not None:
        X = [[oov if w >= vocab_size else w for w in x] for x in X]
    else:
        X = [[w for w in x if w < vocab_size] for x in X]

    X = pad_sentences(X, sentence_length=sentence_length)
    y = np.array(y, dtype=np.int32).reshape((len(y), 1))

    return X, y


def get_google_word2vec_W(fname, vocab, vocab_size=1000000, index_from=3):
    """
    Extract the embedding matrix from the given word2vec binary file and use this
    to initalize a new embedding matrix for words found in vocab.

    Conventions are to save indices for pad, oov, etc.:
    index 0: pad
    index 1: oov (or <unk>)
    index 2: <eos>. But often cases, the <eos> has already been in the
    preprocessed data, so no need to save an index for <eos>
    """
    f = open(fname, 'rb')
    header = f.readline()
    vocab1_size, embedding_dim = list(map(int, header.split()))
    binary_len = np.dtype('float32').itemsize * embedding_dim
    vocab_size = min(len(vocab) + index_from, vocab_size)
    W = np.zeros((vocab_size, embedding_dim))

    found_words = {}
    for i, line in enumerate(range(vocab1_size)):
        word = []
        while True:
            ch = f.read(1)
            if ch == b' ':
                word = b''.join(word)
                break
            if ch != '\n':
                word.append(ch)
        if word in vocab:
            wrd_id = vocab[word] + index_from
            if wrd_id < vocab_size:
                W[wrd_id] = np.fromstring(
                    f.read(binary_len), dtype='float32')
                found_words[wrd_id] = 1
        else:
            f.read(binary_len)

    cnt = 0
    for wrd_id in range(vocab_size):
        if wrd_id not in found_words:
            W[wrd_id] = np.random.uniform(-0.25, 0.25, embedding_dim)
            cnt += 1
    assert cnt + len(found_words) == vocab_size

    f.close()

    return W, embedding_dim, vocab_size
