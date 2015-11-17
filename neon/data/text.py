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
Defines text datatset handling.
"""

import numpy as np

from neon import NervanaObject
import cPickle


class Text(NervanaObject):

    """
    This class defines methods for loading and iterating over text datasets.
    """

    @staticmethod
    def create_valid_file(path, valid_split=0.1):
        """
        Create separate files for training and validation.

        Args:
            path (str) : Path to data file.
            valid_split (float) : Fraction of data to set aside for validation.

        Returns:
            str, str : Paths to train file and validation file
        """
        text = open(path).read()

        # create train and valid paths
        from os.path import splitext
        filename, ext = splitext(path)
        train_path = filename + '_train' + ext
        valid_path = filename + '_valid' + ext

        # split data
        train_split = int(len(text) * (1 - valid_split))
        train_text = text[:train_split]
        valid_text = text[train_split:]

        # write train file
        with open(train_path, 'w') as train_file:
            train_file.write(train_text)

        # write valid file
        with open(valid_path, 'w') as valid_file:
            valid_file.write(valid_text)

        return train_path, valid_path

    @staticmethod
    def get_tokens(string,  tokenizer=None):
        """
        Map string to a list of tokens.

        Args:
            string (str) : String to be tokenized.
            token (object) : Tokenizer object.

        Returns:
            list : A list of tokens
        """
        # (if tokenizer is None, we have a list of characters)
        if tokenizer is None:
            return string
        else:
            return tokenizer(string)

    @staticmethod
    def get_vocab(tokens, vocab=None):
        """
        Construct vocabulary from the given tokens.

        Args:
            tokens (list) : List of tokens.

        Returns:
            python.set : A set of unique tokens
        """
        # (if vocab is not None, we check that it contains all tokens)
        if vocab is None:
            return set(tokens)
        else:
            vocab = set(vocab)
            assert vocab >= set(tokens), "the predefined vocab must contain all the tokens"
            return vocab

    @staticmethod
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

    @staticmethod
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

        X_train = Text.pad_sentences(X_train, sentence_length=sentence_length)
        y_train = np.array(y_train).reshape((len(y_train), 1))

        X_test = Text.pad_sentences(X_test, sentence_length=sentence_length)
        y_test = np.array(y_test).reshape((len(y_test), 1))

        nclass = 1 + max(np.max(y_train), np.max(y_test))

        return (X_train, y_train), (X_test, y_test), nclass

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        pass

    def __init__(self, time_steps, path, vocab=None, tokenizer=None,
                 onehot_input=True):
        """
        Construct a text dataset object.

        Args:
            time_steps (int) : Length of a sequence.
            path (str) : Path to text file.
            vocab (python.set) : A set of unique tokens.
            tokenizer (object) : Tokenizer object.
            onehot_input (boolean): One-hot representation of input
        """
        # figure out how to remove seq_length from the dataloader
        self.seq_length = time_steps
        self.onehot_input = onehot_input
        self.batch_index = 0

        text = open(path).read()
        tokens = self.get_tokens(text, tokenizer)

        # make this a static method
        extra_tokens = len(tokens) % (self.be.bsz * time_steps)
        if extra_tokens:
            tokens = tokens[:-extra_tokens]
        self.nbatches = len(tokens) / (self.be.bsz * time_steps)
        self.ndata = self.nbatches * self.be.bsz  # no leftovers

        self.vocab = sorted(self.get_vocab(tokens, vocab))
        self.nclass = len(self.vocab)

        # vocab dicts
        self.token_to_index = dict((t, i) for i, t in enumerate(self.vocab))
        self.index_to_token = dict((i, t) for i, t in enumerate(self.vocab))

        # map tokens to indices
        X = np.asarray([self.token_to_index[t] for t in tokens], dtype=np.uint32)
        y = np.concatenate((X[1:], X[:1]))

        # reshape to preserve sentence continuity across batches
        self.X = X.reshape(self.be.bsz, self.nbatches, time_steps)
        self.y = y.reshape(self.be.bsz, self.nbatches, time_steps)

        # stuff below this comment needs to be cleaned up and commented
        self.nout = len(self.vocab)
        if self.onehot_input:
            self.shape = (self.nout, time_steps)
            self.dev_X = self.be.iobuf((self.nout, time_steps))
        else:
            self.shape = (time_steps, 1)
            self.dev_X = self.be.iobuf(time_steps, dtype=np.int32)

        self.dev_y = self.be.iobuf((self.nout, time_steps))
        self.dev_lbl = self.be.iobuf(time_steps, dtype=np.int32)
        self.dev_lblflat = self.dev_lbl.reshape((1, -1))

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuple : the next minibatch of data.
        """
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            X_batch = self.X[:, self.batch_index, :].T.astype(np.float32, order='C')
            y_batch = self.y[:, self.batch_index, :].T.astype(np.float32, order='C')

            if self.onehot_input:
                self.dev_lbl.set(X_batch)
                self.dev_X[:] = self.be.onehot(self.dev_lblflat, axis=0)
            else:
                self.dev_X.set(X_batch)

            self.dev_lbl.set(y_batch)
            self.dev_y[:] = self.be.onehot(self.dev_lblflat, axis=0)

            self.batch_index += 1

            yield self.dev_X, self.dev_y
