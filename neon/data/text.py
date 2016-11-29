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
Defines text datatset handling.
"""

import logging
import numpy as np
import os
import h5py

from neon.data.dataiterator import NervanaDataIterator, ArrayIterator
from neon.data.datasets import Dataset
from neon.data.text_preprocessing import pad_sentences, pad_data

logger = logging.getLogger(__name__)


class Text(NervanaDataIterator):
    """
    This class defines methods for loading and iterating over text datasets.
    """

    def __init__(self, time_steps, path, vocab=None, tokenizer=None,
                 onehot_input=True, reverse_target=False, get_prev_target=False):
        """
        Construct a text dataset object.

        Arguments:
            time_steps (int) : Length of a sequence.
            path (str) : Path to text file.
            vocab (python.set) : A set of unique tokens.
            tokenizer (function) : Tokenizer function.
            onehot_input (boolean): One-hot representation of input
            reverse_target (boolean): for sequence to sequence models, set to
                                      True to reverse target sequence. Also
                                      disables shifting target by one.
            get_prev_target (boolean): for sequence to sequence models, set to
                                       True for training data to provide correct
                                       target from previous time step as decoder
                                       input. If condition, shape will be a tuple
                                       of shapes, corresponding to encoder and
                                       decoder inputs.
        """
        super(Text, self).__init__(name=None)

        self.seq_length = time_steps
        self.onehot_input = onehot_input
        self.batch_index = 0
        self.reverse_target = reverse_target
        self.get_prev_target = get_prev_target

        X, y = self._get_data(path, tokenizer, vocab)

        # reshape to preserve sentence continuity across batches
        self.X = X.reshape(self.be.bsz, self.nbatches, time_steps)
        self.y = y.reshape(self.be.bsz, self.nbatches, time_steps)

        # stuff below this comment needs to be cleaned up and commented
        self.nout = self.nclass
        if self.onehot_input:
            self.shape = (self.nout, time_steps)
            self.dev_X = self.be.iobuf((self.nout, time_steps))
            if self.get_prev_target:
                self.dev_Z = self.be.iobuf((self.nout, time_steps))
        else:
            self.shape = (time_steps, 1)
            self.dev_X = self.be.iobuf(time_steps, dtype=np.int32)
            if self.get_prev_target:
                self.dev_Z = self.be.iobuf(time_steps, dtype=np.int32)
        self.decoder_shape = self.shape

        self.dev_y = self.be.iobuf((self.nout, time_steps))
        self.dev_lbl = self.be.iobuf(time_steps, dtype=np.int32)
        self.dev_lblflat = self.dev_lbl.reshape((1, -1))

    def _get_data(self, path, tokenizer, vocab):

        text = open(path).read()
        tokens = self.get_tokens(text, tokenizer)

        # make this a static method
        extra_tokens = len(tokens) % (self.be.bsz * self.seq_length)
        if extra_tokens:
            tokens = tokens[:-extra_tokens]
        self.nbatches = len(tokens) // (self.be.bsz * self.seq_length)
        self.ndata = self.nbatches * self.be.bsz  # no leftovers

        self.vocab = sorted(self.get_vocab(tokens, vocab))
        self.nclass = len(self.vocab)

        # vocab dicts
        self.token_to_index = dict((t, i) for i, t in enumerate(self.vocab))
        self.index_to_token = dict((i, t) for i, t in enumerate(self.vocab))

        # map tokens to indices
        X = np.asarray([self.token_to_index[t] for t in tokens], dtype=np.uint32)
        if self.reverse_target:
            y = X.copy()
        else:
            y = np.concatenate((X[1:], X[:1]))

        return X, y

    @staticmethod
    def create_valid_file(path, valid_split=0.1):
        """
        Create separate files for training and validation.

        Arguments:
            path(str): Path to data file.
            valid_split(float, optional): Fraction of data to set aside for validation.

        Returns:
            str, str : Paths to train file and validation file
        """
        text = open(path).read()

        # create train and valid paths
        filename, ext = os.path.splitext(path)
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
    def get_tokens(string, tokenizer=None):
        """
        Map string to a list of tokens.

        Arguments:
            string(str): String to be tokenized.
            token(object): Tokenizer object.
            tokenizer (function) : Tokenizer function.

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

        Arguments:
            tokens(list): List of tokens.
            vocab:  (Default value = None)

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
        """
        Deprecated, use neon.data.text_preprocessing.pad_sentences.
        """
        logger.error('pad_sentences in the Text class is deprecated.  This function '
                     'is now in neon.data.text_preprocessing.')
        return pad_sentences(sentences,
                             sentence_length=sentence_length,
                             dtype=dtype,
                             pad_val=pad_val)

    @staticmethod
    def pad_data(path, vocab_size=20000, sentence_length=100, oov=2,
                 start=1, index_from=3, seed=113, test_split=0.2):
        """
        Deprecated, use neon.data.text_preprocessing.pad_data.
        """
        logger.error('pad_data in the Text class is deprecated.  This function'
                     'is now in neon.data.text_preprocessing')
        return pad_data(path,
                        vocab_size=vocab_size,
                        sentence_length=sentence_length,
                        oov=oov,
                        start=start,
                        index_from=index_from,
                        seed=seed,
                        test_split=test_split)

    def reset(self):
        """
        Reset the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        self.batch_index = 0

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuple : the next minibatch of data.
        """
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            X_batch = self.X[:, self.batch_index, :].T.astype(np.float32, order='C')
            if self.reverse_target is False:
                y_batch = self.y[:, self.batch_index, :].T.astype(np.float32, order='C')
            else:
                # reverse target sequence
                y_batch = self.y[:, self.batch_index, ::-1].T.astype(np.float32, order='C')

            self.dev_lbl.set(y_batch)
            self.dev_y[:] = self.be.onehot(self.dev_lblflat, axis=0)

            if self.onehot_input:
                self.dev_lbl.set(X_batch)
                self.dev_X[:] = self.be.onehot(self.dev_lblflat, axis=0)
                if self.get_prev_target:
                    self.dev_Z[:, self.be.bsz:] = self.dev_y[:, :-self.be.bsz]
                    self.dev_Z[:, 0:self.be.bsz] = 0  # zero-hot, no input
            else:
                self.dev_X.set(X_batch)
                if self.get_prev_target:
                    self.dev_lbl.set(y_batch)
                    self.dev_Z[1:, :] = self.dev_lbl[:-1, :]
                    self.dev_Z[0, :] = 0

            self.batch_index += 1

            if self.get_prev_target:
                yield (self.dev_X, self.dev_Z), self.dev_y
            else:
                yield self.dev_X, self.dev_y


class TextNMT(Text):
    """
    Datasets for neural machine translation on French / English bilingual datasets.
    Available at http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/bitexts.tgz

    Arguments:
        time_steps (int) : Length of a sequence.
        path (str) : Path to text file.
        tokenizer (function) : Tokenizer function.
        onehot_input (boolean): One-hot representation of input
        get_prev_target (boolean): for sequence to sequence models, set to
                               True for training data to provide correct
                               target from previous time step as decoder
                               input. If condition, shape will be a tuple
                               of shapes, corresponding to encoder and
                               decoder inputs.
        split (str): "train" or "valid" split of the dataset
        dataset (str): 'un2000' for the United Nations dataset or 'eurparl7'
                       for the European Parliament datset.
        subset_pct (float): Percentage of the dataset to use (100 is the full dataset)
    """
    def __init__(self, time_steps, path, tokenizer=None,
                 onehot_input=False, get_prev_target=False, split=None,
                 dataset='un2000', subset_pct=100):
        """
        Load French and English sentence data from file.
        """
        assert dataset in ('europarl7', 'un2000'), "invalid dataset"
        processed_file = os.path.join(path, dataset + '-' + split + '.h5')
        assert os.path.exists(processed_file), "Dataset at '" + processed_file + "' not found"
        self.subset_pct = subset_pct

        super(TextNMT, self).__init__(time_steps, processed_file, vocab=None, tokenizer=tokenizer,
                                      onehot_input=onehot_input, get_prev_target=get_prev_target,
                                      reverse_target=True)

    def _get_data(self, path, tokenizer, vocab):
        """
        Tokenizer and vocab are unused but provided to match superclass method signature
        """
        def vocab_to_dicts(vocab):
            t2i = dict((t, i) for i, t in enumerate(vocab))
            i2t = dict((i, t) for i, t in enumerate(vocab))
            return t2i, i2t

        # get saved processed data
        logger.debug("Loading parsed data from %s", path)
        with h5py.File(path, 'r') as f:
            self.s_vocab = f['s_vocab'][:].tolist()
            self.t_vocab = f['t_vocab'][:].tolist()
            self.s_token_to_index, self.s_index_to_token = vocab_to_dicts(self.s_vocab)
            self.t_token_to_index, self.t_index_to_token = vocab_to_dicts(self.t_vocab)
            X = f['X'][:]
            y = f['y'][:]
        self.nclass = len(self.t_vocab)

        # Trim subset and patial minibatch
        if self.subset_pct < 100:
            X = X[:int(X.shape[0] * self.subset_pct / 100.), :]
            y = y[:int(y.shape[0] * self.subset_pct / 100.), :]
            logger.debug("subset %d%% of data", self.subset_pct*100)

        extra_sentences = X.shape[0] % self.be.bsz
        if extra_sentences:
            X = X[:-extra_sentences, :]
            y = y[:-extra_sentences, :]
            logger.debug("removing %d extra sentences", extra_sentences)
        self.nbatches = X.shape[0] // self.be.bsz
        self.ndata = self.nbatches * self.be.bsz  # no leftovers

        return X, y


class Shakespeare(Dataset):
    """
    Shakespeare data set from http://cs.stanford.edu/people/karpathy/char-rnn.
    """
    def __init__(self, timesteps, path='.'):
        url = 'http://cs.stanford.edu/people/karpathy/char-rnn'
        super(Shakespeare, self).__init__('shakespeare_input.txt',
                                          url,
                                          4573338,
                                          path=path)
        self.timesteps = timesteps

    def load_data(self):
        self.filepath = self.load_zip(self.filename, self.size)
        return self.filepath

    def gen_iterators(self):
        self.load_data()
        train_path, valid_path = Text.create_valid_file(self.filepath)
        self._data_dict = {}
        self._data_dict['train'] = Text(self.timesteps, train_path)
        vocab = self._data_dict['train'].vocab
        self._data_dict['valid'] = Text(self.timesteps, valid_path, vocab=vocab)
        return self._data_dict


class PTB(Dataset):
    """
    Penn Treebank data set from http://arxiv.org/pdf/1409.2329v5.pdf

    Arguments:
        timesteps (int): number of timesteps to embed the data
        onehot_input (bool):
        tokenizer (str): name of the tokenizer function within this
                         class to use on the data
    """
    def __init__(self, timesteps, path='.',
                 onehot_input=True,
                 tokenizer=None,
                 reverse_target=False,
                 get_prev_target=False):
        url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data'
        self.filemap = {'train': 5101618,
                        'test': 449945,
                        'valid': 399782}
        keys = list(self.filemap.keys())
        filenames = [self.gen_filename(phase) for phase in keys]
        sizes = [self.filemap[phase] for phase in keys]
        super(PTB, self).__init__(filenames,
                                  url,
                                  sizes,
                                  path=path)
        self.timesteps = timesteps
        self.onehot_input = onehot_input
        self.tokenizer = tokenizer
        if tokenizer is not None:
            assert hasattr(self, self.tokenizer)
            self.tokenizer_func = getattr(self, self.tokenizer)
        else:
            self.tokenizer_func = None

        self.reverse_target = reverse_target
        self.get_prev_target = get_prev_target

    @staticmethod
    def newline_tokenizer(s):
        """
        Tokenizer which breaks on newlines.

        Arguments:
            s (str): String to tokenize.

        Returns:
            str: String with "<eos>" in place of newlines.

        """
        # replace newlines with '<eos>' so that
        # the newlines count as words
        return s.replace('\n', '<eos>').split()

    @staticmethod
    def gen_filename(phase):
        """
        Filename generator.

        Arguments:
            phase(str): Phase

        Returns:
            string: ptb.<phase>.txt

        """
        return 'ptb.%s.txt' % phase

    def load_data(self):
        self.file_paths = {}
        for phase in self.filemap:
            fn = self.gen_filename(phase)
            size = self.filemap[phase]
            self.file_paths[phase] = self.load_zip(fn, size)
        return self.file_paths

    def gen_iterators(self):
        self.load_data()

        self._data_dict = {}
        self.vocab = None
        for phase in ['train', 'test', 'valid']:
            file_path = self.file_paths[phase]
            get_prev_target = self.get_prev_target if phase is 'train' else False
            self._data_dict[phase] = Text(self.timesteps,
                                          file_path,
                                          tokenizer=self.tokenizer_func,
                                          onehot_input=self.onehot_input,
                                          vocab=self.vocab,
                                          reverse_target=self.reverse_target,
                                          get_prev_target=get_prev_target)
            if self.vocab is None:
                self.vocab = self._data_dict['train'].vocab
        return self._data_dict


class HutterPrize(Dataset):
    """
    Hutter Prize data set from http://prize.hutter1.net/
    """
    def __init__(self, path='.'):
        super(HutterPrize, self).__init__('enwik8.zip',
                                          'http://mattmahoney.net/dc',
                                          35012219,
                                          path=path)

    def load_data(self):
        self.filepath = self.load_zip(self.filename, self.size)
        return self.filepath


class IMDB(Dataset):
    """
    IMDB data set from http://www.aclweb.org/anthology/P11-1015..
    """
    def __init__(self, vocab_size, sentence_length, path='.'):
        url = 'https://s3.amazonaws.com/text-datasets'
        super(IMDB, self).__init__('imdb.pkl',
                                   url,
                                   33213513,
                                   path=path)
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.filepath = None

    def load_data(self):
        self.filepath = self.load_zip(self.filename, self.size)
        return self.filepath

    def gen_iterators(self):
        if self.filepath is None:
            self.load_data()

        data = pad_data(self.filepath, vocab_size=self.vocab_size,
                        sentence_length=self.sentence_length)
        (X_train, y_train), (X_test, y_test), nclass = data

        self._data_dict = {'nclass': nclass}
        self._data_dict['train'] = ArrayIterator(X_train, y_train, nclass=2)
        self._data_dict['test'] = ArrayIterator(X_test, y_test, nclass=2)
        return self._data_dict


class SICK(Dataset):
    """
    Semantic Similarity dataset from qcri.org (Semeval 2014).

    Arguments:
        path (str): path to SICK_data directory
    """
    def __init__(self, path='SICK_data/'):
        url = 'http://alt.qcri.org/semeval2014/task1/data/uploads/'
        self.filemap = {'train': 87341,
                        'test_annotated': 93443,
                        'trial': 16446}
        keys = list(self.filemap.keys())
        self.zip_paths = None
        self.file_paths = [self.gen_filename(phase) for phase in keys]
        self.sizes = [self.filemap[phase] for phase in keys]

        super(SICK, self).__init__(filename=self.file_paths,
                                   url=url,
                                   size=self.sizes,
                                   path=path)

    @staticmethod
    def gen_zipname(phase):
        """
        Zip filename generator.

        Arguments:
            phase(str): Phase of training/evaluation

        Returns:
            string: sick_<phase>.zip
        """
        return "sick_{}.zip".format(phase)

    @staticmethod
    def gen_filename(phase):
        """
        Filename generator for the extracted zip files.

        Arguments:
            phase(str): Phase of training/evaluation

        Returns:
            string: SICK_<phase>.txt
        """
        return "SICK_{}.txt".format(phase)

    def load_data(self):
        """
        Conditional data loader will download and extract zip files if not found locally.
        """
        self.zip_paths = {}
        for phase in self.filemap:
            zn = self.gen_zipname(phase)
            size = self.filemap[phase]
            self.zip_paths[phase] = self.load_zip(zn, size)
        return self.zip_paths

    def load_eval_data(self):
        """
        Load the SICK semantic-relatedness dataset. Data is a tab-delimited txt file,
        in the format: Sentence1\tSentence2\tScore. Data is downloaded and extracted
        from zip files if not found in directory specified by self.path.

        Returns:
            tuple of tuples of np.array: three tuples containing A & B sentences
                for train, dev, and text, along with a fourth tuple containing
                the scores for each AB pair.
        """
        if self.zip_paths is None:
            self.load_data()

        trainA, trainB, devA, devB, testA, testB = [], [], [], [], [], []
        trainS, devS, testS = [], [], []

        with open(self.path + self.gen_filename('train'), 'rb') as f:
            for line in f:
                text = line.strip().split(b'\t')
                trainA.append(text[1])
                trainB.append(text[2])
                trainS.append(text[3])

        with open(self.path + self.gen_filename('trial'), 'rb') as f:
            for line in f:
                text = line.strip().split(b'\t')
                devA.append(text[1])
                devB.append(text[2])
                devS.append(text[3])

        with open(self.path + self.gen_filename('test_annotated'), 'rb') as f:
            for line in f:
                text = line.strip().split(b'\t')
                testA.append(text[1])
                testB.append(text[2])
                testS.append(text[3])

        trainS = [float(s) for s in trainS[1:]]
        devS = [float(s) for s in devS[1:]]
        testS = [float(s) for s in testS[1:]]

        return ((np.array(trainA[1:]), np.array(trainB[1:])),
                (np.array(devA[1:]), np.array(devB[1:])),
                (np.array(testA[1:]), np.array(testB[1:])),
                (np.array(trainS), np.array(devS), np.array(testS)))
