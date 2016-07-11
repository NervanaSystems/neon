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
from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below
import h5py  # noqa
from collections import defaultdict  # noqa
import numpy as np  # noqa
import os  # noqa
from neon import logger as neon_logger  # noqa
from neon.data.text_preprocessing import clean_string  # noqa
from neon.util.compat import pickle  # noqa


def build_data_train(path='.', filepath='labeledTrainData.tsv', vocab_file=None,
                     vocab=None, skip_headers=True, train_ratio=0.8):
    """
    Loads the data file and spits out a h5 file with record of
    {y, review_text, review_int}
    Typically two passes over the data.
    1st pass is for vocab and pre-processing. (WARNING: to get phrases, we need to go
    though multiple passes). 2nd pass is converting text into integers. We will deal with integers
    from thereafter.

    WARNING: we use h5 just as proof of concept for handling large datasets
    Datasets may fit entirely in memory as numpy as array

    """

    fname_h5 = filepath + '.h5'
    if vocab_file is None:
        fname_vocab = filepath + '.vocab'
    else:
        fname_vocab = vocab_file

    if not os.path.exists(fname_h5) or not os.path.exists(fname_vocab):
        # create the h5 store - NOTE: hdf5 is row-oriented store and we slice rows
        # reviews_text holds the metadata and processed text file
        # reviews_int holds the ratings, ints
        h5f = h5py.File(fname_h5, 'w')
        shape, maxshape = (2 ** 16,), (None, )
        dt = np.dtype([('y', np.uint8),
                       ('split', np.bool),
                       ('num_words', np.uint16),
                       # WARNING: vlen=bytes in python 3
                       ('text', h5py.special_dtype(vlen=str))
                       ])
        reviews_text = h5f.create_dataset('reviews', shape=shape, maxshape=maxshape,
                                          dtype=dt, compression='gzip')
        reviews_train = h5f.create_dataset(
            'train', shape=shape, maxshape=maxshape,
            dtype=h5py.special_dtype(vlen=np.int32), compression='gzip')

        reviews_valid = h5f.create_dataset(
            'valid', shape=shape, maxshape=maxshape,
            dtype=h5py.special_dtype(vlen=np.int32), compression='gzip')

        wdata = np.zeros((1, ), dtype=dt)

        # init vocab only for train data
        build_vocab = False
        if vocab is None:
            vocab = defaultdict(int)
            build_vocab = True
        nsamples = 0

        # open the file, skip the headers if needed
        f = open(filepath, 'r')
        if skip_headers:
            f.readline()

        for i, line in enumerate(f):
            _, rating, review = line.strip().split('\t')

            # clean the review
            review = clean_string(review)
            review_words = review.strip().split()
            num_words = len(review_words)
            split = int(np.random.rand() < train_ratio)

            # create record
            wdata['y'] = int(float(rating))
            wdata['text'] = review
            wdata['num_words'] = num_words
            wdata['split'] = split
            reviews_text[i] = wdata

            # update the vocab if needed
            if build_vocab:
                for word in review_words:
                    vocab[word] += 1

            nsamples += 1

        # histogram of class labels, sentence length
        ratings, counts = np.unique(
            reviews_text['y'][:nsamples], return_counts=True)
        sen_len, sen_len_counts = np.unique(
            reviews_text['num_words'][:nsamples], return_counts=True)
        vocab_size = len(vocab)
        nclass = len(ratings)
        reviews_text.attrs['vocab_size'] = vocab_size
        reviews_text.attrs['nrows'] = nsamples
        reviews_text.attrs['nclass'] = nclass
        reviews_text.attrs['class_distribution'] = counts
        neon_logger.display("vocabulary size - {}".format(vocab_size))
        neon_logger.display("# of samples - {}".format(nsamples))
        neon_logger.display("# of classes {}".format(nclass))
        neon_logger.display("class distribution - {} {}".format(ratings, counts))
        sen_counts = list(zip(sen_len, sen_len_counts))
        sen_counts = sorted(sen_counts, key=lambda kv: kv[1], reverse=True)
        neon_logger.display("sentence length - {} {} {}".format(len(sen_len),
                                                                sen_len, sen_len_counts))

        # WARNING: assume vocab is of order ~4-5 million words.
        # sort the vocab , re-assign ids by its frequency. Useful for downstream tasks
        # only done for train data
        if build_vocab:
            vocab_sorted = sorted(
                list(vocab.items()), key=lambda kv: kv[1], reverse=True)
            vocab = {}
            for i, t in enumerate(list(zip(*vocab_sorted))[0]):
                vocab[t] = i

        # map text to integers
        ntrain = 0
        nvalid = 0
        for i in range(nsamples):
            text = reviews_text[i]['text']
            y = int(reviews_text[i]['y'])
            split = reviews_text[i]['split']
            text_int = [y] + [vocab[t] for t in text.strip().split()]
            if split:
                reviews_train[ntrain] = text_int
                ntrain += 1
            else:
                reviews_valid[nvalid] = text_int
                nvalid += 1
        reviews_text.attrs['ntrain'] = ntrain
        reviews_text.attrs['nvalid'] = nvalid
        neon_logger.display(
            "# of train - {0}, # of valid - {1}".format(reviews_text.attrs['ntrain'],
                                                        reviews_text.attrs['nvalid']))
        # close open files
        h5f.close()
        f.close()

    if not os.path.exists(fname_vocab):
        rev_vocab = {}
        for wrd, wrd_id in vocab.items():
            rev_vocab[wrd_id] = wrd
        neon_logger.display("vocabulary from IMDB dataset is saved into {}".format(fname_vocab))
        pickle.dump((vocab, rev_vocab), open(fname_vocab, 'wb'), 2)

    return fname_h5, fname_vocab
