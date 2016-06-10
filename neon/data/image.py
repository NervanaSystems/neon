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
Class definitions for image data sets.
"""
from __future__ import division
from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below
from builtins import str  # noqa

import gzip  # noqa
import logging  # noqa
import numpy as np  # noqa
import os  # noqa
import tarfile  # noqa

from neon.util.compat import pickle  # noqa
from neon.util.compat import pickle_load  # noqa
from neon.data.datasets import Dataset  # noqa
from neon.data.dataiterator import ArrayIterator  # noqa

logger = logging.getLogger(__name__)


class MNIST(Dataset):
    """
    MNIST data set from http://yann.lecun.com/exdb/mnist/
    """
    def __init__(self, path='.', subset_pct=100, normalize=True):
        super(MNIST, self).__init__('mnist.pkl.gz',
                                    'https://s3.amazonaws.com/img-datasets',
                                    15296311,
                                    path=path,
                                    subset_pct=subset_pct)
        self.normalize = normalize

    def load_data(self):
        """
        Fetch the MNIST dataset and load it into memory.

        Arguments:
            path (str, optional): Local directory in which to cache the raw
                                  dataset.  Defaults to current directory.
            normalize (bool, optional): Whether to scale values between 0 and 1.
                                        Defaults to True.

        Returns:
            tuple: Both training and test sets are returned.
        """
        filepath = self._valid_path_append(self.path, self.filename)
        if not os.path.exists(filepath):
            self.fetch_dataset(self.url, self.filename, filepath, self.size)

        with gzip.open(filepath, 'rb') as mnist:
            (X_train, y_train), (X_test, y_test) = pickle_load(mnist)
            X_train = X_train.reshape(-1, 784)
            X_test = X_test.reshape(-1, 784)

            if self.normalize:
                X_train = X_train / 255.
                X_test = X_test / 255.

        return (X_train, y_train), (X_test, y_test), 10

    def gen_iterators(self):
        (X_train, y_train), (X_test, y_test), nclass = self.load_data()
        train = ArrayIterator(X_train,
                              y_train,
                              nclass=nclass,
                              lshape=(1, 28, 28),
                              name='train')
        val = ArrayIterator(X_test,
                            y_test,
                            nclass=nclass,
                            lshape=(1, 28, 28),
                            name='valid')
        self.data_dict = {'train': train,
                          'valid': val}
        return self.data_dict


class CIFAR10(Dataset):
    """
    CIFAR10 data set from https://www.cs.toronto.edu/~kriz/cifar.html

    Arguments:
        path (str): Local path to copy data files.
        normalize (bool): Flag to normalize data.
        whiten (bool): Flag to apply whitening transform.
        pad_classes (bool): Flag to pad out class count to 16
                            for compatibility with conv layers on GPU.
    """
    def __init__(self, path='.', subset_pct=100, normalize=True,
                 contrast_normalize=False, whiten=False, pad_classes=False):
        super(CIFAR10, self).__init__('cifar-10-python.tar.gz',
                                      'http://www.cs.toronto.edu/~kriz',
                                      170498071,
                                      path=path,
                                      subset_pct=subset_pct)
        # CIFAR10 load method specific options
        self.normalize = normalize
        self.contrast_normalize = contrast_normalize
        self.whiten = whiten
        self.pad_classes = pad_classes

    def load_data(self):
        """
        Fetch the CIFAR-10 dataset and load it into memory.

        Arguments:
            path (str, optional): Local directory in which to cache the raw
                                  dataset.  Defaults to current directory.
            normalize (bool, optional): Whether to scale values between 0 and 1.
                                        Defaults to True.

        Returns:
            tuple: Both training and test sets are returned.
        """
        workdir, filepath = self._valid_path_append(self.path, '', self.filename)
        batchdir = os.path.join(workdir, 'cifar-10-batches-py')
        if not os.path.exists(os.path.join(batchdir, 'data_batch_1')):
            if not os.path.exists(filepath):
                self.fetch_dataset(self.url, self.filename, filepath, self.size)
            with tarfile.open(filepath, 'r:gz') as f:
                f.extractall(workdir)

        train_batches = [os.path.join(batchdir, 'data_batch_' + str(i)) for i in range(1, 6)]
        Xlist, ylist = [], []
        for batch in train_batches:
            with open(batch, 'rb') as f:
                d = pickle_load(f)
                Xlist.append(d['data'])
                ylist.append(d['labels'])

        X_train = np.vstack(Xlist)
        y_train = np.vstack(ylist)

        with open(os.path.join(batchdir, 'test_batch'), 'rb') as f:
            d = pickle_load(f)
            X_test, y_test = d['data'], d['labels']

        y_train = y_train.reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)

        if self.contrast_normalize:
            norm_scale = 55.0  # Goodfellow
            X_train = self.global_contrast_normalize(X_train, scale=norm_scale)
            X_test = self.global_contrast_normalize(X_test, scale=norm_scale)

        if self.normalize:
            X_train = X_train / 255.
            X_test = X_test / 255.

        if self.whiten:
            zca_cache = os.path.join(workdir, 'cifar-10-zca-cache.pkl')
            X_train, X_test = self.zca_whiten(X_train, X_test, cache=zca_cache)

        return (X_train, y_train), (X_test, y_test), 10

    def gen_iterators(self):
        datasets = self.load_data()

        (X_train, y_train), (X_test, y_test), nclass = datasets
        if self.pad_classes:
            nclass = 16

        train = ArrayIterator(X_train,
                              y_train,
                              nclass=nclass,
                              lshape=(3, 32, 32),
                              name='train')
        test = ArrayIterator(X_test,
                             y_test,
                             nclass=nclass,
                             lshape=(3, 32, 32),
                             name='valid')
        self.data_dict = {'train': train,
                          'valid': test}
        return self.data_dict

    @staticmethod
    def _compute_zca_transform(imgs, filter_bias=0.1):
        """
        Compute the zca whitening transform matrix.
        """
        logger.info("Computing ZCA transform matrix")
        meanX = np.mean(imgs, 0)

        covX = np.cov(imgs.T)
        D, E = np.linalg.eigh(covX)

        assert not np.isnan(D).any()
        assert not np.isnan(E).any()
        assert D.min() > 0

        D = D ** -.5

        W = np.dot(E, np.dot(np.diag(D), E.T))
        return meanX, W

    @staticmethod
    def zca_whiten(train, test, cache=None):
        """
        Use train set statistics to apply the ZCA whitening transform to
        both train and test sets.
        """
        if cache and os.path.isfile(cache):
            with open(cache, 'rb') as f:
                (meanX, W) = pickle_load(f)
        else:
            meanX, W = CIFAR10._compute_zca_transform(train)
            if cache:
                logger.info("Caching ZCA transform matrix")
                with open(cache, 'wb') as f:
                    pickle.dump((meanX, W), f, 2)

        logger.info("Applying ZCA whitening transform")
        train_w = np.dot(train - meanX, W)
        test_w = np.dot(test - meanX, W)

        return train_w, test_w

    @staticmethod
    def global_contrast_normalize(X, scale=1., min_divisor=1e-8):
        """
        Subtract mean and normalize by vector norm.
        """

        X = X - X.mean(axis=1)[:, np.newaxis]

        normalizers = np.sqrt((X ** 2).sum(axis=1)) / scale
        normalizers[normalizers < min_divisor] = 1.

        X /= normalizers[:, np.newaxis]

        return X
