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
CIFAR-10 contains color images of 10 classes.
More info at: http://www.cs.toronto.edu/~kriz/cifar.html
"""

import logging
import numpy as np
import os
import tarfile

from neon.datasets.dataset import Dataset
from neon.util.compat import range
from neon.util.persist import deserialize


logger = logging.getLogger(__name__)


class CIFAR10(Dataset):

    """
    Sets up a CIFAR-10 dataset.

    Attributes:
        url (str): where to find the source data
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self, **kwargs):
        self.macro_batched = False
        self.__dict__.update(kwargs)

    def initialize(self):
        pass

    def fetch_dataset(self, save_dir):
        save_dir = os.path.expandvars(os.path.expanduser(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        repo_gz_file = os.path.join(save_dir, os.path.basename(self.url))
        if not os.path.exists(repo_gz_file):
            self.download_to_repo(self.url, save_dir)

        data_file = os.path.join(save_dir, 'cifar-10-batches-py', 'test_batch')
        if not os.path.exists(data_file):
            logger.info('untarring: %s', repo_gz_file)
            infile = tarfile.open(repo_gz_file)
            infile.extractall(save_dir)
            infile.close()

    def load_file(self, filename, nclasses):
        logger.info('loading: %s', filename)
        dict = deserialize(filename)

        full_image = np.float32(dict['data'])
        full_image /= 255.

        labels = np.array(dict['labels'])
        onehot = np.zeros((len(labels), nclasses), dtype='float32')
        for col in range(nclasses):
            onehot[:, col] = (labels == col)
        return (full_image, onehot)

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' in self.__dict__:
            self.repo_path = os.path.expandvars(os.path.expanduser(
                self.repo_path))
            ncols = 32 * 32 * 3

            ntrain_total = 50000
            nclasses = 10
            save_dir = os.path.join(self.repo_path,
                                    self.__class__.__name__)
            self.fetch_dataset(save_dir)
            self.inputs['train'] = np.zeros((ntrain_total, ncols),
                                            dtype='float32')
            self.targets['train'] = np.zeros((ntrain_total, nclasses),
                                             dtype='float32')
            for i in range(5):
                filename = os.path.join(save_dir, 'cifar-10-batches-py',
                                        'data_batch_' + str(i + 1))
                data, labels = self.load_file(filename, nclasses)
                nrows = data.shape[0]
                start = i * nrows
                end = (i + 1) * nrows
                self.inputs['train'][start:end] = data
                self.targets['train'][start:end] = labels

            if 'sample_pct' in self.__dict__:
                self.sample_training_data()

            filename = os.path.join(save_dir, 'cifar-10-batches-py',
                                    'test_batch')
            data, labels = self.load_file(filename, nclasses)
            self.inputs['test'] = np.zeros((data.shape[0], ncols),
                                           dtype='float32')
            self.targets['test'] = np.zeros((data.shape[0], nclasses),
                                            dtype='float32')
            self.inputs['test'][:] = data
            self.targets['test'][:] = labels
            self.format()
        else:
            raise AttributeError('repo_path not specified in config')
