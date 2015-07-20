# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
Boston housing dataset.  Contains median house price and other attributes of
506 Boston suburbs.

More info at:
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names

See: Lichman, M. (2013). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School
of Information and Computer Science.
"""

import logging
import numpy as np
import os

from neon.datasets.dataset import Dataset
from neon.util.param import opt_param


logger = logging.getLogger(__name__)


class Housing(Dataset):

    """
    Sets up a Boston housing dataset.

    Attributes:
        raw_dataset_url (str): where to find the source data
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Keyword Args:
        repo_path (str, optional): where to locally host this dataset on disk
        test_pct (float, optional): How much of the dataset to reserve as a
                                    holdout for testing.
    """
    raw_dataset_url = ('https://archive.ics.uci.edu/ml/'
                       'machine-learning-databases/housing/housing.data')

    def __init__(self, **kwargs):
        opt_param(self, ['test_pct'])
        self.__dict__.update(kwargs)

    def initialize(self):
        pass

    def read_raw_file(self, fname, dtype=None):
        """
        Carries out the actual loading of the raw housing data file.  We assume
        the dataset is laid out exactly as in:
        https://archive.ics.uci.edu/ml/datasets/Housing

        Arguments:
            fname (str): path to raw housing dataset on disk
            dtype (type, optional): type to use to store the contents

        Returns:
            tuple: 2 tuple of numpy ndarrays containing the input field
                   attributes, and target median home values.
        """
        try:
            full_data = np.fromfile(fname, sep=" ",
                                    dtype=dtype).reshape((506, 14))
        except ValueError:
            raise ValueError("Incorrectly shaped data.  Ensure %s is from %s" %
                             (fname,
                              "https://archive.ics.uci.edu/ml/datasets/Housing"
                              ))
        return (full_data[:, 0:-1],
                full_data[:, -1].reshape(full_data.shape[0], 1))

    def load(self, backend=None, experiment=None):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' in self.__dict__:
            self.repo_path = os.path.expandvars(os.path.expanduser(
                self.repo_path))
            save_dir = os.path.join(self.repo_path,
                                    self.__class__.__name__)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            name = os.path.basename(self.raw_dataset_url)
            repo_file = os.path.join(save_dir, name)
            if not os.path.exists(repo_file):
                self.download_to_repo(self.raw_dataset_url, save_dir)
            logger.info('loading: %s', name)
            (inp, lab) = self.read_raw_file(repo_file, 'float32')
            if 'test_pct' in self.__dict__ and self.test_pct > 0.0:
                vals = np.random.rand(inp.shape[0]) * 100
                self.inputs['train'] = inp[vals >= self.test_pct]
                self.targets['train'] = lab[vals >= self.test_pct]
                self.inputs['test'] = inp[vals < self.test_pct]
                self.targets['test'] = lab[vals < self.test_pct]
            else:
                self.inputs['train'] = inp
                self.targets['train'] = lab
            if 'sample_pct' in self.__dict__:
                self.sample_training_data()
            if hasattr(self, 'validation_pct'):
                self.split_set(
                    self.validation_pct, from_set='train', to_set='validation')
            self.format()
        else:
            raise AttributeError('repo_path not specified in config')
