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
Sparsenet is the natural image dataset used by Olshausen and Field
More info at: http://redwood.berkeley.edu/bruno/sparsenet/
"""

import logging
import os

import numpy
import pickle

from neon.util.compat import PY3, range

from neon.datasets.dataset import Dataset

if PY3:
    from urllib.parse import urljoin as basejoin
else:
    from urllib import basejoin

logger = logging.getLogger(__name__)


class SPARSENET(Dataset):
    """
    Sets up a Sparsenet dataset.

    Attributes:
        raw_base_url (str): where to find the source data
        raw_train_input_gz (str): URL of the full path to raw train inputs
        raw_train_target_gz (str): URL of the full path to raw train targets
        raw_test_input_gz (str): URL of the full path to raw test inputs
        raw_test_target_gz (str): URL of the full path to raw test targets
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """
    raw_base_url = 'http://redwood.berkeley.edu/bruno/sparsenet/'
    raw_train_whitened = basejoin(raw_base_url, 'IMAGES.mat')
    raw_train_unwhitened = basejoin(raw_base_url, 'IMAGES_RAW.mat')

    def __init__(self, **kwargs):
        self.macro_batched = False
        self.__dict__.update(kwargs)

    def read_image_file(self, fname, dtype=None):
        """
        Carries out the actual reading of Sparsenet image files.
        """
        logger.info("in read_image_file, reading: %s", fname)
        with open(fname, 'rb') as infile:
            array = pickle.load(infile)
            infile.close()
        return array

    def load(self):
        """
        main function
        """
        import scipy.io
        if 'repo_path' in self.__dict__:
            self.repo_path = os.path.expandvars(os.path.expanduser(
                                                self.repo_path))
            save_dir = os.path.join(self.repo_path,
                                    self.__class__.__name__)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            train_idcs = list(range(10000))
            if 'sample_pct' in self.__dict__:
                if self.sample_pct > 1.0:
                    self.sample_pct /= 100.0
                if self.sample_pct < 1.0:
                    numpy.random.seed(self.backend.rng_seed)
                    numpy.random.shuffle(train_idcs)
                train_idcs = train_idcs[0:int(10000 * self.sample_pct)]
            for url in (self.raw_train_unwhitened, self.raw_train_whitened):
                name = os.path.basename(url).rstrip('.mat')
                repo_mat_file = os.path.join(save_dir, name + '.mat')
                repo_file = repo_mat_file.rstrip('.mat')
                # download and create dataset
                if not os.path.exists(repo_file):
                    self.download_to_repo(url, save_dir)
                    infile = scipy.io.loadmat(repo_mat_file)
                    with open(repo_file, 'wb') as outfile:
                        data = infile[infile.keys()[0]]
                        # patches are extracted so they can be cached
                        # doing non-overlapping 16x16 patches (1024 per image)
                        patches = data.reshape(512/16, 16, 512/16, 16, 10)
                        patches = patches.transpose(1, 3, 0, 2, 4)
                        patches = patches.reshape(16, 16, 1024*10)
                        logger.info("Caching to pickle file: %s", outfile)
                        pickle.dump(patches, outfile)
                        outfile.close()
                logger.info('loading: %s', name)
                # load existing data
                if 'IMAGES' in repo_file:
                    indat = self.read_image_file(repo_file, 'float32')
                    # flatten to 1D images
                    indat = indat.reshape((256, 10240)).transpose()[train_idcs]
                    self.inputs['train'] = indat
                else:
                    logger.error('problems loading: %s', name)
            self.format()
        else:
            raise AttributeError('repo_path not specified in config')
            # TODO: try and download and read in directly?
