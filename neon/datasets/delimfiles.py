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
Raw dataset to be read from one or more delimited plain-text input files.
First subdirectory names can be used to separate train, test, validation
partitions.  If no dir or dir name doesn't match one of 'test' or 'validation',
data is assumed to be 'train' data.  Second level subdirectory names denote
target class.  If not present, data is assumed to be unlabelled.
"""

import logging
import numpy as np
import os

from neon.datasets.dataset import Dataset
from neon.util.compat import pickle
from neon.util.param import opt_param


logger = logging.getLogger(__name__)


class DelimFiles(Dataset):
    """
    Sets up a raw delimited text input file based dataset.

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str): where to locally host this dataset on disk
        item_size (int): length of each input when flattened.
        noutputs (int): number of output target classes (should be 1 for
                        regression problems)
    """

    def __init__(self, **kwargs):
        self.live = False
        self.server = None
        opt_param(self, ['batch_size'], default_value=1)
        opt_param(self, ['input_dtype', 'target_dtype'],
                  default_value=np.float32)
        self.__dict__.update(kwargs)

    def load(self, backend=None, experiment=None):
        if self.inputs['train'] is not None or self.inputs['test'] is not None:
            return
        if 'repo_path' not in self.__dict__:
            raise AttributeError('repo_path not specified in config')
        if 'item_size' not in self.__dict__:
            try:
                # infer item_size from DataLayer
                self.item_size = experiment.model.layers[0].nout
            except AttributeError:
                raise AttributeError('item_size not specified in config'
                                     ' and cannot be inferred')
        if 'noutputs' not in self.__dict__:
            try:
                # infer noutputs from CostLayer
                self.noutputs = experiment.model.layers[-1].nin
            except AttributeError:
                raise AttributeError('noutputs not specified in config'
                                     ' and cannot be inferred')

        self.repo_path = os.path.expandvars(os.path.expanduser(self.repo_path))
        self.rootdir = os.path.join(self.repo_path, self.__class__.__name__)
        self.batchdata = np.zeros((self.item_size, self.batch_size),
                                  dtype=self.input_dtype)

        if self.live:
            from neon.ipc.shmem import Server
            self.predict = True
            req_size = (np.dtype(self.input_dtype).itemsize * self.batch_size *
                        self.item_size)
            res_size = (np.dtype(self.target_dtype).itemsize * self.noutputs *
                        self.batch_size)
            self.server = Server(req_size=req_size, res_size=res_size)
            return

    def unload(self):
        logger.info('Unloading...')
        if self.server is not None:
            self.server.stop()

    def serialize(self, obj, save_path):
        fd = open(save_path, 'w')
        pickle.dump(obj, fd, -1)
        fd.close()

    def deserialize(self, load_path):
        fd = open(load_path, 'r')
        obj = pickle.load(fd)
        fd.close()
        return obj

    def receive_batch(self):
        assert self.server is not None
        data, header = self.server.receive()
        self.batchdata[:, 0] = data.view(self.input_dtype)
        dbatchdata = self.backend.allocate_fragment(self.batchdata.shape,
                                                    self.input_data)
        self.backend.set(dbatchdata, self.batchdata)
        return dbatchdata, None

    def read_batch(self, batch_idx):
        raise NotImplementedError("This dataset is still a work in progress")

    def process_result(self, result):
        if self.live:
            assert self.server is not None
            self.server.send(result)

    def get_mini_batch(self, batch_idx):
        if self.live:
            inputs, targets = self.receive_batch()
        else:
            inputs, targets = self.read_batch(batch_idx)
        return inputs, targets
