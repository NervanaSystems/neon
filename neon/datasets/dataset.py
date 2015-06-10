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
Generic Dataset interface.  Defines the operations any dataset should support.
"""

import logging
import numpy as np
import os

from neon.backends.cpu import CPU
from neon.util.compat import PY3, range

if PY3:
    import urllib.request as urllib
else:
    import urllib

logger = logging.getLogger(__name__)


class Dataset(object):

    """
    Base dataset class. Defines interface operations.
    """

    backend = None
    inputs = {'train': None, 'test': None, 'validation': None}
    targets = {'train': None, 'test': None, 'validation': None}

    def __getstate__(self):
        """
        Defines what and how we go about serializing an instance of this class.
        In this case we also want to include any loaded datasets and backend
        references.

        Returns:
            dict: keyword args, plus current inputs, targets, backend
        """
        self.__dict__['backend'] = self.backend
        self.__dict__['inputs'] = self.inputs
        self.__dict__['targets'] = self.targets
        return self.__dict__

    def __setstate__(self, state):
        """
        Defines how we go about deserializing and loading an instance of this
        class from a specified state.

        Arguments:
            state (dict): attribute values to be loaded.
        """
        self.__dict__.update(state)
        if self.backend is None:
            # use CPU as a default backend
            self.backend = CPU()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def load(self, backend=None, experiment=None):
        """
        Makes the dataset data available for use.
        Needs to be implemented in every concrete Dataset child class.

        Arguments:
            backend (neon.backends.backend.Backend, optional): The
                     underlying data structure type used to hold this
                     data once loaded.  If None will use
                     `neon.backends.cpu.CPU`
            experiment (neon.experiments.experiment.Experiment, optional): The
                     object that loads this dataset.

        Raises:
            NotImplementedError: should be overridden in child class
        """
        raise NotImplementedError()

    def unload(self):
        """
        Perform cleanup tasks if any are required.
        """
        pass

    def process_result(self, result):
        """
        Accept and process results of running inference.

        Arguments:
            result (ndarray): Array containing predictions obtained by
                    processing a minibatch of input data.
        """
        pass

    def download_to_repo(self, url, repo_path):
        """
        Fetches the dataset to a local repository for future use.

        Arguments:
            url (str): The external URI to a specific dataset
            repo_path (str): The local path to write the fetched dataset to.
        """
        repo_path = os.path.expandvars(os.path.expanduser(repo_path))
        logger.info("fetching: %s, saving to: %s (this may take some time "
                    "depending on dataset size)", url, repo_path)
        urllib.urlretrieve(url, os.path.join(repo_path,
                                             os.path.basename(url)))

    def get_inputs(self, backend=None, train=True, test=False,
                   validation=False):
        """
        Loads and returns one or more input datasets.

        Arguments:
            backend (neon.backends.backend.Backend, optional): The underlying
                    data structure type used to hold this data once loaded.
                    If None will use whatever is set for this class
            train (bool, optional): load a training target outcome dataset.
                                    Defaults to True.
            test (bool, optional): load a hold-out test target outcome dataset.
                                   Defaults to False.
            validation (bool, optional): Load a separate validation target
                                         outcome dataset.  Defaults to False.

        Returns:
            dict: of loaded datasets with keys train, test, validation
                  based on what was requested.  Each dataset is a
                  neon.backends.backend.Tensor instance.
        """
        res = dict()
        if self.inputs['train'] is None:
            if backend is not None:
                self.load(backend)
            else:
                self.load()
        if train and self.inputs['train'] is not None:
            res['train'] = self.inputs['train']
        if test and self.inputs['test'] is not None:
            res['test'] = self.inputs['test']
        if validation and self.inputs['validation'] is not None:
            res['validation'] = self.inputs['validation']
        return res

    def get_targets(self, backend=None, train=True, test=False,
                    validation=False):
        """
        Loads and returns one or more labelled outcome datasets.

        Arguments:
            backend (neon.backends.backend.Backend, None): The underlying
                    data structure type used to hold this data once loaded.
                    If None will use whatever is set for this class
            train (bool, optional): load a training target outcome dataset.
                                    Defaults to True.
            test (bool, optional): load a hold-out test target outcome dataset.
                                   Defaults to False.
            validation (bool, optional): Load a separate validation target
                                         outcome dataset.  Defaults to False.

        Returns:
            dict: of loaded datasets with keys train, test, validation
                  based on what was requested.  Each dataset is a
                  neon.backends.backend.Tensor instance.
        """
        # can't have targets without inputs, ensure these are loaded
        res = dict()
        if self.inputs['train'] is None:
            self.load()
        if train and self.inputs['train'] is not None:
            res['train'] = self.targets['train']
        if test and self.inputs['test'] is not None:
            res['test'] = self.targets['test']
        if validation and self.inputs['validation'] is not None:
            res['validation'] = self.targets['validation']
        return res

    def sample_training_data(self):
        """
        Carries out actual downsampling of data, to the percentage specified in
        self.sample_pct.
        """
        if self.sample_pct != 100:
            train_idcs = np.arange(self.inputs['train'].shape[0])
            ntrain_actual = (self.inputs['train'].shape[0] *
                             int(self.sample_pct) / 100)
            np.random.seed(self.backend.rng_seed)
            np.random.shuffle(train_idcs)
            train_idcs = train_idcs[0:ntrain_actual]
            self.inputs['train'] = self.inputs['train'][train_idcs]
            self.targets['train'] = self.targets['train'][train_idcs]

    def transpose_batches(self, data, dtype):
        """
        Transpose and distribute each minibatch within a dataset.

        Arguments:
            data (ndarray): Dataset to be sliced into mini batches,
                            transposed, and loaded to appropriate device
                            memory.
        Returns:
            list: List of device loaded mini-batches of data.
        """
        bs = self.backend.actual_batch_size
        if data.shape[0] % bs != 0:
            logger.warning('Incompatible batch size. Discarding %d samples...',
                           data.shape[0] % bs)
        nbatches = data.shape[0] // bs
        batchwise = []
        for batch in range(nbatches):
            batchdata = np.empty((data.shape[1], bs))
            batchdata[...] = data[batch * bs:(batch + 1) * bs].transpose()
            dev_batchdata = self.backend.distribute(batchdata, dtype)
            batchwise.append(dev_batchdata)
        return batchwise

    def format(self, dtype=np.float32):
        """
        Transforms the loaded data into the format expected by the
        backend. If a hardware accelerator device is being used,
        this function also copies the data to the device memory.
        """
        assert self.backend is not None
        for dataset in (self.inputs, self.targets):
            for key in dataset:
                item = dataset[key]
                if item is not None:
                    dataset[key] = self.transpose_batches(item, dtype)

    def get_batch(self, data, batch):
        """
        Extract and return a single batch from the data specified.

        Arguments:
            data (list): List of device loaded batches of data
            batch (int): 0-based index specifying the batch number to get

        Returns:
            neon.backends.Tensor: Single batch of data

        See Also:
            transpose_batches
        """
        return data[batch]

    def has_set(self, setname):
        """
        Indicate whether the specified setname type is part of this dataset.

        Arguments:
            setname (str): The type of data to look for. Typically this is one
                           of 'train', 'test', 'validation'.

        Returns:
            bool: True if this dataset contains setname type of data, and False
                  otherwise.
        """
        inputs_dic = self.get_inputs(train=True, validation=True,
                                     test=True)
        return True if (setname in inputs_dic) else False

    def init_mini_batch_producer(self, batch_size, setname, predict):
        """
        Setup the ability to generate mini-batches.

        Arguments:
            batch_size (int): The number of data examples will be contained in
                              each mini-batch
            setname (str): The type of data to produce mini-batches for:
                           'train', 'test', 'validation'
            predict (bool): Set this to False when training a model, or True
                            when generating batches to be used for prediction.

        Returns:
            int: The total number of examples to be mini-batched.

        Notes:
            This is the implementation for non-macro batched data.
            macro-batched datasets will override this (e.g. ImageNet)
        """
        self.cur_inputs = self.get_inputs(train=True, validation=True,
                                          test=True)[setname]
        self.cur_tgts = self.get_targets(train=True, validation=True,
                                         test=True)[setname]
        self.predict_mode = predict
        return len(self.inputs[setname])

    def get_mini_batch(self, batch_idx):
        """
        Return the specified mini-batch of input and target data.

        Arguments:
            batch_idx (int): 0-based index specifying the mini-batch number to
                             retrieve.

        Returns:
            tuple: 2-tuple of neon.backend.Tensor objects containing the
                   corresponding input, and target mini-batches.

        Notes:
            This is the implementation for non-macro batched data.
            macro-batched datasets will override this (e.g. ImageNet)
        """
        return self.get_batch(self.cur_inputs, batch_idx), self.get_batch(
            self.cur_tgts, batch_idx)

    def del_mini_batch_producer(self):
        """
        Perform any cleanup needed once all mini-batches have been produced.

        Notes:
            macro-batched datasets will likely override this
        """
        pass
