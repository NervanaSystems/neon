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
Simple restricted Boltzmann Machine model.
"""

import logging

from neon.backends.backend import Block
from neon.models.model import Model
from neon.util.compat import range

logger = logging.getLogger(__name__)


class RBM(Model):

    """
    Restricted Boltzmann Machine with binary visible and binary hidden units
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for req_param in ['layers', 'batch_size']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)
        self.cost.initialize(kwargs)

    def link(self, initlayer=None):
        """
        To make legacy config files work.
        """
        pass

    def initialize(self):
        """
        To make legacy config files work.
        """
        pass

    def fit(self, dataset):
        """
        Learn model weights on the given datasets.
        """
        for layer in self.layers:
            logger.info("%s", str(layer))
        inputs = dataset.get_inputs(train=True)['train']
        nin = self.layers[0].nin
        self.nlayers = len(self.layers)
        if 'temp_dtype' not in self.__dict__:
            self.temp_dtype = None
        self.temp = self.backend.empty((nin, self.batch_size), self.temp_dtype)

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = len(inputs)
        logger.info('commencing model fitting')
        error = self.backend.empty((1, 1))
        while self.epochs_complete < self.num_epochs:
            self.backend.begin(Block.epoch, self.epochs_complete)
            error.fill(0.0)
            for batch in range(num_batches):
                self.backend.begin(Block.minibatch, batch)
                inputs_batch = dataset.get_batch(inputs, batch)
                self.backend.begin(Block.fprop, batch)
                self.positive(inputs_batch)
                self.backend.end(Block.fprop, batch)
                self.backend.begin(Block.bprop, batch)
                self.negative(inputs_batch)
                self.backend.end(Block.bprop, batch)
                self.backend.add(error, self.cost.apply_function(inputs_batch),
                                 error)
                self.backend.begin(Block.update, batch)
                self.update(self.epochs_complete)
                self.backend.end(Block.update, batch)
                self.backend.end(Block.minibatch, batch)
            self.epochs_complete += 1
            logger.info('epoch: %d, total training error: %0.5f',
                        self.epochs_complete,
                        error.asnumpyarray() / num_batches)
            self.backend.end(Block.epoch, self.epochs_complete - 1)
            self.save_snapshot()

    def positive(self, inputs):
        """Wrapper for RBMLayer.positive"""
        self.layers[0].positive(inputs)
        return None

    def negative(self, inputs):
        """Wrapper for RBMLayer.negative"""
        self.layers[0].negative(inputs)
        return None

    def update(self, epoch):
        """Wrapper for RBMLayer.update"""
        self.layers[0].update(epoch)
