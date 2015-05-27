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
Contains code to train stacked autoencoder models and run inference.
"""

import logging

from neon.backends.backend import Block
from neon.models.mlp import MLP
from neon.util.compat import range

logger = logging.getLogger(__name__)


class Autoencoder(MLP):
    """
    Adaptation of multi-layer perceptron.
    """

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        for layer in self.layers:
            logger.info("%s", str(layer))
        ds = datasets[0]
        inputs = ds.get_inputs(train=True)['train']
        targets = ds.get_inputs(train=True)['train']

        num_batches = len(inputs)
        logger.info('commencing model fitting')
        error = self.backend.empty((1, 1))
        while self.epochs_complete < self.num_epochs:
            self.backend.begin(Block.epoch, self.epochs_complete)
            error.fill(0.0)
            for batch in range(num_batches):
                self.backend.begin(Block.minibatch, batch)
                inputs_batch = ds.get_batch(inputs, batch)
                targets_batch = ds.get_batch(targets, batch)
                self.backend.begin(Block.fprop, batch)
                self.fprop(inputs_batch)
                self.backend.end(Block.fprop, batch)
                self.backend.begin(Block.bprop, batch)
                self.bprop(targets_batch, inputs_batch)
                self.backend.end(Block.bprop, batch)
                self.backend.add(error,
                                 self.cost.apply_function(targets_batch),
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
