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
Simple deep belief net.
"""

import logging
import math

from neon.backends.backend import Block
from neon.models.model import Model
from neon.util.compat import range

logger = logging.getLogger(__name__)


class DBN(Model):

    """
    deep belief net
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for req_param in ['layers', 'batch_size']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        for layer in self.layers:
            logger.info("%s", str(layer))
        inputs = datasets[0].get_inputs(train=True)['train']
        nrecs, nin = inputs.shape
        self.nlayers = len(self.layers)
        self.temp = self.backend.zeros((self.batch_size, nin))

        logger.info('commencing model fitting')
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        # Part 1: Unsupervised pretraining
        for i in range(self.nlayers):
            if i > 0:
                logger.info('layer %d: setting inputs to output of previous '
                            'layer', i)
                # transform all inputs to generate data for next layer
                out_shape = (inputs.shape[0],
                             self.layers[i - 1].s_hid_plus.shape[1] - 1)
                outputs = self.backend.zeros(out_shape)
                for batch in range(num_batches):
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, nrecs)
                    self.positive(inputs[start_idx:end_idx], i - 1)
                    prev_end = self.layers[i - 1].s_hid_plus.shape[1] - 1
                    outputs[start_idx:end_idx] = (self.layers[i -
                                                  1].s_hid_plus[:, 0:prev_end])
                inputs = outputs
                logger.info('inputs (%d, %d) weights (%d,%d)', inputs.shape[0],
                            inputs.shape[1], self.layers[i].weights.shape[0],
                            self.layers[i].weights.shape[1])
                # If we are in the penultimate layer, append labels to the
                # visibles ...
            while self.epochs_complete < self.num_epochs:
                self.backend.begin(Block.epoch, self.epochs_complete)
                error = 0.0
                for batch in range(num_batches):
                    self.backend.begin(Block.minibatch, batch)
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, nrecs)
                    batch_in = inputs[start_idx:end_idx]
                    self.backend.begin(Block.fprop, batch)
                    self.positive(batch_in, i)
                    self.backend.end(Block.fprop, batch)
                    self.backend.begin(Block.bprop, batch)
                    self.negative(batch_in, i)
                    self.backend.end(Block.bprop, batch)
                    self.backend.begin(Block.update, batch)
                    self.update(self.epochs_complete, i)
                    self.backend.end(Block.update, batch)
                    batch_out = self.layers[i].x_minus[:,
                                                       0:(self.layers[i].
                                                          x_minus.shape[1] - 1)
                                                       ]
                    error += self.cost.apply_function(self.backend,
                                                      batch_in,
                                                      batch_out,
                                                      self.temp)
                    self.backend.end(Block.minibatch, batch)
                self.epochs_complete += 1
                logger.info('epoch: %d, total training error: %0.5f',
                            self.epochs_complete,
                            error / num_batches)
                self.backend.end(Block.epoch, self.epochs_complete - 1)
                self.save_snapshot()
            self.epochs_complete = 0  # reset for next layer
        # Part 2: up-down finetuning ... [not implemented yet]

    def positive(self, inputs, i):
        """Wrapper for RBMLayer.positive"""
        self.layers[i].positive(inputs)
        return None

    def negative(self, inputs, i):
        """Wrapper for RBMLayer.negative"""
        self.layers[i].negative(inputs)
        return None

    def update(self, epsilon, epoch, i):
        """Wrapper for RBMLayer.update"""
        self.layers[i].update(epoch)
