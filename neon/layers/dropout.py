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
Single neural network layer in which activations are randomly turned off
according to a specific Bernoulli probability threshold.
"""

import logging
from neon.layers.layer import Layer
from neon.util.param import opt_param

logger = logging.getLogger(__name__)


class DropOutLayer(Layer):
    """
    Dropout layer randomly kills activations from being passed on at each
    fprop call.
    Uses parameter 'keep' as the threshhold above which to retain activation.
    During training, the mask is applied, but during inference, we switch
    off the random dropping.
    Make sure to set train mode to False during inference.

    Attributes:
        keep (numeric, optional): The Bernoulli success probability, indicating
                                  the cutoff below which we keep an activation.
                                  Defaults to 0.5, and should lie in range
                                  [0, 1].
    """
    def initialize(self, kwargs):
        opt_param(self, ['keep'], 0.5)
        super(DropOutLayer, self).initialize(kwargs)
        self.keepmask = self.backend.empty((self.nin, self.batch_size),
                                           dtype=self.weight_dtype)
        self.train_mode = True
        self.allocate_output_bufs()

    def set_previous_layer(self, pl):
        if pl.is_local:
            self.is_local = True
            self.nifm = self.nofm = pl.nofm
            self.ifmshape = self.ofmshape = pl.ofmshape
        self.nout = self.nin = pl.nout
        self.prev_layer = pl

    def fprop(self, inputs):
        if (self.train_mode):
            self.backend.make_binary_mask(self.keepmask, self.keep)
            self.backend.multiply(inputs, self.keepmask, out=self.output)
        else:
            self.backend.multiply(inputs, self.keep, out=self.output)

    def bprop(self, error):
        if self.deltas is not None:
            self.backend.multiply(error, self.keepmask, out=self.deltas)

    def set_train_mode(self, mode):
        self.train_mode = mode
