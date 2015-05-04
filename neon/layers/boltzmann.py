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
Boltzmann distribution based layers.
"""

import logging
from neon.layers.layer import WeightLayer
from neon.util.param import opt_param

logger = logging.getLogger(__name__)


class RBMLayer(WeightLayer):
    """
    CD1 training layer for RBM
    """
    def initialize(self, kwargs):
        super(RBMLayer, self).initialize(kwargs)

        self.set_weight_shape()

        self.allocate_output_bufs()
        self.allocate_param_bufs()

        self.p_hid_plus = self.backend.empty((self.nout, self.batch_size))
        self.s_hid_plus = self.backend.empty((self.nout, self.batch_size))
        self.p_hid_minus = self.backend.empty((self.nout, self.batch_size))
        self.p_plus = self.backend.empty((self.nout, self.nin))
        self.p_minus = self.backend.empty((self.nout, self.nin))
        self.diff = self.backend.empty((self.nout, self.nin))
        self.learning_rule.allocate_state(self.diff)
        self.neg_pre_act = self.backend.empty((self.nin, self.batch_size))
        self.x_minus = self.backend.empty((self.nin, self.batch_size))
        self.output = self.backend.empty((self.nin, self.batch_size))

    def set_weight_shape(self):
        opt_param(self, ['weight_shape'], (self.nout, self.nin))

    def positive(self, inputs):
        """
        Positive / upward pass of the CD1 RBM

        Arguments:
           inputs (neon.datasets.dataset.Dataset): dataset upon which
                                                      to operate
        """
        self.backend.dot(self.weights, inputs, out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_plus)
        self.backend.dot(self.p_hid_plus, inputs.transpose(), out=self.p_plus)
        self.random_numbers = self.backend.uniform(size=self.p_hid_plus.shape)
        self.backend.greater(self.p_hid_plus, self.random_numbers,
                             out=self.s_hid_plus)

    def negative(self, inputs):
        """
        Negative / downward pass of the CD1 RBM

        Arguments:
           inputs (neon.datasets.dataset.Dataset): dataset upon which
                                                      to operate
        """
        self.backend.dot(self.weights.transpose(), self.s_hid_plus,
                         out=self.neg_pre_act)
        self.activation.apply_function(self.backend, self.neg_pre_act,
                                       self.x_minus)
        self.backend.dot(self.weights, self.x_minus, out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_minus)
        self.output[:] = self.x_minus

    def update(self, epoch):
        """
        CD1 weight update

        Arguments:
            epoch: not used, for future compatibility
        """
        self.backend.dot(self.p_hid_minus, self.x_minus.transpose(),
                         out=self.p_minus)
        self.backend.subtract(self.p_plus, self.p_minus, out=self.diff)
        self.learning_rule.apply_rule([self.weights], [self.diff], epoch)
