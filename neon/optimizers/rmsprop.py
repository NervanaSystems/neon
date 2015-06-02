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
Classes used to control how updates are applied to coefficients
i.e. how the learning should proceed.
"""

import logging
from neon.optimizers.gradient_descent import GradientDescentMomentum

logger = logging.getLogger(__name__)


class RMSProp(GradientDescentMomentum):

    """
    RMSProp based learning rule updates.
    See Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA

    Arguments:
        gamma (float): exponential weighting param for updating square of
                       gradient (default 0.9)
        epsilon (float): value added to denominator of 1/rms for stability
                         (default 1E-6)

    Notes:
        The suggested values for learning_rate is 0.001, using larger learning
        rates seems to lead to much worse performance.
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        if param_dtype is not None:
            self.param_dtype = param_dtype
        super(RMSProp, self).__init__(name, lr_params)
        if 'gamma' in lr_params:
            self.gamma = lr_params['gamma']
        else:
            self.gamma = 0.9
        if 'epsilon' in lr_params:
            self.epsilon = lr_params['epsilon']
        else:
            self.epsilon = 0.000001

        self.running_squares_dtype = self.param_dtype
        self.scratch_space_dtype = self.param_dtype
        self.velocity_dtype = self.param_dtype

        self.running_squares = []
        self.scratch_space = []
        self.param_names = ['running_squares', 'velocity', 'scratch_space']

    def allocate_state(self, params):
        assert len(self.running_squares) == 0
        super(RMSProp, self).allocate_state(params)
        for item in params:
            self.running_squares.append(
                self.backend.zeros(item.shape, self.running_squares_dtype))
            self.scratch_space.append(
                self.backend.zeros(item.shape, self.scratch_space_dtype))

    def apply_rule(self, params, updates, epoch):
        learning_rate = self.get_learning_rate(epoch)
        momentum_coef = self.get_momentum_coef(epoch)
        for ps_item, us_item, rs_item, vs_item, ss_item in zip(
                params, updates, self.running_squares,
                self.velocity, self.scratch_space):
            self.backend.rms_update(ps_item, us_item, rs_item, vs_item,
                                    ss_item, self.gamma, self.epsilon,
                                    learning_rate, momentum_coef)
