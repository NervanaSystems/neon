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
from neon.optimizers.learning_rule import LearningRule

logger = logging.getLogger(__name__)


class AdaDelta(LearningRule):

    """
    Adadelta based learning rule updates.  See Zeiler2012 for instance.
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        if param_dtype is not None:
            self.param_dtype = param_dtype
        super(AdaDelta, self).__init__(name, lr_params)
        if 'rho' in lr_params:
            self.rho = lr_params['rho']
        else:
            self.rho = 0.95
        if 'epsilon' in lr_params:
            self.epsilon = lr_params['epsilon']
        else:
            self.epsilon = 0.000001
        self.exp_gradsq_dtype = self.param_dtype
        self.exp_deltsq_dtype = self.param_dtype
        self.scratch_space_dtype = self.param_dtype
        self.lrates_dtype = self.param_dtype
        self.lrates_dtype = self.param_dtype
        self.exp_gradsq = []
        self.exp_deltsq = []
        self.lrates = []
        self.scratch_space = []
        self.param_names = ['exp_gradsq', 'exp_deltsq', 'lrates',
                            'scratch_space']

    def allocate_state(self, params):
        assert len(self.exp_gradsq) == 0
        for item in params:
            self.exp_gradsq.append(self.backend.zeros(item.shape,
                                                      self.exp_gradsq_dtype))
            self.exp_deltsq.append(self.backend.zeros(item.shape,
                                                      self.exp_deltsq_dtype))
            self.lrates.append(self.backend.zeros(item.shape,
                                                  self.lrates_dtype))
            self.scratch_space.append(self.backend.zeros(
                item.shape, self.scratch_space_dtype))

    def apply_rule(self, params, updates, epoch):
        for ps_item, us_item, gs_item, ds_item, ls_item, ss_item in zip(
                params, updates, self.exp_gradsq,
                self.exp_deltsq, self.lrates, self.scratch_space):
            self.backend.ada_update(ps_item, us_item, gs_item, ds_item,
                                    ls_item, ss_item, self.rho, self.epsilon)
