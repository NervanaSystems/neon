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
Generic parent class used to control how updates are applied to coefficients
i.e. how the learning should proceed.
"""

from neon.util.param import opt_param
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LearningRule(object):

    """
    Base object for applying learning rule on the parameters to be updated

    Attributes:
        name (str): Used to identify this LearningRule when logging.
        batch_size (int): Number of examples presented at this iteration
    """

    def __init__(self, name, lr_params):
        self.name = name
        opt_param(self, ['velocity_dtype', 'param_dtype', 'gradient_dtype'],
                  np.float32)

    def initialize(self, backend):
        self.backend = backend

    def __str__(self):
        be_nm = ''
        if hasattr(self, 'backend'):
            be_nm = ", utilizing {} backend".format(
                    self.backend.__class__.__name__)
        return ("LearningRule {upd_nm}: {upd_tp} upd_rl{be_nm}\n\t".format(
                upd_nm=self.name, upd_tp=self.__class__.__name__, be_nm=be_nm))

    def allocate_state(self, params):
        pass

    def set_pretrain_mode(self, pretrain_mode):
        pass

    def apply_rule(self, params, updates, epoch):
        raise NotImplementedError()

    def get_params(self):
        np_params = dict()
        for p in self.param_names:
            if hasattr(self, p):
                p_list = getattr(self, p)
                np_params[p] = []
                for p_tensor in p_list:
                    np_params[p].append(np.array(
                        p_tensor.asnumpyarray(), dtype=p_tensor.dtype).reshape(
                            p_tensor.shape))
        return np_params

    def set_params(self, params_dict):
        for p in self.param_names:
            if p in params_dict:
                for i in range(len(params_dict[p])):
                    getattr(self, p)[i][:] = params_dict[p][i]
