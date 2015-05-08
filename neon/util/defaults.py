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
Contains functions to create default values for parameters.
"""

from neon.params.val_init import GaussianValGen
from neon.metrics.misclass import MisclassPercentage


def default_weight_init():
    return GaussianValGen(loc=0.0, scale=0.01)


def default_lrule_init():
    gdm = {
        'type': 'gradient_descent_momentum',
        'lr_params': {
            'learning_rate': 0.1,
            'momentum_params': {
                'type': 'constant',
                'coef': 0.9
            }
        }
    }
    return gdm


def default_metric():
    metric = {
        'train': [MisclassPercentage()],
        'test': [MisclassPercentage()],
    }
    return metric
