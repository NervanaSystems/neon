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
import numpy
from neon.optimizers.learning_rule import LearningRule

logger = logging.getLogger(__name__)


class GradientDescent(LearningRule):

    """
    Vanilla gradient descent based update rule that can optionally support use
    of weight decay.
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(GradientDescent, self).__init__(name, lr_params)
        if 'learning_rate' in lr_params:
            self.learning_rate = lr_params['learning_rate']
        else:
            raise AttributeError("Missing required learning rate")

        if 'weight_decay' in lr_params:
            self.weight_decay = lr_params['weight_decay']
        else:
            self.weight_decay = 0.0
        self.param_names = []

    def apply_rule(self, params, updates, epoch):
        for ps_item, us_item in zip(params, updates):
            self.backend.multiply(us_item, self.learning_rate, out=us_item)
            self.backend.subtract(ps_item, us_item, out=ps_item)


class GradientDescentPretrain(GradientDescent):

    """
    Gradient descent based variant that also supports a separate learning
    rate during pre-training.
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(GradientDescentPretrain, self).__init__(name, lr_params)
        if 'pretrain_learning_rate' in lr_params:
            self.pretrain_learning_rate = lr_params['pretrain_learning_rate']
        else:
            raise AttributeError("Missing required pretrain learning rate")

        self.train_learning_rate = self.learning_rate
        self.pretrain_mode = False

    def set_pretrain_mode(self, pretrain_mode):
        if (pretrain_mode):
            self.learning_rate = self.pretrain_learning_rate
        else:
            self.learning_rate = self.train_learning_rate

    def apply_rule(self, params, updates, epoch):
        for ps_item, us_item in zip(params, updates):
            self.backend.multiply(us_item, self.learning_rate, out=us_item)
            self.backend.subtract(ps_item, us_item, out=ps_item)


class GradientDescentMomentum(GradientDescent):

    """
    Gradient descent learning rate variant that supports different types of
    momentum based updates
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(GradientDescentMomentum, self).__init__(name, lr_params)
        if 'momentum_params' in lr_params:
            self.momentum_params = lr_params['momentum_params']
        else:
            self.momentum_params = None
        self.velocity = []
        self.velocity_dtype = param_dtype
        if 'schedule' in lr_params:
            self.schedule_flag = True
            self.schedule = lr_params['schedule']
        else:
            self.schedule_flag = False
        self.param_names = ['velocity']

    def allocate_state(self, params):
        self.velocity = []
        for item in params:
            self.velocity.append(self.backend.zeros(item.shape,
                                                    self.velocity_dtype))

    def apply_rule(self, params, updates, epoch):
        """
        Steps for momentum:
        1. velo = mu * velo    scale down old velocity (momentum coef)
        2. upda = eps * upda   scale down new updates (lerning rate)
        3. velo = velo - upda  combine old and new part
        4. para = para + velo  update the actual weights.
        """
        learning_rate = self.get_learning_rate(epoch)
        momentum_coef = self.get_momentum_coef(epoch)
        for ps_item, us_item, vs_item in zip(params, updates, self.velocity):
            # temporarily making backend dependent checks until we completely
            # switch MOP over to optree approach
            if ((self.backend.__module__ == 'neon.backends.cc2') or
                    (self.backend.__module__ == 'neon.backends.gpu')):
                # wrapping all calls into a single, lazy-eval kernel
                self.backend.gdm_compound(ps_item=ps_item, us_item=us_item,
                                          vs_item=vs_item,
                                          momentum_coef=momentum_coef,
                                          learning_rate=self.learning_rate,
                                          epoch=epoch)
            else:
                self.backend.multiply(vs_item, momentum_coef, out=vs_item)
                self.backend.multiply(us_item, learning_rate, out=us_item)
                self.backend.subtract(vs_item, us_item, out=vs_item)
                self.backend.add(ps_item, vs_item, out=ps_item)

    def get_learning_rate(self, epoch):
        if self.schedule_flag:
            if self.schedule['type'] == 'step':
                div_factor = numpy.floor(
                    (epoch + 1) / self.schedule['step_epochs'])
                return float(self.learning_rate *
                             self.schedule['ratio'] ** div_factor)
            else:
                raise NotImplementedError("learning rate schedule type not "
                                          "supported")
        else:
            return self.learning_rate

    def get_momentum_coef(self, epoch):
        """
        Uses the following parameters from self.momentum_params
        initial_coef:   momentum coefficient used from first epoch on
        saturated_coef: momentum after saturate_epoch is reached
        start_epoch:    start increasing momentum at this epoch
        saturate_epoch: saturated_coef is reached and held
        ...
        """
        if self.momentum_params is None:
            return 0.0
        coef = 0.0
        if 'coef' in self.momentum_params:
            coef = self.momentum_params['coef']

        if 'initial_coef' in self.momentum_params:
            init_coef = self.momentum_params['initial_coef']
        else:
            init_coef = coef

        if 'saturated_coef' in self.momentum_params:
            saturated_coef = self.momentum_params['saturated_coef']
        else:
            saturated_coef = coef

        if 'start_epoch' in self.momentum_params:
            start_epoch = self.momentum_params['start_epoch']
        else:
            start_epoch = None

        if 'saturate_epoch' in self.momentum_params:
            saturate_epoch = self.momentum_params['saturate_epoch']
        else:
            saturate_epoch = None

        if self.momentum_params['type'] == 'constant':
            if 'coef' not in self.momentum_params:
                coef = init_coef
        elif self.momentum_params['type'] == 'linear_monotone':
            coef = init_coef
            if start_epoch is not None and epoch >= start_epoch:
                if saturate_epoch is not None and epoch <= saturate_epoch:
                    if start_epoch == saturate_epoch:
                        coef = saturated_coef
                    else:
                        init_proportion = 1 - ((epoch - start_epoch + 0.0) /
                                               (saturate_epoch - start_epoch))
                        coef = (init_proportion * init_coef +
                                (1.0 - init_proportion) * saturated_coef)
                elif saturate_epoch is not None and epoch > saturate_epoch:
                    coef = saturated_coef
            else:
                pass
        elif self.momentum_params['type'] == 'nesterov':
            raise NotImplementedError("TODO!")
        else:
            raise AttributeError("invalid momentum_params specified")
        return coef


class GradientDescentMomentumWeightDecay(GradientDescentMomentum):
    """
    Adds weight decay regularization
    """
    def apply_rule(self, params, updates, epoch):
        """
        Steps for momentum:
        1. velo = mu * velo    scale down old velocity
        2. upda = eps * upda   scale down new updates
        3. velo = velo - upda  combine old and new part
        Extra steps for weight decay:
        4. tmp = W * decay
        5. tmp = tmp * eps
        6. velo = velo - tmp_decay term.
        and add update
        """
        learning_rate = self.get_learning_rate(epoch)
        momentum_coef = self.get_momentum_coef(epoch)
        for ps_item, us_item, vs_item in zip(params, updates, self.velocity):
            # temporarily making backend dependent checks until we completely
            # switch MOP over to optree approach
            if ((self.backend.__module__ == 'neon.backends.cc2') or
                    (self.backend.__module__ == 'neon.backends.gpu')):
                # wrapping all calls into a single, lazy-eval kernel
                self.backend.gdmwd_compound(ps_item=ps_item, us_item=us_item,
                                            vs_item=vs_item,
                                            momentum_coef=momentum_coef,
                                            learning_rate=self.learning_rate,
                                            wd=self.weight_decay,
                                            epoch=epoch)
            else:
                self.backend.multiply(vs_item, momentum_coef, out=vs_item)
                self.backend.multiply(us_item, learning_rate, out=us_item)
                self.backend.subtract(vs_item, us_item, out=vs_item)
                # reuse us_item for weight decay term
                # note: usually want to only apply for weights, not biases
                self.backend.multiply(ps_item, self.weight_decay, out=us_item)
                self.backend.multiply(us_item, learning_rate, out=us_item)
                self.backend.subtract(vs_item, us_item, out=vs_item)

                self.backend.add(ps_item, vs_item, out=ps_item)
