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

from neon import NervanaObject
import numpy as np


def get_param_list(layer_list):
    '''
    returns a flattened list of params
    '''
    plist = []
    for l in layer_list:
        ptuple = l.get_params()
        plist.extend(ptuple) if isinstance(ptuple, list) else plist.append(ptuple)
    return plist


class Optimizer(NervanaObject):

    '''
    Optimizers will take a param, update, and state
    will be responsible for keeping track of a schedule
    '''

    def optimize(self, layer_list, epoch):
        raise NotImplementedError()

    def clip_gradient_norm(self, param_list, clip_norm):
        """
        Scale the magnitude of the network's gradients

        Arguments:
            param_list (list): a list of layer parameters
            clip_norm (float, optional): Value to scale gradients'
                                         magnitude by.
        """
        scale_factor = 1
        if clip_norm:
            grad_list = [grad for (param, grad), states in param_list]
            grad_square_sums = sum(self.be.sum(self.be.square(grad)) for grad in grad_list)
            grad_norm = self.be.zeros((1, 1))
            grad_norm[:] = self.be.sqrt(grad_square_sums)/self.be.bsz
            scale_factor = clip_norm / max(float(grad_norm.get()), float(clip_norm))
        return scale_factor

    def clip_gradient_value(self, grad, clip_value):
        """
        Element-wise clip a list of gradients.

        Arguments:
            grad (list): a list of gradients of a single layer
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
        """
        if clip_value:
            return self.be.clip(grad, -abs(clip_value), abs(clip_value))
        else:
            return grad


class Schedule(NervanaObject):
    """
    Learning rate schedule for constant or step learning rates.

    By default implements a constant learning rate.
    """
    def __init__(self, step_config=None, change=1.):
        """
        Arguments:
            step_config (int or list, optional): Configure the epoch step rate (int)
                or step times (list of epoch indices). Defaults to None (constant).

            change (float or list, optional): In step mode, learning rate is
                multiplied by ``change ** steps``, where ``steps`` is the number of
                steps in the step schedule that have passed. If ``change`` is a list,
                ``step_config`` must also be a list. Then at ``step[i]``, the
                learning rate is set to ``change[i]``.
        """

        if isinstance(step_config, list) and isinstance(change, list):
            assert len(step_config) == len(change), "change and step_config must have the same" \
                "length after step_config is deduplicated to do epoch-level LR assignment."

        self.step_config = step_config
        self.change = change
        self.steps = 0

    def get_learning_rate(self, learning_rate, epoch):
        """
        Get the current learning rate given the epoch and initial rate

        Arguments:
            learning_rate (float): the initial learning rate
            epoch (int): the current epoch, used to calculate the new effective learning rate.
        """

        if isinstance(self.step_config, list) and isinstance(self.change, list):
            if epoch in self.step_config:
                # steps will store the current lr
                self.steps = self.change[self.step_config.index(epoch)]
            if self.steps == 0:
                return learning_rate
            else:
                return self.steps

        elif isinstance(self.step_config, int):
            self.steps = np.floor((epoch + 1) / self.step_config)

        elif isinstance(self.step_config, list):
            self.steps = np.sum(epoch >= np.array(self.step_config))

        return float(learning_rate * self.change ** self.steps)


class ExpSchedule(Schedule):
    """
    Exponential learning rate schedule.

    Arguments:
        decay (float): how much exponential decay to apply to the learning rate
    """
    def __init__(self, decay):
        self.decay = decay

    def get_learning_rate(self, learning_rate, epoch):
        return float(learning_rate / (1. + self.decay * epoch))


class PolySchedule(Schedule):
    """
    Polynomial learning rate schedule.

    Arguments:
        total_epochs (int): total number of epochs over which to calculate interpolated decay
        power (float): total decay parameter
    """

    def __init__(self, total_epochs, power):
        self.total_epochs = np.float32(total_epochs)
        self.power = power

    def get_learning_rate(self, learning_rate, epoch):
        return float(learning_rate * (1. - (epoch / self.total_epochs)) ** self.power)


class GradientDescentMomentum(Optimizer):

    """
    Stochastic gradient descent with momentum
    """

    def __init__(self, learning_rate, momentum_coef, stochastic_round=False,
                 wdecay=0.0, gradient_clip_norm=None, gradient_clip_value=None,
                 name="gdm", schedule=Schedule()):
        """
        Arguments:
            learning_rate (float): the multiplicative coefficient of updates
            momentum_coef (float): the coefficient of momentum
            stochastic_round (bool, optional): Set this to True for stochastic
                                               rounding.  If False (default)
                                               rounding will be to nearest.  If
                                               True use default width
                                               stochastic rounding.  Note that
                                               this only affects the GPU
                                               backend.
            wdecay (float, optional): Amount of weight decay.  Defaults to 0
            gradient_clip_norm (float, optional): Value to scale gradients'
                                                  magnitude by.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
            name (str, optional): the optimizer's layer's pretty-print name.
                                  Defaults to "gdm".
            schedule (neon.optimizers.optimizer.Schedule, optional): Learning
                rate schedule.  Defaults to a constant learning rate.
        """
        super(GradientDescentMomentum, self).__init__(name=name)
        self.learning_rate, self.momentum_coef = (learning_rate, momentum_coef)
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.wdecay = wdecay
        self.schedule = schedule
        self.stochastic_round = stochastic_round

    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            layer_list (list): a list of Layer objects to optimize.
            epoch (int): the current epoch, needed for the Schedule object.
        """
        lrate = self.schedule.get_learning_rate(self.learning_rate, epoch)
        param_list = get_param_list(layer_list)

        scale_factor = self.clip_gradient_norm(param_list, self.gradient_clip_norm)

        for (param, grad), states in param_list:
            param.rounding = self.stochastic_round
            if len(states) == 0:
                states.append(self.be.zeros_like(grad))
            grad = grad / self.be.bsz
            grad = self.clip_gradient_value(grad, self.gradient_clip_value)

            velocity = states[0]
            velocity[:] = velocity * self.momentum_coef \
                - lrate * (scale_factor * grad + self.wdecay * param)
            param[:] = param + velocity


class RMSProp(Optimizer):

    """
    Root Mean Square propagation.
    """

    def __init__(self, stochastic_round=False, decay_rate=0.95, learning_rate=2e-3, epsilon=1e-6,
                 gradient_clip_norm=None, gradient_clip_value=None, name="rmsprop",
                 schedule=Schedule()):
        """
        Arguments:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            decay_rate (float): decay rate of states
            learning_rate (float): the multiplication coefficent of updates
            epsilon (float): smoothing epsilon to avoid divide by zeros
            gradient_clip_norm (float, optional): Value to scale gradients'
                                                  magnitude by.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
            schedule (neon.optimizers.optimizer.Schedule, optional): Learning rate schedule.
                                                                     Defaults to a constant.
        Notes:
            Only constant learning rate is supported currently.
        """
        self.state_list = None

        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.stochastic_round = stochastic_round

    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            layer_list (list): a list of Layer objects to optimize.
            epoch (int): the current epoch, needed for the Schedule object.
        """
        lrate = self.schedule.get_learning_rate(self.learning_rate, epoch)
        epsilon, decay = (self.epsilon, self.decay_rate)
        param_list = get_param_list(layer_list)

        scale_factor = self.clip_gradient_norm(param_list, self.gradient_clip_norm)

        for (param, grad), states in param_list:

            param.rounding = self.stochastic_round
            if len(states) == 0:
                states.append(self.be.zeros_like(grad))

            grad = grad / self.be.bsz
            grad = self.clip_gradient_value(grad, self.gradient_clip_value)

            # update state
            state = states[0]
            state[:] = decay * state + self.be.square(grad) * (1.0 - decay)

            param[:] = param \
                - (scale_factor * grad * lrate) / (self.be.sqrt(state + epsilon) + epsilon)


class Adagrad(Optimizer):

    """
     AdaGrad learning rule updates.  See Duchi2011 for instance
    """

    def __init__(self, stochastic_round=False, learning_rate=0.01, epsilon=1e-6,
                 gradient_clip_norm=None, gradient_clip_value=None, name="adagrad"):
        """
        Arguments:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            learning_rate (float): the multiplication coefficent of updates
            epsilon (float): smoothing epsilon to avoid divide by zeros
            gradient_clip_norm (float, optional): Value to scale gradients'
                                                  magnitude by.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
        Notes:
            Only constant learning rate is supported currently.
        """
        self.state_list = None
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.stochastic_round = stochastic_round

    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            layer_list (list): a list of Layer objects to optimize.
            epoch (int): the current epoch, needed for the Schedule object.
        """
        lrate, epsilon = (self.learning_rate, self.epsilon)
        param_list = get_param_list(layer_list)

        scale_factor = self.clip_gradient_norm(param_list, self.gradient_clip_norm)

        for (param, grad), states in param_list:

            param.rounding = self.stochastic_round
            if len(states) == 0:
                states.append(self.be.zeros_like(grad))

            grad = grad / self.be.bsz
            grad = self.clip_gradient_value(grad, self.gradient_clip_value)

            # update state
            state = states[0]
            state[:] = state + self.be.square(grad)
            param[:] = param - (scale_factor * grad * lrate) / (self.be.sqrt(state + epsilon))


class Adadelta(Optimizer):

    """
    Adadelta based learning rule updates.
    See Zeiler2012 for instance.
    """

    def __init__(self, stochastic_round=False, decay=0.95, epsilon=1e-6, name="ada"):
        """
        Args:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            decay: decay parameter in Adadelta
            epsilon: epsilon parameter in Adadelta
        """
        super(Adadelta, self).__init__(name=name)
        self.decay = decay
        self.epsilon = epsilon
        self.stochastic_round = stochastic_round

    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            param_list (list): a list of tuples of the form ((param, grad), state),
                               corresponding to parameters, grads,
                               and states of layers to be updated
            epoch (int): the current epoch, needed for the Schedule object.
        """
        epsilon, decay = (self.epsilon, self.decay)

        param_list = get_param_list(layer_list)

        for (param, grad), states in param_list:
            param.rounding = self.stochastic_round

            if len(states) == 0:
                # E[Grad^2], E[Delt^2], updates
                states.extend([self.be.zeros_like(grad) for i in range(3)])

            grad = grad / self.be.bsz
            states[0][:] = states[0] * decay + (1. - decay) * grad * grad
            states[2][:] = self.be.sqrt((states[1] + epsilon) / (states[0] + epsilon)) * grad
            states[1][:] = states[1] * decay + (1. - decay) * states[2] * states[2]

            param[:] = param - states[2]


class Adam(Optimizer):

    """
    Adam based learning rule updates. http://arxiv.org/pdf/1412.6980v8.pdf
    """

    def __init__(self, stochastic_round=False, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, name="adam"):
        """
        Args:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            learning_rate (float): the multiplicative coefficient of updates
            beta_1 (float): Adam parameter beta1
            beta_2 (float): Adam parameter beta2
            epsilon (float): numerical stability parameter
        """
        super(Adam, self).__init__(name=name)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.stochastic_round = stochastic_round

    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            param_list (list): a list of tuples of the form ((param, grad), state),
                corresponding to parameters, grads, and states of layers to be updated
            epoch (int): the current epoch, needed for the Schedule object.
        """
        t = epoch + 1
        l = self.learning_rate * self.be.sqrt(1 - self.beta_2 ** t) / (1 - self.beta_1 ** t)

        param_list = get_param_list(layer_list)

        for (param, grad), states in param_list:
            param.rounding = self.stochastic_round
            if len(states) == 0:
                # running_1st_mom, running_2nd_mom
                states.extend([self.be.zeros_like(grad) for i in range(2)])

            grad = grad / self.be.bsz
            m, v = states
            m[:] = m * self.beta_1 + (1. - self.beta_1) * grad
            v[:] = v * self.beta_2 + (1. - self.beta_2) * grad * grad

            param[:] = param - l * m / (self.be.sqrt(v) + self.epsilon)


class MultiOptimizer(Optimizer):

    """
    A wrapper class for using multiple Optimizers within the same model.
    """

    def __init__(self, optimizer_mapping, name="multiopt"):
        """

        Args:
            optimizer_mapping (dict): dictionary specifying the mapping of layers to optimizers.
                Key: Layer class name or Layer `name` attribute. The latter takes
                precedence over the former for finer layer-to-layer control.
                Don't name your layers ``'default'``. Value: the optimizer object to use for those
                layers. For instance, ``{'default': optimizer1, 'Bias': optimizer2,
                'special_bias': optimizer3}`` will use ``optimizer3`` for the layer named
                ``special_bias``, ``optimizer2`` for all other Bias layers, and ``optimizer1``
                for all other layers.
        """
        super(MultiOptimizer, self).__init__(name=name)
        self.optimizer_mapping = optimizer_mapping
        assert 'default' in self.optimizer_mapping, "Must specify a default" \
            "optimizer in layer type to optimizer mapping"

        self.map_list = None

    def map_optimizers(self, layer_list):
        """
        maps the optimizers to their corresponding layers
        """
        map_list = dict()
        for layer in layer_list:
            classname = layer.__class__.__name__
            name = layer.name
            opt = None
            if name in self.optimizer_mapping:
                opt = self.optimizer_mapping[name]
            elif classname in self.optimizer_mapping:
                opt = self.optimizer_mapping[classname]
            else:
                opt = self.optimizer_mapping['default']

            if opt not in map_list:
                map_list[opt] = [layer]
            else:
                map_list[opt].append(layer)
        return map_list

    def reset_mapping(self, new_mapping):
        """
        Pass this optimizer a new mapping, and on subsequent optimize call, the
        mapping will be refreshed (since map_list will be recreated)
        """
        self.optimizer_mapping = new_mapping
        self.map_list = None

    def optimize(self, layer_list, epoch):
        """
        Determine which optimizer in the container should go with which layers,
        then apply their optimize functions to those layers.

        Notes:

        We can recalculate ``map_list`` in case ``optimizer_mapping`` changes
        during training.
        """

        if self.map_list is None:
            self.map_list = self.map_optimizers(layer_list)

        for opt in self.map_list:
            opt.optimize(self.map_list[opt], epoch)

    def get_description(self):
        desc = {'type': self.__class__.__name__}
        for key in self.optimizer_mapping:
            desc[key] = self.optimizer_mapping[key].get_description()
        return desc
