# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
from __future__ import division
from neon import NervanaObject
from neon.util.persist import load_class
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_param_list(layer_list):
    '''
    Returns a flattened list of parameters. Each element in the list
    is a tuple ``((W, dW), states)`` for the parameters ``W``, parameter updates ``dW``,
    and the current set of ``states``.

    Args:
        layer_list (list): List of layers

    Returns:
        param_list (list): List of parameters.
    '''
    plist = []
    for l in layer_list:
        ptuple = l.get_params()
        plist.extend(ptuple) if isinstance(ptuple, list) else plist.append(ptuple)
    return plist


class Optimizer(NervanaObject):

    '''
    The optimizer class handles the gradient update stage of training a neural network.
    Given the current parameters :math:`w`, update parameters
    :math:`\Delta w`, and current state :math:`s`, the optimizer specifies an
    algorithm for performing the update.

    This base class contains to helper functions for scaling the gradients.
    specifices the abstract method optimize, which subclasses should implement. The optimize
    method is called at every minibatch to update the layer parameters.
    '''
    def __init__(self, name=None):
        """
        Class constructor.
        """
        super(Optimizer, self).__init__(name=name)

    def optimize(self, layer_list, epoch):
        """
        Update the parameters for a provided list of layers.

        Args:
            layer_list (list): List of layers to optimize
            epoch (integer): Epoch count of training
        """
        raise NotImplementedError()

    def clip_gradient_norm(self, param_list, clip_norm):
        """
        Returns a scaling factor to apply to the gradients.

        The scaling factor is computed such that the root mean squared
        average of the scaled gradients across all layers will be less than
        or equal to the provided clip_norm value. This factor is always <1, so
        never scales up the gradients.

        Arguments:
            param_list (list): List of layer parameters
            clip_norm (float, optional): Target norm for the gradients. If not provided
                                         the returned scale_factor will equal 1.

        Returns:
            scale_factor (float): Computed scale factor.
        """
        scale_factor = 1
        if clip_norm:
            grad_list = [grad for (param, grad), states in param_list]
            grad_square_sums = sum(self.be.sum(self.be.square(grad)) for grad in grad_list)
            grad_norm = self.be.zeros((1, 1))
            grad_norm[:] = self.be.sqrt(grad_square_sums) / self.be.bsz
            scale_factor = clip_norm / max(float(grad_norm.get()), float(clip_norm))
        return scale_factor

    def clip_value(self, v, abs_bound=None):
        """
        Element-wise clip a gradient or parameter tensor to between
        ``-abs_bound`` and ``+abs_bound``.

        Arguments:
            v (tensor): Tensor of gradients or parameters for a single layer
            abs_bound (float, optional): Value to element-wise clip gradients
                                         or parameters. Defaults to None.

        Returns:
            v (tensor): Tensor of clipped gradients or parameters.

        """
        if abs_bound:
            return self.be.clip(v, -abs(abs_bound), abs(abs_bound))
        else:
            return v


class Schedule(NervanaObject):
    """
    Learning rate schedule.

    By default implements a constant learning rate:

    .. code-block:: python

        # Constant learning rate of 0.01 across training epochs
        optimizer = GradientDescentMomentum(0.01, 0.9, schedule = Schedule())

    Otherwise, the schedule multiplies the learning rate by change at every element in
    ``step_config``.
    For example,

    .. code-block:: python

        schedule = Schedule(step_config=[2, 6], change=0.5)
        optimizer = GradientDescentMomentum(1.0, 0.9, schedule = Schedule())

    will yield a learning rate schedule of:

    .. csv-table::
        :header: "Epoch", "LR"
        :widths: 20, 10

        0, 1.0
        1, 1.0
        2, 0.5
        3, 0.5
        4, 0.5
        5, 0.5
        6, 0.25
        7, 0.25
        8, 0.25
        9, 0.25
    """

    def __init__(self, step_config=None, change=1.):
        """
        Class constructor.

        Arguments:
            step_config (list, optional): Configure the step times (list of epoch indices).
                                          Defaults to None (constant).
            change (int, optional): The learning rate is
                                    multiplied by ``change ** steps``, where ``steps`` is the
                                    number of steps in the step schedule that have passed.
        """

        if isinstance(step_config, list) and isinstance(change, list):
            assert len(step_config) == len(change), "change and step_config must have the same" \
                "length after step_config is deduplicated to do epoch-level LR assignment."

            logger.warn("This functionality will be removed from Schedule in the future. "
                        "Please use the StepSchedule class instead.")

        if isinstance(step_config, int):
            logger.warn("This functionality will be removed from Schedule in the future. "
                        "Please use the PowerSchedule class instead.")

        self.step_config = step_config
        self.change = change
        self.steps = 0

    def get_learning_rate(self, learning_rate, epoch):
        """
        Returns the current learning rate given the epoch and initial learning rate.

        Arguments:
            learning_rate (float): Initial learning rate
            epoch (int): Current epoch, used to calculate the adjusted learning rate

        Returns:
            (float): The adjusted learning rate
        """

        # will be moved to StepSchedule in the future
        if isinstance(self.step_config, list) and isinstance(self.change, list):
            if epoch in self.step_config:
                # steps will store the current lr
                self.steps = self.change[self.step_config.index(epoch)]
            if self.steps == 0:
                return learning_rate
            else:
                return self.steps

        # will be moved to PowerSchedule in the future
        elif isinstance(self.step_config, int):
            self.steps = np.floor(epoch / self.step_config)

        elif isinstance(self.step_config, list):
            self.steps = np.sum(epoch >= np.array(self.step_config))

        return float(learning_rate * self.change ** self.steps)


class StepSchedule(Schedule):
    """
    Steps the learning rate over training time.

    To set a step schedule, pass as arguments ``step_config`` and ``change``. The schedule
    will set the learning rate at ``step[i]`` to ``change[i]``. For example, the call:

    .. code-block:: python

        schedule = Schedule(step_config=[2, 6], change=[0.6, 0.4])

    will set the learning rate to 0.6 at step 2, and to 0.4 at step 6.
    """
    def __init__(self, step_config, change):
        """
        Class constructor.

        Arguments:
            step_config (list): Configure the step times (list of epoch indices)
            change (list): List of learning rates. Must be same length as step_config
        """

        assert isinstance(step_config, list) and isinstance(change, list), \
            "The arguments change and step_config must be lists."

        assert len(step_config) == len(change), \
            "The arguments change and step_config must have the same length."

        self.step_config = step_config
        self.change = change
        self.steps = 0

    def get_learning_rate(self, learning_rate, epoch):
        """
        Returns the current learning rate given the epoch and initial learning rate.

        Arguments:
            learning_rate (float): Initial learning rate
            epoch (int): Current epoch, used to calculate the adjusted learning rate

        Returns:
            (float): The adjusted learning rate
        """
        if epoch in self.step_config:
            # steps will store the current lr
            self.steps = self.change[self.step_config.index(epoch)]
        if self.steps == 0:
            return learning_rate
        else:
            return self.steps


class PowerSchedule(Schedule):
    """
    Multiplies the learning rate by a factor at regular epoch intervals.

    This schedule will multiply the learning rate by
    the factor ``change`` every ``step_config`` epochs. For example,

    .. code-block:: python

        schedule = Schedule(step_config=2, change=0.5)
        optimizer = GradientDescentMomentum(0.1, 0.9, schedule=schedule)

    will yield a learning rate schedule of:

    .. csv-table::
        :header: "Epoch", "LR"
        :widths: 20, 10

        0, 0.1
        1, 0.1
        2, 0.05
        3, 0.05
        4, 0.025
        5, 0.025
        6, 0.0125
        7, 0.0125
    """
    def __init__(self, step_config, change):
        """
        Class constructor.

        Arguments:
            step_config (int): Learning rate update interval (in epochs)
            change (int): Update factor
        """
        assert isinstance(step_config, int), \
            "The argument step_config must be an integer."

        assert not isinstance(change, list), \
            "The argument change must be a float or integer."

        self.step_config = step_config
        self.change = change
        self.steps = 0

    def get_learning_rate(self, learning_rate, epoch):
        """
        Returns the current learning rate given the epoch and initial learning rate.

        Arguments:
            learning_rate (float): Initial learning rate
            epoch (int): Current epoch, used to calculate the adjusted learning rate.

        Returns:
            (float): The adjusted learning rate.
        """
        self.steps = np.floor(epoch / self.step_config)

        return float(learning_rate * self.change ** self.steps)


class ExpSchedule(Schedule):
    """
    Exponential learning rate schedule. This schedule implements

    .. math::
        \\alpha(t) = \\frac{\\alpha_\\circ}{1 + \\beta t}

    where :math:`\\beta` is the decay rate, and :math:`\\alpha_\\circ` is the
    initial learning rate.
    """
    def __init__(self, decay):
        """
        Class constructor.

        Arguments:
            decay (float): Decay rate.
        """
        self.decay = decay

    def get_learning_rate(self, learning_rate, epoch):
        """
        Returns the current learning rate given the epoch and initial learning rate.

        Arguments:
            learning_rate (float): Initial learning rate
            epoch (int): Current epoch, used to calculate the adjusted learning rate.

        Returns:
            (float): The adjusted learning rate.
        """
        return float(learning_rate / (1. + self.decay * epoch))


class PolySchedule(Schedule):
    """
    Polynomial learning rate schedule.

    This schedule takes as input the total number of epochs :math:`T` and a power :math:`\\beta`,
    and produces the learning schedule:

    .. math::
        \\alpha(t) = \\alpha_\\circ \\times\\left(1-\\frac{t}{T}\\right)^\\beta

    where :math:`\\alpha_\\circ` is the initial learning rate.
    """

    def __init__(self, total_epochs, power):
        """
        Class constructor.

        Arguments:
            total_epochs (int): Total number of epochs over which to calculate interpolated decay
            power (float): Total decay parameter
        """
        self.total_epochs = np.float32(total_epochs)
        self.power = power

    def get_learning_rate(self, learning_rate, epoch):
        """
        Returns the current learning rate given the epoch and initial learning rate.

        Arguments:
            learning_rate (float): Initial learning rate
            epoch (int): Current epoch, used to calculate the adjusted learning rate.

        Returns:
            (float): The adjusted learning rate.
        """
        return float(learning_rate * (1. - epoch // self.total_epochs) ** self.power)


class ShiftSchedule(Schedule):
    """
    Binary shift learning rate schedule.

    Arguments:
        interval (int): interval in epochs the learning rate is shifted
        shift_size (int): amount to shift
    """
    def __init__(self, interval, shift_size=1):
        self.interval = interval
        self.shift_size = shift_size

    def get_learning_rate(self, learning_rate, epoch):
        total_shift = -1 * self.shift_size * int(epoch/self.interval)
        return float(self.be.shift(learning_rate, total_shift, value=False).get())


class GradientDescentMomentum(Optimizer):

    """
    Stochastic gradient descent with momentum.

    Given the parameters :math:`\\theta`, the learning rate :math:`\\alpha`,
    and the gradients :math:`\\nabla J(\\theta; x)`
    computed on the minibatch data :math:`x`, SGD updates the parameters via

    .. math::
        \\theta' = \\theta - \\alpha\\nabla J(\\theta; x)

    Here we implement SGD with momentum. Momentum tracks the history of
    gradient updates to help the system move faster through saddle points.
    Given the additional parameters: momentum :math:`\gamma`, weight decay :math:`\lambda`,
    and current velocity :math:`v`, we use the following update equations

    .. math::
        v' = \\gamma v - \\alpha(\\nabla J(\\theta; x) + \\lambda\\theta)
        theta' = \\theta + v'

    The optional `nesterov` parameter implements Nesterov Accelerated Gradient.
    If this is set, we use the following update equations instead
    .. math::
        v' = \\gamma^2 v + \\alpha (\\gamma + 1) (\\nabla J(\\theta; x) + \\lambda\\theta)
        theta' = \\theta + v'

    Example usage:

    .. code-block:: python

        from neon.optimizers import GradientDescentMomentum

        # use SGD with learning rate 0.01 and momentum 0.9, while
        # clipping the gradient magnitude to between -5 and 5.
        opt = GradientDescentMomentum(0.01, 0.9, gradient_clip_value = 5)
    """

    def __init__(self, learning_rate, momentum_coef, stochastic_round=False,
                 wdecay=0.0, gradient_clip_norm=None, gradient_clip_value=None,
                 param_clip_value=None, name=None, schedule=Schedule(),
                 nesterov=False):
        """
        Class constructor.

        Arguments:
            learning_rate (float): Multiplicative coefficient of updates
            momentum_coef (float): Coefficient of momentum
            stochastic_round (bool, optional): Set this to True for stochastic
                                               rounding.  If False (default)
                                               rounding will be to nearest.  If
                                               True use default width
                                               stochastic rounding.  Note that
                                               this only affects the GPU
                                               backend.
            wdecay (float, optional): Amount of weight decay.  Defaults to 0
            gradient_clip_norm (float, optional): Target gradient norm.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
            param_clip_value (float, optional): Value to element-wise clip
                                                parameters.
                                                Defaults to None.
            name (str, optional): the optimizer's layer's pretty-print name.
                                  Defaults to "gdm".
            schedule (neon.optimizers.optimizer.Schedule, optional): Learning
                        rate schedule.  Defaults to a constant learning rate.
            nesterov (bool, optional): Use nesterov accelerated gradient.
                                       Defaults to False.
        """
        super(GradientDescentMomentum, self).__init__(name=name)
        self.learning_rate, self.momentum_coef = (learning_rate, momentum_coef)
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.param_clip_value = param_clip_value
        self.wdecay = wdecay
        self.schedule = schedule
        self.stochastic_round = stochastic_round
        self.nesterov = nesterov
        if self.momentum_coef == 0 and self.nesterov:
            raise ValueError("nesterov requires non-zero momentum")

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
            if len(states) == 0 and self.momentum_coef != 0:
                states.append(self.be.zeros_like(grad))

            grad = grad / self.be.bsz
            grad = self.clip_value(grad, self.gradient_clip_value)

            if self.momentum_coef == 0:
                param[:] = (- lrate * scale_factor) * grad +\
                           (1 - lrate * self.wdecay) * param
                param = self.clip_value(param, self.param_clip_value)
            else:
                grad = scale_factor * grad + self.wdecay * param
                velocity = states[0]
                velocity[:] = self.momentum_coef * velocity - lrate * grad

                # Nesterov accelerated gradient (NAG) is implemented the same
                # as in torch's "sgd.lua". It's a reformulation of Sutskever's
                # NAG equation found in "On the importance of initialization
                # and momentum in deep learning".
                if self.nesterov:
                    param[:] = self.clip_value(
                               param + self.momentum_coef * velocity
                               - lrate * grad, self.param_clip_value)
                else:
                    param[:] = self.clip_value(
                                param + velocity, self.param_clip_value)


class RMSProp(Optimizer):

    """
    Root Mean Square propagation.

    Root Mean Square (RMS) propagation protects against vanishing and
    exploding gradients. In RMSprop, the gradient is divided by a running
    average of recent gradients. Given the parameters :math:`\\theta`, gradient :math:`\\nabla J`,
    we keep a running average :math:`\\mu` of the last :math:`1/\\lambda` gradients squared.
    The update equations are then given by

    .. math::

        \\mu' &= \\lambda\\mu + (1-\\lambda)(\\nabla J)^2

    .. math::

        \\theta' &= \\theta - \\frac{\\alpha}{\\sqrt{\\mu + \\epsilon} + \\epsilon}\\nabla J

    where we use :math:`\\epsilon` as a (small) smoothing factor to prevent from dividing by zero.
    """

    def __init__(self, stochastic_round=False, decay_rate=0.95, learning_rate=2e-3, epsilon=1e-6,
                 gradient_clip_norm=None, gradient_clip_value=None, param_clip_value=None,
                 name=None, schedule=Schedule()):
        """
        Class constructor.

        Arguments:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            decay_rate (float): decay rate of states
            learning_rate (float): the multiplication coefficent of updates
            epsilon (float): smoothing epsilon to avoid divide by zeros
            gradient_clip_norm (float, optional): Target gradient norm.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
            param_clip_value (float, optional): Value to element-wise clip
                                                parameters.
                                                Defaults to None.
            schedule (neon.optimizers.optimizer.Schedule, optional): Learning rate schedule.
                                                                     Defaults to a constant.
        Notes:
            Only constant learning rate is supported currently.
        """
        super(RMSProp, self).__init__(name=name)
        self.state_list = None

        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.param_clip_value = param_clip_value
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
            grad = self.clip_value(grad, self.gradient_clip_value)

            # update state
            state = states[0]
            state[:] = decay * state + self.be.square(grad) * (1.0 - decay)

            param[:] = self.clip_value(
                        param - (scale_factor * grad * lrate)
                        / (self.be.sqrt(state + epsilon) + epsilon),
                        self.param_clip_value)


class Adagrad(Optimizer):

    """
    Adagrad optimization algorithm.

    Adagrad is an algorithm that adapts the learning rate individually for each parameter
    by dividing by the :math:`L_2`-norm of all previous gradients. Given the parameters
    :math:`\\theta`, gradient :math:`\\nabla J`, accumulating norm :math:`G`, and smoothing
    factor :math:`\\epsilon`, we use the update equations:

    .. math::

        G' = G + (\\nabla J)^2

    .. math::

        \\theta' = \\theta - \\frac{\\alpha}{\sqrt{G' + \\epsilon}} \\nabla J

    where the smoothing factor :math:`\\epsilon` prevents from dividing by zero.
    By adjusting the learning rate individually for each parameter, Adagrad adapts
    to the geometry of the error surface. Differently scaled weights have appropriately scaled
    update steps.

    Example usage:

    .. code-block:: python

        from neon.optimizers import Adagrad

        # use Adagrad with a learning rate of 0.01
        optimizer = Adagrad(learning_rate=0.01, epsilon=1e-6)

    """

    def __init__(self, stochastic_round=False, learning_rate=0.01, epsilon=1e-6,
                 gradient_clip_norm=None, gradient_clip_value=None,
                 param_clip_value=None, name=None):
        """
        Class constructor.

        Arguments:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            learning_rate (float): the multiplication coefficent of updates
            epsilon (float): smoothing epsilon to avoid divide by zeros
            gradient_clip_norm (float, optional): Target gradient norm.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
            param_clip_value (float, optional): Value to element-wise clip
                                                parameters.
                                                Defaults to None.
        Notes:
            Only constant learning rate is supported currently.
        """
        super(Adagrad, self).__init__(name=name)
        self.state_list = None
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.param_clip_value = param_clip_value
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
            grad = self.clip_value(grad, self.gradient_clip_value)

            # update state
            state = states[0]
            state[:] = state + self.be.square(grad)
            param[:] = self.clip_value(
                        param - (scale_factor * grad * lrate)
                        / (self.be.sqrt(state + epsilon)), self.param_clip_value)


class Adadelta(Optimizer):

    """
    Adadelta optimization algorithm.

    Similar to RMSprop, Adadelta tracks the running average of the
    gradients, :math:`\\mu_J`, over a window size :math:`1/\\lambda`, where
    :math:`\\lambda` is the parameter ``decay``. Adadelta also tracks an average of the
    recent update steps, which we denote as :math:`\\mu_\\theta`, and sets the learning rate
    as the ratio of the two averages:

    .. math::

        \\mu_J' &= \\lambda\\mu_J + (1-\\lambda) (\\nabla J)^2

    .. math::

        \\Delta \\theta &= \\sqrt{\\frac{\\mu_\\theta + \\epsilon}{\\mu_J' + \\epsilon}} \\nabla J

    .. math::

        \\mu_\\theta &= \\lambda \\mu_\\theta + (1-\\rho) (\\Delta \\theta)^2

    .. math::

        \\theta &= \\theta - \\Delta \\theta

    Note that the learning rate is a ratio of the average updates from the
    previous step, :math:`\\mu_\\theta`, divided by the average gradients including the current
    step, :math:`\\mu'_J`.

    Example usage:

    .. code-block:: python

        from neon.optimizers import Adadelta
        # use Adagrad with a learning rate of 0.01
        optimizer = Adadelta(decay=0.95, epsilon=1e-6)

    """

    def __init__(self, stochastic_round=False, decay=0.95, epsilon=1e-6,
                 param_clip_value=None, name=None):
        """
        Class constructor.

        Args:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            decay: decay parameter in Adadelta
            epsilon: epsilon parameter in Adadelta
            param_clip_value (float, optional): Value to element-wise clip
                                                parameters.
                                                Defaults to None.
        """
        super(Adadelta, self).__init__(name=name)
        self.decay = decay
        self.epsilon = epsilon
        self.stochastic_round = stochastic_round
        self.param_clip_value = param_clip_value

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

            param[:] = self.clip_value(param - states[2], self.param_clip_value)


class Adam(Optimizer):

    """
    Adam optimizer.

    The Adam optimizer combines features from RMSprop and Adagrad. We
    accumulate both the first and second moments of the gradient with decay
    rates :math:`\\beta_1` and :math:`\\beta_2` corresponding to window sizes of
    :math:`1/\\beta_1` and :math:`1/\\beta_2`, respectively.

    .. math::
        m' &= \\beta_1 m + (1-\\beta_1) \\nabla J

    .. math::
        v' &= \\beta_2 v + (1-\\beta_2) (\\nabla J)^2

    We update the parameters by the ratio of the two moments:

    .. math::
        \\theta = \\theta - \\alpha \\frac{\\hat{m}'}{\\sqrt{\\hat{v}'}+\\epsilon}

    where we compute the bias-corrected moments :math:`\\hat{m}'` and :math:`\\hat{v}'` via

    .. math::
        \\hat{m}' &= m'/(1-\\beta_1^t)

    .. math::
        \\hat{v}' &= v'/(1-\\beta_1^t)

    Example usage:

    .. code-block:: python

        from neon.optimizers import Adam

        # use Adam
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    """

    def __init__(self, stochastic_round=False, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, gradient_clip_norm=None, gradient_clip_value=None,
                 param_clip_value=None, name="adam"):
        """
        Class constructor.

        Args:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            learning_rate (float): the multiplicative coefficient of updates
            beta_1 (float): Adam parameter beta1
            beta_2 (float): Adam parameter beta2
            epsilon (float): numerical stability parameter
            gradient_clip_norm (float, optional): Target gradient norm.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip gradients.
                                                   Defaults to None.
            param_clip_value (float, optional): Value to element-wise clip parameters.
                                                Defaults to None.
        """
        super(Adam, self).__init__(name=name)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.stochastic_round = stochastic_round
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.param_clip_value = param_clip_value
        self.t = 0

    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            param_list (list): a list of tuples of the form ((param, grad), state),
                corresponding to parameters, grads, and states of layers to be updated
            epoch (int): the current epoch, needed for the Schedule object.
        """
        self.t = self.t + 1
        l = (self.learning_rate * self.be.sqrt(1 - self.beta_2 ** self.t) /
             (1 - self.beta_1 ** self.t))

        param_list = get_param_list(layer_list)

        scale_factor = self.clip_gradient_norm(param_list, self.gradient_clip_norm)

        for (param, grad), states in param_list:
            param.rounding = self.stochastic_round
            if len(states) == 0:
                # running_1st_mom, running_2nd_mom
                states.extend([self.be.zeros_like(grad) for i in range(2)])

            grad = grad / self.be.bsz
            grad = self.clip_value(grad, self.gradient_clip_value)

            m, v = states
            m[:] = m * self.beta_1 + (1. - self.beta_1) * grad
            v[:] = v * self.beta_2 + (1. - self.beta_2) * grad * grad

            param[:] = self.clip_value(
                        param - (scale_factor * l * m)
                        / (self.be.sqrt(v) + self.epsilon), self.param_clip_value)


class ShiftAdaMax(Optimizer):

    """
    Shift based AdaMax. http://arxiv.org/pdf/1602.02830v3.pdf
    """

    def __init__(self, stochastic_round=False, learning_rate=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, schedule=Schedule(), name="ShiftAdaMax"):
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
            schedule (neon.optimizers.optimizer.Schedule, optional): Learning rate schedule.
                                                                     Defaults to a constant.
        """
        super(ShiftAdaMax, self).__init__(name=name)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.stochastic_round = stochastic_round
        self.schedule = schedule

    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            param_list (list): a list of tuples of the form ((param, grad), state),
                corresponding to parameters, grads, and states of layers to be updated
            epoch (int): the current epoch, needed for the Schedule object.
        """
        t = epoch + 1
        lrate = self.schedule.get_learning_rate(self.learning_rate, epoch)
        l = lrate / (1 - self.beta_1 ** t)

        param_list = get_param_list(layer_list)

        for (param, grad), states in param_list:
            param.rounding = self.stochastic_round
            if len(states) == 0:
                # running_1st_mom, running_2nd_mom
                states.extend([self.be.zeros_like(grad) for i in range(3)])

            grad = grad / self.be.bsz
            m, v, inv_v = states
            m[:] = m * self.beta_1 + (1. - self.beta_1) * grad
            v[:] = self.be.maximum(v * self.beta_2, self.be.absolute(grad))

            inv_v[:] = 1.0 / (v + self.epsilon)
            param[:] = param - self.be.shift(self.be.shift(m, inv_v), l)
            self.be.clip(param, -1, 1, param)


class MultiOptimizer(Optimizer):

    """
    A wrapper class for using multiple Optimizers within the same model.

    To assign different optimizers to different layers we first define
    the different optimizers:

    .. code-block:: python

        from neon.optimizers import GradientDescentMomentum, RMSprop

        optimizer_A = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9)
        optimizer_B = GradientDescentMomentum(learning_rate=0.05, momentum_coef=0.9)
        optimizer_C = RMSprop(learning_rate=2e-3, decay_rate=0.95)

    Then, we instantiate this class and pass a
    dictionary mapping layers to optimizers. The keys can either be:
    ``default``, a layer class name (e.g. ``Bias``), or the Layer's name
    attribute. The latter takes precedence for finer layer-to-layer control.

    For example, if we have the following layers,

    .. code-block:: python

        layers = []
        layers.append(Linear(nout = 100, init=Gaussian(), name="layer_one"))
        layers.append(Linear(nout = 50, init=Gaussian(), name="layer_two"))
        layers.append(Affine(nout = 5, init=Gaussian(), activation=Softmax()))

    we can define multiple optimizers with

    .. code-block:: python

        from neon.optimizers import MultiOptimizer

        # dictionary of mappings
        mapping = {'default': optimizer_A, # default optimizer
                   'Linear': optimizer_B, # all layers from the Linear class
                   'layer_two': optimizer_C} # this overrides the previous entry

        # use multiple optimizers
        opt = MultiOptimizer(mapping)

    After definition, we have the following mapping

    +----------------------+----------------------------+
    | Layer                | Optimizer                  |
    +======================+============================+
    | ``layer_one``        | ``optimizer_B``            |
    +----------------------+----------------------------+
    | ``layer_two``        | ``optimizer_C``            |
    +----------------------+----------------------------+
    | ``Affine.Linear``    | ``optimizer_B``            |
    +----------------------+----------------------------+
    | ``Affine.Bias``      | ``optimizer_A``            |
    +----------------------+----------------------------+
    | ``Affine.Softmax``   | ``None (no parameters)``   |
    +----------------------+----------------------------+
    """

    def __init__(self, optimizer_mapping, name=None):
        """
        Class constructor.

        Args:
            optimizer_mapping (dict): dictionary specifying the mapping of layers to optimizers.
                Key: ``'default'``, layer class name or layer `name` attribute.
                Don't name your layers ``'default'``. Value: the optimizer object to
                use for those layers.
        """
        super(MultiOptimizer, self).__init__(name=name)
        self.optimizer_mapping = optimizer_mapping
        assert 'default' in self.optimizer_mapping, "Must specify a default" \
            "optimizer in layer type to optimizer mapping"

        self.map_list = None
        self.map_list_cache = dict()

    @classmethod
    def gen_class(cls, pdict):
        for key in pdict['optimizer_mapping']:
            # these should be optimizers
            typ = pdict['optimizer_mapping'][key]['type']
            ocls = load_class(typ)
            if 'config' not in pdict['optimizer_mapping'][key]:
                pdict['optimizer_mapping'][key]['config'] = {}
            conf = pdict['optimizer_mapping'][key]['config']
            pdict['optimizer_mapping'][key] = ocls.gen_class(conf)
        return cls(**pdict)

    def get_description(self):
        desc = {'type': self.modulenm}
        desc['config'] = {'optimizer_mapping': {}}
        for key in self.optimizer_mapping:
            opt_desc = self.optimizer_mapping[key].get_description()
            desc['config']['optimizer_mapping'][key] = opt_desc
        return desc

    def _map_optimizers(self, layer_list):
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

    def _reset_mapping(self, new_mapping):
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

        if not id(layer_list) in self.map_list_cache:
            self.map_list = self._map_optimizers(layer_list)
            self.map_list_cache[id(layer_list)] = self.map_list
        else:
            self.map_list = self.map_list_cache[id(layer_list)]

        for opt in self.map_list:
            opt.optimize(self.map_list[opt], epoch)
