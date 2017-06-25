# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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
from neon.transforms.transform import Transform


class Identity(Transform):
    """
    Identity activation function, :math:`f(x) = x`
    """
    def __init__(self, name=None):
        """
        Class constructor.
        """
        super(Identity, self).__init__(name)

    def __call__(self, x):
        """
        Returns the input as output.

        Arguments:
            x (Tensor or optree): input value

        Returns:
            Tensor or optree: identical to input
        """
        return x

    def bprop(self, x):
        """
        Returns the derivative.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Integer value 1.

        """
        return 1


class Rectlin(Transform):
    """
    Rectified Linear Unit (ReLu) activation function, :math:`f(x) = \max(x, 0)`.
    Can optionally set a slope which will make this a Leaky ReLu.
    """
    def __init__(self, slope=0, name=None):
        """
        Class constructor.

        Args:
            slope (float, optional): Slope for negative domain. Defaults to 0.
            name (string, optional): Name to assign this class instance.
        """
        super(Rectlin, self).__init__(name)
        self.slope = slope
        self.is_mklop = True

    def __call__(self, x, nglayer=None):
        """
        Returns the Exponential Linear activation

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: output activation
        """
        return self.be.fprop_relu(nglayer, x, self.slope)

    def bprop(self, x, nglayer=None, error=None, deltas=None):
        """
        Returns the derivative.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Derivative
        """
        return self.be.bprop_relu(nglayer, x, error, deltas, self.slope)


class Rectlinclip(Transform):
    """
    Clipped ReLu activation function
    Computes the function f(x) = min(max(0, x),cutoff)
    """
    def __init__(self, slope=0, name=None, xcut=20.0):
        super(Rectlinclip, self).__init__(name)
        self.xcut = xcut
        self.slope = slope

    def __call__(self, x):
        return self.be.minimum(self.be.maximum(x, 0) + self.slope * self.be.minimum(x, 0),
                               self.xcut)

    def bprop(self, x):
        return (self.be.greater(x, 0) + self.slope * self.be.less(x, 0)) *\
                self.be.greater(self.xcut, x)


class Explin(Transform):
    """
    Exponential Linear activation function, :math:`f(x) = \max(x, 0) + \\alpha (e^{\min(x, 0)}-1)`

    From: Clevert, Unterthiner and Hochreiter, ICLR 2016.
    """
    def __init__(self, alpha=1.0, name=None):
        """
        Class constructor.

        Arguments:
            alpha (float): weight of exponential factor for negative values (default: 1.0).
            name (string, optional): Name (default: None)
        """
        super(Explin, self).__init__(name)
        self.alpha = alpha

    def __call__(self, x):
        """
        Returns the Exponential Linear activation

        Arguments:
            x (Tensor or optree): input value

        Returns:
            Tensor or optree: output activation
        """
        return self.be.maximum(x, 0) + self.alpha * (self.be.exp(self.be.minimum(x, 0)) - 1)

    def bprop(self, x):
        """
        Returns the derivative.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Derivative
        """
        return self.be.greater(x, 0) + self.be.minimum(x, 0) + self.alpha * self.be.less(x, 0)


class Normalizer(Transform):
    """
    Normalize inputs by a fixed divisor.
    """
    def __init__(self, name=None, divisor=128.):
        """
        Class constructor.

        Arguments:
            divisor (float, optional): Normalization factor (default: 128)
            name (string, optional): Name (default: None)
        """
        super(Normalizer, self).__init__(name)
        self.divisor = divisor

    def __call__(self, x):
        """
        Returns the normalized value.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output :math:`x / N`
        """
        return x / self.divisor

    def bprop(self, x):
        """
        Returns the derivative.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Derivative
        """
        return 1.0 / self.divisor


class Softmax(Transform):
    """
    SoftMax activation function. Ensures that the activation output sums to 1.
    """
    def __init__(self, axis=0, name=None, epsilon=2**-23):
        """
        Class constructor.

        Arguments:
            name (string, optional): Name (default: none)
            epsilon (float, optional): Not used.
            axis (int, optional): axis to perform softmax (default: 0)
        """
        super(Softmax, self).__init__(name)
        self.epsilon = epsilon
        self.axis = axis

    def __call__(self, x):
        """
        Returns the Softmax value.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output activation
        """
        return self.be.fprop_softmax(x, self.axis)

    def bprop(self, x):
        """
        Returns the derivative.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Integer value 1
        """
        return 1


class PixelwiseSoftmax(Transform):
    """
    Pixelwise SoftMax activation function.
    Computes the function f(x_k) = exp(x_k) / sum_i(exp(x_i))
    """
    def __init__(self, c, name=None, epsilon=2**-23):
        super(PixelwiseSoftmax, self).__init__(name)
        self.epsilon = epsilon
        self.c = c

    def __call__(self, x):
        y = x.reshape((self.c, -1))
        y[:] = (self.be.reciprocal(self.be.sum(self.be.exp(y - self.be.max(y, axis=0)), axis=0)) *
                self.be.exp(y - self.be.max(y, axis=0)))
        return x

    def bprop(self, x):
        return 1


class Tanh(Transform):
    """
    Hyperbolic tangent activation function, :math:`f(x) = \\tanh(x)`.
    """
    def __init__(self, name=None):
        """
        Class constructor.
        """
        super(Tanh, self).__init__(name)

    def __call__(self, x):
        """
        Returns the hyperbolic tangent.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output activation
        """
        return self.be.tanh(x)

    def bprop(self, x):
        """
        Returns the derivative.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Derivative, :math:`1-x^2`
        """
        return (1.0 - self.be.square(x))


class Logistic(Transform):
    """
    Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`

    Squashes the input from range :math:`[-\infty,+\infty]` to :math:`[0, 1]`
    """
    def __init__(self, name=None, shortcut=False):
        """
        Initialize Logistic based on whether shortcut is True or False. Shortcut
        should be set to true when Logistic is used in conjunction with a CrossEntropy cost.
        Doing so allows a shortcut calculation to be used during backpropagation.

        Args:
            shortcut (bool): If True, shortcut calculation will be used during backpropagation.

        """
        super(Logistic, self).__init__(name=name)

        self.set_shortcut(shortcut)

    def set_shortcut(self, shortcut):
        """
        Sets the backpropagation to use the shortcut when gradients do not
        need to be calculated.

        If True, a shortcut calculation is used. If False, the actual derivative
        is return during backpropagation.

        Arguments:
            shortcut (bool): If True, shortcut calculation will be used during backpropagation.
        """
        self.shortcut = shortcut

        if shortcut:
            self.bprop_func = lambda x: 1
        else:
            self.bprop_func = lambda x: x * (1.0 - x)

    def __call__(self, x):
        """
        Returns the sigmoidal activation.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output activation
        """
        return self.be.sig(x)

    def bprop(self, y):
        """

        Returns the derivative of the logistic (sigmoid) function at y (output)

        Args:
            y (Tensor or OpTree): input. y = f(x)

        Returns:
            OpTree: Derivative of the Logistic (sigmoid)
                    Returns 1 if shortcut is True.
                    Returns derivative (y*(1-y)) if shortcut is False.

        """
        return self.bprop_func(y)


class Sign(Transform):
    """
    Sign activation function.
    Computes the function f(x) = Sign(x).
    Uses straight-through estimator for bprop.
    """
    def __init__(self, name=None):
        super(Sign, self).__init__(name)

    def __call__(self, x):
        self.inputs = self.be.array(x.get())
        return self.be.binarize(x, x, stochastic=False)

    def bprop(self, x):
        return self.be.less_equal(self.be.absolute(self.inputs), 1)
