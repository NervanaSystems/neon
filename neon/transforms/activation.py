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
from neon.transforms.transform import Transform


class Identity(Transform):
    def __init__(self, name='identity'):
        super(Identity, self).__init__(name)

    def __call__(self, x):
        return x

    def bprop(self, x):
        return 1


class Rectlin(Transform):
    """
    ReLu activation function (Nair and  Hinton, ICML 2010)
    Can optionally set a slope which will make this a Leaky ReLu
    Computes the function f(x) = max(0, x)
    """
    def __init__(self, slope=0, name='relu'):
        super(Rectlin, self).__init__(name)
        self.slope = slope

    def __call__(self, x):
        return self.be.maximum(x, 0) + self.slope * self.be.minimum(0, x)

    def bprop(self, x):
        return self.be.greater(x, 0) + self.slope * self.be.less(x, 0)


class Explin(Transform):
    """
    ELU activation function (Clevert, Unterthiner and Hochreiter, ICLR 2016 submission)
    """
    def __init__(self, alpha=1.0, name='elu'):
        super(Explin, self).__init__(name)
        self.alpha = alpha

    def __call__(self, x):
        return self.be.maximum(x, 0) + self.alpha * (self.be.exp(self.be.minimum(x, 0)) - 1)

    def bprop(self, x):
        return self.be.greater(x, 0) + self.be.minimum(x, 0) + self.alpha * self.be.less(x, 0)


class Normalizer(Transform):
    """
    Normalize inputs by a fixed divisor
    """
    def __init__(self, name='normalizer', divisor=128.):
        super(Normalizer, self).__init__(name)
        self.divisor = divisor

    def __call__(self, x):
        return x / self.divisor

    def bprop(self, x):
        return x


class Softmax(Transform):
    """
    SoftMax activation function.
    Computes the function f(x_k) = exp(x_k) / sum_i(exp(x_i))
    """
    def __init__(self, name='softmax', epsilon=2**-23):
        super(Softmax, self).__init__(name)
        self.epsilon = epsilon

    def __call__(self, x):
        return (self.be.reciprocal(self.be.sum(
                self.be.exp(x - self.be.max(x, axis=0)), axis=0)) *
                self.be.exp(x - self.be.max(x, axis=0)))

    def bprop(self, x):
        return 1


class Tanh(Transform):
    """
    Hyperbolic tangent activation function.
    Computes the function f(x) = (1 - exp(-2x))  / (1 + exp(-2x))
    """
    def __init__(self, name='tanh'):
        super(Tanh, self).__init__(name)

    def __call__(self, x):
        return self.be.tanh(x)

    def bprop(self, x):
        return (1.0 - self.be.square(x))


class Logistic(Transform):
    """
    Logistic sigmoid activation function.
    Computes the function f(x) = 1  / (1 + exp(-x))
    """
    def __init__(self, shortcut=False):
        """Initialize Logistic based on whether shortcut is True or False

        Args:
            shortcut (bool): if True shortcut is used
                             if False, actual derivative is returned in bprop

        """
        super(Logistic, self).__init__(name='logistic')

        self.set_shortcut(shortcut)

    def set_shortcut(self, shortcut):
        """Method to set the bprop func to use shortcut
           when gradients do not need to be calculated.

           Arguments:
               shortcut (bool): if True shortcut is used
               if False, actual derivative is returned in bprop
        """
        self.shortcut = shortcut

        if shortcut:
            self.bprop_func = lambda x: 1
        else:
            self.bprop_func = lambda x: x * (1.0 - x)

    def __call__(self, x):
        return self.be.reciprocal(self.be.exp(-x) + 1.0)

    def bprop(self, y):
        """Returns the derivative of the logistic (sigmoid) function at y (output)
        Args:
            y (Tensor or OpTree): input. y = f(x)

        Returns:
            OpTree: Derivative of the Logistic (sigmoid)
                    Returns 1 if shortcut is True
                    Returns derivative (y*(1-y)) if shortcut is False

        """
        return self.bprop_func(y)
