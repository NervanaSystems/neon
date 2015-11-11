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

import numpy as np

from neon import NervanaObject


class Initializer(NervanaObject):
    """
    Abstract base class from which parameter tensor initializers inherit.
    """
    def fill(self, param):
        raise NotImplementedError()


class Constant(Initializer):
    """
    A class for initializing parameter tensors with a single value.

    Args:
        val (float, optional): The value to assign to all tensor elements
    """
    def __init__(self, val=0.0, name="constantInit"):
        super(Constant, self).__init__(name=name)
        self.val = val

    def fill(self, param):
        param[:] = self.val


class Uniform(Initializer):
    """
    A class for initializing parameter tensors with values drawn from
    a uniform distribution.

    Args:
        low  (float, optional): Lower bound of range from which we draw values.
        high (float, optional): Upper bound of range from which we draw values.
    """
    def __init__(self, low=0.0, high=1.0, name="uniformInit"):
        super(Uniform, self).__init__(name=name)
        self.low, self.high = (low, high)

    def fill(self, param):
        param[:] = self.be.rng.uniform(self.low, self.high, param.shape)


class Gaussian(Initializer):
    """
    A class for initializing parameter tensors with values drawn from
    a normal distribution.

    Args:
        loc   (float, optional): The mean of the normal (mu).
        scale (float, optional): The standard deviation of the normal (sigma).
    """
    def __init__(self, loc=0.0, scale=1.0, name="gaussianInit"):
        super(Gaussian, self).__init__(name=name)
        self.loc, self.scale = (loc, scale)

    def fill(self, param):
        param[:] = self.be.rng.normal(self.loc, self.scale, param.shape)


class GlorotUniform(Initializer):
    """
    A class for initializing parameter tensors with values drawn from
    a uniform distribution over a region whose bounds have been determined
    using the policy described in:
    "Understanding the difficulty of training deep feedforward neural networks"
    (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

    We normalize the range by the scaled average of the input dimension
    and the output dimension of the tensor in question.
    """
    def __init__(self, name="autouniformInit"):
        super(GlorotUniform, self).__init__(name=name)

    def fill(self, param):
        k = np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
        param[:] = self.be.rng.uniform(-k, k, param.shape)


class Xavier(Initializer):
    """
    Alternate form of Glorot where only input nodes are used for scaling range.

    Args:
        local   (bool, optional): Whether the layer type is local (Convolutional) or not.
                                  default is True.
    """

    def __init__(self, local=True, name="xavier"):
        super(Xavier, self).__init__(name=name)
        self.local = local

    def fill(self, param):
        fan_in = param.shape[0 if self.local else 1]
        scale = np.sqrt(3./fan_in)
        param[:] = self.be.rng.uniform(-scale, scale, param.shape)


class Orthonormal(Initializer):
    """
    Implementation taken from Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """

    def __init__(self, scale=1.1, name="orthonormal"):
        super(Orthonormal, self).__init__(name=name)
        self.scale = scale

    def fill(self, param):
        a = np.random.normal(0.0, 1.0, param.shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == param.shape else v
        param[:] = self.scale * q
