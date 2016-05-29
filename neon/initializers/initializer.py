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
import numpy as np

from neon import NervanaObject
from neon.backends.backend import Tensor


class Initializer(NervanaObject):
    """
    Abstract base class from which parameter tensor initializers inherit.

    Subclasses should implement the ``fill`` method which takes as input a Tensor
    and fills the values based on the initialization scheme.
    """
    def fill(self, param):
        """
        Initialize the provided tensor with values.

        Args:
            param (Tensor): Input Tensor.
        """
        raise NotImplementedError()


class Constant(Initializer):
    """
    Initializes parameters as a constant.
    """
    def __init__(self, val=0.0, name="constantInit"):
        """
        Class constructor.

        Args:
            val (float, optional): The value to assign to all tensor elements
        """
        super(Constant, self).__init__(name=name)
        self.val = val

    def fill(self, param):
        """
        Fills the provided tensor.

        Args:
            param (tensor): target tensor to fill
        """
        if isinstance(self.val, Tensor):
            assert self.val.shape == param.shape, "Constant(Array) initializer can"\
                                                  " only fill a matching shape tensor"
        param[:] = self.val


class Array(Constant):
    """
    Initializes parameters with values specified by a provided numpy array.

    Same functionality as Constant except serialization needs to dump
    tensor values into np array

    Args:
        vals (ndarray or tensor, optional): Values to assign to the tensor elements
    """
    def get_description(self):
        """
        Returns description of the object as a dict. Transfers the
        tensors back to a numpy array.
        """
        desc = super(Array, self).get_description()
        if isinstance(desc['config']['val'], Tensor):
            desc['config']['val'] = desc['config']['val'].get()
        return desc


class Uniform(Initializer):
    """
    Initializes parameters with random values drawn from a uniform distribution.
    """
    def __init__(self, low=0.0, high=1.0, name="uniformInit"):
        """
        Class constructor.

        Args:
            low  (float, optional): Lower bound of range.
            high (float, optional): Upper bound of range.
        """
        super(Uniform, self).__init__(name=name)
        self.low, self.high = (low, high)

    def fill(self, param):
        """
        Fill the provided tensor with random values drawn from a uniform
        distribution.

        Args:
            params (tensor): Tensor to fill
        """
        param[:] = self.be.rng.uniform(self.low, self.high, param.shape)


class Gaussian(Initializer):
    """
    Initializes parameters with a gaussian distribution with the provided mean
    and standard deviation. Defaults to (loc=0, scale=1)
    """
    def __init__(self, loc=0.0, scale=1.0, name="gaussianInit"):
        """
        Class constructor.

        Args:
            loc   (float, optional): Mean parameter (mu). Defaults to 0.
            scale (float, optional): Standard deviation parameter (sigma). Defaults to 1.
            name (string, optional): Name to assign an instance of this class.
        """
        super(Gaussian, self).__init__(name=name)
        self.loc, self.scale = (loc, scale)

    def fill(self, param):
        """
        Fill the provided tensor with random values drawn from a gaussian
        distribution.

        Args:
            params (tensor): Tensor to fill
        """
        param[:] = self.be.rng.normal(self.loc, self.scale, param.shape)


class GlorotUniform(Initializer):
    """
    Initializes parameter tensors with values drawn from a uniform distribution
    ranging from :math:`-K` to :math:`K`. We define :math:`K=\sqrt{6 / (n_{in} + n_{out})}`,
    where :math:`n_{in}` and :math:`n_{out}` are the input and output dimensions, respectively,
    of the parameter tensor. This approach normalizes the range of the initialized values
    by the tensor dimensions.

    From: "Understanding the difficulty of training deep feedforward neural networks"
    (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
    """
    def __init__(self, name="autouniformInit"):
        """
        Class constructor.

        Args:
            name (string, optional): Name to assign an instance of this class
        """
        super(GlorotUniform, self).__init__(name=name)

    def fill(self, param):
        """
        Fill the provided tensor with random values drawn from the Uniform
        distribution, using normalized bounds.

        Args:
            params (tensor): Tensor to fill
        """
        k = np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
        param[:] = self.be.rng.uniform(-k, k, param.shape)


class Xavier(Initializer):
    """
    Initializes parameter tensors with values drawn from a uniform distribution
    ranging from :math:`-K` to :math:`K` We define :math:`K=\sqrt{3 / (n_{in})}`,
    where :math:`n_{in}` is the number of input nodes.

    Similar to Glorot except the range is normalized by the input size only.
    """

    def __init__(self, local=True, name="xavier"):
        """
        Class constructor.

        Args:
            local (bool, optional): Whether the layer type is local (Convolutional) or not.
                                      Default is True.
            name (string, optional): Name to assign an instance of this class.
        """
        super(Xavier, self).__init__(name=name)
        self.local = local

    def fill(self, param):
        """
        Fill the provided tensor with random values drawn from the Uniform
        distribution, using normalized bounds.

        Args:
            params (tensor): Tensor to fill
        """
        fan_in = param.shape[0 if self.local else 1]
        scale = np.sqrt(3. / fan_in)
        param[:] = self.be.rng.uniform(-scale, scale, param.shape)


class Kaiming(Initializer):
    """
    Initializes parameters with a zero-mean Gaussian distribution. The standard deviation
    is automatically set as :math:`\sigma=\sqrt{2 / n_{in}}`, where :math:`n_{in}` is
    the input dimension of the tensor.


    Based on the initializer described in: http://arxiv.org/pdf/1502.01852.pdf.
    """
    def __init__(self, local=True, name="Kaiming"):
        """
        Class constructor.

        Args:
            local (bool, optional): Whether the layer type is local (Convolutional) or not.
                                      Default is True.
            name (string, optional): Name to assign an instance of this class.
        """
        super(Kaiming, self).__init__(name=name)
        self.local = local

    def fill(self, param):
        """
        Fill the provided tensor with random values drawn from a gaussian
        distribution.

        Args:
            params (tensor): Tensor to fill
        """
        fan_in = param.shape[0 if self.local else 1]
        scale = np.sqrt(2. / fan_in)
        param[:] = self.be.rng.normal(0, scale, param.shape)


class IdentityInit(Initializer):
    """
    Initializes parameters with the identity matrix.
    """
    def __init__(self, local=True, name="Identity"):
        """
        Class constructor.

        Args:
            local (bool, optional): Whether the layer type is local (Convolutional) or not.
                                      Default is True.
            name (string, optional): Name to assign an instance of this class.
        """
        super(IdentityInit, self).__init__(name=name)
        self.local = local

    def fill(self, param):
        """
        Fill the provided tensor with the identity matrix.

        Args:
            params (tensor): Tensor to fill
        """
        (nin, nout) = param.shape
        w_ary = np.zeros((nin, nout), dtype=np.float32)
        w_ary[:, :nin] = np.eye(nin)
        param[:] = w_ary


class Orthonormal(Initializer):
    """
    Initializes parameters with the single value decomposition of a
    random gaussian matrix.

    Implementation taken from Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """

    def __init__(self, scale=1.1, name="orthonormal"):
        """
        Class constructor.

        Args:
            scale (float, optional): Scaling factor of values. Defaults to 1.1.
            name (string, optional): Name to assign an instance of this class.
        """
        super(Orthonormal, self).__init__(name=name)
        self.scale = scale

    def fill(self, param):
        """
        Fill the provided tensor using the Orthonormal method.

        Args:
            params (tensor): Tensor to fill
        """
        a = np.random.normal(0.0, 1.0, param.shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == param.shape else v
        param[:] = self.scale * q
