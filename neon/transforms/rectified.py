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
Rectified linear (ReLU) transform functions and classes.
"""

from neon.transforms.activation import Activation


class RectLin(Activation):
    """
    Embodiment of a rectified linear activation function.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def apply_function(self, backend, inputs, outputs):
        """
        Apply the rectified linear activation function.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectlin(inputs, outputs)

    def apply_derivative(self, backend, inputs, outputs):
        """
        Apply the rectified linear activation function derivative.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectlin_derivative(inputs, outputs)

    def fprop_func(self, backend, inputs, outputs):
        """
        Function to apply during fprop

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also acts
                                 as storage for the output of the derivative
                                 function.
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectlin(inputs, outputs)

    def pre_act_buffer(self, backend, output, dtype):
        """
        overrides the pre_act_buffer with output to save memory

        Arguments:
            backend (Backend): The backend class to use for computation.
            output (array_like): Output data buffer.
            dtype: dtype for pre_act_buffer
        """
        return output

    def bprop_func(self, backend, pre_act, error, skip_act=False):
        """
        Function to perform during the bprop

        Arguments:
            backend (Backend): The backend class to use for computation.
            pre_act (array_like): pre_activation buffer
            error (array_like): error buffer
            skip_act (Boolean): whether to skip the multiplication
        """
        backend.greater(pre_act, 0, out=pre_act)
        super(RectLin, self).bprop_func(backend, pre_act, error, skip_act)


class RectLeaky(Activation):
    """
    Embodiment of a leaky rectified linear activation function.  Instead of
    the hard zero gradient for all non-active values, a small, non-zero
    gradient exists instead.

    See Maas2013 for details.
    """

    def __init__(self, slope=0.01, **kwargs):
        self.slope = slope
        self.__dict__.update(kwargs)

    def apply_function(self, backend, inputs, outputs):
        """
        Apply the leaky rectified linear activation function.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectleaky(inputs, self.slope, outputs)

    def apply_derivative(self, backend, inputs, outputs):
        """
        Apply the leaky rectified linear activation function derivative.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectleaky_derivative(inputs, self.slope, outputs)

    def fprop_func(self, backend, inputs, outputs):
        """
        Function to apply during fprop

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also acts
                                 as storage for the output of the derivative
                                 function.
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectleaky(inputs, self.slope, outputs)

    def bprop_func(self, backend, pre_act, error, skip_act=False):
        """
        Function to perform during the bprop

        Arguments:
            backend (Backend): The backend class to use for computation.
            pre_act (array_like): pre_activation buffer
            error (array_like): error buffer
            skip_act (Boolean): whether to skip the multiplication
        """
        super(RectLeaky, self).bprop_func(backend, pre_act, error, skip_act)
