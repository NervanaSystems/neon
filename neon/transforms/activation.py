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
Contains activation function related code.
"""


class Activation(object):
    """
    Abstract activation function class.  Defines operations any concrete
    activation function child must support.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.gain = 1.0

    def apply_function(self, backend, inputs, outputs):
        """
        Computes the activation function value by applying it to each element
        of the dataset passed.

        Arguments:
            dataset (array_like): The dataset upon which to apply the
                                  activation function.

        Returns:
            array_like: A transformed copy of the input dataset with the same
                        type and shape.

        Raises:
            NotImplementedError: Must be implemented in a child class.
        """
        raise NotImplementedError("apply_function should be" +
                                  "overridden in child class.")

    def apply_derivative(self, backend, inputs, outputs):
        """
        Computes the activation function derivative value by applying it to
        each element of the dataset passed.

        Arguments:
            dataset (array_like): The dataset upon which to apply the
                                  activation function derivative.

        Returns:
            array_like: A transformed copy of the input dataset with the same
                        type and shape.

        Raises:
            NotImplementedError: Must be implemented in a child class.
        """
        raise NotImplementedError("apply_derivative should be" +
                                  "overridden in child class.")

    def pre_act_buffer(self, backend, output, dtype):
        """
        Creates the pre_act_buffer

        Arguments:
            backend (Backend): The backend class to use for computation.
            output (array_like): Output data buffer.
            dtype: dtype for pre_act_buffer
        """
        return backend.zeros(output.shape, dtype)

    def fprop_func(self, backend, inputs, outputs):
        """
        Function to apply during fprop
        Typically computes the activation function and its derivative by
        applying it to each element of the dataset passed, but there are
        exceptions (RectLin).

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also
                                 acts as storage for the output of the
                                 derivative function.
            outputs (array_like): Storage for the transformed output.

        Raises:
            NotImplementedError: Must be implemented in a child class.
        """
        raise NotImplementedError("apply_both should be" +
                                  "overridden in child class.")

    def bprop_func(self, backend, pre_act, error, skip_act=False):
        """
        Function to apply during bprop
        Typically empty, but can be used to compute derivative during bprop
        instead of storing it during fprop (used in RectLin).

        Arguments:
            backend (Backend): The backend class to use for computation.
            pre_act (array_like): pre_activation buffer
            error (array_like): error buffer
            skip_act (Boolean): whether to skip the multiplication
        """
        if skip_act is False:
            backend.multiply(error, pre_act, out=error)
