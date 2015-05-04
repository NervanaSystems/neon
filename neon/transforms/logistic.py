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
Logistic transform functions and classes.
"""

from neon.transforms.activation import Activation


class Logistic(Activation):

    """
    Embodiment of a logistic activation function.
    """
    def __init__(self):
        self.tmp = None

    def apply_function(self, backend, inputs, outputs):
        """
        Applies logistic transform to the dataset passed.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.logistic(inputs, outputs)

    def apply_derivative(self, backend, inputs, outputs):
        """
        Applies derivative of the logistic transform to the dataset passed.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        if not self.tmp or self.tmp.shape != inputs.shape:
            self.tmp = backend.zeros(inputs.shape)

        backend.logistic(inputs, outputs)
        backend.subtract(1.0, outputs, out=self.tmp)
        backend.multiply(outputs, self.tmp, outputs)

    def fprop_func(self, backend, inputs, outputs):
        """
        Applies logistic function and its derivative to the dataset passed.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also
                                 acts as storage for the output of the
                                 derivative function.
            outputs (array_like): Storage for the transformed output.
        """
        # Apply the logistic function.
        backend.logistic(inputs, outputs)

        # Apply the derivative of the logistic function, storing the result in
        # inputs
        backend.subtract(1.0, outputs, out=inputs)
        backend.multiply(inputs, outputs, out=inputs)
