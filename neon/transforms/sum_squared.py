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
Sum of squares transform functions and classes.
"""

from neon.transforms.cost import Cost


def sum_squared_diffs(backend, outputs, targets, temp,
                      scale_by_batchsize=False):
    """
    Evaluates sum of squared difference on pairwise elements from outputs and
    targets.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        scalar: Calculated sum of squared diff values for each element.
    """
    backend.subtract(outputs, targets, temp[0])
    backend.multiply(temp[0], temp[0], temp[0])
    if scale_by_batchsize:
        backend.divide(temp[0], temp[0].shape[1], temp[0])
    result = backend.empty((1, 1), dtype=outputs.dtype)
    backend.sum(temp[0], axes=None, out=result)
    return backend.multiply(result, 0.5, result)


def sum_squared_diffs_derivative(backend, outputs, targets, temp, scale=1.0):
    """
    Applies derivative of the sum of squared differences to pairwise elements
    from outputs and targets (with respect to the outputs).

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated diff values for each corresponding element.
                    Will have the same shape as outputs.
    """

    backend.subtract(outputs, targets, temp[0])
    backend.multiply(temp[0], scale, out=temp[0])
    return temp[0]


class SumSquaredDiffs(Cost):
    """
    Embodiment of a sum of squared differences cost function.
    """
    def __init__(self, **kwargs):
        super(SumSquaredDiffs, self).__init__(**kwargs)

    def set_outputbuf(self, databuf):
        if not self.outputbuf or self.outputbuf.shape != databuf.shape:
            tempbuf = self.backend.empty(databuf.shape, self.temp_dtype)
            self.temp = [tempbuf]
        self.outputbuf = databuf

    def get_deltabuf(self):
        return self.temp[0]

    def apply_function(self, targets, scale_by_batchsize=False):
        """
        Apply the sum of squared differences cost function to the datasets
        passed.
        """
        result = sum_squared_diffs(self.backend, self.outputbuf, targets,
                                   self.temp, scale_by_batchsize)
        return self.backend.multiply(result, self.scale, result)

    def apply_derivative(self, targets):
        """
        Apply the derivative of the sum of squared differences cost function
        to the datasets passed.
        """
        return sum_squared_diffs_derivative(self.backend,
                                            self.outputbuf, targets,
                                            self.temp, self.scale)
