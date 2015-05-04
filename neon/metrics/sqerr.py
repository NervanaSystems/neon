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
Contains squared error related metrics
"""

import numpy as np

from neon.metrics.metric import Metric


class SSE(Metric):
    """
    Sum of Squared Errors.

    See Also:
        MSE

    """

    def __init__(self, **kwargs):
        super(SSE, self).__init__(**kwargs)

    def add(self, reference, outputs):
        """
        Add the the expected reference and predicted outputs passed to the set
        of values used to calculate this metric.

        Arguments:
            reference (neon.backend.Tensor): Ground truth, expected outcomes.
            outputs (neon.backend.Tensor): Predicted outputs.  Must have the
                                           same dimensions as reference.
        """

        diff = reference.asnumpyarray() - outputs.asnumpyarray()
        self.value += np.sum(diff * diff)

    def report(self):
        """
        Report the metric

        Returns:
            float: value of this metric
        """
        return self.value

    def clear(self):
        """
        Reset this metric's calculated value
        """
        self.value = 0.0


class MSE(SSE):
    """
    Mean Sqaured Error.

    See Also:
        SSE

    """

    def add(self, reference, outputs):
        """
        Add the the expected reference and predicted outputs passed to the set
        of values used to calculate this metric.

        Arguments:
            reference (neon.backend.Tensor): Ground truth, expected outcomes.
            outputs (neon.backend.Tensor): Predicted outputs.  Must have the
                                           same dimensions as reference.
        """
        super(MSE, self).add(reference, outputs)
        self.rec_count += np.prod(reference.shape)

    def report(self):
        """
        Report the metric

        Returns:
            float: value of this metric
        """
        return super(MSE, self).report() / self.rec_count

    def clear(self):
        """
        Reset this metric's calculated value
        """
        super(MSE, self).clear()
        self.rec_count = 0.0
