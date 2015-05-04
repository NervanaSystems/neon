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
Contains various loss related metrics ex. log loss
"""

import numpy

from neon.metrics.metric import Metric
from neon.util.param import opt_param


class LogLossSum(Metric):
    """
    Logistic loss (aka cross-entropy loss) for a multi-class classification
    task.  Defined to be the negative log of the likelihood summed across all
    data points received.

    Arguments:
        eps (float, optional): Amount to clip values by to prevent potential
                               numeric difficulties (taking log of 0).

    See Also:
        LogLossMean

    References:
        Bishop2006 (p. 209)
    """
    def __init__(self, **kwargs):
        super(LogLossSum, self).__init__(**kwargs)
        opt_param(self, ['eps'], 1e-15)

    def add(self, reference, outputs):
        """
        Add the the expected reference and predicted outputs passed to the set
        of values used to calculate this metric.

        Arguments:
            reference (neon.backend.Tensor): Ground truth, expected outcomes.
                                             If each outcome is a vector, we
                                             expect it to be a column vector,
                                             with each case in a separate
                                             (one-hot encoded) column.
            outputs (neon.backend.Tensor): Predicted outputs.  Must have the
                                           same dimensions as reference.  To
                                           prevent numeric difficulties, output
                                           probabilities will be scaled to lie
                                           within [self.eps, 1 - self.eps]
        """
        ismixed = ((reference.shape[0] == 1) and (outputs.shape[0] > 1) and
                   (reference.shape[1] == outputs.shape[1]))
        if (reference.shape != outputs.shape) and (not ismixed):
            raise ValueError("reference dimensions: %s, incompatible with "
                             "outputs dimensions: %s" %
                             (str(reference.shape), str(outputs.shape)))
        # clip and normalize predictions
        preds = outputs.asnumpyarray().clip(self.eps, (1.0 - self.eps))
        preds = numpy.log(preds / preds.sum(axis=0))
        if ismixed:
            ref = reference.asnumpyarray().ravel().astype(int)
            reference = numpy.eye(outputs.shape[0], dtype=int)[ref].T
            self.logloss += (reference * preds).sum()
        else:
            self.logloss += (reference.asnumpyarray() * preds).sum()

    def report(self):
        """
        Report the log loss value

        Returns:
            float: log loss value
        """
        return - self.logloss

    def clear(self):
        """
        Reset this metric's calculated value
        """
        self.logloss = 0.0


class LogLossMean(LogLossSum):
    """
    Logistic loss (aka cross-entropy loss) for a multi-class classification
    task.  Defined to be the negative log of the likelihood averaged across all
    data points received.

    Arguments:
        eps (float, optional): Amount to clip values by to prevent potential
                               numeric difficulties (taking log of 0).

    See Also:
        LogLossSum
    """

    def add(self, reference, outputs):
        """
        Add the the expected reference and predicted outputs passed to the set
        of values used to calculate this metric.

        Arguments:
            reference (neon.backend.Tensor): Ground truth, expected outcomes.
                                             If each outcome is a vector, we
                                             expect it to be a column vector,
                                             with each case in a separate
                                             (one-hot encoded) column.
            outputs (neon.backend.Tensor): Predicted outputs.  Must have the
                                           same dimensions as reference.  To
                                           prevent numeric difficulties, output
                                           probabilities will be scaled to lie
                                           within [self.eps, 1 - self.eps]
        """
        super(LogLossMean, self).add(reference, outputs)
        self.rec_count += reference.shape[-1]

    def report(self):
        """
        Report the mean log loss value

        Returns:
            float: log loss mean value
        """
        return super(LogLossMean, self).report() / self.rec_count

    def clear(self):
        """
        Reset this metric's calculated value
        """
        super(LogLossMean, self).clear()
        self.rec_count = 0.0
