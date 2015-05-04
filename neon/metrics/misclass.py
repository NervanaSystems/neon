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
Misclassification related metrics.
"""

import logging

from neon.metrics.metric import Metric
from neon.util.param import opt_param

logger = logging.getLogger(__name__)


class MisclassSum(Metric):
    """
    Metric that counts the number of misclassifications made (prediction does
    not match the reference target exactly).

    Arguments:
        error_rank (int, optional): Prediction is only considered misclassified
                                    if not within the first error_rank guesses.
                                    Defaults to 1.

    See Also:
        MisclassRate, MisclassPercentage
    """
    def __init__(self, **kwargs):
        super(MisclassSum, self).__init__(**kwargs)
        opt_param(self, ['error_rank'], 1)

    def __str__(self):
        return self.__class__.__name__ + "_TOP_" + str(self.error_rank)

    def add(self, reference, outputs):
        """
        Add the expected reference and predicted outputs passed to the set
        of values used to calculate this metric.

        Arguments:
            reference (neon.backend.Tensor): Ground truth, expected outcomes.
                                             If each outcome is a vector, we
                                             expect it to be a column vector,
                                             with each case in a separate
                                             column.
            outputs (neon.backend.Tensor): Predicted outputs.  Must have the
                                           same dimensions as reference.  If
                                           each prediction is a vector, we
                                           treat the inidividual values as
                                           probabilities for that class.
        """
        ismixed = ((reference.shape[0] == 1) and (outputs.shape[0] > 1) and
                   (reference.shape[1] == outputs.shape[1]))
        if (reference.shape != outputs.shape) and (not ismixed):
            raise ValueError("reference dimensions: %s, incompatible with "
                             "outputs dimensions: %s" %
                             (str(reference.shape), str(outputs.shape)))
        self.rec_count += reference.shape[-1]
        if len(outputs.shape) > 1 and outputs.shape[0] > 1:
            # vector of outputs per case.  Check if ground truth index in top-k
            # predictions (ordered by decreasing probability)
            if ismixed:
                true_idcs = reference.asnumpyarray()
            else:
                true_idcs = reference.asnumpyarray().argmax(axis=0)
            top_pred_idcs = - outputs.asnumpyarray()
            top_pred_idcs = top_pred_idcs.argpartition(self.error_rank,
                                                       axis=0)
            top_pred_idcs = top_pred_idcs[:self.error_rank, :]
            self.misclass_sum += (true_idcs != top_pred_idcs).all(axis=0).sum()
        else:
            if self.error_rank > 1:
                raise ValueError("can't compute top-k error on indices")
            self.misclass_sum += (reference.asnumpyarray().ravel() !=
                                  outputs.asnumpyarray().ravel()).sum()

    def report(self):
        """
        Report the misclassification count.

        Returns:
            int: Misclassification count

        """
        if self.rec_count == 0:
            raise ValueError("No records to count misclassifications on")
        return self.misclass_sum

    def clear(self):
        """
        Reset this metric's calculated value(s)
        """
        self.misclass_sum = 0
        self.rec_count = 0


class MisclassRate(MisclassSum):
    """
    Metric that reports the fraction of misclassifications made (prediction
    does not match the reference target exactly) relative to the total numbe
    of predictions.

    See Also: MisclassSum, MisclassPercentage
    """
    def report(self):
        """
        Report the misclassification rate.

        Returns:
            float: The misclassification rate (will lie between 0.0 and 1.0)
        """
        if self.rec_count == 0:
            raise ValueError("No records to report misclassifications on.")
        else:
            return (self.misclass_sum + 0.0) / self.rec_count


class MisclassPercentage(MisclassRate):

    def report(self):
        """
        Report the misclassification percentage (0-100).

        Returns:
            float: The misclassification percentage (will lie between 0.0 and
                   100.0)
        """
        return super(MisclassPercentage, self).report() * 100.0
