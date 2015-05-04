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
Contains performance metrics related to ROC (receiver operating
characteristic) curves.
"""

import logging
import numpy

from neon.metrics.metric import Metric
from neon.util.param import opt_param
from neon.util.compat import range

logger = logging.getLogger(__name__)


class AUC(Metric):
    """
    Area under the ROC curve (for binary classification tasks).

    See Hand2001 http://link.springer.com/article/10.1023%2FA%3A1010920819831
    for implementation details

    Arguments:
        pos_label (int, optional): Which value represents the positive class.
                                   Defaults to 1.  For one-hot encoded data,
                                   this value specifies which row represents
                                   the positive class.
    """
    def __init__(self, **kwargs):
        super(AUC, self).__init__(**kwargs)
        opt_param(self, ['pos_label'], 1)

    def __str__(self):
        return self.__class__.__name__ + "_POS_" + str(self.pos_label)

    def add(self, reference, outputs):
        """
        Add the the expected reference and predicted outputs passed to the set
        of values used to calculate this metric.

        Arguments:
            reference (neon.backend.Tensor): Ground truth, expected outcomes.
                                             If each outcome is a vector, we
                                             expect it to be a (one-hot)
                                             column vector, with each case in
                                             a separate column.
            outputs (neon.backend.Tensor): Predicted outputs.  Must have the
                                           same dimensions as reference.  If
                                           each case is a vector, we assume it
                                           represents probabiity distributions
                                           over each class.
        """
        ismixed = ((reference.shape[0] == 1) and (outputs.shape[0] > 1) and
                   (reference.shape[1] == outputs.shape[1]))
        if (reference.shape != outputs.shape) and (not ismixed):
            raise ValueError("reference dimensions: %s, incompatible with "
                             "outputs dimensions: %s" %
                             (str(reference.shape), str(outputs.shape)))
        old_num_recs = len(self.probs)
        if len(outputs.shape) > 1 and outputs.shape[0] > 1:
            # vector of outputs per case
            self.probs.extend(outputs.asnumpyarray()[self.pos_label, :])
            if ismixed:
                self.labels.extend(reference.asnumpyarray().
                                   ravel().astype(int))
            else:
                self.labels.extend(reference.asnumpyarray().argmax(axis=0))
        else:
            logger.error("Should you really compute an AUC with hard model "
                         "predictions?")
            self.probs.extend(outputs.asnumpyarray().ravel())
            if ismixed:
                ref = reference.asnumpyarray().ravel().astype(int)
                reference = numpy.eye(outputs.shape[0], dtype=int)[ref].T
                self.labels.extend(reference.ravel())
            else:
                self.labels.extend(reference.asnumpyarray().ravel())
        num_new_pos = self.labels[old_num_recs:].count(self.pos_label)
        self.num_pos += num_new_pos
        self.num_neg += len(self.probs) - old_num_recs - num_new_pos
        self.stale_auc = True

    def report(self):
        """
        Report the area under the curve

        Returns:
            float of AUC value

        Raises:
            ValueError: if no records have been added yet.
        """
        if self.num_pos == self.num_neg == 0:
            raise ValueError("No records to compute AUC from")
        if self.stale_auc:
            ranks = self.get_ranks(self.probs)
            sum_pos_ranks = sum([ranks[i] for i in range(len(ranks)) if
                                 self.labels[i] == self.pos_label])
            self.auc = (sum_pos_ranks - self.num_pos * (self.num_pos + 1) /
                        2.0) / (self.num_pos * self.num_neg + 0.0)
            self.stale_auc = False
        return self.auc

    def clear(self):
        """
        Reset this metric's calculated value(s)
        """
        self.num_pos = 0
        self.num_neg = 0
        self.auc = float("nan")
        self.stale_auc = True
        self.probs = []
        self.labels = []

    def get_ranks(self, values):
        """
        Computes the rank of the list of values passed from lowest to highest.
        Note that ties are given equal ranking value (the average of their
        positions)

        Arguments:
            values (list): The list of numeric values to be ranked.

        Returns:
            list: Same length as values with the positional rank of each
                  original value (1-based).
        """
        num_vals = len(values)
        srt_vals = sorted(zip(values, list(range(num_vals))))
        ranks = [0 for i in values]
        val = srt_vals[0][0]
        high_rank = 0
        for i in range(num_vals):
            if val != srt_vals[i][0]:
                val = srt_vals[i][0]
                for j in range(high_rank, i):
                    ranks[srt_vals[j][1]] = float(high_rank + i + 1) / 2.0
                high_rank = i
            if i == (num_vals - 1):
                for j in range(high_rank, i + 1):
                    ranks[srt_vals[j][1]] = float(high_rank + i + 2) / 2.0
        return ranks
