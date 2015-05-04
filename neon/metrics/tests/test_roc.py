#!/usr/bin/env python
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

from neon.backends.cpu import CPUTensor
from neon.metrics.roc import AUC


class TestROC(object):

    def test_auc_add_binary(self):
        auc = AUC()
        assert auc.num_pos == 0
        assert auc.num_neg == 0
        refs = CPUTensor([[0, 1, 0, 0]])
        preds = CPUTensor([[1, 1, 0, 1]])
        auc.add(refs, preds)
        assert auc.num_pos == 1
        assert auc.num_neg == 3
        assert len(auc.probs) == 4
        assert len(auc.labels) == 4

    def test_auc_add_mixed(self):
        auc = AUC()
        assert auc.num_pos == 0
        assert auc.num_neg == 0
        refs = CPUTensor([[0, 1, 0]])
        preds = CPUTensor([[0.00,    1,    0],
                           [0.09,  0.0, 0.75],
                           [0.01,    0, 0.15],
                           [0.90,    0, 0.10]])
        auc.add(refs, preds)
        assert auc.num_pos == 1
        assert auc.num_neg == 2
        assert len(auc.probs) == 3
        assert len(auc.labels) == 3

    def test_auc_add_probs(self):
        auc = AUC()
        assert auc.num_pos == 0
        assert auc.num_neg == 0
        refs = CPUTensor([[0.03, 0.80, 0.01],
                          [0.20, 0.02, 0.80],
                          [0.31, 0.08, 0.01],
                          [0.46, 0.10, 0.03]])
        preds = CPUTensor([[0.00,    1,    0],
                           [0.09,  0.0, 0.75],
                           [0.01,    0, 0.15],
                           [0.90,    0, 0.10]])
        auc.add(refs, preds)
        assert auc.num_pos == 1
        assert auc.num_neg == 2
        assert len(auc.probs) == 3
        assert len(auc.labels) == 3

    def test_auc_unique_ranks(self):
        auc = AUC()
        assert auc.get_ranks([0.1, 0.8, 0.4, 0.5]) == [1.0, 4.0, 2.0, 3.0]

    def test_auc_tied_ranks(self):
        auc = AUC()
        assert auc.get_ranks([0.1, 0.8, 0.4, 0.5, 0.4]) == [1.0, 5.0, 2.5, 4.0,
                                                            2.5]

    def test_auc_report_binary(self):
        auc = AUC()
        refs = CPUTensor([[0, 1, 0, 0]])
        preds = CPUTensor([[1, 1, 0, 1]])
        auc.add(refs, preds)
        assert auc.report() == (2.0 / 3.0)

    def test_auc_report_probs(self):
        auc = AUC()
        refs = CPUTensor([[0, 0, 1, 1]])
        preds = CPUTensor([[0.1, 0.4, 0.35, 0.8]])
        auc.add(refs, preds)
        assert auc.report() == .75
