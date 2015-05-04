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
from neon.metrics.misclass import MisclassSum, MisclassRate, MisclassPercentage


class TestMisclass(object):

    def test_misclass_sum_add_binary(self):
        mcs = MisclassSum()
        assert mcs.rec_count == 0
        assert mcs.misclass_sum == 0
        refs = CPUTensor([[0, 1, 0, 0]])
        preds = CPUTensor([[1, 1, 0, 1]])
        mcs.add(refs, preds)
        assert mcs.rec_count == 4
        assert mcs.misclass_sum == 2

    def test_misclass_sum_mixed(self):
        mcs = MisclassSum()
        assert mcs.rec_count == 0
        assert mcs.misclass_sum == 0
        refs = CPUTensor([[3, 0, 1]])
        preds = CPUTensor([[0.00,    1,    0],
                           [0.09,  0.0, 0.55],
                           [0.01,    0, 0.75],
                           [0.90,    0, 0.34]])
        mcs.add(refs, preds)
        assert mcs.rec_count == 3
        assert mcs.misclass_sum == 1

    def test_misclass_sum_probs(self):
        mcs = MisclassSum()
        assert mcs.rec_count == 0
        assert mcs.misclass_sum == 0
        refs = CPUTensor([[0.03, 0.80, 0.81],
                          [0.20, 0.02, 0.15],
                          [0.31, 0.08, 0.01],
                          [0.46, 0.10, 0.03]])
        preds = CPUTensor([[0.00,    1,    0],
                           [0.09,  0.0, 0.55],
                           [0.01,    0, 0.75],
                           [0.90,    0, 0.34]])
        mcs.add(refs, preds)
        assert mcs.rec_count == 3
        assert mcs.misclass_sum == 1

    def test_misclass_sum_top3probs(self):
        mcs = MisclassSum(error_rank=3)
        assert mcs.rec_count == 0
        assert mcs.misclass_sum == 0
        refs = CPUTensor([[0.03, 0.80, 0.81],
                          [0.20, 0.02, 0.15],
                          [0.31, 0.08, 0.01],
                          [0.46, 0.10, 0.03]])
        preds = CPUTensor([[0.00,    1, 0.34],
                           [0.09,  0.0, 0.55],
                           [0.01,    0, 0.75],
                           [0.90,    0, 0.00]])
        mcs.add(refs, preds)
        assert mcs.rec_count == 3
        assert mcs.misclass_sum == 0

    def test_misclass_sum_report(self):
        mcs = MisclassSum()
        assert mcs.rec_count == 0
        assert mcs.misclass_sum == 0
        refs = CPUTensor([[0, 1, 0, 0]])
        preds = CPUTensor([[1, 1, 0, 1]])
        mcs.add(refs, preds)
        assert mcs.rec_count == 4
        assert mcs.misclass_sum == 2
        assert mcs.report() == 2

    def test_misclass_rate_report(self):
        mcr = MisclassRate()
        assert mcr.rec_count == 0
        assert mcr.misclass_sum == 0
        refs = CPUTensor([[0, 1, 0, 0]])
        preds = CPUTensor([[1, 1, 0, 1]])
        mcr.add(refs, preds)
        assert mcr.rec_count == 4
        assert mcr.misclass_sum == 2
        assert mcr.report() == 0.5

    def test_misclass_pct_report(self):
        mcp = MisclassPercentage()
        assert mcp.rec_count == 0
        assert mcp.misclass_sum == 0
        refs = CPUTensor([[0, 1, 0, 0]])
        preds = CPUTensor([[1, 1, 0, 1]])
        mcp.add(refs, preds)
        assert mcp.rec_count == 4
        assert mcp.misclass_sum == 2
        assert mcp.report() == 50.0

    def test_misclass_tied(self):
        mcs = MisclassSum()
        assert mcs.rec_count == 0
        assert mcs.misclass_sum == 0
        refs = CPUTensor([[0]])
        preds = CPUTensor([[0.5],
                           [0.5]])
        mcs.add(refs, preds)
        assert mcs.rec_count == 1
        assert mcs.misclass_sum == 0
        assert mcs.report() == 0
        mcs.clear()
        assert mcs.rec_count == 0
        assert mcs.misclass_sum == 0
        refs = CPUTensor([[1]])
        mcs.add(refs, preds)
        assert mcs.rec_count == 1
        assert mcs.misclass_sum == 1
        assert mcs.report() == 1
