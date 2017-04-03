# ----------------------------------------------------------------------------
# Copyright 2015-2017 Nervana Systems Inc.
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

import numpy as np
import pytest

from neon.benchmark import Benchmark

print_stats = Benchmark.print_stats


class NotImplementedOutput(object):
    pass


class TestOutput(object):
    def __init(self):
        self.display_output = ''

    def display(self, data):
        self.display_output = data

test_stats = {'feature_name': [10, 10.5, 11, 11.6]}


def test_empty_functions():
    test_console = TestOutput()

    print_stats(test_stats, functions=[], output=test_console)

    assert all(s in test_console.display_output for s in
               ['Mean', 'Median', 'Amin', 'Amax', 'feature_name'])


def test_custom_functions():
    test_console = TestOutput()

    print_stats(test_stats, functions=[np.average], output=test_console)

    assert not any(s in test_console.display_output for s in ['Mean', 'Median', 'Amin', 'Amax'])

    assert 'Average' in test_console.display_output
    assert 'feature_name' in test_console.display_output


def test_output_not_implemented():
    test_console = NotImplementedOutput()
    with pytest.raises(TypeError):
        print_stats(test_stats, output=test_console)


def test_output_no_stats():
    with pytest.raises(ValueError):
        print_stats(None)


def test_output_empty_stats():
    with pytest.raises(ValueError):
        print_stats([])
