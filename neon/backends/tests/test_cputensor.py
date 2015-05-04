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

import numpy as np

from neon.backends.cpu import CPUTensor
from neon.util.testing import assert_tensor_equal


class TestCPUTensor(object):

    def __init__(self):
        # this code gets called prior to each test
        pass

    def test_empty_creation(self):
        tns = CPUTensor([])
        expected_shape = (0, )
        while len(expected_shape) < tns._min_dims:
            expected_shape += (1, )
        assert tns.shape == expected_shape

    def test_1d_creation(self):
        tns = CPUTensor([1, 2, 3, 4])
        expected_shape = (4, )
        while len(expected_shape) < tns._min_dims:
            expected_shape += (1, )
        assert tns.shape == expected_shape

    def test_2d_creation(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)

    def test_2d_ndarray_creation(self):
        tns = CPUTensor(np.array([[1.5, 2.5], [3.3, 9.2],
                                  [0.111111, 5]]))
        assert tns.shape == (3, 2)

    def test_higher_dim_creation(self):
        shapes = ((1, 1, 1), (1, 2, 3, 4), (1, 2, 3, 4, 5, 6, 7))
        for shape in shapes:
            tns = CPUTensor(np.empty(shape))
            assert tns.shape == shape

    def test_str(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        assert str(tns) == "[[ 1.  2.]\n [ 3.  4.]]"

    def test_scalar_slicing(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        res = tns[1, 0]
        expected_shape = (1, )
        while len(expected_shape) < res._min_dims:
            expected_shape += (1, )
        assert res.shape == expected_shape
        assert_tensor_equal(res, CPUTensor(3))

    def test_range_slicing(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        res = tns[0:2, 0]
        expected_shape = (2, )
        while len(expected_shape) < res._min_dims:
            expected_shape += (1, )
        assert res.shape == expected_shape
        assert_tensor_equal(res, CPUTensor([1, 3]))

    def test_scalar_slice_assignment(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        tns[1, 0] = 9
        assert_tensor_equal(tns, CPUTensor([[1, 2], [9, 4]]))

    def test_asnumpyarray(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        res = tns.asnumpyarray()
        assert isinstance(res, np.ndarray)
        assert_tensor_equal(res, np.array([[1, 2], [3, 4]]))

    def test_transpose(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        res = tns.transpose()
        assert_tensor_equal(res, CPUTensor([[1, 3], [2, 4]]))

    def test_fill(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        tns.fill(-9.5)
        assert_tensor_equal(tns, CPUTensor([[-9.5, -9.5], [-9.5, -9.5]]))
