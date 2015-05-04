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

from nose.plugins.attrib import attr
from nose.tools import nottest
import numpy as np

from neon.util.testing import assert_tensor_equal


@attr('cuda')
class TestGPUTensor(object):

    def setup(self):
        from neon.backends.cc2 import GPUTensor
        self.gpt = GPUTensor

    def test_empty_creation(self):
        tns = self.gpt([])
        expected_shape = (0, )
        while len(expected_shape) < tns._min_dims:
            expected_shape += (1, )
        assert tns.shape == expected_shape

    def test_1d_creation(self):
        tns = self.gpt([1, 2, 3, 4])
        expected_shape = (4, )
        while len(expected_shape) < tns._min_dims:
            expected_shape += (1, )
        assert tns.shape == expected_shape

    def test_2d_creation(self):
        tns = self.gpt([[1, 2], [3, 4]])
        expected_shape = (2, 2)
        while len(expected_shape) < tns._min_dims:
            expected_shape += (1, )
        assert tns.shape == expected_shape

    def test_2d_ndarray_creation(self):
        tns = self.gpt(np.array([[1.5, 2.5], [3.3, 9.2],
                                 [0.111111, 5]]))
        assert tns.shape == (3, 2)

    @nottest  # TODO: add >2 dimension support to cudanet
    def test_higher_dim_creation(self):
        shapes = ((1, 1, 1), (1, 2, 3, 4), (1, 2, 3, 4, 5, 6, 7))
        for shape in shapes:
            tns = self.gpt(np.empty(shape))
            assert tns.shape == shape

    def test_str(self):
        tns = self.gpt([[1, 2], [3, 4]])
        assert str(tns) == "[[ 1.  2.]\n [ 3.  4.]]"

    def test_scalar_slicing(self):
        tns = self.gpt([[1, 2], [3, 4]])
        res = tns[1, 0]
        assert res.shape == (1, 1)
        assert_tensor_equal(res, self.gpt([[3]]))

    def test_range_slicing(self):
        tns = self.gpt([[1, 2], [3, 4]])
        res = tns[0:2, 0]
        assert res.shape == (2, 1)
        assert_tensor_equal(res, self.gpt([1, 3]))

    @nottest  # TODO: add scalar assignment to self.gpt class
    def test_scalar_slice_assignment(self):
        tns = self.gpt([[1, 2], [3, 4]])
        tns[1, 0] = 9
        assert_tensor_equal(tns, self.gpt([[1, 2], [9, 4]]))

    def test_asnumpyarray(self):
        tns = self.gpt([[1, 2], [3, 4]])
        res = tns.asnumpyarray()
        assert isinstance(res, np.ndarray)
        assert_tensor_equal(res, np.array([[1, 2], [3, 4]]))

    @nottest  # TODO: fix this for self.gpt
    def test_transpose(self):
        tns = self.gpt([[1, 2], [3, 4]])
        res = tns.transpose()
        assert_tensor_equal(res, self.gpt([[1, 3], [2, 4]]))

    def test_fill(self):
        tns = self.gpt([[1, 2], [3, 4]])
        tns.fill(-9.5)
        assert_tensor_equal(tns, self.gpt([[-9.5, -9.5], [-9.5, -9.5]]))
