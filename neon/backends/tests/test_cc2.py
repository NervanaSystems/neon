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

from neon.util.testing import assert_tensor_equal, assert_tensor_near_equal


@attr('cuda')
class TestGPU(object):

    def setup(self):
        from neon.backends.cc2 import GPU, GPUTensor
        # this code gets called prior to each test
        self.be = GPU(rng_seed=0)
        self.gpt = GPUTensor

    def test_empty_creation(self):
        tns = self.be.empty((4, 3))
        assert tns.shape == (4, 3)

    def test_array_creation(self):
        tns = self.be.array([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)
        assert_tensor_equal(tns, self.gpt([[1, 2], [3, 4]]))

    def test_zeros_creation(self):
        tns = self.be.zeros([3, 1])
        assert tns.shape == (3, 1)
        assert_tensor_equal(tns, self.gpt([[0], [0], [0]]))

    def test_ones_creation(self):
        tns = self.be.ones([1, 4])
        assert tns.shape == (1, 4)
        assert_tensor_equal(tns, self.gpt([[1, 1, 1, 1]]))

    def test_all_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[1, 1], [1, 1]]))

    def test_some_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.array([[0, 1], [0, 1]])
        out = self.be.empty([2, 2])
        self.be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[0, 1], [0, 1]]))

    def test_none_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.zeros([2, 2])
        out = self.be.empty([2, 2])
        self.be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[0, 0], [0, 0]]))

    def test_all_not_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.zeros([2, 2])
        out = self.be.empty([2, 2])
        self.be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[1, 1], [1, 1]]))

    def test_some_not_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.array([[0, 1], [0, 1]])
        out = self.be.empty([2, 2])
        self.be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[1, 0], [1, 0]]))

    def test_none_not_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[0, 0], [0, 0]]))

    def test_greater(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.greater(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[0, 0], [0, 1]]))

    def test_greater_equal(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.greater_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[0, 0], [1, 1]]))

    def test_less(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.less(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[1, 1], [0, 0]]))

    def test_less_equal(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.less_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, self.gpt([[1, 1], [1, 0]]))

    @nottest  # TODO: cudanet doesn't currently support noaxis argmin
    def test_argmin_noaxis(self):
        tsr = self.be.array([[-1, 0], [1, 92]])
        out = self.be.empty([1, 1])
        self.be.argmin(tsr, None, out)
        assert_tensor_equal(out, self.gpt([[0]]))

    def test_argmin_axis0(self):
        tsr = self.be.array([[-1, 0], [1, 92]])
        out = self.be.empty((1, 2))
        self.be.argmin(tsr, 0, out)
        assert_tensor_equal(out, self.gpt([[0, 0]]))

    def test_argmin_axis1(self):
        tsr = self.be.array([[-1, 10], [11, 9]])
        out = self.be.empty((2, 1))
        self.be.argmin(tsr, 1, out)
        assert_tensor_equal(out, self.gpt([[0], [1]]))

    @nottest  # TODO: cudanet doesn't currently support noaxis argmax
    def test_argmax_noaxis(self):
        tsr = self.be.array([[-1, 0], [1, 92]])
        out = self.be.empty([1, 1])
        self.be.argmax(tsr, None, out)
        assert_tensor_equal(out, self.gpt(3))

    def test_argmax_axis0(self):
        tsr = self.be.array([[-1, 0], [1, 92]])
        out = self.be.empty((1, 2))
        self.be.argmax(tsr, 0, out)
        assert_tensor_equal(out, self.gpt([[1, 1]]))

    def test_argmax_axis1(self):
        tsr = self.be.array([[-1, 10], [11, 9]])
        out = self.be.empty((2, 1))
        self.be.argmax(tsr, 1, out)
        assert_tensor_equal(out, self.gpt([[1], [0]]))

    def test_2norm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        rpow = 1. / 2
        # -> sum([[1, 0], [1, 9]], axis=0)**.5 -> sqrt([2, 9])
        out = self.be.empty((1, 2))
        assert_tensor_equal(self.be.norm(tsr, order=2, axis=0, out=out),
                            self.gpt([[2**rpow, 9**rpow]]))
        # -> sum([[1, 0], [1, 9]], axis=1)**.5 -> sqrt([1, 10])
        out = self.be.empty((2, 1))
        assert_tensor_equal(self.be.norm(tsr, order=2, axis=1, out=out),
                            self.gpt([1**rpow, 10**rpow]))

    def test_1norm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        # -> sum([[1, 0], [1, 3]], axis=0)**1 -> [2, 3]
        out = self.be.empty((1, 2))
        assert_tensor_equal(self.be.norm(tsr, order=1, axis=0, out=out),
                            self.gpt([[2, 3]]))
        # -> sum([[1, 0], [1, 3]], axis=1)**1 -> [1, 4]
        out = self.be.empty((2, 1))
        assert_tensor_equal(self.be.norm(tsr, order=1, axis=1, out=out),
                            self.gpt([1, 4]))

    def test_0norm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        # -> sum(tsr != 0, axis=0) -> [2, 1]
        out = self.be.empty((1, 2))
        assert_tensor_equal(self.be.norm(tsr, order=0, axis=0, out=out),
                            self.gpt([[2, 1]]))
        # -> sum(tsr != 0, axis=1) -> [1, 2]
        out = self.be.empty((2, 1))
        assert_tensor_equal(self.be.norm(tsr, order=0, axis=1, out=out),
                            self.gpt([1, 2]))

    def test_infnorm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        # -> max(abs(tsr), axis=0) -> [1, 3]
        assert_tensor_equal(self.be.norm(tsr, order=float('inf'), axis=0),
                            self.gpt([[1, 3]]))
        # -> max(abs(tsr), axis=1) -> [1, 3]
        assert_tensor_equal(self.be.norm(tsr, order=float('inf'), axis=1),
                            self.gpt([1, 3]))

    def test_neginfnorm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        # -> min(abs(tsr), axis=0) -> [1, 0]
        assert_tensor_equal(self.be.norm(tsr, order=float('-inf'), axis=0),
                            self.gpt([[1, 0]]))
        # -> min(abs(tsr), axis=1) -> [0, 1]
        assert_tensor_equal(self.be.norm(tsr, order=float('-inf'), axis=1),
                            self.gpt([0, 1]))

    def test_lrgnorm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        rpow = 1. / 5
        # -> sum([[1, 0], [1, 243]], axis=0)**rpow -> rpow([2, 243])
        out = self.be.empty((1, 2))
        assert_tensor_equal(self.be.norm(tsr, order=5, axis=0, out=out),
                            self.gpt([[2**rpow, 243**rpow]]))
        # -> sum([[1, 0], [1, 243]], axis=1)**rpow -> rpow([1, 244])
        # 244**.2 == ~3.002465 hence the near_equal test
        out = self.be.empty((2, 1))
        assert_tensor_near_equal(self.be.norm(tsr, order=5, axis=1, out=out),
                                 self.gpt([1**rpow, 244**rpow]), 1e-6)

    def test_negnorm(self):
        tsr = self.be.array([[-1, -2], [1, 3]])
        rpow = -1. / 3
        # -> sum([[1, .125], [1, .037037]], axis=0)**rpow -> rpow([2, .162037])
        out = self.be.empty((1, 2))
        assert_tensor_equal(self.be.norm(tsr, order=-3, axis=0, out=out),
                            self.gpt([[2**rpow, .162037037037**rpow]]))
        # -> sum([[1, .125], [1, .037037]], axis=1)**rpow ->
        # rpow([1.125, 1.037037])
        out = self.be.empty((2, 1))
        assert_tensor_near_equal(self.be.norm(tsr, order=-3, axis=1, out=out),
                                 self.gpt([1.125**rpow, 1.037037**rpow]),
                                 1e-6)
