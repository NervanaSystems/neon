# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
# pylint: skip-file

"""
Test of basic math operations on the Tensors and compare with numpy results
The Tensor types includes GPU and CPU Tensors
"""

from neon.backends import gen_backend
from neon.backends.tests.utils import assert_tensors_allclose
import numpy as np


class TestTensor(object):

    def setup(self):

        self.gpu = gen_backend("gpu", stochastic_round=False)
        self.cpu = gen_backend("cpu")
        self.dims = (1024, 1024)

    def teardown(self):
        self.gpu.ctx.pop()
        del(self.gpu)

    def init_helper(self, lib, inA, inB, dtype):

        A = lib.array(inA, dtype=dtype)
        B = lib.array(inB, dtype=dtype)
        C = lib.empty(inB.shape, dtype=dtype)

        return A, B, C

    def math_helper(self, lib, op, inA, inB, dtype):

        A, B, C = self.init_helper(lib, inA, inB, dtype)

        if op == '+':
            C[:] = A + B
        elif op == '-':
            C[:] = A - B
        elif op == '*':
            C[:] = A * B
        elif op == '/':
            C[:] = A / B
        elif op == '>':
            C[:] = A > B
        elif op == '>=':
            C[:] = A >= B
        elif op == '<':
            C[:] = A < B
        elif op == '<=':
            C[:] = A <= B

        return C

    def compare_helper(self, op, inA, inB, dtype):

        numpy_result = self.math_helper(np, op, inA, inB, dtype=np.float32)

        if np.dtype(dtype).kind == 'i' or np.dtype(dtype).kind == 'u':
            numpy_result = np.around(numpy_result)
            numpy_result = numpy_result.clip(
                np.iinfo(dtype).min, np.iinfo(dtype).max)
        numpy_result = numpy_result.astype(dtype)

        nervanaGPU_result = self.math_helper(
            self.gpu, op, inA, inB, dtype=dtype)
        nervanaCPU_result = self.math_helper(
            self.cpu, op, inA, inB, dtype=dtype)

        assert_tensors_allclose(numpy_result, nervanaGPU_result, rtol=0, atol=1e-5)

        if dtype in (np.float64, np.float32, np.float16):
            assert_tensors_allclose(numpy_result, nervanaCPU_result, rtol=0, atol=1e-5)

    def rand_unif(self, dtype, dims):
        if np.dtype(dtype).kind == 'f':
            return np.random.uniform(-1, 1, dims).astype(dtype)
        else:
            iinfo = np.iinfo(dtype)
            return np.around(np.random.uniform(iinfo.min, iinfo.max, dims)) \
                .clip(iinfo.min, iinfo.max)

    def test_math(self):

        for dtype in (np.float32, np.float16, np.int8, np.uint8):
            randA = self.rand_unif(dtype, self.dims)
            randB = self.rand_unif(dtype, self.dims)

            self.compare_helper('+', randA, randB, dtype)
            self.compare_helper('-', randA, randB, dtype)
            self.compare_helper('*', randA, randB, dtype)
            # self.compare_helper('/', randA, randB, dtype)
            self.compare_helper('>', randA, randB, dtype)
            self.compare_helper('>=', randA, randB, dtype)
            self.compare_helper('<', randA, randB, dtype)
            self.compare_helper('<=', randA, randB, dtype)
