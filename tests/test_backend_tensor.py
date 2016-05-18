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

import numpy as np
import itertools as itt
import pytest

from utils import tensors_allclose


def init_helper(lib, inA, inB, dtype):
    A = lib.array(inA, dtype=dtype)
    B = lib.array(inB, dtype=dtype)
    C = lib.empty(inB.shape, dtype=dtype)
    return A, B, C


def math_helper(lib, op, inA, inB, dtype):
    A, B, C = init_helper(lib, inA, inB, dtype)

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


def compare_helper(op, inA, inB, ng, nc, dtype):
    numpy_result = math_helper(np, op, inA, inB, dtype=np.float32).astype(dtype)

    nervanaGPU_result = math_helper(ng, op, inA, inB, dtype=dtype).get()
    np.allclose(numpy_result, nervanaGPU_result, rtol=0, atol=1e-5)

    nervanaCPU_result = math_helper(nc, op, inA, inB, dtype=dtype).get()
    np.allclose(numpy_result, nervanaCPU_result, rtol=0, atol=1e-5)


def rand_unif(dtype, dims):
    if np.dtype(dtype).kind == 'f':
        return np.random.uniform(-1, 1, dims).astype(dtype)
    else:
        iinfo = np.iinfo(dtype)
        return np.around(np.random.uniform(iinfo.min, iinfo.max, dims)).clip(iinfo.min, iinfo.max)


def pytest_generate_tests(metafunc):
    """
    Build a list of test arguments.

    """
    dims = [(64, 327),
            (64, 1),
            (1, 1023),
            (4, 3),
            ]

    if 'fargs_tests' in metafunc.fixturenames:
        fargs = itt.product(dims)
        metafunc.parametrize("fargs_tests", fargs)


@pytest.mark.hasgpu
def test_math(fargs_tests, backend_pair_dtype):
    dims = fargs_tests[0]
    ng, nc = backend_pair_dtype
    dtype = ng.default_dtype

    randA = rand_unif(dtype, dims)
    randB = rand_unif(dtype, dims)

    compare_helper('+', randA, randB, ng, nc, dtype)
    compare_helper('-', randA, randB, ng, nc, dtype)
    compare_helper('*', randA, randB, ng, nc, dtype)
    compare_helper('>', randA, randB, ng, nc, dtype)
    compare_helper('>=', randA, randB, ng, nc, dtype)
    compare_helper('<', randA, randB, ng, nc, dtype)
    compare_helper('<=', randA, randB, ng, nc, dtype)


@pytest.mark.hasgpu
def test_slicing(fargs_tests, backend_pair_dtype):
    dims = fargs_tests[0]

    gpu, cpu = backend_pair_dtype
    dtype = gpu.default_dtype

    array_np = np.random.uniform(-1, 1, dims).astype(dtype)
    array_ng = gpu.array(array_np, dtype=dtype)
    array_nc = cpu.array(array_np, dtype=dtype)

    assert tensors_allclose(array_ng[0], array_nc[0], rtol=0, atol=1e-3)
    assert tensors_allclose(array_ng[-1], array_nc[-1], rtol=0, atol=1e-3)
    assert tensors_allclose(array_ng[0, :], array_nc[0, :], rtol=0, atol=1e-3)
    assert tensors_allclose(array_ng[0:], array_nc[0:], rtol=0, atol=1e-3)
    assert tensors_allclose(array_ng[:-1], array_nc[:-1], rtol=0, atol=1e-3)
    assert tensors_allclose(array_ng[:, 0], array_nc[:, 0], rtol=0, atol=1e-3)
    assert tensors_allclose(array_ng[:, 0:1], array_nc[:, 0:1], rtol=0, atol=1e-3)
    assert tensors_allclose(array_ng[-1, 0:], array_nc[-1:, 0:], rtol=0, atol=1e-3)

    array_ng[0] = 0
    array_nc[0] = 0

    assert tensors_allclose(array_ng, array_nc, rtol=0, atol=1e-3)
