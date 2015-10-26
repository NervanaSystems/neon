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
import numpy as np
import itertools as itt


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


def compare_helper(op, inA, inB, dtype):
    numpy_result = math_helper(np, op, inA, inB, dtype=np.float32)

    if np.dtype(dtype).kind == 'i' or np.dtype(dtype).kind == 'u':
        numpy_result = np.around(numpy_result)
        numpy_result = numpy_result.clip(
            np.iinfo(dtype).min, np.iinfo(dtype).max)
    numpy_result = numpy_result.astype(dtype)

    if dtype in (np.float32, np.float16):
        gpu = gen_backend(backend='gpu', datatype=dtype)
        nervanaGPU_result = math_helper(gpu, op, inA, inB, dtype=dtype)
        nervanaGPU_result = nervanaGPU_result.get()
        np.allclose(numpy_result, nervanaGPU_result, rtol=0, atol=1e-5)

    cpu = gen_backend(backend='cpu', datatype=dtype)
    nervanaCPU_result = math_helper(cpu, op, inA, inB, dtype=dtype)
    nervanaCPU_result = nervanaCPU_result.get()
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
    dtypes = [np.float32, np.float16]

    if 'fargs_tests' in metafunc.fixturenames:
        fargs = itt.product(dims, dtypes)
        metafunc.parametrize("fargs_tests", fargs)


def test_math(fargs_tests):

    dims, dtype = fargs_tests

    randA = rand_unif(dtype, dims)
    randB = rand_unif(dtype, dims)

    compare_helper('+', randA, randB, dtype)
    compare_helper('-', randA, randB, dtype)
    compare_helper('*', randA, randB, dtype)
    compare_helper('>', randA, randB, dtype)
    compare_helper('>=', randA, randB, dtype)
    compare_helper('<', randA, randB, dtype)
    compare_helper('<=', randA, randB, dtype)
