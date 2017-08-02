# Copyright 2016 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import itertools as itt
import pytest
from utils import tensors_allclose


def pytest_generate_tests(metafunc):
    """
    Build a list of test arguments for test_copy_transpose.

    Run a full but slow set if --all is specified as a py.test arg, or just
    run sanity tests otherwise.
    """
    # sanity tests
    shapes = [(32, 24, 28, 16),
              (32, 20, 1),
              (16, 4),
              (30, 217, 32),
              ]

    inputs = [
        ("normal dist", lambda shape: np.random.uniform(-1.0, 1.0, shape)),
    ]

    if 'shape_inp' in metafunc.fixturenames:
        fargs = itt.product(shapes, inputs)
        metafunc.parametrize("shape_inp", fargs)


def test_copy_transpose_mkl_32(shape_inp, backend_pair_dtype_mkl_32):
    """
    Parameterized test case, uses pytest_generate_test to enumerate dim_inp
    tuples that drive the test.
    """

    shape, (name, inp_gen) = shape_inp
    nm, nc = backend_pair_dtype_mkl_32
    np_inp = inp_gen(shape).astype(nc.default_dtype)
    ndims = len(shape)

    axes = [None] + list(itt.permutations(range(ndims), ndims))
    axes.remove(tuple(range(ndims)))
    for be, ax in itt.product([nm, nc], axes):
        be_inp = be.array(np_inp)
        np_trans = np.transpose(np_inp, axes=ax)
        be_trans = be.zeros(np_trans.shape)
        be.copy_transpose(be_inp, be_trans, axes=ax)
        assert tensors_allclose(np_trans, be_trans)


@pytest.mark.skip(reason="mkl backend does not support float16")
def test_copy_transpose_mkl_16(shape_inp, backend_pair_dtype_mkl_16):
    """
    Parameterized test case, uses pytest_generate_test to enumerate dim_inp
    tuples that drive the test.
    """

    shape, (name, inp_gen) = shape_inp
    nm, nc = backend_pair_dtype_mkl_16
    np_inp = inp_gen(shape).astype(nc.default_dtype)
    ndims = len(shape)

    axes = [None] + list(itt.permutations(range(ndims), ndims))
    axes.remove(tuple(range(ndims)))
    for be, ax in itt.product([nm, nc], axes):
        be_inp = be.array(np_inp)
        np_trans = np.transpose(np_inp, axes=ax)
        be_trans = be.zeros(np_trans.shape)
        be.copy_transpose(be_inp, be_trans, axes=ax)
        assert tensors_allclose(np_trans, be_trans)


@pytest.mark.hasgpu
def test_copy_transpose(shape_inp, backend_pair_dtype):
    """
    Parameterized test case, uses pytest_generate_test to enumerate dim_inp
    tuples that drive the test.
    """

    shape, (name, inp_gen) = shape_inp
    ng, nc = backend_pair_dtype
    np_inp = inp_gen(shape).astype(nc.default_dtype)
    ndims = len(shape)

    axes = [None] + list(itt.permutations(range(ndims), ndims))
    axes.remove(tuple(range(ndims)))
    for be, ax in itt.product([ng, nc], axes):
        be_inp = be.array(np_inp)
        np_trans = np.transpose(np_inp, axes=ax)
        be_trans = be.zeros(np_trans.shape)
        be.copy_transpose(be_inp, be_trans, axes=ax)
        assert tensors_allclose(np_trans, be_trans)
