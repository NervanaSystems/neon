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
from neon.backends.nervanacpu import NervanaCPU
from neon.backends.nervanagpu import NervanaGPU
from neon.backends.tests.utils import assert_tensors_allclose


def test_copy_transpose(shape_dtype_inp):
    """
    Parameterized test case, uses pytest_generate_test to enumerate dim_dtype_inp
    tuples that drive the test.
    """

    shape, dtype, (name, inp_gen) = shape_dtype_inp
    # import pdb; pdb.set_trace()
    ng = NervanaGPU(default_dtype=dtype)
    nc = NervanaCPU(default_dtype=dtype)
    np_inp = inp_gen(shape).astype(dtype)
    ndims = len(shape)

    axes = [None] + list(itt.permutations(range(ndims), ndims))
    axes.remove(tuple(range(ndims)))
    for be, ax in itt.product([ng, nc], axes):
        if ax is not None and ax[-1] == ndims - 1:
            continue
        be_inp = be.array(np_inp, dtype=dtype)
        np_trans = np.transpose(np_inp, axes=ax)
        be_trans = be.zeros(np_trans.shape)
        be.copy_transpose(be_inp, be_trans, axes=ax)
        assert_tensors_allclose(np_trans, be_trans)
    del(ng)
    del(nc)


def pytest_generate_tests(metafunc):
    """
    Build a list of test arguments for test_copy_transpose.

    Run a full but slow set if --all is specified as a py.test arg, or just
    run sanity tests otherwise.
    """
    # sanity tests
    shapes = [(64, 56, 28, 16),
              (32, 20, 1),
              (16, 4)]

    dtypes = [np.float32, np.float16]
    inputs = [
        ("normal dist", lambda shape: np.random.uniform(-1.0, 1.0, shape)),
    ]

    if 'shape_dtype_inp' in metafunc.fixturenames:
        fargs = itt.product(shapes, dtypes, inputs)
        metafunc.parametrize("shape_dtype_inp", fargs)
