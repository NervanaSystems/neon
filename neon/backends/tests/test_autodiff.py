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

import itertools
import numpy as np
from neon.backends import gen_backend
from neon.backends.autodiff import Autodiff

from neon.backends.tests.utils import call_func, gen_backend_tensors
from neon.backends.tests.utils import assert_tensors_allclose


def get_audiff_gradient(f, be, tensors):
    """
    get autodiff gradient w.r.t the tensors
    """
    op_tree = f(be, *tensors)
    ad = Autodiff(op_tree, be)
    return ad.get_grad_asnumpyarray(tensors)


def get_numerical_gradient(f, tensors, delta=1e-5):
    """
    sum all of f's elements to make the last layer error as one
    """
    # buffer for gradients
    gradients = []
    for i in range(len(tensors)):
        tensors[i] = tensors[i].astype(np.float64)
        gradients.append(np.zeros(tensors[i].shape))

    # iterate throuth each tensor
    for tensor, gradient in zip(tensors, gradients):

        tensor_flat = tensor.reshape((-1, ))
        gradient_flat = gradient.reshape((-1, ))

        # iterate throuth each element
        for idx in range(len(tensor_flat)):
            # backup
            backup = tensor_flat[idx]
            # increment
            tensor_flat[idx] = tensor_flat[idx] + delta
            f_inc = np.sum(f(np, *tensors))
            # decrement
            tensor_flat[idx] = backup - delta
            f_dec = np.sum(f(np, *tensors))
            # recover
            tensor_flat[idx] = backup
            # gradient
            gradient_flat[idx] = (f_inc - f_dec) / (2.0 * delta)

    return gradients


class TestFuncs():

    """
    A collection of functions to be tested
    """
    @staticmethod
    def func_basic_ops(be, x0, x1, x2, x3, x4):
        return (x0 + x2) + x0 * x4 + x1 * x3

    @staticmethod
    def func_real(be, x0, x1, x2, x3, x4):
        return x1 + be.absolute(x2 + x3) + x4 - (x1 + be.square(x2 + x3) + x4)

    @staticmethod
    def func_dot(be, x0, x1, x2, x3, x4):
        return (x0 + x3) + be.dot(x1, x2) - (x1 - x2) - be.dot(x3, x4)

    @staticmethod
    def func_dot_reduction_mix(be, x0, x1, x2, x3, x4):
        f1 = be.max(x0, axis=1, keepdims=True)
        f2 = be.min(x1, axis=0, keepdims=True)
        f3 = be.dot(1. / x3, x2 + x4)
        f4 = be.min(x3, axis=0, keepdims=True)
        return f1 + f2 + f3 + f4

    @staticmethod
    def func_scalar_broadcast(be, x0, x1, x2, x3, x4):
        return (0.2 * x0 - x1 * x2 / 3 * 4 * x1 + x0 * x0 / x0 / x3 + x4)


def pytest_generate_tests(metafunc):
    # number of test to repeat
    test_indices = range(1)

    # test params
    test_funcs = [
        TestFuncs.func_basic_ops,
        TestFuncs.func_real,
        TestFuncs.func_dot,
        TestFuncs.func_dot_reduction_mix,
        TestFuncs.func_scalar_broadcast,
    ]
    test_tensor_flags = ['pos_rand', 'neg_rand', 'rand']
    test_tensor_dims = [(2, 2)]
    test_dtypes = [np.float16, np.float32]
    test_backends = ["gpu", "cpu"]

    # generate params for testing
    if 'custom_args' in metafunc.fixturenames:
        fargs = itertools.product(test_indices, test_funcs, test_tensor_flags,
                                  test_tensor_dims, test_dtypes, test_backends)
        # parameterize test call
        metafunc.parametrize("custom_args", fargs)


def test_gradients(custom_args):
    test_idx, f, flag, dim, dtype, backend_type = custom_args
    be = gen_backend(backend_type, default_dtype=dtype)

    # tensors
    tensors = gen_backend_tensors(
        [np, be], 5, [dim] * 5, [flag] * 5, dtype=dtype)

    # compare function value and gradient
    numpy_func_val = call_func(f, np, tensors[0])
    backend_func_val = call_func(f, be, tensors[1])
    numerical_gradient = get_numerical_gradient(f, tensors[0])
    autodiff_gradient = get_audiff_gradient(f, be, tensors[1])

    # TODO: stricter test to fix numerical issues
    assert_tensors_allclose(
        numpy_func_val, backend_func_val, rtol=1e-2, atol=1e-2)
    assert_tensors_allclose(
        numerical_gradient, autodiff_gradient, rtol=1e-02, atol=1e-3)
