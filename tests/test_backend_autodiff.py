# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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

from builtins import zip
import itertools
import numpy as np

from neon import NervanaObject
from neon.backends.autodiff import Autodiff
from utils import call_func, gen_backend_tensors, tensors_allclose
import pytest


def get_audiff_gradient(f, be, tensors):
    """
    get autodiff gradient w.r.t the tensors
    """
    op_tree = f(be, *tensors)
    ad = Autodiff(op_tree, be)
    return ad


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


class Funcs(object):

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

    @staticmethod
    def func_transpose(be, x0, x1, x2, x3, x4):
        f1 = ((x0.T.T.T + x1).T + (x2 - x3.T.T + x4).T).T
        f2 = (x0 + x0.T - f1.T.T - x1.T).T.T.T - x4
        return f1 + f2


def pytest_generate_tests(metafunc):
    # number of test to repeat
    test_indices = list(range(1))

    # test params
    test_funcs = [
        Funcs.func_basic_ops,
        Funcs.func_real,
        Funcs.func_dot,
        Funcs.func_dot_reduction_mix,
        Funcs.func_scalar_broadcast,
        Funcs.func_transpose
    ]
    test_tensor_flags = ['pos_rand', 'neg_rand', 'rand']
    test_tensor_dims = [(2, 2)]

    # generate params for testing
    if 'custom_args' in metafunc.fixturenames:
        fargs = itertools.product(test_indices,
                                  test_funcs,
                                  test_tensor_flags,
                                  test_tensor_dims)
        # parameterize test call
        metafunc.parametrize("custom_args", fargs)


def test_gradients_mkl(backend_tests_mkl, custom_args):
    test_idx, f, flag, dim = custom_args

    # backend_tests fixture will parameterize over cpu and mkl
    # backends as well as float16 and float32
    # pull the be and dtype from the actions of the fixture
    be = NervanaObject.be
    dtype = be.default_dtype

    # tensors
    tensors = gen_backend_tensors([np, be], [dim] * 5, [flag] * 5, dtype=dtype)

    # compare function value and gradient
    numpy_func_val = call_func(f, np, tensors[0])
    backend_func_val = call_func(f, be, tensors[1])
    numerical_gradient = get_numerical_gradient(f, tensors[0])
    ad = get_audiff_gradient(f, be, tensors[1])
    autodiff_gradient = ad.get_grad_asnumpyarray(tensors[1])

    # TODO: stricter test to fix numerical issues
    assert tensors_allclose(numpy_func_val, backend_func_val, rtol=1e-2, atol=1e-2)
    assert tensors_allclose(numerical_gradient, autodiff_gradient, rtol=1e-02, atol=1e-3)

    # cleanup diff tree
    ad.cleanup()
    dtype = None
    be = None


@pytest.mark.hasgpu
def test_gradients(backend_tests, custom_args):
    test_idx, f, flag, dim = custom_args

    # backend_tests fixture will parameterize over cpu, gpu, and mkl
    # backends as well as float16 and float32
    # pull the be and dtype from the actions of the fixture
    be = NervanaObject.be
    dtype = be.default_dtype

    # tensors
    tensors = gen_backend_tensors([np, be], [dim] * 5, [flag] * 5, dtype=dtype)

    # compare function value and gradient
    numpy_func_val = call_func(f, np, tensors[0])
    backend_func_val = call_func(f, be, tensors[1])
    numerical_gradient = get_numerical_gradient(f, tensors[0])
    ad = get_audiff_gradient(f, be, tensors[1])
    autodiff_gradient = ad.get_grad_asnumpyarray(tensors[1])

    # TODO: stricter test to fix numerical issues
    assert tensors_allclose(numpy_func_val, backend_func_val, rtol=1e-2, atol=1e-2)
    assert tensors_allclose(numerical_gradient, autodiff_gradient, rtol=1e-02, atol=1e-3)

    # cleanup diff tree
    ad.cleanup()
    dtype = None
    be = None
