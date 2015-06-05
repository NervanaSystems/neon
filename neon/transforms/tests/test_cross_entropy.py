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
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.cross_entropy import (cross_entropy,
                                           cross_entropy_derivative)
from neon.util.testing import assert_tensor_near_equal


def test_cross_entropy_cputensor():
    be = CPU(rng_seed=0)
    outputs = CPUTensor([0.5, 0.9, 0.1, 0.0001])
    targets = CPUTensor([0.5, 0.99, 0.01, 0.2])
    eps = 2**-23
    temp = [be.zeros(outputs.shape), be.zeros(outputs.shape)]
    expected_result = np.sum((- targets.asnumpyarray()) *
                             np.log(outputs.asnumpyarray() + eps) -
                             (1 - targets.asnumpyarray()) *
                             np.log(1 - outputs.asnumpyarray() + eps),
                             keepdims=True)
    assert_tensor_near_equal(expected_result, cross_entropy(be, outputs,
                                                            targets, temp,
                                                            eps))


@attr('cuda')
def test_cross_entropy_cc2tensor():
    from neon.backends.cc2 import GPU, GPUTensor
    be = GPU(rng_seed=0)  # to ensure cublas_init() is called.
    outputs = GPUTensor([0.5, 0.9, 0.1, 0.0001])
    targets = GPUTensor([0.5, 0.99, 0.01, 0.2])
    eps = 2**-23
    temp = [be.zeros(outputs.shape), be.zeros(outputs.shape)]
    expected_result = np.sum((- targets.asnumpyarray()) *
                             np.log(outputs.asnumpyarray() + eps) -
                             (1 - targets.asnumpyarray()) *
                             np.log(1 - outputs.asnumpyarray() + eps),
                             keepdims=True)
    assert_tensor_near_equal(expected_result, cross_entropy(be, outputs,
                                                            targets, temp,
                                                            eps),
                             tolerance=1e-6)


def test_cross_entropy_limits():
    be = CPU(rng_seed=0)
    outputs = CPUTensor([0.5, 1.0, 0.0, 0.0001])
    targets = CPUTensor([0.5, 0.0, 1.0, 0.2])
    eps = 2**-23
    temp = [be.zeros(outputs.shape), be.zeros(outputs.shape)]
    expected_result = np.sum((- targets.asnumpyarray()) *
                             np.log(outputs.asnumpyarray() + eps) -
                             (1 - targets.asnumpyarray()) *
                             np.log(1 - outputs.asnumpyarray() + eps),
                             keepdims=True)
    assert_tensor_near_equal(expected_result, cross_entropy(be, outputs,
                                                            targets, temp,
                                                            eps))


def test_cross_entropy_derivative_cputensor():
    be = CPU(rng_seed=0)
    outputs = CPUTensor([0.5, 0.9, 0.1, 0.0001])
    targets = CPUTensor([0.5, 0.99, 0.01, 0.2])
    temp = [be.zeros(outputs.shape), be.zeros(outputs.shape)]
    expected_result = ((outputs.asnumpyarray() - targets.asnumpyarray()) /
                       (outputs.asnumpyarray() * (1 - outputs.asnumpyarray())))
    assert_tensor_near_equal(expected_result,
                             cross_entropy_derivative(be, outputs,
                                                      targets, temp))


@attr('cuda')
def test_cross_entropy_derivative_cc2tensor():
    from neon.backends.cc2 import GPU, GPUTensor
    be = GPU(rng_seed=0)
    outputs = GPUTensor([0.5, 0.9, 0.1, 0.0001])
    targets = GPUTensor([0.5, 0.99, 0.01, 0.2])
    temp = [be.zeros(outputs.shape), be.zeros(outputs.shape)]
    expected_result = ((outputs.asnumpyarray() - targets.asnumpyarray()) /
                       (outputs.asnumpyarray() * (1 - outputs.asnumpyarray())))
    assert_tensor_near_equal(expected_result,
                             cross_entropy_derivative(be, outputs,
                                                      targets, temp))
