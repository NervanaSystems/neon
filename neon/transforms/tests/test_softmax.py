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
from neon.transforms.softmax import Softmax
from neon.util.testing import assert_tensor_near_equal


def test_softmax_cputensor():
    sftmx = Softmax()
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    temp = be.zeros((3, 1))
    outputs = np.exp(inputs-1) / np.sum(np.exp(inputs-1))
    sftmx.apply_function(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_softmax_cc2tensor():
    sftmx = Softmax()
    from neon.backends.cc2 import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.exp(inputs) / np.sum(np.exp(inputs))
    be = GPU(rng_seed=0)
    temp = be.zeros((3, 1))
    sftmx.apply_function(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(GPUTensor(outputs), temp)


def test_softmax_derivative_cputensor():
    sftmx = Softmax()
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    outputs = np.exp(inputs) / np.sum(np.exp(inputs))
    errmat = np.ones(inputs.shape)
    a = np.einsum('ij,ji->i', errmat.T, outputs)
    outputs = outputs * (errmat - a[np.newaxis, :])
    temp = be.zeros(inputs.shape)
    sftmx.apply_derivative(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_softmax_derivative_cc2tensor():
    sftmx = Softmax()
    from neon.backends.cc2 import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.exp(inputs) / np.sum(np.exp(inputs))
    errmat = np.ones(inputs.shape)
    a = np.einsum('ij,ji->i', errmat.T, outputs)
    outputs = outputs * (errmat - a[np.newaxis, :])
    be = GPU(rng_seed=0)
    temp = be.zeros(inputs.shape)
    sftmx.apply_derivative(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(GPUTensor(outputs), temp)
