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

"""
test batched_dot behaviors between NervanaCPU, NervanaMKL, and NervanaGPU backend
against numpy.
In NervanaGPU, it supports both N as inside dimension or as outer dimension.
In NervanaCPU, it only supports N as inside dimension, since this is what we use.
In NervanaMKL, it supports both N as inside dimension or as outer dimension.
"""
from __future__ import print_function
import numpy as np
import pytest

from utils import tensors_allclose

size = 32  # size input for GPU - 32, 64, 128, None=auto


def setup_test_data(X, N, C, K, dtype):
    dimW = (K, C)
    dimI = (X, C, N)
    dimO = (X, K, N)

    cpuI = np.random.uniform(-1.0, 1.0, dimI).astype(dtype)
    cpuE = np.random.uniform(-1.0, 1.0, dimO).astype(dtype)
    cpuW = np.random.uniform(-1.0, 1.0, dimW).astype(dtype)

    return cpuI, cpuE, cpuW


def run_batched_dot(lib, I, E, W, X, dtype):

    devI = lib.array(I, dtype=dtype)
    devE = lib.array(E, dtype=dtype)
    devW = lib.array(W, dtype=dtype)

    devO = lib.zeros(E.shape, dtype=dtype)
    devB = lib.zeros(I.shape, dtype=dtype)
    devU = lib.zeros(W.shape, dtype=dtype)

    if (lib.__class__.__name__.endswith("CPU") | lib.__class__.__name__.endswith("MKL")):
        lib.batched_dot(devW, devI, devO)  # fprop
        lib.batched_dot(devW.T, devE, devB)  # bprop
        lib.batched_dot(devE, devI.T, devU)  # update
    elif lib.__class__.__name__.endswith("GPU"):
        lib.batched_dot(devW, devI, devO, size=size)  # fprop
        lib.batched_dot(devW.T, devE, devB, size=size)  # bprop
        lib.batched_dot(devE, devI.T, devU, size=size)  # update
    else:
        for i in range(X):
            devO[i] = np.dot(W, I[i])  # fprop
            devB[i] = np.dot(W.T, E[i])  # bprop
            devU += np.dot(E[i], I[i].T)  # update

    return devO, devB, devU


def test_batched_dot_mkl(backend_pair_bench_mkl):
    np.set_printoptions(threshold=8192 * 4, linewidth=600,
                        formatter={'int': lambda x: "%2d" % x, 'float': lambda x: "%2.0f" % x})

    nm, nc = backend_pair_bench_mkl

    dtype = np.float32  # np.float16 or np.float32

    X = 100   # Batch Size
    N = 32   # Minibatch Size
    C = 1536  # Input  Features
    K = 768  # Output Features

    cpuI, cpuE, cpuW = setup_test_data(X, N, C, K, dtype)

    ncO, ncB, ncU = run_batched_dot(nc, cpuI, cpuE, cpuW, X, dtype)
    npO, npB, npU = run_batched_dot(np, cpuI, cpuE, cpuW, X, dtype)
    nmO, nmB, nmU = run_batched_dot(nm, cpuI, cpuE, cpuW, X, dtype)

    assert tensors_allclose(npO, nmO, rtol=0, atol=1e-3)
    assert tensors_allclose(npB, nmB, rtol=0, atol=1e-3)
    assert tensors_allclose(npU, nmU, rtol=0, atol=1e-3)

    assert tensors_allclose(npO, ncO, rtol=0, atol=1e-3)
    assert tensors_allclose(npB, ncB, rtol=0, atol=1e-3)
    assert tensors_allclose(npU, ncU, rtol=0, atol=1e-3)


@pytest.mark.hasgpu
def test_batched_dot(backend_pair_bench):
    np.set_printoptions(threshold=8192 * 4, linewidth=600,
                        formatter={'int': lambda x: "%2d" % x, 'float': lambda x: "%2.0f" % x})

    ng, nc = backend_pair_bench

    dtype = np.float32  # np.float16 or np.float32

    X = 100   # Batch Size
    N = 32   # Minibatch Size
    C = 1536  # Input  Features
    K = 768  # Output Features

    cpuI, cpuE, cpuW = setup_test_data(X, N, C, K, dtype)

    ncO, ncB, ncU = run_batched_dot(nc, cpuI, cpuE, cpuW, X, dtype)
    npO, npB, npU = run_batched_dot(np, cpuI, cpuE, cpuW, X, dtype)

    if ng.compute_capability > (5, 0):
        ngO, ngB, ngU = run_batched_dot(ng, cpuI, cpuE, cpuW, X, dtype)

        assert tensors_allclose(npO, ngO, rtol=0, atol=1e-3)
        assert tensors_allclose(npB, ngB, rtol=0, atol=1e-3)
        assert tensors_allclose(npU, ngU, rtol=0, atol=1e-3)

    assert tensors_allclose(npO, ncO, rtol=0, atol=1e-3)
    assert tensors_allclose(npB, ncB, rtol=0, atol=1e-3)
    assert tensors_allclose(npU, ncU, rtol=0, atol=1e-3)
