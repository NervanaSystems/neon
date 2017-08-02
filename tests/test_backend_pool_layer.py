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
"""
To test pool layer operations between NervanaGPU, NervanaCPU, NervanaMKL against numpy.
"""
import itertools as itt
import numpy as np
import pytest

from neon import logger as neon_logger
from utils import allclose_with_out

# how many times to repeat the fprop and bprop
repeat = 5


def sliceable(dim, pad=0):
    """
    collapse outer dimensions into one and preserve inner dimension
    this allows for easy cpu operations in numpy
    """
    dim0 = np.prod(dim[:-1]) + pad
    return (dim0, dim[-1])


def pixel_indices(pool, kj, mt, pr, qs):

    C = pool.C
    J, T, R, S = pool.JTRS
    D, H, W = pool.DHW
    HW = H * W
    DHW = D * H * W
    idx = []

    for j in range(J):
        c = kj + j
        ci = c * DHW
        cb = c >= 0 and c < C

        for t in range(T):
            z = mt + t
            zi = ci + z * HW
            zb = cb and z >= 0 and z < D

            for r in range(R):
                y = pr + r
                yi = zi + y * W
                yb = zb and y >= 0 and y < H

                for s in range(S):
                    x = qs + s
                    if yb and x >= 0 and x < W:
                        xi = yi + x
                        idx.append(xi)
    return idx


def run_backend_pool(lib, layer, I, E, dtype):

    beI = lib.array(I, dtype=dtype)
    beE = lib.array(E, dtype=dtype)
    beO = lib.zeros(layer.dimO, dtype=dtype)
    beA = lib.zeros(layer.dimO, dtype=np.int8)
    beB = lib.zeros(layer.dimI, dtype=dtype)

    for i in range(repeat):
        lib.fprop_pool(layer, beI, beO, beA)
        lib.bprop_pool(layer, beE, beB, beA)

    return beO, beB


def run_numpy_pool(op, cpuI, cpuE, dytpe, be_layer):
    # pass in the backend layer for the parameters

    dimI = be_layer.dimI
    dimO = be_layer.dimO
    op = be_layer.op
    K = be_layer.K
    N = be_layer.N
    M, P, Q = be_layer.MPQ
    pad_j, pad_d, pad_h, pad_w = be_layer.padding
    str_j, str_d, str_h, str_w = be_layer.strides

    # cpu output arrays
    cpuO = np.empty(dimO, dtype=dytpe)
    cpuB = np.zeros(sliceable(dimI, 1), dtype=dytpe)

    for i in range(repeat):
        cpuB.fill(0)
        for k in range(K):
            kj = k * str_j - pad_j

            for m in range(M):
                mt = m * str_d - pad_d

                for p in range(P):
                    pr = p * str_h - pad_h

                    for q in range(Q):
                        qs = q * str_w - pad_w

                        idx = pixel_indices(be_layer, kj, mt, pr, qs)

                        if op == "max":
                            cpuO[k, m, p, q, :] = np.max(cpuI[idx, :], axis=0)
                            b_idx = np.argmax(cpuI[idx, :], axis=0)
                            for n in range(N):
                                cpuB[idx[b_idx[n]], n] += cpuE[k, m, p, q, n]
                        elif op == "avg":
                            cpuO[k, m, p, q, :] = np.mean(cpuI[idx, :], axis=0)
                            cpuB[idx, :] += cpuE[k, m, p, q, :] * (1.0 / len(idx))
                        elif op == "l2":
                            cpuO[k, m, p, q, :] = np.sqrt(
                                np.sum(cpuI[idx, :] ** 2, axis=0))

    return cpuO, cpuB


def pytest_generate_tests(metafunc):
    if 'poolargs' in metafunc.fixturenames:
        fargs = []

        op_list = ["avg", "max"]
        fargs = itt.product(op_list)
        metafunc.parametrize('poolargs', fargs)


def test_pool_layer_mkl(poolargs, backend_pair_bench_mkl):

    op = poolargs[0]

    dtype = np.float32
    nm, nc = backend_pair_bench_mkl

    N, C = 32, 32
    D, H, W = 1, 32, 32
    J, T, R, S = 2, 1, 3, 3
    padding_j, padding_d, padding_h, padding_w = 0, 0, 0, 0
    strides_j, strides_d, strides_h, strides_w = 2, 1, 2, 2

    pool_nm = nm.pool_layer(
        dtype,
        op,
        N,
        C, D, H, W,
        J, T, R, S,
        padding_j, padding_d, padding_h, padding_w,
        strides_j, strides_d, strides_h, strides_w)

    pool_nc = nc.pool_layer(
        dtype,
        op,
        N,
        C, D, H, W,
        J, T, R, S,
        padding_j, padding_d, padding_h, padding_w,
        strides_j, strides_d, strides_h, strides_w)

    assert pool_nm.dimI == pool_nc.dimI
    assert pool_nm.dimO == pool_nc.dimO

    dimI = pool_nm.dimI
    dimO = pool_nm.dimO

    # generating input arrays for inputs and errors
    cpuI = np.random.uniform(0.0, 1.0, sliceable(dimI, 1)).astype(
        np.float16).astype(dtype)
    cpuE = np.random.uniform(-0.2, 0.2, dimO).astype(dtype)

    # zero pad the last row of cpu input for the sake of numpy
    if op == "max":
        cpuI[-1, :] = np.finfo(dtype).min
    else:
        cpuI[-1, :] = 0

    # ========= MKL, CPU and numpy ==========
    beI = cpuI[:-1, :].reshape(dimI)
    beE = cpuE

    nmO, nmB = run_backend_pool(nm, pool_nm, beI, beE, dtype)
    ncO, ncB = run_backend_pool(nc, pool_nc, beI, beE, dtype)
    cpuO, cpuB = run_numpy_pool(op, cpuI, cpuE, dtype, pool_nm)

    for opA, nmA, ncA, cpuA in (
            ("fprop", nmO, ncO, cpuO),
            ("bprop", nmB, ncB.reshape(dimI), cpuB[:-1, :].reshape(dimI))):

        neon_logger.display(opA)
        assert allclose_with_out(nmA.get(), ncA.get(), rtol=0, atol=1e-4)
        assert allclose_with_out(ncA.get(), cpuA, rtol=0, atol=1e-5)


@pytest.mark.hasgpu
def test_pool_layer(poolargs, backend_pair_bench):

    op = poolargs[0]

    dtype = np.float32
    ng, nc = backend_pair_bench

    N, C = 32, 32
    D, H, W = 1, 32, 32
    J, T, R, S = 2, 1, 3, 3
    padding_j, padding_d, padding_h, padding_w = 0, 0, 0, 0
    strides_j, strides_d, strides_h, strides_w = 2, 1, 2, 2

    pool_ng = ng.pool_layer(
        dtype,
        op,
        N,
        C, D, H, W,
        J, T, R, S,
        padding_j, padding_d, padding_h, padding_w,
        strides_j, strides_d, strides_h, strides_w)

    pool_nc = nc.pool_layer(
        dtype,
        op,
        N,
        C, D, H, W,
        J, T, R, S,
        padding_j, padding_d, padding_h, padding_w,
        strides_j, strides_d, strides_h, strides_w)

    assert pool_ng.dimI == pool_nc.dimI
    assert pool_ng.dimO == pool_nc.dimO

    dimI = pool_ng.dimI
    dimO = pool_ng.dimO

    # generating input arrays for inputs and errors
    cpuI = np.random.uniform(0.0, 1.0, sliceable(dimI, 1)).astype(
        np.float16).astype(dtype)
    cpuE = np.random.uniform(-0.2, 0.2, dimO).astype(dtype)

    # zero pad the last row of cpu input for the sake of numpy
    if op == "max":
        cpuI[-1, :] = np.finfo(dtype).min
    else:
        cpuI[-1, :] = 0

    # ========= GPU, CPU and numpy ==========
    beI = cpuI[:-1, :].reshape(dimI)
    beE = cpuE

    ngO, ngB = run_backend_pool(ng, pool_ng, beI, beE, dtype)
    ncO, ncB = run_backend_pool(nc, pool_nc, beI, beE, dtype)
    cpuO, cpuB = run_numpy_pool(op, cpuI, cpuE, dtype, pool_ng)

    for opA, ngA, ncA, cpuA in (
            ("fprop", ngO, ncO, cpuO),
            ("bprop", ngB, ncB.reshape(dimI), cpuB[:-1, :].reshape(dimI))):

        neon_logger.display(opA)
        assert allclose_with_out(ngA.get(), ncA.get(), rtol=0, atol=1e-4)
        assert allclose_with_out(ncA.get(), cpuA, rtol=0, atol=1e-5)


if __name__ == '__main__':
    fargs = ["max"]
    test_pool_layer(fargs)
    test_pool_layer_mkl(fargs)
