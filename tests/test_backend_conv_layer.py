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
To test conv layer operations between NervanaGPU, NervanaCPU against numpy.
The numpy implementation is different from what is done underneath NervanaCPU to
be a valid checking. It requires externally pad the input, while NervanaCPU does
not require so
"""
import itertools as itt
import numpy as np
import pytest
from timeit import default_timer
from utils import allclose_with_out
from neon import logger as neon_logger


def slicable(dim, pad=0):
    """
    colapse outer dimensions into one and preserve inner dimension
    this allows for easy cpu convolution in numpy

    Arguments:
        dim (tuple): dimensions list in a tuple
        pad (int):  how many pixel paddings
    """
    dim0 = np.prod(dim[:-1]) + pad
    return (dim0, dim[-1])


def pixel_indices(conv, mt, pr, qs):

    T, R, S = conv.TRS
    D, H, W = conv.DHW
    C = conv.C
    HW = H * W
    DHW = D * H * W
    imax = C * DHW

    idx = []
    for c in range(C):
        ci = c * DHW

        for t in range(T):
            z = mt + t
            zi = ci + z * HW
            zb = z >= 0 and z < D

            for r in range(R):
                y = pr + r
                yi = zi + y * W
                yb = zb and y >= 0 and y < H

                for s in range(S):
                    x = qs + s
                    if yb and x >= 0 and x < W:
                        xi = yi + x
                    else:
                        xi = imax  # out of bounds

                    idx.append(xi)
    return idx


def run_backend_conv(lib, layer, I, F, E, dtype):
    beI = lib.array(I, dtype=dtype)
    beF = lib.array(F, dtype=dtype)
    beE = lib.array(E, dtype=dtype)

    beO = lib.zeros(layer.dimO, dtype=dtype)
    lib.fprop_conv(layer, beI, beF, beO)

    beB = lib.zeros(layer.dimI, dtype=dtype)
    lib.bprop_conv(layer, beF, beE, beB)

    beU = lib.zeros(layer.dimF, dtype=dtype)
    lib.update_conv(layer, beI, beE, beU)

    return beO, beB, beU


def pytest_generate_tests(metafunc):
    """
    Build a list of test arguments.

    """
    N_C_K = [
        (64, 64, 64),
        (32, 1, 128),
    ]

    D_H_W = [
        (3, 7, 58),
        (3, 1, 68),
    ]

    T_R_S = [
        (3, 3, 3),
        (1, 3, 3),
        (1, 1, 11),
    ]

    pad_d_h_w = [
        (0, 1, 1),
        (0, 0, 1),
    ]

    str_d_h_w = [
        (1, 1, 1),
        (1, 1, 2),
    ]

    if 'fargs_tests' in metafunc.fixturenames:
        fargs = itt.product(N_C_K, D_H_W, T_R_S, pad_d_h_w, str_d_h_w)
        metafunc.parametrize("fargs_tests", fargs)


def test_conv_layer_mkl(fargs_tests, backend_pair_mkl):

    dtype = np.float32
    nm, nc = backend_pair_mkl

    N, C, K = fargs_tests[0]
    D, H, W = fargs_tests[1]
    T, R, S = fargs_tests[2]
    padding_d, padding_h, padding_w = fargs_tests[3]
    strides_d, strides_h, strides_w = fargs_tests[4]

    conv_nm = nm.conv_layer(
        dtype,
        N, C, K,
        D, H, W,
        T, R, S,
        padding_d, padding_h, padding_w,
        strides_d, strides_h, strides_w)

    conv_nc = nc.conv_layer(
        dtype,
        N, C, K,
        D, H, W,
        T, R, S,
        padding_d, padding_h, padding_w,
        strides_d, strides_h, strides_w)

    assert conv_nc.dimI == conv_nm.dimI
    assert conv_nc.dimF == conv_nm.dimF
    assert conv_nc.dimO == conv_nm.dimO
    assert conv_nc.M == conv_nm.M

    dimI = conv_nm.dimI
    dimF = conv_nm.dimF
    dimO = conv_nm.dimO

    if any(np.array(dimO) <= 0):
        return

    # cpu input arrays
    cpuI = np.random.uniform(-0.8, 0.8, slicable(dimI, 1)).astype(np.float32)
    cpuF = np.random.uniform(0.0, 0.3, slicable(dimF)).astype(np.float32)
    cpuE = np.random.uniform(-0.2, 0.2, dimO).astype(np.float32)

    # zero pad the last row of cpu input for the sake of numpy
    cpuI[-1, :] = 0.0

    # =======MKL and CPU==========
    beI = cpuI[:-1, :].reshape(dimI)
    beF = cpuF.reshape(dimF)
    beE = cpuE

    start_mkl = default_timer()
    nmO, nmB, nmU = run_backend_conv(nm, conv_nm, beI, beF, beE, dtype)
    end_mkl = default_timer()

    start_cpu = default_timer()
    ncO, ncB, ncU = run_backend_conv(nc, conv_nc, beI, beF, beE, dtype)
    end_cpu = default_timer()

    neon_logger.display("mkltime: %s, cputime %s" %
                        (end_mkl - start_mkl, end_cpu - start_cpu))

    # ======numpy===========
    # cpu output arrays
    cpuO = np.zeros(dimO, dtype=dtype)
    cpuB = np.zeros(slicable(dimI, 1), dtype=dtype)
    cpuU = np.zeros(slicable(dimF), dtype=dtype)

    D, H, W = conv_nc.DHW
    T, R, S = conv_nc.TRS
    M, P, Q = conv_nc.MPQ

    pad_d, pad_h, pad_w = conv_nc.padding
    str_d, str_h, str_w = conv_nc.strides

    for m in range(M):
        mt = m * str_d - pad_d

        for p in range(P):
            pr = p * str_h - pad_h

            for q in range(Q):
                qs = q * str_w - pad_w

                idx = pixel_indices(conv_nc, mt, pr, qs)

                cpuO[:, m, p, q, :] = np.dot(cpuF.T, cpuI[idx, :])

                cpuB[idx, :] += np.dot(cpuF, cpuE[:, m, p, q, :])

                cpuU += np.dot(cpuI[idx, :], cpuE[:, m, p, q, :].T)

    for op, nmA, ncA, cpuA, w in (
            ("fprop", nmO, ncO, cpuO, Q),
            ("bprop", nmB, ncB.reshape(dimI), cpuB[:-1, :].reshape(dimI), W),
            ("update", nmU, ncU.reshape(dimF), cpuU.reshape(dimF), S)):

        neon_logger.display(op)
        ncAnp = ncA.get().astype(np.float32)
        nmAnp = nmA.get().astype(np.float32)
        ncdif = cpuA - ncAnp
        nmdif = cpuA - nmAnp
        maxval = abs(cpuA).max()
        ncmaxdif = abs(ncdif).max()
        nmmaxdif = abs(nmdif).max()
        ncRatio = ncmaxdif / float(maxval)
        nmRatio = nmmaxdif / float(maxval)

        assert ncRatio < 1e-5
        assert nmRatio < 1e-5

        assert allclose_with_out(ncA.get(), cpuA, rtol=0, atol=1e-5)
        assert allclose_with_out(nmA.get(), cpuA, rtol=0, atol=1e-3)


@pytest.mark.hasgpu
def test_conv_layer(fargs_tests, backend_pair):

    dtype = np.float32
    ng, nc = backend_pair

    if ng.compute_capability < (5, 0):
        pytest.skip(msg="Test requires Maxwell or higher")

    N, C, K = fargs_tests[0]
    D, H, W = fargs_tests[1]
    T, R, S = fargs_tests[2]
    padding_d, padding_h, padding_w = fargs_tests[3]
    strides_d, strides_h, strides_w = fargs_tests[4]

    conv_ng = ng.conv_layer(
        dtype,
        N, C, K,
        D, H, W,
        T, R, S,
        padding_d, padding_h, padding_w,
        strides_d, strides_h, strides_w)

    conv_nc = nc.conv_layer(
        dtype,
        N, C, K,
        D, H, W,
        T, R, S,
        padding_d, padding_h, padding_w,
        strides_d, strides_h, strides_w)

    assert conv_nc.dimI == conv_ng.dimI
    assert conv_nc.dimF == conv_ng.dimF
    assert conv_nc.dimO == conv_ng.dimO
    assert conv_nc.M == conv_ng.M

    dimI = conv_ng.dimI
    dimF = conv_ng.dimF
    dimO = conv_ng.dimO

    if any(np.array(dimO) <= 0):
        return

    # cpu input arrays
    cpuI = np.random.uniform(-0.8, 0.8, slicable(dimI, 1)).astype(np.float32)
    cpuF = np.random.uniform(0.0, 0.3, slicable(dimF)).astype(np.float32)
    cpuE = np.random.uniform(-0.2, 0.2, dimO).astype(np.float32)

    # zero pad the last row of cpu input for the sake of numpy
    cpuI[-1, :] = 0.0

    # =======GPU and CPU==========
    beI = cpuI[:-1, :].reshape(dimI)
    beF = cpuF.reshape(dimF)
    beE = cpuE

    start_gpu = default_timer()
    ngO, ngB, ngU = run_backend_conv(ng, conv_ng, beI, beF, beE, dtype)
    end_gpu = default_timer()

    start_cpu = default_timer()
    ncO, ncB, ncU = run_backend_conv(nc, conv_nc, beI, beF, beE, dtype)
    end_cpu = default_timer()

    neon_logger.display("gputime: %s, cputime %s" %
                        (end_gpu - start_gpu, end_cpu - start_cpu))

    # ======numpy===========
    # cpu output arrays
    cpuO = np.zeros(dimO, dtype=dtype)
    cpuB = np.zeros(slicable(dimI, 1), dtype=dtype)
    cpuU = np.zeros(slicable(dimF), dtype=dtype)

    D, H, W = conv_nc.DHW
    T, R, S = conv_nc.TRS
    M, P, Q = conv_nc.MPQ

    pad_d, pad_h, pad_w = conv_nc.padding
    str_d, str_h, str_w = conv_nc.strides

    for m in range(M):
        mt = m * str_d - pad_d

        for p in range(P):
            pr = p * str_h - pad_h

            for q in range(Q):
                qs = q * str_w - pad_w

                idx = pixel_indices(conv_nc, mt, pr, qs)

                cpuO[:, m, p, q, :] = np.dot(cpuF.T, cpuI[idx, :])

                cpuB[idx, :] += np.dot(cpuF, cpuE[:, m, p, q, :])

                cpuU += np.dot(cpuI[idx, :], cpuE[:, m, p, q, :].T)

    for opA, ngA, ncA, cpuA, w in (
            ("fprop", ngO, ncO, cpuO, Q),
            ("bprop", ngB, ncB.reshape(dimI), cpuB[:-1, :].reshape(dimI), W),
            ("update", ngU, ncU.reshape(dimF), cpuU.reshape(dimF), S)):

        neon_logger.display(opA)
        ncAnp = ncA.get().astype(np.float32)
        ngAnp = ngA.get().astype(np.float32)
        ncdif = cpuA - ncAnp
        ngdif = cpuA - ngAnp
        maxval = abs(cpuA).max()
        ncmaxdif = abs(ncdif).max()
        ngmaxdif = abs(ngdif).max()
        ncRatio = ncmaxdif / float(maxval)
        ngRatio = ngmaxdif / float(maxval)

        assert ncRatio < 1e-5
        assert ngRatio < 1e-5

        assert allclose_with_out(ncA.get(), cpuA, rtol=1e-5, atol=1e-4)
        assert allclose_with_out(ngA.get(), cpuA, rtol=1e-5, atol=1e-3)
