#!/usr/bin/python
# Copyright 2014 Nervana Systems Inc. All rights reserved.
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
import pycuda.driver as drv
from neon.backends.nervanagpu import NervanaGPU
from operator import mul
from neon.backends.tests.utils import assert_tensors_allclose


np.set_printoptions(threshold=8193, linewidth=600,
                    formatter={'int': lambda x: "%10d" % x, 'float': lambda x: "% .3f" % x})

ng = NervanaGPU(stochastic_round=False, bench=True)

print(drv.Context.get_current().get_device().name())


def test_pooling():
    configs = {}
    configs['overlap'] = (1, 1, 3, 3,    # J,T,R,S
                          0, 0, 1, 1,    # padding
                          1, 1, 2, 2)    # strides
    configs['nooverlap'] = (1, 1, 2, 2,    # J,T,R,S
                            0, 0, 0, 0,    # padding
                            1, 1, 2, 2)    # stridesx
    ones = 0
    cpu = 1
    repeat = 1
    alpha = 1.1
    beta = 1.2
    for dtype in [np.float32, np.float16]:
        for op in ["max", "avg"]:  # ["max"]:  #
            # ['nooverlap', 'overlap']:
            for config in ['nooverlap', 'overlap']:
                pool = ng.pool_layer(dtype,
                                     op,
                                     32,         # N
                                     32, 1, 28, 28,  # C,D,H,W
                                     *configs[config])

                pool_helper(
                    dtype, ones, cpu, repeat, alpha, beta, ng, pool, config, op)


def pool_helper(dtype, ones, cpu, repeat, alpha, beta, ng, pool, config, op):

    err_string = "Error in dtype: '%s' op: '%s' config: '%s'" % (
        str(dtype), op, config)

    dimI = pool.dimI
    dimO = pool.dimO

    # colapse pooling dimensions into one
    # this allows for easy cpu pooling in numpy
    def slicable(dim, pad=0):
        dim0 = reduce(mul, dim[:-1], 1) + pad
        return (dim0, dim[-1])

    # cpu input arrays
    # Note that we truncte these to 16 bits so that the cpu and gpu will agree
    # on an index if there is a tie.
    if ones:
        cpuI = np.ones(slicable(dimI), dtype=np.float32)
        cpuB = np.ones(slicable(dimI), dtype=np.float32)
        cpuE = np.ones(dimO, dtype=np.float32)
        cpuO = np.ones(dimO, dtype=np.float32)

    else:
        # .astype(np.float16)
        cpuI = np.random.uniform(-1.0, 1.0, slicable(dimI)
                                 ).astype(np.float16).astype(np.float32)
        cpuB = np.random.uniform(-1.0, 1.0, slicable(dimI)
                                 ).astype(np.float16).astype(np.float32)
        cpuE = np.random.uniform(-1.0, 1.0,
                                 dimO).astype(np.float16).astype(np.float32)
        cpuO = np.random.uniform(-1.0, 1.0,
                                 dimO).astype(np.float16).astype(np.float32)

    cpuA = np.empty(dimO, dtype=np.int32)

    # give gpu the input array without zero padding (not needed)
    devI = ng.array(cpuI.reshape(dimI), dtype=dtype)
    devB = ng.array(cpuB.reshape(dimI), dtype=dtype)
    devE = ng.array(cpuE, dtype=dtype)
    devO = ng.array(cpuO, dtype=dtype)
    devA = ng.empty(dimO, dtype=np.uint8)

    ng.fprop_pool(
        pool, devI, devO, devA, alpha=alpha, beta=beta, repeat=repeat)

    ng.bprop_pool(
        pool, devE, devB, devA, alpha=alpha, beta=beta, repeat=repeat)

    cpuO *= beta
    cpuB *= beta

    def pixel_indices(kj, mt, pr, qs):

        C = pool.C
        J, T, R, S = pool.JTRS
        D, H, W = pool.DHW
        HW = H*W
        DHW = D*H*W
        idx = []

        for j in range(J):
            c = kj + j
            ci = c*DHW
            cb = c >= 0 and c < C

            for t in range(T):
                z = mt + t
                zi = ci + z*HW
                zb = cb and z >= 0 and z < D

                for r in range(R):
                    y = pr + r
                    yi = zi + y*W
                    yb = zb and y >= 0 and y < H

                    for s in range(S):
                        x = qs + s
                        if yb and x >= 0 and x < W:
                            xi = yi + x
                            idx.append(xi)
        return idx

    # numpy pooling implementation
    if cpu:

        op = pool.op
        K = pool.K
        N = pool.N
        M, P, Q = pool.MPQ
        pad_j, pad_d, pad_h, pad_w = pool.padding
        str_j, str_d, str_h, str_w = pool.strides

        for k in range(K):
            kj = k*str_j - pad_j

            for m in range(M):
                mt = m*str_d - pad_d

                for p in range(P):
                    pr = p*str_h - pad_h

                    for q in range(Q):
                        qs = q*str_w - pad_w

                        idx = pixel_indices(kj, mt, pr, qs)
                        # print idx
                        # exit()

                        if op == "max":

                            # set_trace()
                            cpuO[
                                k, m, p, q, :] += np.max(cpuI[idx, :], axis=0) * alpha

                            b_idx = np.argmax(cpuI[idx, :], axis=0)
                            cpuA[k, m, p, q, :] = b_idx.astype(np.int32)

                            # There's probably a more elegant numpy way to do
                            # this..
                            for n in range(N):
                                cpuB[
                                    idx[b_idx[n]], n] += cpuE[k, m, p, q, n] * alpha

                        elif op == "avg":
                            cpuO[
                                k, m, p, q, :] += np.mean(cpuI[idx, :], axis=0) * alpha

                            cpuB[idx, :] += cpuE[k, m, p, q, :] * \
                                (1.0/len(idx)) * alpha

                        # bprop not implemented yet
                        elif op == "l2":
                            cpuO[k, m, p, q, :] = np.sqrt(
                                np.sum(cpuI[idx, :]**2, axis=0))

        # drop zero padding
        cpuI = cpuI.reshape(dimI)
        cpuB = cpuB.reshape(dimI)

        devA = devA.get().astype(np.int32)
        devO = devO.get().astype(np.float32)
        devB = devB.get().astype(np.float32)

        # difA = np.absolute(cpuA - devA)

        # np.savetxt("out_cpuB.txt", cpuB.reshape((-1,pool.N))[:,0:8], fmt='%5.2f')
        # np.savetxt("out_devB.txt", devB.reshape((-1,pool.N))[:,0:8], fmt='%5.2f')

        difO = np.absolute(cpuO - devO)
        maxD = difO.max()
        maxO = np.absolute(cpuO).max()
        print("difO max: %.6f cpuO max: %5.2f ratio: %.6f" %
              (maxD, maxO, maxD / maxO))
        assert_tensors_allclose(
            cpuO, devO, rtol=0, atol=1e-2, err_msg="fprop:" + err_string)

        difB = np.absolute(cpuB - devB)
        maxD = difB.max()
        maxB = np.absolute(cpuB).max()
        print("difB max: %.6f cpuB max: %5.2f ratio: %.6f" %
              (maxD, maxB, maxD / maxB))
        assert_tensors_allclose(
            cpuB, devB, rtol=0, atol=1e-2, err_msg="bprop:" + err_string)


if __name__ == '__main__':
    test_pooling()
