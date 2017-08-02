# Copyright 2014-2016 Nervana Systems Inc. All rights reserved.
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
import pytest
from neon import logger as neon_logger


def slicable(dim, pad=0):
    dim0 = np.prod(dim[:-1]) + pad
    return (dim0, dim[-1])


def test_pooling_mkl(backend_pair_bench_mkl):
    nm, nc = backend_pair_bench_mkl
    layer_args = dict(dtype=np.float32, N=122, C=16, D=1, H=32, W=32, J=5)
    pool_test_args = dict(ones=0, cpu=1, nm=nm, nc=nc,
                          alpha=1.0,  # not supported in CPU
                          ascale=1.2,
                          beta=0.0,  # not supported in CPU
                          bpower=0.5,
                          layer_m=nm.lrn_layer(**layer_args),  # returns a pool layer
                          layer_c=nc.lrn_layer(**layer_args),
                          **layer_args)

    lrn_helper_mkl(**pool_test_args)


def lrn_helper_mkl(dtype, ones, cpu, alpha, beta, ascale, bpower,
                   nm, nc, layer_m, layer_c, N, C, D, H, W, J):

    dimI = layer_m.dimI
    dimO = layer_m.dimO

    # cpu input arrays
    # Note that we truncte these to 16 bits so that the cpu and gpu
    # will agree on an index if there is a tie.
    if ones:
        cpuI = np.ones(slicable(dimI), dtype=np.float32)
        cpuB = np.ones(slicable(dimI), dtype=np.float32)
        cpuE = np.ones(dimO, dtype=np.float32)
        cpuO = np.ones(dimO, dtype=np.float32)
    else:
        cpuI = np.random.uniform(-1.0, 1.0, slicable(dimI)).astype(np.float16).astype(np.float32)
        cpuB = np.random.uniform(-1.0, 1.0, slicable(dimI)).astype(np.float16).astype(np.float32)
        cpuE = np.random.uniform(-1.0, 1.0, dimO).astype(np.float16).astype(np.float32)
        cpuO = np.random.uniform(-1.0, 1.0, dimO).astype(np.float16).astype(np.float32)

    # give gpu the input array without zero padding (not needed)
    devI = nm.array(cpuI.reshape(dimI), dtype=dtype)
    devB = nm.array(cpuB.reshape(dimI), dtype=dtype)  # delta "backprop"
    devE = nm.array(cpuE, dtype=dtype)
    devO = nm.array(cpuO, dtype=dtype)
    devD = nm.empty(dimO, dtype=dtype)  # denom

    cccI = nc.array(cpuI.reshape(dimI), dtype=dtype)
    cccB = nc.array(cpuB.reshape(dimI), dtype=dtype)  # delta "backprop"
    cccE = nc.array(cpuE, dtype=dtype)
    cccO = nc.array(cpuO, dtype=dtype)
    cccD = nc.empty(dimO, dtype=dtype)  # denom

    # layer, I, O, denom, ascale=1, bpower=1
    nm.fprop_lrn(layer_m, devI, devO, devD, alpha, beta, ascale, bpower)  # I, O, denom
    nc.fprop_lrn(layer_c, cccI, cccO, cccD, None, None, ascale, bpower)  # CPU has no alpha, beta

    neon_logger.display("== denom ==")
    neon_logger.display("CPU fprop")
    neon_logger.display(cccD.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display("MKL fprop")
    neon_logger.display(devD.get().reshape(C * D * H * W, N)[0:4, 0:4])

    neon_logger.display("== output ==")
    neon_logger.display("CPU fprop")
    neon_logger.display(cccO.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display("MKL fprop")
    neon_logger.display(devO.get().reshape(C * D * H * W, N)[0:4, 0:4])

    # I, O, E, delta, denom
    nm.bprop_lrn(layer_m, devI, devO, devE, devB, devD, alpha, beta, ascale, bpower)
    nc.bprop_lrn(layer_c, cccI, cccO, cccE, cccB, cccD, None, None, ascale, bpower)

    neon_logger.display("== bprop ==")
    neon_logger.display("CPU bprop")
    neon_logger.display(cccB.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display("MKL bprop")
    neon_logger.display(devB.get().reshape(C * D * H * W, N)[0:4, 0:4])


@pytest.mark.hasgpu
def test_pooling(backend_pair_bench):
    ng, nc = backend_pair_bench
    layer_args = dict(dtype=np.float32, N=122, C=16, D=1, H=32, W=32, J=5)
    pool_test_args = dict(ones=0, cpu=1, ng=ng, nc=nc,
                          alpha=1.0,  # not supported in CPU
                          ascale=1.2,
                          beta=0.0,  # not supported in CPU
                          bpower=0.5,
                          layer_g=ng.lrn_layer(**layer_args),  # returns a pool layer
                          layer_c=nc.lrn_layer(**layer_args),
                          **layer_args)

    lrn_helper(**pool_test_args)


def lrn_helper(dtype, ones, cpu, alpha, beta, ascale, bpower,
               ng, nc, layer_g, layer_c, N, C, D, H, W, J):

    dimI = layer_g.dimI
    dimO = layer_g.dimO

    # cpu input arrays
    # Note that we truncte these to 16 bits so that the cpu and gpu
    # will agree on an index if there is a tie.
    if ones:
        cpuI = np.ones(slicable(dimI), dtype=np.float32)
        cpuB = np.ones(slicable(dimI), dtype=np.float32)
        cpuE = np.ones(dimO, dtype=np.float32)
        cpuO = np.ones(dimO, dtype=np.float32)
    else:
        cpuI = np.random.uniform(-1.0, 1.0, slicable(dimI)).astype(np.float16).astype(np.float32)
        cpuB = np.random.uniform(-1.0, 1.0, slicable(dimI)).astype(np.float16).astype(np.float32)
        cpuE = np.random.uniform(-1.0, 1.0, dimO).astype(np.float16).astype(np.float32)
        cpuO = np.random.uniform(-1.0, 1.0, dimO).astype(np.float16).astype(np.float32)

    # give gpu the input array without zero padding (not needed)
    devI = ng.array(cpuI.reshape(dimI), dtype=dtype)
    devB = ng.array(cpuB.reshape(dimI), dtype=dtype)  # delta "backprop"
    devE = ng.array(cpuE, dtype=dtype)
    devO = ng.array(cpuO, dtype=dtype)
    devD = ng.empty(dimO, dtype=dtype)  # denom

    cccI = nc.array(cpuI.reshape(dimI), dtype=dtype)
    cccB = nc.array(cpuB.reshape(dimI), dtype=dtype)  # delta "backprop"
    cccE = nc.array(cpuE, dtype=dtype)
    cccO = nc.array(cpuO, dtype=dtype)
    cccD = nc.empty(dimO, dtype=dtype)  # denom

    # layer, I, O, denom, ascale=1, bpower=1
    ng.fprop_lrn(layer_g, devI, devO, devD, alpha, beta, ascale, bpower)  # I, O, denom
    nc.fprop_lrn(layer_c, cccI, cccO, cccD, None, None, ascale, bpower)  # CPU has no alpha, beta

    neon_logger.display("== denom ==")
    neon_logger.display("CPU fprop")
    neon_logger.display(cccD.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display("GPU fprop")
    neon_logger.display(devD.get().reshape(C * D * H * W, N)[0:4, 0:4])

    neon_logger.display("== output ==")
    neon_logger.display("CPU fprop")
    neon_logger.display(cccO.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display("GPU fprop")
    neon_logger.display(devO.get().reshape(C * D * H * W, N)[0:4, 0:4])

    # I, O, E, delta, denom
    ng.bprop_lrn(layer_g, devI, devO, devE, devB, devD, alpha, beta, ascale, bpower)
    nc.bprop_lrn(layer_c, cccI, cccO, cccE, cccB, cccD, None, None, ascale, bpower)

    neon_logger.display("== bprop ==")
    neon_logger.display("CPU bprop")
    neon_logger.display(cccB.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display("GPU bprop")
    neon_logger.display(devB.get().reshape(C * D * H * W, N)[0:4, 0:4])


if __name__ == '__main__':
    test_pooling(0)
    test_pooling_mkl(0)
