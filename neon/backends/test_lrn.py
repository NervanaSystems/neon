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
from neon.backends.nervanacpu import NervanaCPU


np.set_printoptions(threshold=8193, linewidth=600,
                    formatter={'int': lambda x: "%10d" % x, 'float': lambda x: "% .3f" % x})

ng = NervanaGPU(stochastic_round=False, bench=True)
nc = NervanaCPU()

print(drv.Context.get_current().get_device().name())


def slicable(dim, pad=0):
    dim0 = np.prod(dim[:-1]) + pad
    return (dim0, dim[-1])


def test_pooling():
    layer_args = dict(dtype=np.float32, N=122, C=16, D=1, H=32, W=32, J=5)
    pool_test_args = dict(ones=0, cpu=1,
                          alpha=1.0,  # not supported in CPU
                          ascale=1.2,
                          beta=0.0,  # not supported in CPU
                          bpower=0.5,
                          layer_g=ng.lrn_layer(**layer_args),  # returns a pool layer
                          layer_c=nc.lrn_layer(**layer_args),
                          **layer_args)

    lrn_helper(**pool_test_args)


def lrn_helper(dtype, ones, cpu, alpha, beta, ascale, bpower,
               ng, layer_g, layer_c, N, C, D, H, W, J):

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
    nc.fprop_lrn(layer_c, cccI, cccO, cccD, ascale, bpower)  # CPU has no alpha, beta

    print "== denom =="
    print "CPU fprop"
    print cccD.get().reshape(C*D*H*W, N)[0:4, 0:4]
    print "GPU fprop"
    print devD.get().reshape(C*D*H*W, N)[0:4, 0:4]

    print "== output =="
    print "CPU fprop"
    print cccO.get().reshape(C*D*H*W, N)[0:4, 0:4]
    print "GPU fprop"
    print devO.get().reshape(C*D*H*W, N)[0:4, 0:4]

    # I, O, E, delta, denom
    ng.bprop_lrn(layer_g, devI, devO, devE, devB, devD, alpha, beta, ascale, bpower)
    nc.bprop_lrn(layer_c, cccI, cccO, cccE, cccB, cccD, ascale, bpower)

    print "== bprop =="
    print "CPU bprop"
    print cccB.get().reshape(C*D*H*W, N)[0:4, 0:4]
    print "GPU bprop"
    print devB.get().reshape(C*D*H*W, N)[0:4, 0:4]

    # import ipdb; ipdb.set_trace()
    # cpuO *= beta
    # cpuB *= beta

    #     # drop zero padding
    #     cpuI = cpuI.reshape(dimI)
    #     cpuB = cpuB.reshape(dimI)

    #     devA = devA.get().astype(np.int32)
    #     devO = devO.get().astype(np.float32)
    #     devB = devB.get().astype(np.float32)
    #     cpuA = np.empty(dimO, dtype=np.int32)

    #     difA = np.absolute(cpuA - devA)

    #     # np.savetxt("out_cpuB.txt", cpuB.reshape((-1,layer_g.N))[:,0:8], fmt='%5.2f')
    #     # np.savetxt("out_devB.txt", devB.reshape((-1,layer_g.N))[:,0:8], fmt='%5.2f')

    #     difO = np.absolute(cpuO - devO)
    #     maxD = difO.max()
    #     maxO = np.absolute(cpuO).max()
    #     print("difO max: %.6f cpuO max: %5.2f ratio: %.6f" % (maxD, maxO, maxD / maxO))
    #     assert_tensors_allclose(cpuO, devO, rtol=0, atol=1e-2, err_msg="fprop:" + err_string)

    #     difB = np.absolute(cpuB - devB)
    #     maxD = difB.max()
    #     maxB = np.absolute(cpuB).max()
    #     print("difB max: %.6f cpuB max: %5.2f ratio: %.6f" % (maxD, maxB, maxD / maxB))
    #     assert_tensors_allclose(cpuB, devB, rtol=0, atol=1e-2, err_msg="bprop:" + err_string)


if __name__ == '__main__':
    test_pooling()
