#!/usr/bin/env python
# Copyright 2016 Nervana Systems Inc. All rights reserved.
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
from __future__ import division
from builtins import str
import numpy         as np
import pycuda.driver as drv
from neon import logger as neon_logger
from neon.backends.nervanagpu import NervanaGPU
from neon.backends.nervanacpu import NervanaCPU

from neon.backends.convolution import (_ceil_div,
    FpropCuda,   BpropCuda,   UpdateCuda,
    FpropDirect, BpropDirect, UpdateDirect)

from neon.backends.winograd_conv import (
    FpropWinograd_2x2_3x3, BpropWinograd_2x2_3x3, UpdateWinograd_3x3_2x2,
    FpropWinograd_4x4_3x3, BpropWinograd_4x4_3x3, UpdateWinograd_3x3_4x4,
    FpropWinograd_2x2_5x5, BpropWinograd_2x2_5x5)

fprop_kernels  = (FpropCuda,  FpropDirect,  FpropWinograd_2x2_3x3,  FpropWinograd_4x4_3x3, FpropWinograd_2x2_5x5)
bprop_kernels  = (BpropCuda,  BpropDirect,  BpropWinograd_2x2_3x3,  BpropWinograd_4x4_3x3, BpropWinograd_2x2_5x5)
update_kernels = (UpdateCuda, UpdateDirect, UpdateWinograd_3x3_2x2, UpdateWinograd_3x3_4x4)

ng = NervanaGPU(0)
nc = NervanaCPU()

neon_logger.display(drv.Context.get_current().get_device().name())

def main():
    out =  0
    ones = 0

    #                D,   H,   W,  T, R, S,    pad,   str
    conv_1x1     = ( 1,  14,  14,  1, 1, 1,  0,0,0, 1,1,1)
    conv_3x3     = ( 1,  14,  14,  1, 3, 3,  0,1,1, 1,1,1)
    conv_3x3p0   = ( 1,  14,  14,  1, 3, 3,  0,0,0, 1,1,1)
    conv_3x3p2   = ( 1,  14,  14,  1, 3, 3,  0,2,2, 1,1,1)
    conv_3x3s2   = ( 1,  14,  14,  1, 3, 3,  0,1,1, 1,2,2)
    conv_1x3     = ( 1,  14,  14,  1, 1, 3,  0,0,1, 1,1,1)
    conv_3x1     = ( 1,  14,  14,  1, 3, 1,  0,1,0, 1,1,1)
    conv_5x5     = ( 1,  14,  14,  1, 5, 5,  0,2,2, 1,1,1)
    conv_11x11s4 = ( 1, 224, 224,  1,11,11,  0,2,2, 1,4,4)
    conv_1x1x1   = ( 7,   7,   7,  1, 1, 1,  0,0,0, 1,1,1)
    conv_3x3x3   = ( 7,   7,   7,  3, 3, 3,  1,1,1, 1,1,1)
    conv_3x3x3s2 = ( 7,   7,   7,  3, 3, 3,  1,1,1, 2,2,2)
    conv_3x3L    = ( 1, 200, 200,  1, 3, 3,  0,1,1, 1,1,1)
    conv_1D      = ( 1, 13, 3263,  1,13,11,  0,0,0, 1,1,3)

    # configs = [

    # ]

    configs = [
        # Kernel                  N,   C,   K  Determ Cmpnd  Xtern, conv
        (FpropCuda,              32,  32,  32, True,  True,  None,  (conv_3x3,) ),
        (BpropCuda,              32,  32,  32, True,  True,  None,  (conv_3x3,) ),
        (UpdateCuda,             32,  32,  32, True,  True,  None,  (conv_3x3,) ),
        (FpropCuda,              32,  32,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5) ),
        (BpropCuda,              32,  32,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5) ),
        (UpdateCuda,             32,  32,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5) ),
        (FpropCuda,              32,   3,  64, True,  False, None,  (conv_11x11s4,) ),
        (UpdateCuda,             32,   3,  64, True,  False, None,  (conv_11x11s4,) ),

        (FpropDirect,            32,  32,  64, True,  True,  None,  (conv_3x3,conv_3x3L) ),
        (BpropDirect,            32,  64,  32, True,  True,  None,  (conv_3x3,conv_3x3L) ),
        (UpdateDirect,           32,  32,  32, True,  True,  None,  (conv_3x3,conv_3x3L) ),
        (UpdateDirect,           32,  32,  32, False, True,  None,  (conv_3x3,conv_3x3L) ),

        (FpropDirect,            32,  32,  64, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (BpropDirect,            32,  64,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (UpdateDirect,           32,  32,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),

        (FpropDirect,            32,   3,  64, True,  False, None,  (conv_11x11s4,) ),
        (UpdateDirect,           32,   3,  32, True,  False, None,  (conv_11x11s4,) ),

        (FpropDirect,            32,  64, 128, True,  True,  None,  (conv_3x3,) ),
        (FpropDirect,            32,  32,  63, True,  True,  None,  (conv_3x3,conv_3x3L) ),
        (FpropDirect,            32,  32,   1, True,  True,  None,  (conv_3x3,conv_3x3L) ),

        (FpropDirect,            16,  32,  64, True,  False, None,  (conv_3x3,) ),
        (FpropDirect,             8,  32,  64, True,  False, None,  (conv_3x3,) ),
        (FpropDirect,             4,  32,  64, True,  False, None,  (conv_3x3,) ),
        (FpropDirect,             2,  32,  64, True,  False, None,  (conv_3x3,) ),
        (FpropDirect,             1,  32,  64, True,  True,  None,  (conv_3x3,) ),

        (UpdateDirect,           16,  32,  63, True,  False, None,  (conv_3x3,) ),
        (UpdateDirect,            8,  32,  64, True,  False, None,  (conv_3x3,) ),
        (UpdateDirect,            4,  32, 128, True,  False, None,  (conv_3x3,) ),

        (FpropDirect,            32,   1, 512, True,  False, None,  (conv_1D,) ),
        (FpropDirect,            16,   1, 512, True,  False, None,  (conv_1D,) ),
        (FpropDirect,             8,   1, 512, True,  False, None,  (conv_1D,) ),
        (FpropDirect,             4,   1, 512, True,  False, None,  (conv_1D,) ),
        (FpropDirect,             2,   1, 512, True,  False, None,  (conv_1D,) ),
        (FpropDirect,             1,   1, 512, True,  False, None,  (conv_1D,) ),

        (UpdateDirect,           32,   1, 512, True,  False, None,  (conv_1D,) ),
        (UpdateDirect,           16,   1, 512, True,  False, None,  (conv_1D,) ),
        (UpdateDirect,            8,   1, 512, True,  False, None,  (conv_1D,) ),
        (UpdateDirect,            4,   1, 512, True,  False, None,  (conv_1D,) ),

        # Kernel                  N,   C,   K  Determ Cmpnd  Xtern, conv
        (FpropDirect,            64,  32,  64, True,  True,  None,  (conv_3x3,) ),
        (FpropDirect,            64,  32, 128, True,  True,  None,  (conv_3x3,) ),
        (FpropDirect,           128,  32,  32, True,  True,  None,  (conv_3x3,) ),
        (FpropDirect,           128,  32,  64, True,  True,  None,  (conv_3x3,) ),
        (FpropDirect,           128,  32, 128, True,  True,  None,  (conv_3x3,) ),
        (BpropDirect,            64,  64,  32, True,  True,  None,  (conv_3x3,) ),
        (BpropDirect,            64, 128,  32, True,  True,  None,  (conv_3x3,) ),
        (BpropDirect,           128,  32,  32, True,  True,  None,  (conv_3x3,) ),
        (BpropDirect,           128,  64,  32, True,  True,  None,  (conv_3x3,) ),
        (BpropDirect,           128, 128,  32, True,  True,  None,  (conv_3x3,) ),

        (FpropDirect,            64,  32,  64, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (FpropDirect,            64,  32, 128, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (FpropDirect,           128,  32,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (FpropDirect,           128,  32,  64, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (FpropDirect,           128,  32, 128, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (BpropDirect,            64,  64,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (BpropDirect,            64, 128,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (BpropDirect,           128,  32,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (BpropDirect,           128,  64,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),
        (BpropDirect,           128, 128,  32, True,  False, None,  (conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2) ),

        (FpropDirect,            64,   3,  64, True,  False, None,  (conv_11x11s4,) ),
        (FpropDirect,            64,   3, 128, True,  False, None,  (conv_11x11s4,) ),
        (FpropDirect,           128,   3,  32, True,  False, None,  (conv_11x11s4,) ),
        (FpropDirect,           128,   3,  64, True,  False, None,  (conv_11x11s4,) ),

        (FpropDirect,            64,  33,  56, True,  True,  None,  (conv_3x3s2,) ),
        (FpropDirect,            64,  33, 120, True,  True,  None,  (conv_3x3s2,) ),
        (FpropDirect,           128,  33,  56, True,  True,  None,  (conv_3x3s2,) ),
        (FpropDirect,           128,  33, 120, True,  True,  None,  (conv_3x3s2,) ),
        (FpropDirect,           128,  33, 248, True,  True,  None,  (conv_3x3s2,) ),

        # Kernel                  N,   C,   K  Determ Cmpnd  Xtern, conv
        (FpropWinograd_2x2_3x3,  32,  32,  32, True,  True,  False, (conv_3x3,conv_3x3L) ),
        (FpropWinograd_2x2_3x3,  32,  32,  32, True,  True,  True,  (conv_3x3,) ),
        (BpropWinograd_2x2_3x3,  32,  32,  32, True,  True,  False, (conv_3x3,conv_3x3L) ),
        (BpropWinograd_2x2_3x3,  32,  32,  32, True,  True,  True,  (conv_3x3,) ),
        (UpdateWinograd_3x3_2x2, 32,  32,  32, True,  True,  None,  (conv_3x3,) ),
        (UpdateWinograd_3x3_2x2, 32,  32,  32, False, True,  None,  (conv_3x3,) ),
        (FpropWinograd_4x4_3x3,  32,  32,  32, True,  True,  False, (conv_3x3,) ),
        (FpropWinograd_4x4_3x3,  32,  32,  32, True,  True,  True,  (conv_3x3,conv_3x3L) ),
        (BpropWinograd_4x4_3x3,  32,  32,  32, True,  True,  False, (conv_3x3,) ),
        (BpropWinograd_4x4_3x3,  32,  32,  32, True,  True,  True,  (conv_3x3,conv_3x3L) ),
        (UpdateWinograd_3x3_4x4, 32,  32,  32, True,  True,  None,  (conv_3x3,) ),
        (UpdateWinograd_3x3_4x4, 32,  32,  32, False, True,  None,  (conv_3x3,) ),

        (FpropWinograd_2x2_3x3,  32,  32,  32, True,  False, True,  (conv_3x3p0,conv_3x3p2) ),
        (BpropWinograd_2x2_3x3,  32,  32,  32, True,  False, True,  (conv_3x3p0,conv_3x3p2) ),
        (UpdateWinograd_3x3_2x2, 32,  32,  32, True,  False, None,  (conv_3x3p0,conv_3x3p2) ),
        (FpropWinograd_4x4_3x3,  32,  32,  32, True,  False, True,  (conv_3x3p0,conv_3x3p2) ),
        (BpropWinograd_4x4_3x3,  32,  32,  32, True,  False, True,  (conv_3x3p0,conv_3x3p2) ),
        (UpdateWinograd_3x3_4x4, 32,  32,  32, True,  False, None,  (conv_3x3p0,conv_3x3p2) ),

        (FpropWinograd_2x2_3x3,   1,  63,  63, True,  False, True,  (conv_3x3,conv_3x3L) ),
        (BpropWinograd_2x2_3x3,   1,  63,  63, True,  False, True,  (conv_3x3,conv_3x3L) ),
        (UpdateWinograd_3x3_2x2,  1,  63,  63, True,  False, None,  (conv_3x3,) ),
        (FpropWinograd_4x4_3x3,   1,  63,  63, True,  False, True,  (conv_3x3,conv_3x3L) ),
        (BpropWinograd_4x4_3x3,   1,  63,  63, True,  False, True,  (conv_3x3,conv_3x3L) ),
        (UpdateWinograd_3x3_4x4,  1,  63,  63, True,  False, None,  (conv_3x3,) ),

        (FpropWinograd_2x2_5x5,  32,  32,  32, False, True,  None,  (conv_5x5,) ),
        (BpropWinograd_2x2_5x5,  32,  32,  32, False, True,  None,  (conv_5x5,) ),

        (FpropWinograd_2x2_5x5,  32,  64, 192, False, False, None,  (conv_5x5,) ),
        (BpropWinograd_2x2_5x5,  32,  64, 192, False, False, None,  (conv_5x5,) ),
        (FpropWinograd_2x2_5x5,  16,  64, 192, False, False, None,  (conv_5x5,) ),
        (FpropWinograd_2x2_5x5,   8,  64, 192, False, False, None,  (conv_5x5,) ),
        (FpropWinograd_2x2_5x5,   4,  64, 192, False, False, None,  (conv_5x5,) ),
        (FpropWinograd_2x2_5x5,   2,  64, 192, False, False, None,  (conv_5x5,) ),
    ]

    fprop_opts = [
        dict(),
        dict(slope=0.0, relu=True),
        dict(slope=0.1, relu=True),
        dict(bias=True),
        dict(bias=True, slope=0.0, relu=True),
        dict(bias=True, slope=0.1, relu=True),
        dict(bsum=True),
    ]
    bprop_opts = [
        dict(),
        dict(X=True, slope=0.0, brelu=True),
        dict(X=True, slope=0.1, brelu=True),
        dict(X=True, bsum=True, slope=0.0, brelu=True),
        dict(X=True, bsum=True, slope=0.1, brelu=True),
        dict(X=True, alpha=2.0, beta=3.0),
        dict(alpha=2.0, beta=3.0),
    ]
    update_opts = [
        dict(alpha=2.0, beta=3.0),
        dict(),
    ]

    for config in configs:

        kernelClass, N, C, K, determ, compound, override, convs = config

        for conv in convs:

            D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w = conv

            ng.deterministic = determ

            layer = nc.conv_layer(np.float64,
                N, C, K, D, H, W, T, R, S,
                pad_d, pad_h, pad_w,
                str_d, str_h, str_w)

            (M, P, Q) = layer.MPQ

            if kernelClass in (FpropCuda, BpropCuda, UpdateCuda):
                dtypes = (np.float32,)
            else:
                dtypes = (np.float32, np.float16)

            for dtype in (dtypes):

                ng.scratch_buffer_reset()

                if override is None:
                    kernel = kernelClass(ng, np.dtype(dtype),
                        N, C, K, D, H, W, T, R, S, M, P, Q,
                        pad_d, pad_h, pad_w,
                        str_d, str_h, str_w)
                else:
                    kernel = kernelClass(ng, np.dtype(dtype),
                        N, C, K, D, H, W, T, R, S, M, P, Q,
                        pad_d, pad_h, pad_w,
                        str_d, str_h, str_w, override)

                neon_logger.display(kernel)

                back = False
                if kernelClass in fprop_kernels:
                    dimI1 = layer.dimI
                    dimI2 = layer.dimF
                    dimO  = layer.dimO
                    opts  = fprop_opts
                    func  = layer.xprop_conv
                elif kernelClass in bprop_kernels:
                    dimI1 = layer.dimO
                    dimI2 = layer.dimF
                    dimO  = layer.dimI
                    opts  = bprop_opts
                    func  = layer.xprop_conv
                    back  = True
                elif kernelClass in update_kernels:
                    dimI1 = layer.dimI
                    dimI2 = layer.dimO
                    dimO  = layer.dimF
                    opts  = update_opts
                    func  = layer.update_conv
                else:
                    raise TypeError("Unknown Kernel Class")

                if not compound:
                    opts = [ dict() ]

                if ones:
                    vals = 1.0
                else:
                    vals = (0.5 - ng.rand()) * 2

                devI1    = ng.empty(dimI1, dtype=dtype)
                devI2    = ng.empty(dimI2, dtype=dtype)
                devO     = ng.empty(dimO,  dtype=dtype)
                devI1[:] = vals
                devI2[:] = vals
                devO[:]  = vals
                cpuI1    = nc.array(devI1.get(), dtype=np.float64)
                cpuI2    = nc.array(devI2.get(), dtype=np.float64)
                cpuO     = nc.array(devO.get(),  dtype=np.float64)

                if compound and opts is not update_opts:
                    devB    = ng.empty((dimO[0], 1), dtype=np.float32)
                    devS    = ng.empty((dimO[0], 1), dtype=np.float32)
                    devB[:] = vals
                    devS[:] = vals
                    cpuB    = nc.array(devB.get(), dtype=np.float64)
                    cpuS    = nc.array(devS.get(), dtype=np.float64)

                if opts is bprop_opts:
                    devX    = ng.empty(dimO,  dtype=dtype)
                    devX[:] = vals
                    cpuX    = nc.array(devX.get(),  dtype=np.float64)

                for opt in opts:
                    dev_opts = dict(opt)
                    cpu_opts = dict(opt)

                    if back:
                        cpu_opts["backward"] = True

                    if "bias" in dev_opts:
                        dev_opts["bias"] = devB
                        cpu_opts["bias"] = cpuB
                    if "bsum" in dev_opts:
                        dev_opts["bsum"] = devS
                        cpu_opts["bsum"] = cpuS
                    if "X" in dev_opts:
                        dev_opts["X"] = devX
                        cpu_opts["X"] = cpuX

                    kernel.bind_params(devI1, devI2, devO, **dev_opts)
                    kernel.execute()

                    func(cpuI1, cpuI2, cpuO, **cpu_opts)

                    devA = devO.get()
                    cpuA = cpuO._tensor
                    difA = cpuA - devA

                    if out:
                        np.savetxt("out.txt",  difA.reshape((-1,dimO[-1])), fmt='%6.3f')
                        np.savetxt("outC.txt", cpuA.reshape((-1,dimO[-1])), fmt='%6.3f')
                        np.savetxt("outD.txt", devA.reshape((-1,dimO[-1])), fmt='%6.3f')

                    maxval = abs(cpuA).max()
                    maxdif = abs(difA).max()
                    ratio  = maxdif / maxval

                    if "bsum" in dev_opts:
                        devZ = devS.get()
                        cpuZ = cpuS._tensor
                        difZ = abs(cpuZ - devZ) / abs(cpuZ).max()
                        ratio2  = difZ.max()
                        #print difZ

                        # def output_slice(p, P, B):
                        #     p0 = p * B
                        #     p1 = p0 + B
                        #     if p1 > P:
                        #         p1 = P
                        #     return slice(p0, p1)

                        # B = 4
                        # Yw = _ceil_div(P, B)
                        # Xw = _ceil_div(Q, B)

                        # bsum = np.empty((K, Yw, Xw))

                        # for y in range(Yw):
                        #     pSlice = output_slice(y, P, B)
                        #     for x in range(Xw):
                        #         qSlice = output_slice(x, Q, B)
                        #         for k in range(K):
                        #             bsum[k,y,x] = np.sum(cpuA[k,0,pSlice,qSlice,:])

                        # bsum = bsum.reshape((K,-1))

                        # np.savetxt("outC.txt", bsum, fmt='%6.1f')
                        # np.savetxt("outD.txt", kernel.bsum.ary, fmt='%6.1f')
                        # np.savetxt("out.txt",  abs(bsum - kernel.bsum.ary), fmt='%6.1f')

                        #print abs(devZ - np.sum(kernel.bsum.ary, axis=1, keepdims=True))


                    else:
                        ratio2  = 0.0

                    bad = ratio > 0.01 or ratio2 > 0.01

                    if bad:
                        neon_logger.display("=================FAIL==============")

                    neon_logger.display("%17.12f %17.12f %s" % (ratio, ratio2, str(opt)))

                    if bad: exit()

                devI1 = devI2 = devO = devB = devS = devX = None
                cpuI1 = cpuI2 = cpuO = cpuB = cpuS = cpuX = None


if __name__ == '__main__':
    main()
