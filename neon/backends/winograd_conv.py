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
"""
Python code to wrap convolution kernels
"""

import sys
import logging
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize
import kernel_specs
from neon.backends.cuda_templates import _common_round, _common_fp16_to_fp32, _ew_types
import os.path
import shelve
from convolution import KernelGroup, _get_shuffle_kernel, _get_sm_count, _ceil_div, _magic64, _magic32, _flatten, _fp_convert, _closest_divisor


logger = logging.getLogger(__name__)


class XpropWinograd(KernelGroup):

    def __init__(self, op, lib, dtype,
                 N, C, K,
                 H, W, P, Q,
                 pad_h, pad_w,
                 relu, bsum):

        super(XpropWinograd, self).__init__(lib, dtype)

        SMs = _get_sm_count()

        self.autotune_key = " ".join(str(x) for x in (op + "_2x2_3x3",
           SMs, dtype.itemsize, N, C, K, H, W, P, Q))
        self.autotune_db_file = os.path.join(lib.cache_dir, "autotune.db")

        # allow for .5 seconds worth of warmup when autotuning
        # assume 10 Tflops on 24 SMs
        self.warmup = min(max(int(5e12 / (P*Q*K*N*C*9*2.0) * (SMs / 24.0)), 1), 1000)

        self.params = (N, C, K, H, W, P, Q, pad_h, pad_w, relu, bsum)
        self.init(self.params)

        lib.set_scratch_size(self.trans_size)

    def init(self, params, autotune=0, filter_extern=0):

        R, S = 3,3
        N, C, K, H, W, P, Q, pad_h, pad_w, relu, bsum = self.params

        if not autotune:
            if N >= 32:
                if self.dtype.type is np.float32:
                    filter_extern = C < 512 and K < 512
                else:
                    filter_extern = True
                self.initialized = True
            else:
                autotune_db = shelve.open(self.autotune_db_file)

                if self.autotune_key in autotune_db:
                    filter_extern = autotune_db[self.autotune_key]
                    self.initialized = True
                else:
                    filter_extern    = True
                    self.initialized = False

                autotune_db.close()

            #print filter_extern, self.autotune_key

        filter_trans = "_FX" if filter_extern else ""

        kernel_name = "%s_winograd_2x2_3x3_32x32%s" % (self.clss, filter_trans)

        if N == 1:
            shiftN = 0
        elif N < 32:
            shiftN = len(bin(N-1))-2
        else:
            shiftN = 5
        blkN = 1 << shiftN

        shiftY, shiftX, superY, superX = {
            1 : (3,4,0x203,0x300), # 4x8
            2 : (3,3,0x203,0x201), # 4x4
            4 : (2,3,0x104,0x202), # 2x4
            8 : (2,2,0x104,0x103), # 2x2
            16: (1,2,0x000,0x104), # 1x2
            32: (1,1,0x000,0x000), # 1x1
        }.get(blkN)

        #print "%d, 0x%03x, 0x%03x" % (blkN, superY, superX)

        gridK = _ceil_div(K, 32)
        gridY = _ceil_div(P, 1<<shiftY)
        gridX = _ceil_div(Q, 1<<shiftX)
        gridN = _ceil_div(N, blkN)
        Y2    = gridY // 2
        X2    = gridX  * 2
        XNp   = W*N   * self.dtype.itemsize
        YXNp  = H*W*N * self.dtype.itemsize
        SKp   = S*K   * self.dtype.itemsize
        RSKp  = R*S*K * self.dtype.itemsize
        C4Kp  = gridK * self.dtype.itemsize * 4 * 512

        # process not more than 4 groups of K32 tiles at a time
        # perhaps tune this for smaller cache sizes
        groupK = _closest_divisor(gridK, 4)
        YXGK   = gridY*gridX*groupK
        X2GK   = X2*groupK

        magic_YXGK   = _magic64(YXGK)
        magic_X2GK   = _magic64(X2GK)
        magic_groupK = _magic32(X2GK, groupK)

        grid  = ( gridK*gridY*gridX, gridN, 1 )
        block = (256, 1, 1)

        self.kernel = [ kernel_name, grid, block, None, None, None, None, None, None, None, None ]
        self.kernel.extend( _flatten([
            K, C, H, W, N, P, Q, W*N, H*W*N, R*S*K, Q*N, P*Q*N,
            shiftY, shiftX, shiftN, superY, superX, gridX, gridK, Y2,
            YXGK, X2GK, groupK, magic_YXGK, magic_X2GK, magic_groupK,
            2*XNp, XNp, 4*YXNp-3*XNp, 2*SKp, SKp, 4*RSKp-SKp, C4Kp, pad_h, pad_w ]))

        self.flags  = (relu and 2) + (bsum and 4)

        return filter_extern

    def autotune(self, I, F, O):

        start, stop = self.lib.get_events()

        # Only need to do warmup once
        if not self.lib.warmup:
            self.lib.warmup = True
            self.init(self.params, autotune=1, filter_extern=1)
            self.bind_params(I, F, O, alpha=1.0, beta=0.0, bsum=None, flags=1)
            self.execute(repeat=self.warmup, unbind=False)

        results = []
        for external in (0,1):

            self.init(self.params, autotune=1, filter_extern=external)
            self.bind_params(I, F, O, alpha=1.0, beta=0.0, bsum=None, flags=1)
            start.record(stream=self.lib.stream)
            self.execute(repeat=self.warmup, unbind=False)
            stop.record(stream=self.lib.stream)
            stop.synchronize()
            msecs = stop.time_since(start) / self.warmup

            results.append((msecs, external))

        results.sort()
        external = results[0][1]
        # print self.autotune_key
        # for res in results:
        #     print res

        autotune_db = shelve.open(self.autotune_db_file)
        autotune_db[self.autotune_key] = external
        autotune_db.close()

        self.init(self.params, autotune=0, filter_extern=external)


class FpropWinograd(XpropWinograd):

    def __init__(self, lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w, relu, bsum):

        super(FpropWinograd, self).__init__("fprop", lib, dtype,
            N, C, K, H, W, P, Q, pad_h, pad_w, relu, bsum)


    def init(self, params, autotune=0, filter_extern=0):

        filter_extern = super(FpropWinograd, self).init(params, autotune, filter_extern)

        R, S  = 3,3
        C, K  = self.params[1:3]
        gridK = _ceil_div(K, 32)

        if filter_extern:

            self.trans_size   = C * gridK * 512 * self.dtype.itemsize
            self.trans_shared = 512 * self.dtype.itemsize
            self.trans_args   = [(gridK, C, 1), (32, 1, 1), None,
                                 None, None, R*S*K, S*K, S*K*2, K]
        else:
            self.trans_size = 0
            self.trans_args = None

    def bind_params(self, I, F, O, alpha, beta, bsum, flags=0):

        assert I.dtype == F.dtype == O.dtype

        if not self.initialized:
            self.initialized = True
            self.autotune(I, F, O)

        bsum_gpudata, flags = self.init_bsum(bsum, flags)

        if self.trans_size:
            filter_temp = self.lib.scratch_buffer(self.trans_size)
            self.trans_args[2:5] = (self.lib.stream, filter_temp, F.gpudata)
        else:
            filter_temp = F.gpudata

        self.kernel[3:11] = (self.lib.stream, bsum_gpudata, O.gpudata, I.gpudata, filter_temp,
                             alpha, beta, flags)

    def execute(self, repeat=1, unbind=True):

        kernel = kernel_specs.get_kernel(self.kernel[0])
        if self.trans_size:
            trans_kernel = _get_fprop_filter_trans_kernel(self.dtype_str)

        for r in range(repeat):

            if self.trans_size:
                trans_kernel.prepared_async_call(*self.trans_args, shared_size=self.trans_shared)

            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)

            kernel.prepared_async_call(*self.kernel[1:])

        if unbind:
            self.bsum_zero = None
            self.kernel[3:11] = (None,) * 8
            if self.trans_size:
                self.trans_args[2:5] = (None,) * 3

    def __str__(self):
        return "FpropWinograd " + self.kernel[0]

class BpropWinograd(XpropWinograd):

    def __init__(self, lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w, relu, bsum):

        # Swap C<=>K and HW<=>PQ, invert padding
        super(BpropWinograd, self).__init__("bprop", lib, dtype,
            N, K, C, P, Q, H, W, 2-pad_h, 2-pad_w, relu, bsum)

    def init(self, params, autotune=0, filter_extern=0):

        filter_extern = super(BpropWinograd, self).init(params, autotune, filter_extern)

        R, S  = 3,3
        K, C  = self.params[1:3]
        gridC = _ceil_div(C, 32)
        gridK = _ceil_div(K, 32)
        self.filter_extern = filter_extern

        # transform plus dim shuffle CRSK => KRSC
        if filter_extern:

            cBlks = _ceil_div(gridC*32, 16)
            kBlks = _ceil_div(K, 16)

            self.trans_size = K * gridC * 512 * self.dtype.itemsize
            self.trans_args = [(kBlks,cBlks,1), (256,1,1), None,
                               None, None, R*S*K, S*K, S*K*2, K, C, cBlks]

        # plain dim shuffle CRSK => KRSC
        else:
            self.trans_size = C*R*S*K * self.dtype.itemsize
            self.trans_args = [(gridK, gridC, R*S), (32, 8, 1), None, None, None]
            self.trans_args.extend(_flatten([
                R*S*K, R*S*K, S*K, K,
                R*S*C, R*S*C, S*C, C,
                R*S, 1, R, S, _magic32(R*S, R*S), _magic32(R*S, S)]))

    def bind_params(self, I, F, O, alpha, beta, bsum, flags=0):

        assert I.dtype == F.dtype == O.dtype

        if not self.initialized:
            self.initialized = True
            self.autotune(I, F, O)

        bsum_gpudata, flags = self.init_bsum(bsum, flags)

        filter_temp = self.lib.scratch_buffer(self.trans_size)
        self.trans_args[2:5] = (self.lib.stream, filter_temp, F.gpudata)

        self.kernel[3:11] = (self.lib.stream, bsum_gpudata, O.gpudata, I.gpudata, filter_temp,
                             alpha, beta, flags)

    def execute(self, repeat=1, unbind=True):

        kernel = kernel_specs.get_kernel(self.kernel[0])

        if self.filter_extern:
            trans_kernel = _get_bprop_filter_trans_kernel(self.dtype_str)
        else:
            trans_kernel = _get_shuffle_kernel(self.dtype_str)

        for r in range(repeat):

            if self.trans_size:
                trans_kernel.prepared_async_call(*self.trans_args)

            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)

            kernel.prepared_async_call(*self.kernel[1:])

        if unbind:
            self.bsum_zero = None
            self.kernel[3:11] = (None,) * 8
            if self.trans_size:
                self.trans_args[2:5] = (None,) * 3

    def __str__(self):
        return "BpropWinograd " + self.kernel[0]

class UpdateWinograd(KernelGroup):

    internal_superblock = {
        #blkN : (shlY, shlX,  supY,  supX,  supN)
            1 : (   2,    2, 0x101, 0x100, 0x000), # 2x2 cccccyx
            2 : (   1,    2, 0x000, 0x101, 0x100), # 1x2 cccccxn
            4 : (   1,    1, 0x000, 0x000, 0x200), # 1x1 cccccnn
    }
    external_superblock = {
        #blkN : (shlY, shlX,  supY,  supX,  supN)
            1 : (   1,    1, 0x106, 0x105, 0x000), # 2x2 yxccccc
            2 : (   0,    1, 0x000, 0x106, 0x105), # 1x2 xnccccc
            4 : (   0,    0, 0x000, 0x000, 0x205), # 1x1 nnccccc
    }

    def __init__(self, lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w):

        # Support N = 1,2 and multiples of 4 for now
        assert N in (1,2) or N % 4 == 0

        super(UpdateWinograd, self).__init__(lib, dtype)

        SMs = _get_sm_count()

        self.autotune_key = [str(x) for x in ("update_3x3_2x2",
           SMs, 0, dtype.itemsize, N, C, K, H, W, P, Q)]
        self.autotune_db_file = os.path.join(lib.cache_dir, "autotune.db")

        self.params = (N, C, K, H, W, P, Q, pad_h, pad_w)
        self.init(self.params)

        self.output_size = (self.determ_size or (dtype.type != np.float32 and 9*C*K)) * 4

        lib.set_scratch_size(self.output_size, self.trans_size)

        # allow for .5 seconds worth of warmup when autotuning
        # assume 10 Tflops on 24 SMs
        self.warmup = min(max(int(5e12 / (P*Q*K*N*C*9*2.0) * (SMs / 24.0)), 1), 1000)


    def init(self, params, autotune=False):

        N, C, K, H, W, P, Q, pad_h, pad_w = self.params
        R, S    = 3, 3
        loopN   = 4 if N >= 4 else N
        blkN    = 4 if N >= 3 else N
        superI  = UpdateWinograd.external_superblock.get(blkN) # I = image
        superE  = UpdateWinograd.internal_superblock.get(blkN) # E = error
        blkXI   = 1 << superI[1]
        blkY    = 1 << superE[0]
        blkX    = 1 << superE[1]
        GY      = _ceil_div(P, 2)
        GX      = _ceil_div(Q, 2)
        GYS     = self.GYS = _ceil_div(P, blkY)
        GXS     = self.GXS = _ceil_div(Q, blkX)

        if autotune:
            strideY, strideX, external = autotune
        else:
            autotune_db  = shelve.open(self.autotune_db_file)
            autotune_key = " ".join(self.autotune_key)

            if autotune_key in autotune_db:
                strideY, strideX, external = autotune_db[autotune_key]
                #print strideY, strideX, external, autotune_key
                self.initialized = True
            else:
                if GYS * GXS > 768:
                    strideY  = 768
                    strideX  = 1
                else:
                    strideY  = GYS
                    strideX  = GXS
                external = True
                self.initialized = False

            autotune_db.close()

        loopXI  = N * (strideX*blkXI - 1)
        loopXE  = N * (strideX*blkX  - 1)
        Np      = N     * self.dtype.itemsize
        XNp     = W*N   * self.dtype.itemsize
        XN2p    = W*N*2 * self.dtype.itemsize
        QNp     = Q*N   * self.dtype.itemsize

        gridK  = _ceil_div(K, 32)
        gridC  = _ceil_div(C, 32)
        # find even grid divisors closest to 4 (priorize larger values of equal distance)
        k_size = _closest_divisor(gridK, 4)
        c_size = _closest_divisor(gridC, 4)
        C_size = gridC // c_size

        kc          = k_size  * c_size
        Qkc         = strideX * kc
        PQkc        = strideY * Qkc
        CPQkc       = C_size  * PQkc
        gridPQKC    = strideY * strideX * gridK * gridC
        magic_CPQkc = _magic64(CPQkc)
        magic_PQkc  = _magic64(PQkc)
        magic_Qkc   = _magic64(Qkc)
        magic_kc    = _magic32(Qkc, kc)
        magic_c     = _magic32(kc, c_size)

        self.blocksCK = gridK * gridC

        if external:
            # External Image transform
            IX  = "_IX"
            WN  = GX*N
            HWN = GY*WN

            self.image_transform(N, C, K, H, W, P, Q, pad_h, pad_w)
        else:
            # Internal Image transform
            IX     = ""
            WN     = W*N
            HWN    = H*WN
            superI = superE

            self.trans_size = 0

        # If output grid is 1, don't use atomics.  Kernel is deterministic by default
        if strideY*strideX == 1:
            determ = "D"
            self.determ_size  = 0
            self.determ_shape = False
            self.zero         = False
        elif self.lib.deterministic:
            determ = "D"
            self.determ_size  = strideY*strideX * C*R*S*K
            self.determ_shape = (strideY*strideX, C*R*S*K)
            self.zero         = False
        else:
            determ = ""
            self.determ_size  = 0
            self.determ_shape = False
            self.zero         = True

        kernel_name = "%s_winograd%s_3x3_2x2_32x32%s" % (self.clss, determ, IX)

        # print "blks/sm:%.2f blks:%d gridKC:(%d,%d) gridYX:(%d,%d) stride:(%d,%d)" % (
        #    gridPQKC/24.0, gridPQKC, gridK, gridC, GYS, GXS, strideY, strideX)

        self.kernel = [ kernel_name, (gridPQKC,1,1), (256,1,1), None, None, None, None, None ]
        self.kernel.extend(_flatten([
            H, W, P, Q, C, K, N, pad_h, pad_w,
            GY, GX, GYS, GXS, superI, superE, loopXI, loopXE, loopN, strideY, strideX,
            WN, HWN, Q*N, P*Q*N, S*K, R*S*K, Np, XNp, XN2p, QNp,
            CPQkc, PQkc, Qkc, kc, c_size, k_size,
            magic_CPQkc, magic_PQkc, magic_Qkc, magic_kc, magic_c, C*R*S*K ]))

    def image_transform(self, N, C, K, H, W, P, Q, pad_h, pad_w):

        if N == 1:
            shlN = 0
        elif N < 16:
            shlN = len(bin(N-1))-2
        else:
            shlN = 4

        GY = _ceil_div(P, 2)
        GX = _ceil_div(Q, 2)

        # maximize points of GX in superblock for large GX (most contiguous access)
        if shlN > 1 or GX >= 32:
            shlY, shlX, maskY, shrY, maskX, shrX, maskN = {
                0 : (1, 5, 0x0, 0, 0xf, 0, 0x0), # 1x16 xxxx
                1 : (1, 4, 0x0, 0, 0xe, 1, 0x1), # 1x8  xxxn
                2 : (1, 3, 0x0, 0, 0xc, 2, 0x3), # 1x4  xxnn
                3 : (1, 2, 0x0, 0, 0x8, 3, 0x7), # 1x2  xnnn
                4 : (1, 1, 0x0, 0, 0x0, 0, 0xf), # 1x1  nnnn
            }.get(shlN)
        # Only use GY in superblock for small N
        elif shlN > 0 or GX >= 16:
            shlY, shlX, maskY, shrY, maskX, shrX, maskN = {
                0 : (2, 4, 0x8, 3, 0x7, 0, 0x0), # 2x8  yxxx
                1 : (2, 3, 0x8, 3, 0x6, 1, 0x1), # 2x4  yxxn
            }.get(shlN)
        # if GX is small use more points of GY in superblock
        else:
            shlY, shlX, maskY, shrY, maskX, shrX, maskN = {
                0 : (3, 3, 0xc, 2, 0x3, 0, 0x0), # 4x4  yyxx
            }.get(shlN)

        blkN = 1 << shlN
        blkY = 1 << shlY
        blkX = 1 << shlX

        gridC  = _ceil_div(C, 32)

        GC  = _ceil_div(gridC*32, 16)
        GYS = _ceil_div(P, blkY)
        GXS = _ceil_div(Q, blkX)
        GN  = _ceil_div(N, blkN)
        GYS2 = GYS // 2
        GXS2 = GXS  * 2

        # Since our shared memory shuffle block is 16x16, try to process
        # C and N in blocks of 2.  This gives the same performance as a 32x32 block.
        groupC  = 2 # GC always divisible by 2
        groupN  = 1 if GN & 1 else 2
        shift_c = groupC - 1
        shift_n = groupN - 1

        X2cn       = GXS2 * groupC * groupN
        magic_X2cn = _magic64(X2cn)

        self.trans_size = gridC * GY * GX * N * 512 * self.dtype.itemsize
        self.trans_args = [(GYS*GXS*groupC*groupN, GN//groupN, GC//groupC),
                           (256,1,1), None, None, None,
                            C, H, W, N, pad_h, pad_w,
                            GY, GX, GXS, GYS2, X2cn,
                            magic_X2cn[0], magic_X2cn[1], shift_c, shift_n,
                            shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN,
                            H*W*N, W*N, GY*GX*N*512, GX*N*512, N*512 ]

    def autotune(self, I, E, O):

        autotune_key = " ".join(self.autotune_key)
        #print "autotune: ", self.autotune_key

        start, stop = self.lib.get_events()

        # Only need to do warmup once
        if not self.lib.warmup:
            self.lib.warmup = True
            # warmup  with a conservative set of params
            self.init(self.params, autotune=(self.GYS, 1, 1))
            self.bind_params(I, E, O, 1.0)
            self.execute(repeat=self.warmup, unbind=False)

        # we want at least this many blocks
        block_slots = _get_sm_count()
        # loops for given size of N
        loopsN = max(self.params[0] // 4, 1)
        # don't bother with internal mode for N>4
        modes = (0,1) if self.params[0] <= 4 else (1,)

        gys = float(self.GYS)
        gxs = float(self.GYS)
        small_set = gys * gxs <= 512

        results = []
        sys.stdout.write("Autotune " + str(self))
        progress = 0
        for threshold in (True, False):
            for external in modes:
                for strideY in range(1, self.GYS+1):
                    for strideX in range(1, self.GXS+1):
                        if progress % 32 == 0:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                        progress += 1

                        # CRSK copies in determ mode
                        outputs = strideY * strideX

                        # minimal occupancy filter
                        blocks = self.blocksCK * strideY * strideX
                        # gemm loop size filter
                        depth = (gys / strideY) * (gxs / strideX) * loopsN

                        filters = outputs <= 768 and  blocks >= block_slots and depth >= 32.0

                        # In case we filter out all settings, run though the loops again
                        # this time looking only at settings that didn't pass.
                        if small_set or (threshold and filters) or (not threshold and not filters):

                            settings = (strideY, strideX, external)
                            #print settings

                            self.init(self.params, autotune=settings)
                            self.bind_params(I, E, O, 1.0)
                            start.record(stream=self.lib.stream)
                            self.execute(repeat=2, unbind=False)
                            stop.record(stream=self.lib.stream)
                            stop.synchronize()
                            msecs = stop.time_since(start) / 2.0
                            results.append((msecs, settings))

                        # else:
                        #     print strideY, strideX, external, blocks, round(depth,1)

            # if we got any results, no need to disable the filter
            if len(results) > 0:
                break
        sys.stdout.write('\n')

        results.sort()
        settings = results[0][1]
        # for res in results[0:10]:
        #     print res

        autotune_db = shelve.open(self.autotune_db_file)
        autotune_db[autotune_key] = settings

        # add a copy if this layer has small strides
        # deterministic vs non-determ should make no speed difference here
        if settings[0] * settings[1] <= 8:
            self.autotune_key[2] = str(1 - int(self.autotune_key[2]))
            autotune_key = " ".join(self.autotune_key)
            autotune_db[autotune_key] = settings

        autotune_db.close()

        self.init(self.params, autotune=settings)


    def bind_params(self, I, E, O, alpha):

        assert I.dtype == E.dtype

        if not self.initialized:
            self.initialized = True
            self.autotune(I, E, O)

        if O.dtype.type is not np.float32 or self.determ_size:

            update_temp       = self.lib.scratch_buffer(self.output_size)
            self.convert_args = [ update_temp, "f4", O, self.determ_shape ]

            if self.trans_size:
                input_temp = self.lib.scratch_buffer_offset(self.trans_size)

        else:
            update_temp = O.gpudata
            self.convert_args = False

            if self.trans_size:
                input_temp = self.lib.scratch_buffer(self.trans_size)

        if self.trans_size:
            self.trans_args[2:5] = (self.lib.stream, input_temp, I.gpudata)
        else:
            input_temp = I.gpudata

        if self.zero:
            self.zero_args = [update_temp, 0, O.size, self.lib.stream]

        self.kernel[3:8] = (self.lib.stream, update_temp, input_temp, E.gpudata, alpha)

    def execute(self, repeat=1, unbind=True):

        kernel = kernel_specs.get_kernel(self.kernel[0])

        if self.trans_size:
            trans_kernel = _get_update_image_trans_kernel(self.dtype_str)

        for r in range(repeat):

            if self.trans_size:
                trans_kernel.prepared_async_call(*self.trans_args)

            if self.zero:
                drv.memset_d32_async(*self.zero_args)

            kernel.prepared_async_call(*self.kernel[1:])

            if self.convert_args:
                _fp_convert(*self.convert_args)

        if unbind:
            self.zero_args = self.convert_args = None
            self.kernel[3:8] = (None,) * 5

    def __str__(self):
        N, C, K, H, W, P, Q, pad_h, pad_w = self.params
        return "%s NCK:(%3d,%3d,%3d) HW:(%3d,%3d)" % (self.kernel[0], N, C, K, H, W)


class XpropWinograd_4x4_3x3(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 H, W, P, Q,
                 pad_h, pad_w,
                 relu, bsum):

        super(XpropWinograd_4x4_3x3, self).__init__(lib, dtype)

        itemsize    = self.dtype.itemsize
        kernel_name = "%s_winograd_4x4_3x3_32x32" % self.clss

        if N == 1:
            shlN = 0
        elif N < 32:
            shlN = len(bin(N-1))-2
        else:
            shlN = 5

        shlY, shlX, maskY, shrY, maskX, shrX, maskN, supY, supX = {
            0 : (4, 5, 0x18, 3, 0x07, 0, 0x00, 0x203, 0x300), # 4x8  yyxxx
            1 : (4, 4, 0x18, 3, 0x06, 1, 0x01, 0x203, 0x201), # 4x4  yyxxn
            2 : (3, 4, 0x10, 4, 0x0c, 2, 0x03, 0x104, 0x202), # 2x4  yxxnn
           #3 : (3, 3, 0x10, 4, 0x08, 3, 0x07, 0x104, 0x103), # 2x2  yxnnn
            3 : (2, 4, 0x00, 0, 0x18, 3, 0x07, 0x000, 0x203), # 1x4  xxnnn
            4 : (2, 3, 0x00, 0, 0x10, 4, 0x0f, 0x000, 0x104), # 1x2  xnnnn
            5 : (2, 2, 0x00, 0, 0x00, 0, 0x1f, 0x000, 0x000), # 1x1  nnnnn
        }.get(shlN)

        R, S = 3, 3
        GYS  = _ceil_div(P, 1 << shlY)
        GXS  = _ceil_div(Q, 1 << shlX)
        GN   = _ceil_div(N, 1 << shlN)
        GK   = _ceil_div(K, 32)
        GYS2 = GYS // 2
        GXS2 = GXS  * 2
        k    = _closest_divisor(GK, 4)
        Xk   = GXS*k
        YXk  = GYS*Xk

        magic_GXS2 = _magic64(GXS2)
        magic_YXk  = _magic64(YXk)
        magic_Xk   = _magic64(Xk)
        magic_k    = _magic32(Xk, k)

        self.image_size   = itemsize*1152*C*GXS*GYS*GN
        self.image_args   = [
            ( GN, GYS*GXS, C ), (32,1,1), None, None, None,
            H, W, N, pad_h, pad_w,
            GXS, GYS2, GXS2, magic_GXS2[0], magic_GXS2[1],
            shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN,
            H*W*N, W*N, GYS*GXS*C*1152, GXS*C*1152, C*1152]

        #print GYS, GXS, GYS*GXS*GK*GN, GYS*GXS*GK*GN/24.0, k

        self.kernel = [
            kernel_name, (GYS*GXS*GK, GN, 1), (640, 1, 1), None,
            None, None, None, None, None, None, None ]
        self.kernel.extend( _flatten([
            C, K, N, YXk, Xk, k, magic_YXk, magic_Xk, magic_k,
            C*1152, GXS*C*1152, GYS*GXS*C*1152,
            P, Q, Q*N, P*Q*N, N*itemsize, Q*N*itemsize, Q*N*3*itemsize,
            max(P*Q*N - Q*N*3, 0)*itemsize, (P*Q*N*15 - Q*N*3)*itemsize,
            maskN, shlX, shlY, supX, supY ]))

        lib.set_scratch_size(self.image_size, self.filter_size)

        self.flags  = (relu and 2) + (bsum and 4)
        self.mode = ""

    def bind_params(self, I, F, O, alpha, beta, bsum, flags=0):

        assert I.dtype == F.dtype == O.dtype

        bsum_gpudata, flags = self.init_bsum(bsum, flags)

        # Warning: beta and bsum mutually exclusive in 4x4 kernels
        if beta:
            self.mode = "_beta"
            #assert not bsum
        elif bsum:
            self.mode = "_bsum"
        else:
            self.mode = ""

        image_temp = self.lib.scratch_buffer(self.image_size)
        self.image_args[2:5] = (self.lib.stream, image_temp, I.gpudata)

        filter_temp = self.lib.scratch_buffer_offset(self.filter_size)
        self.filter_args[2:5] = (self.lib.stream, filter_temp, F.gpudata)

        self.kernel[3:11] = (self.lib.stream, bsum_gpudata, O.gpudata, image_temp, filter_temp,
                             alpha, beta, flags)

    def execute(self, repeat=1, unbind=True):

        image_kernel  = _get_xprop_image_trans_4x4_kernel(self.dtype_str)
        filter_kernel = self.filter_func(self.dtype_str)
        kernel        = kernel_specs.get_kernel(self.kernel[0] + self.mode)

        for r in range(repeat):

            image_kernel.prepared_async_call(*self.image_args)
            filter_kernel.prepared_async_call(*self.filter_args)

            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)

            kernel.prepared_async_call(*self.kernel[1:])

        if unbind:
            self.bsum_zero = None
            self.kernel[3:11] = (None,) * 8
            self.image_args[2:5] = (None,) * 3
            self.filter_args[2:5] = (None,) * 3

class FpropWinograd_4x4_3x3(XpropWinograd_4x4_3x3):

    def __init__(self, lib, dtype,
                 N, C, K,
                 H, W, P, Q,
                 pad_h, pad_w,
                 relu, bsum):

        R, S = 3, 3
        GK   = _ceil_div(K, 32)

        self.filter_func = _get_fprop_filter_trans_4x4_kernel
        self.filter_size = dtype.itemsize*1152*C*GK
        self.filter_args = [
            (GK,C,1), (32,1,1), None, None, None,
            R*S*K, S*K, S*K*2, K, C*1152]

        super(FpropWinograd_4x4_3x3, self).__init__(
                 lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w, relu, bsum)

    def __str__(self):
        return "FpropWinograd " + self.kernel[0]

class BpropWinograd_4x4_3x3(XpropWinograd_4x4_3x3):

    def __init__(self, lib, dtype,
                 N, C, K,
                 H, W, P, Q,
                 pad_h, pad_w,
                 relu, bsum):

        R, S = 3, 3
        GC32 = _ceil_div(C, 32)
        GC16 = _ceil_div(GC32*32, 16)
        GK16 = _ceil_div(K, 16)

        self.filter_func = _get_bprop_filter_trans_4x4_kernel
        self.filter_size = dtype.itemsize*1152*K*GC32
        self.filter_args = [
            (GK16,GC16,1), (256,1,1), None, None, None,
            R*S*K, S*K, S*K*2, K, C, K*1152]

        super(BpropWinograd_4x4_3x3, self).__init__(
                 lib, dtype, N, K, C, P, Q, H, W, 2-pad_h, 2-pad_w, relu, bsum)

    def __str__(self):
        return "BpropWinograd " + self.kernel[0]

class UpdateWinograd_3x3_4x4(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 H, W, P, Q,
                 pad_h, pad_w):

        super(UpdateWinograd_3x3_4x4, self).__init__(lib, dtype)

        SMs = _get_sm_count()

        self.autotune_key = [str(x) for x in ("update_3x3_4x4",
           SMs, 0, dtype.itemsize, N, C, K, H, W, P, Q)]
        self.autotune_db_file = os.path.join(lib.cache_dir, "autotune.db")

        self.params = (N, C, K, H, W, P, Q, pad_h, pad_w)
        self.init(self.params)

        self.output_size = (self.determ_size or (dtype.type != np.float32 and 9*C*K)) * 4

        lib.set_scratch_size(self.image_size, self.delta_size, self.output_size)

        # allow for .5 seconds worth of warmup when autotuning
        # assume 10 Tflops on 24 SMs
        self.warmup = min(max(int(5e12 / (P*Q*K*N*C*9*2.0) * (SMs / 24.0)), 1), 1000)

    def init(self, params, autotune=False):

        N, C, K, H, W, P, Q, pad_h, pad_w = self.params
        itemsize = self.dtype.itemsize
        R, S = 3, 3

        if N == 1:
            shlN = 0
        elif N < 16:
            shlN = len(bin(N-1))-2
        else:
            shlN = 4

        GC32 = _ceil_div(C, 32)
        GK32 = _ceil_div(K, 32)
        GC16 = _ceil_div(GC32*32, 16)
        GK16 = _ceil_div(GK32*32, 16)
        GY   = _ceil_div(P, 4)
        GX   = _ceil_div(Q, 4)

        # maximize points of GX in superblock for large GX (most contiguous access)
        if shlN > 1 or GX >= 32:
            shlY, shlX, maskY, shrY, maskX, shrX, maskN = {
                0 : (2, 6, 0x0, 0, 0xf, 0, 0x0), # 1x16 xxxx
                1 : (2, 5, 0x0, 0, 0xe, 1, 0x1), # 1x8  xxxn
                2 : (2, 4, 0x0, 0, 0xc, 2, 0x3), # 1x4  xxnn
                3 : (2, 3, 0x0, 0, 0x8, 3, 0x7), # 1x2  xnnn
                4 : (2, 2, 0x0, 0, 0x0, 0, 0xf), # 1x1  nnnn
            }.get(shlN)
        # As GX gets smaller, add points of y
        elif shlN > 0 or GX >= 16:
            shlY, shlX, maskY, shrY, maskX, shrX, maskN = {
                0 : (3, 5, 0x8, 3, 0x7, 0, 0x0), # 2x8  yxxx
                1 : (3, 4, 0x8, 3, 0x6, 1, 0x1), # 2x4  yxxn
            }.get(shlN)
        # for smallest dimensions, make the superblock square
        else:
            shlY, shlX, maskY, shrY, maskX, shrX, maskN = {
                0 : (4, 4, 0xc, 2, 0x3, 0, 0x0), # 4x4  yyxx
            }.get(shlN)

        GYS     = _ceil_div(P, 1 << shlY)
        GXS     = _ceil_div(Q, 1 << shlX)
        GN16    = _ceil_div(N, 1 << shlN)
        GYS2    = GYS // 2
        GXS2    = GXS  * 2
        groupC  = 2
        groupK  = 2
        groupN  = 1 if GN16 & 1 else 2
        shift_c = groupC - 1
        shift_k = groupK - 1
        shift_n = groupN - 1

        X2cn       = GXS2*groupC*groupN
        Xkn        = GXS*groupK*groupN
        magic_X2cn = _magic64(X2cn)
        magic_Xkn  = _magic64(Xkn)

        self.image_size = GC32*GY*GX*N*1152*itemsize
        self.image_args = [
            ( GYS*GXS*groupC*groupN, GN16//groupN, GC16//groupC ), (256,1,1), None, None, None,
            C, H, W, N, pad_h, pad_w,
            GY, GX, GXS, GYS2, X2cn, magic_X2cn[0], magic_X2cn[1], shift_c, shift_n,
            shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN,
            H*W*N, W*N, GY*GX*N*1152, GX*N*1152, N*1152]

        self.delta_size = GK32*GY*GX*N*1152*itemsize
        self.delta_args = [
            ( GYS*GXS*groupK*groupN, GN16//groupN, GK16//groupK ), (256,1,1), None, None, None,
            K, P, Q, N, GY, GX,
            Xkn, magic_Xkn[0], magic_Xkn[1], shift_k, shift_n,
            shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN,
            P*Q*N, Q*N, GY*GX*N*1152, GX*N*1152, N*1152]

        Gc = _closest_divisor(GC32, 4)
        Gk = _closest_divisor(GK32, 4)
        GC = GC32 // Gc
        GK = GK32 // Gk
        kc = Gk*Gc
        YXN  = GY*GX*N
        YXN2 = self.YXN2 = _ceil_div(YXN, 2)

        self.maxYXN2 = max(1, YXN2 // N)

        if autotune:
            strideYXN = autotune
        else:
            autotune_db  = shelve.open(self.autotune_db_file)
            autotune_key = " ".join(self.autotune_key)

            if autotune_key in autotune_db:
                strideYXN = autotune_db[autotune_key]
                #print strideYXN, autotune_key
                self.initialized = True
            else:
                strideYXN = self.maxYXN2
                self.initialized = False

            autotune_db.close()

        self.blocksCK = GC32 * GK32
        magic_sYXN    = _magic64(strideYXN)
        magic_kc      = _magic64(kc)
        magic_c       = _magic32(kc, Gc)

        # If output grid is 1, don't use atomics.  Kernel is deterministic by default
        if strideYXN == 1:
            determ = "D"
            self.determ_size  = 0
            self.determ_shape = False
            self.zero         = False
        elif self.lib.deterministic:
            determ = "D"
            self.determ_size  = strideYXN * C*R*S*K
            self.determ_shape = (strideYXN, C*R*S*K)
            self.zero         = False
        else:
            determ = ""
            self.determ_size  = 0
            self.determ_shape = False
            self.zero         = True

        kernel_name = "%s_winograd%s_3x3_4x4_32x32" % (self.clss, determ)

        #print strideYXN*Gk*Gc, strideYXN, Gk, Gc, strideYXN*Gk*Gc*GC*GK, YXN2//strideYXN

        self.kernel = [
            kernel_name, (strideYXN*Gk*Gc, GC, GK), (640, 1, 1), None,
            None, None, None, None]

        self.kernel.extend( _flatten([
            K, C, Gk, Gc, kc, magic_kc, magic_c, YXN2, strideYXN, magic_sYXN,
            strideYXN*2*1152*itemsize, YXN, YXN*1152, R*S*K, C*R*S*K,
            K*4, S*K*4, (R*S*K*15 - S*K*2)*4 ]))

    def autotune(self, I, E, O):

        autotune_key = " ".join(self.autotune_key)
        #print "autotune: ", self.autotune_key

        # Only need to do warmup once
        if not self.lib.warmup:
            self.lib.warmup = True
            # warmup  with a conservative set of params
            self.init(self.params, autotune=self.maxYXN2)
            self.bind_params(I, E, O, 1.0)
            self.execute(repeat=self.warmup, unbind=False)

        start, stop = self.lib.get_events()
        block_slots = _get_sm_count()
        small_set   = self.YXN2 < 512
        YXN2        = float(self.YXN2)
        results     = []
        sys.stdout.write("Autotune " + str(self))
        progress = 0
        for threshold in (True, False):
            for strideYXN in range(1, self.maxYXN2 + 1):
                if progress % 32 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                progress += 1
                # minimal occupancy filter
                blocks  = self.blocksCK * strideYXN
                # gemm loop count filter
                depth   = YXN2 / strideYXN

                filters = blocks >= block_slots and blocks <= 24*block_slots and depth >= 32.0

                # In case we filter out all settings, run though the loops again
                # this time looking only at settings that didn't pass.
                if small_set or (threshold and filters) or (not threshold and not filters):

                    self.init(self.params, autotune=strideYXN)
                    self.bind_params(I, E, O, 1.0)
                    start.record(stream=self.lib.stream)
                    self.execute(repeat=2, unbind=False)
                    stop.record(stream=self.lib.stream)
                    stop.synchronize()
                    msecs = stop.time_since(start) / 2.0
                    results.append((msecs, strideYXN))

                    #print strideYXN, msecs, blocks, round(depth,1)

                # else:
                #     print strideYXN, blocks, round(depth,1)

            # if we got any results, no need to disable the filter
            if len(results) > 0:
                break
        sys.stdout.write('\n')

        results.sort()
        strideYXN = results[0][1]
        # for res in results[0:10]:
        #     print res

        autotune_db = shelve.open(self.autotune_db_file)
        autotune_db[autotune_key] = strideYXN

        # add a copy if this layer has small strides
        # deterministic vs non-determ should make no speed difference here
        if strideYXN <= 8:
            self.autotune_key[2] = str(1 - int(self.autotune_key[2]))
            autotune_key = " ".join(self.autotune_key)
            autotune_db[autotune_key] = strideYXN

        autotune_db.close()

        self.init(self.params, autotune=strideYXN)

    def bind_params(self, I, E, O, alpha):

        assert I.dtype == E.dtype

        if not self.initialized:
            self.initialized = True
            self.autotune(I, E, O)

        if O.dtype.type is not np.float32 or self.determ_size:

            updat_temp = self.lib.scratch_buffer(self.output_size)
            image_temp = self.lib.scratch_buffer_offset(self.image_size)
            delta_temp = self.lib.scratch_buffer_offset(self.delta_size)

            self.convert_args = [ updat_temp, "f4", O, self.determ_shape ]
        else:
            updat_temp = O.gpudata
            image_temp = self.lib.scratch_buffer(self.image_size)
            delta_temp = self.lib.scratch_buffer_offset(self.delta_size)

            self.convert_args = False

        self.image_args[2:5] = (self.lib.stream, image_temp, I.gpudata)
        self.delta_args[2:5] = (self.lib.stream, delta_temp, E.gpudata)

        if self.zero:
            self.zero_args = [updat_temp, 0, O.size, self.lib.stream]

        self.kernel[3:8] = (self.lib.stream, updat_temp, image_temp, delta_temp, alpha)

    def execute(self, repeat=1, unbind=True):

        kernel = kernel_specs.get_kernel(self.kernel[0])

        image_kernel = _get_update_image_trans_4x4_kernel(self.dtype_str)
        delta_kernel = _get_update_delta_trans_4x4_kernel(self.dtype_str)

        for r in range(repeat):

            if self.zero:
                drv.memset_d32_async(*self.zero_args)

            image_kernel.prepared_async_call(*self.image_args)
            delta_kernel.prepared_async_call(*self.delta_args)

            kernel.prepared_async_call(*self.kernel[1:])

            if self.convert_args:
                _fp_convert(*self.convert_args)

        if unbind:
            self.zero_args = self.convert_args = None
            self.kernel[3:8]     = (None,) * 5
            self.image_args[2:5] = (None,) * 3
            self.delta_args[2:5] = (None,) * 3

    def __str__(self):
        N, C, K, H, W, P, Q, pad_h, pad_w = self.params
        return "%s NCK:(%3d,%3d,%3d) HW:(%3d,%3d)" % (self.kernel[0], N, C, K, H, W)


@context_dependent_memoize
def _get_fprop_filter_trans_kernel(dtype):

    code = r"""
%(common)s

__global__ void fprop_filter_trans(%(type4)s* T, const %(type)s* F, int RSK, int SK, int SK2, int K)
{
    extern %(type)s  __shared__ share[];
    extern %(type4)s __shared__ share4[];

    int tid  = threadIdx.x;
    int blkK = blockIdx.x;
    int c    = blockIdx.y;
    int k    = (blkK<<5) + tid;

    bool valid_k = k < K;

    int f_r0s0 = c*RSK  + k;
    int f_r0s1 = f_r0s0 + K;
    int f_r0s2 = f_r0s1 + K;

    int f_r2s0 = f_r0s0 + SK2;
    int f_r2s1 = f_r0s1 + SK2;
    int f_r2s2 = f_r0s2 + SK2;

    int f_r1s0 = f_r0s0 + SK;
    int f_r1s1 = f_r0s1 + SK;
    int f_r1s2 = f_r0s2 + SK;

    float r0s0 = valid_k ? %(cvt_in)s(__ldg(F + f_r0s0)) : 0.0f;
    float r0s1 = valid_k ? %(cvt_in)s(__ldg(F + f_r0s1)) : 0.0f;
    float r0s2 = valid_k ? %(cvt_in)s(__ldg(F + f_r0s2)) : 0.0f;

    float r2s0 = valid_k ? %(cvt_in)s(__ldg(F + f_r2s0)) : 0.0f;
    float r2s1 = valid_k ? %(cvt_in)s(__ldg(F + f_r2s1)) : 0.0f;
    float r2s2 = valid_k ? %(cvt_in)s(__ldg(F + f_r2s2)) : 0.0f;

    float r1s0 = valid_k ? %(cvt_in)s(__ldg(F + f_r1s0)) : 0.0f;
    float r1s1 = valid_k ? %(cvt_in)s(__ldg(F + f_r1s1)) : 0.0f;
    float r1s2 = valid_k ? %(cvt_in)s(__ldg(F + f_r1s2)) : 0.0f;

    float temp00 = __fmul_rn(r0s1, 0.5f);
    float temp01 = __fadd_rn(r0s0, r0s2);
    float F01    = __fmaf_rn(temp01, 0.5f,  temp00);
    float F02    = __fmaf_rn(temp01, 0.5f, -temp00);
    share[tid + 32*0] = %(cvt_out)s(r0s0);
    share[tid + 32*1] = %(cvt_out)s(F01);
    share[tid + 32*2] = %(cvt_out)s(F02);
    share[tid + 32*3] = %(cvt_out)s(r0s2);
    float temp02 = __fadd_rn(r2s0, r2s2);
    float temp08 = __fmul_rn(r2s1, 0.5f);
    float F13    = __fmaf_rn(temp02, 0.5f,  temp08);
    float F14    = __fmaf_rn(temp02, 0.5f, -temp08);
    share[tid + 32*12] = %(cvt_out)s(r2s0);
    share[tid + 32*13] = %(cvt_out)s(F13);
    share[tid + 32*14] = %(cvt_out)s(F14);
    share[tid + 32*15] = %(cvt_out)s(r2s2);
    float temp10 = __fadd_rn(temp01, temp02);
    float temp05 = __fadd_rn(r0s1,   r2s1);
    float temp07 = __fadd_rn(r1s0,   r1s2);
    float temp09 = __fmul_rn(r1s1,   0.25f);
    float temp11 = __fadd_rn(temp10,  temp05);
    float temp14 = __fadd_rn(temp10, -temp05);
    float temp13 = __fmaf_rn(temp07, 0.25f,  temp09);
    float temp15 = __fmaf_rn(temp07, 0.25f, -temp09);
    float F05    = __fmaf_rn(temp11, 0.25f,  temp13);
    float F09    = __fmaf_rn(temp11, 0.25f, -temp13);
    float F06    = __fmaf_rn(temp14, 0.25f,  temp15);
    float F10    = __fmaf_rn(temp14, 0.25f, -temp15);
    share[tid + 32* 5] = %(cvt_out)s(F05);
    share[tid + 32* 9] = %(cvt_out)s(F09);
    share[tid + 32* 6] = %(cvt_out)s(F06);
    share[tid + 32*10] = %(cvt_out)s(F10);
    float temp03 = __fmul_rn(r1s0, 0.5f);
    float temp06 = __fadd_rn(r0s0, r2s0);
    float temp04 = __fmul_rn(r1s2, 0.5f);
    float F04    = __fmaf_rn(temp06, 0.5f,  temp03);
    float F08    = __fmaf_rn(temp06, 0.5f, -temp03);
    share[tid + 32*4] = %(cvt_out)s(F04);
    share[tid + 32*8] = %(cvt_out)s(F08);
    float temp12 = __fadd_rn(r0s2, r2s2);
    float F07    = __fmaf_rn(temp12, 0.5f,  temp04);
    float F11    = __fmaf_rn(temp12, 0.5f, -temp04);
    share[tid + 32* 7] = %(cvt_out)s(F07);
    share[tid + 32*11] = %(cvt_out)s(F11);

    %(type4)s batch0 = share4[tid +  0];
    %(type4)s batch1 = share4[tid + 32];
    %(type4)s batch2 = share4[tid + 64];
    %(type4)s batch3 = share4[tid + 96];

    int offset = c*gridDim.x*128 + blkK*128 + tid;

    T[offset +  0] = batch0;
    T[offset + 32] = batch1;
    T[offset + 64] = batch2;
    T[offset + 96] = batch3;
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    code = code % {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "type4"   : _ew_types[dtype]["type4"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }

    module = SourceModule(code)
    kernel = module.get_function("fprop_filter_trans")
    kernel.prepare("PPIIII")
    return kernel

@context_dependent_memoize
def _get_bprop_filter_trans_kernel(dtype):

    code = r"""
%(common)s

__global__ void bprop_filter_trans(
    %(type)s* T, const %(type)s* F,
    int RSK, int SK, int SK2, int K, int C, int cBlks)
{
    // Add padding to avoid all but 1 shared bank conflict on loads
    %(type)s __shared__ share[16][16*16 + 2];

    int tid  = threadIdx.x;
    int blkK = blockIdx.x;
    int blkC = blockIdx.y;

    int cs = tid >> 4;
    int ks = tid & 15;

    int c = blkC * 16 + cs;
    int k = blkK * 16 + ks;

    bool valid = c < C && k < K;

    // Miror RS
    int f_r2s2 = c*RSK  + k;
    int f_r2s1 = f_r2s2 + K;
    int f_r2s0 = f_r2s1 + K;

    int f_r1s2 = f_r2s2 + SK;
    int f_r1s1 = f_r2s1 + SK;
    int f_r1s0 = f_r2s0 + SK;

    int f_r0s2 = f_r2s2 + SK2;
    int f_r0s1 = f_r2s1 + SK2;
    int f_r0s0 = f_r2s0 + SK2;

    float r0s0 = valid ? %(cvt_in)s(__ldg(F + f_r0s0)) : 0.0f;
    float r0s1 = valid ? %(cvt_in)s(__ldg(F + f_r0s1)) : 0.0f;
    float r0s2 = valid ? %(cvt_in)s(__ldg(F + f_r0s2)) : 0.0f;

    float r2s0 = valid ? %(cvt_in)s(__ldg(F + f_r2s0)) : 0.0f;
    float r2s1 = valid ? %(cvt_in)s(__ldg(F + f_r2s1)) : 0.0f;
    float r2s2 = valid ? %(cvt_in)s(__ldg(F + f_r2s2)) : 0.0f;

    float r1s0 = valid ? %(cvt_in)s(__ldg(F + f_r1s0)) : 0.0f;
    float r1s1 = valid ? %(cvt_in)s(__ldg(F + f_r1s1)) : 0.0f;
    float r1s2 = valid ? %(cvt_in)s(__ldg(F + f_r1s2)) : 0.0f;

    float temp00 = __fmul_rn(r0s1, 0.5f);
    float temp01 = __fadd_rn(r0s0, r0s2);
    float F01    = __fmaf_rn(temp01, 0.5f,  temp00);
    float F02    = __fmaf_rn(temp01, 0.5f, -temp00);
    share[ks][cs + 16*0] = %(cvt_out)s(r0s0);
    share[ks][cs + 16*1] = %(cvt_out)s(F01);
    share[ks][cs + 16*2] = %(cvt_out)s(F02);
    share[ks][cs + 16*3] = %(cvt_out)s(r0s2);
    float temp02 = __fadd_rn(r2s0, r2s2);
    float temp08 = __fmul_rn(r2s1, 0.5f);
    float F13    = __fmaf_rn(temp02, 0.5f,  temp08);
    float F14    = __fmaf_rn(temp02, 0.5f, -temp08);
    share[ks][cs + 16*12] = %(cvt_out)s(r2s0);
    share[ks][cs + 16*13] = %(cvt_out)s(F13);
    share[ks][cs + 16*14] = %(cvt_out)s(F14);
    share[ks][cs + 16*15] = %(cvt_out)s(r2s2);
    float temp10 = __fadd_rn(temp01, temp02);
    float temp05 = __fadd_rn(r0s1,   r2s1);
    float temp07 = __fadd_rn(r1s0,   r1s2);
    float temp09 = __fmul_rn(r1s1,   0.25f);
    float temp11 = __fadd_rn(temp10,  temp05);
    float temp14 = __fadd_rn(temp10, -temp05);
    float temp13 = __fmaf_rn(temp07, 0.25f,  temp09);
    float temp15 = __fmaf_rn(temp07, 0.25f, -temp09);
    float F05    = __fmaf_rn(temp11, 0.25f,  temp13);
    float F09    = __fmaf_rn(temp11, 0.25f, -temp13);
    float F06    = __fmaf_rn(temp14, 0.25f,  temp15);
    float F10    = __fmaf_rn(temp14, 0.25f, -temp15);
    share[ks][cs + 16* 5] = %(cvt_out)s(F05);
    share[ks][cs + 16* 9] = %(cvt_out)s(F09);
    share[ks][cs + 16* 6] = %(cvt_out)s(F06);
    share[ks][cs + 16*10] = %(cvt_out)s(F10);
    float temp03 = __fmul_rn(r1s0, 0.5f);
    float temp06 = __fadd_rn(r0s0, r2s0);
    float temp04 = __fmul_rn(r1s2, 0.5f);
    float F04    = __fmaf_rn(temp06, 0.5f,  temp03);
    float F08    = __fmaf_rn(temp06, 0.5f, -temp03);
    share[ks][cs + 16*4] = %(cvt_out)s(F04);
    share[ks][cs + 16*8] = %(cvt_out)s(F08);
    float temp12 = __fadd_rn(r0s2, r2s2);
    float F07    = __fmaf_rn(temp12, 0.5f,  temp04);
    float F11    = __fmaf_rn(temp12, 0.5f, -temp04);
    share[ks][cs + 16* 7] = %(cvt_out)s(F07);
    share[ks][cs + 16*11] = %(cvt_out)s(F11);

    __syncthreads();

    // now make c contiguous
    cs = tid & 15;
    ks = tid >> 4;

    k = blkK*16 + ks;

    if (k < K)
    {
        T += k*((cBlks+1)>>1)*512 + (blkC>>1)*512 + (blkC&1)*16 + cs;

        T[32* 0] = share[ks][cs + 16* 0];
        T[32* 1] = share[ks][cs + 16* 1];
        T[32* 2] = share[ks][cs + 16* 2];
        T[32* 3] = share[ks][cs + 16* 3];
        T[32* 4] = share[ks][cs + 16* 4];
        T[32* 5] = share[ks][cs + 16* 5];
        T[32* 6] = share[ks][cs + 16* 6];
        T[32* 7] = share[ks][cs + 16* 7];
        T[32* 8] = share[ks][cs + 16* 8];
        T[32* 9] = share[ks][cs + 16* 9];
        T[32*10] = share[ks][cs + 16*10];
        T[32*11] = share[ks][cs + 16*11];
        T[32*12] = share[ks][cs + 16*12];
        T[32*13] = share[ks][cs + 16*13];
        T[32*14] = share[ks][cs + 16*14];
        T[32*15] = share[ks][cs + 16*15];
    }
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    code = code % {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }
    #print code

    module = SourceModule(code)
    kernel = module.get_function("bprop_filter_trans")
    kernel.prepare("PPIIIIII")
    return kernel


@context_dependent_memoize
def _get_update_image_trans_kernel(dtype):

    code = r"""
#include <stdio.h>

%(common)s

__device__ __forceinline__ int div64(int value, int magic, int shift)
{
    int result;
    // if the divisor is a power of two the magic will be 1 and it's just a simple right shift
    if (magic == 1)
        result = value >> shift;
    // Otherwise multiply by magic and right shift just the high bits
    else
        asm(".reg .u64 res64;\n\t"
            ".reg .u32 lo32, hi32;\n\t"
            "mul.wide.u32 res64, %%1, %%2;\n\t"
            "mov.b64 {lo32, hi32}, res64;\n\t"
            "shr.b32 %%0, hi32, %%3;\n\t"
            : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}

__global__ void update_image_trans(
    %(type)s* T, const %(type)s* I,
    int C, int Y, int X, int N, int pad_y, int pad_x,
    int GY, int GX, int GXS, int GYS2,
    int X2cn, int magic_X2cn, int shift_X2cn, int shift_c, int shift_n,
    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,
    int YXN, int XN, int GYGXN512, int GXN512, int N512)
{
    // Add padding to avoid all but 1 shared bank conflict on loads
    %(type)s __shared__ share[16][16*16 + 2];

    int tid        = threadIdx.x;
    int blk_YXcn   = blockIdx.x;
    int blk_N      = blockIdx.y;
    int blk_C      = blockIdx.z;

    // unpack y,x,c,n from blockIdx.x
    int gy2     = div64(blk_YXcn, magic_X2cn, shift_X2cn);
    int blk_Xcn = blk_YXcn - gy2*X2cn;

    int shift_cn = shift_c + shift_n;

    int gx2    = blk_Xcn >> shift_cn;
    int blk_cn = blk_Xcn - (gx2 << shift_cn);

    int blk_c = blk_cn >> shift_n;
    int blk_n = blk_cn - (blk_c << shift_n);

    int blkN = (blk_N << shift_n) + blk_n;
    int blkC = (blk_C << shift_c) + blk_c;

    // Implement a square wave block id remapping
    // (for all but last row (if odd number of rows))
    int gy = gy2 << 1;
    int gx = gx2;
    if (gy2 != GYS2)
    {
        gy += (gx2 & 1) ^ ((gx2 & 2) >> 1);
        gx  = gx2 >> 1;
    }
    // Scan backwards on odd rows
    if (gy2 & 1)
        gx = GXS - gx - 1;

    int ns = tid & 15;
    int cs = tid >> 4;

    // Super block YXN coordinates
    int y0 = (gy << shlY) + (((ns & maskY) >> shrY) << 1) - pad_y;
    int x0 = (gx << shlX) + (((ns & maskX) >> shrX) << 1) - pad_x;
    int n  = (blkN << shlN) + (ns & maskN);
    int c  = (blkC << 4) + cs;

    bool valid = c < C && n < N;

    int x1 = x0 + 1;
    int x2 = x0 + 2;
    int x3 = x0 + 3;

    bool x0in = x0 >= 0 && x0 < X && valid;
    bool x1in = x1 >= 0 && x1 < X && valid;
    bool x2in = x2 >= 0 && x2 < X && valid;
    bool x3in = x3 >= 0 && x3 < X && valid;

    int y1 = y0 + 1;
    int y2 = y0 + 2;
    int y3 = y0 + 3;

    bool y0in = y0 >= 0 && y0 < Y;
    bool y1in = y1 >= 0 && y1 < Y;
    bool y2in = y2 >= 0 && y2 < Y;
    bool y3in = y3 >= 0 && y3 < Y;

    const %(type)s* Iy0x0 = I + c*YXN + y0*XN + x0*N + n;
    const %(type)s* Iy0x1 = Iy0x0 + N;
    const %(type)s* Iy0x2 = Iy0x1 + N;
    const %(type)s* Iy0x3 = Iy0x2 + N;

    float y0x0 = y0in && x0in ? %(cvt_in)s(__ldg(Iy0x0)) : 0.0f;
    float y0x1 = y0in && x1in ? %(cvt_in)s(__ldg(Iy0x1)) : 0.0f;
    float y0x2 = y0in && x2in ? %(cvt_in)s(__ldg(Iy0x2)) : 0.0f;
    float y0x3 = y0in && x3in ? %(cvt_in)s(__ldg(Iy0x3)) : 0.0f;

    const %(type)s* Iy1x0 = Iy0x0 + XN;
    const %(type)s* Iy1x1 = Iy1x0 + N;
    const %(type)s* Iy1x2 = Iy1x1 + N;
    const %(type)s* Iy1x3 = Iy1x2 + N;

    float y1x0 = y1in && x0in ? %(cvt_in)s(__ldg(Iy1x0)) : 0.0f;
    float y1x1 = y1in && x1in ? %(cvt_in)s(__ldg(Iy1x1)) : 0.0f;
    float y1x2 = y1in && x2in ? %(cvt_in)s(__ldg(Iy1x2)) : 0.0f;
    float y1x3 = y1in && x3in ? %(cvt_in)s(__ldg(Iy1x3)) : 0.0f;

    const %(type)s* Iy2x0 = Iy1x0 + XN;
    const %(type)s* Iy2x1 = Iy2x0 + N;
    const %(type)s* Iy2x2 = Iy2x1 + N;
    const %(type)s* Iy2x3 = Iy2x2 + N;

    float y2x0 = y2in && x0in ? %(cvt_in)s(__ldg(Iy2x0)) : 0.0f;
    float y2x1 = y2in && x1in ? %(cvt_in)s(__ldg(Iy2x1)) : 0.0f;
    float y2x2 = y2in && x2in ? %(cvt_in)s(__ldg(Iy2x2)) : 0.0f;
    float y2x3 = y2in && x3in ? %(cvt_in)s(__ldg(Iy2x3)) : 0.0f;

    const %(type)s* Iy3x0 = Iy2x0 + XN;
    const %(type)s* Iy3x1 = Iy3x0 + N;
    const %(type)s* Iy3x2 = Iy3x1 + N;
    const %(type)s* Iy3x3 = Iy3x2 + N;

    float y3x0 = y3in && x0in ? %(cvt_in)s(__ldg(Iy3x0)) : 0.0f;
    float y3x1 = y3in && x1in ? %(cvt_in)s(__ldg(Iy3x1)) : 0.0f;
    float y3x2 = y3in && x2in ? %(cvt_in)s(__ldg(Iy3x2)) : 0.0f;
    float y3x3 = y3in && x3in ? %(cvt_in)s(__ldg(Iy3x3)) : 0.0f;

    float A0  = y0x0 - y2x0;
    float A1  = y0x1 - y2x1;
    float A2  = y0x2 - y2x2;
    float A3  = y0x3 - y2x3;
    float B0  = y1x0 + y2x0;
    float B1  = y1x1 + y2x1;
    float B2  = y1x2 + y2x2;
    float B3  = y1x3 + y2x3;
    float C0  = y2x0 - y1x0;
    float C1  = y2x1 - y1x1;
    float C2  = y2x2 - y1x2;
    float C3  = y2x3 - y1x3;
    float D0  = y3x0 - y1x0;
    float D1  = y3x1 - y1x1;
    float D2  = y3x2 - y1x2;
    float D3  = y3x3 - y1x3;

    share[ns][cs + 16* 0] = %(cvt_out)s(A0 - A2);
    share[ns][cs + 16* 1] = %(cvt_out)s(A1 + A2);
    share[ns][cs + 16* 2] = %(cvt_out)s(A2 - A1);
    share[ns][cs + 16* 3] = %(cvt_out)s(A3 - A1);
    share[ns][cs + 16* 4] = %(cvt_out)s(B0 - B2);
    share[ns][cs + 16* 5] = %(cvt_out)s(B1 + B2);
    share[ns][cs + 16* 6] = %(cvt_out)s(B2 - B1);
    share[ns][cs + 16* 7] = %(cvt_out)s(B3 - B1);
    share[ns][cs + 16* 8] = %(cvt_out)s(C0 - C2);
    share[ns][cs + 16* 9] = %(cvt_out)s(C1 + C2);
    share[ns][cs + 16*10] = %(cvt_out)s(C2 - C1);
    share[ns][cs + 16*11] = %(cvt_out)s(C3 - C1);
    share[ns][cs + 16*12] = %(cvt_out)s(D0 - D2);
    share[ns][cs + 16*13] = %(cvt_out)s(D1 + D2);
    share[ns][cs + 16*14] = %(cvt_out)s(D2 - D1);
    share[ns][cs + 16*15] = %(cvt_out)s(D3 - D1);

    __syncthreads();

    // now make c contiguous
    cs = tid & 15;
    ns = tid >> 4;

    // apply the super block to just grid coordinates this time
    gy = (gy << (shlY-1)) + ((ns & maskY) >> shrY);
    gx = (gx << (shlX-1)) + ((ns & maskX) >> shrX);
    n  = (blkN << shlN)   + (ns  & maskN);

    if (n < N && gy < GY && gx < GX)
    {
        // output dim: (C32,GY,GX,N,c32)
        // where c32 is the 512 element transform data for 32 values of c
        // We group two blkC's to form each transform row

        T += (blkC >> 1)*GYGXN512 + gy*GXN512 + gx*N512 + n*512 + (blkC & 1)*16 + cs;

        #pragma unroll
        for (int i = 0; i < 16; i++)
            T[32*i] = share[ns][cs + 16*i];
    }
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    code = code % {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }
    #print code

    module = SourceModule(code)
    kernel = module.get_function("update_image_trans")
    kernel.prepare("PPIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    return kernel


@context_dependent_memoize
def _get_xprop_image_trans_4x4_kernel(dtype):

    code = r"""
%(common)s

__device__ __forceinline__ int div64(int value, int magic, int shift)
{
    int result;
    // if the divisor is a power of two the magic will be 1 and it's just a simple right shift
    if (magic == 1)
        result = value >> shift;
    // Otherwise multiply by magic and right shift just the high bits
    else
        asm(".reg .u64 res64;\n\t"
            ".reg .u32 lo32, hi32;\n\t"
            "mul.wide.u32 res64, %%1, %%2;\n\t"
            "mov.b64 {lo32, hi32}, res64;\n\t"
            "shr.b32 %%0, hi32, %%3;\n\t"
            : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}

__global__ void xprop_image_trans_4x4(
    %(type)s* Out, const %(type)s* In,
    int Y, int X, int N, int pad_y, int pad_x,
    int GXS, int GYS2, int GXS2, int magic_GXS2, int shift_GXS2,
    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,
    int YXN, int XN, int GYS_GXS_C_1152, int GXS_C_1152, int C_1152)
{
    int tid   = threadIdx.x;
    int blkN  = gridDim.x - blockIdx.x - 1;
    int blkYX = gridDim.y - blockIdx.y - 1;
    int c     = gridDim.z - blockIdx.z - 1;

    // unpack y,x from blockIdx.x
    int gy2 = div64(blkYX, magic_GXS2, shift_GXS2);
    int gx2 = blkYX - gy2*GXS2;

    // Implement a square wave block id remapping
    // (for all but last row (if odd number of rows))
    int gy = gy2 << 1;
    int gx = gx2;
    if (gy2 != GYS2)
    {
        gy += (gx2 & 1) ^ ((gx2 & 2) >> 1);
        gx  = gx2 >> 1;
    }
    // Scan backwards on odd rows
    if (gy2 & 1)
        gx = GXS - gx - 1;

    // Super block YXN coordinates
    int y0 = (gy << shlY) + (((tid & maskY) >> shrY) << 2) - pad_y;
    int x0 = (gx << shlX) + (((tid & maskX) >> shrX) << 2) - pad_x;
    int n  = (blkN << shlN) + (tid & maskN);

    int out_offset = blkN*GYS_GXS_C_1152 + gy*GXS_C_1152 + gx*C_1152 + c*1152 + tid;

    bool valid = n < N;

    bool xin[6], yin[6];
    float I[6][6];

    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        xin[i] = x0 + i >= 0 && x0 + i < X && valid;
        yin[i] = y0 + i >= 0 && y0 + i < Y;
    }

    int offset = c*YXN + y0*XN + x0*N + n;

    #pragma unroll
    for (int y = 0; y < 6; y++)
    {
        if (y) offset += XN;

        #pragma unroll
        for (int x = 0; x < 6; x++)
        {
            %(type)s val = 0;
            if (yin[y] && xin[x])
                val = __ldg(In + offset + x*N);
            I[y][x] = %(cvt_in)s(val);
        }
    }

    // float T[6][6];
    // float rcp4  = 1.0f/4.0f;
    // float rcp6  = 1.0f/6.0f;
    // float rcp12 = 1.0f/12.0f;
    // float rcp24 = 1.0f/24.0f;
    // #pragma unroll
    // for (int i = 0; i < 6; i++)
    // {
    //     float t0 = __fmaf_rn(I[2][i], 4.0f, -I[4][i]) * rcp6;
    //     float t1 = __fmaf_rn(I[1][i], 4.0f, -I[3][i]) * rcp6;
    //     float t2 = (I[4][i] - I[2][i]) * rcp24;
    //     float t3 = (I[3][i] - I[1][i]) * rcp12;
    //     float t4 = __fmaf_rn(I[2][i], -5.0f, I[4][i]);
    //     float t5 = __fmaf_rn(I[3][i], -5.0f, I[5][i]);
    //     T[0][i] = __fmaf_rn(t4, rcp4, I[0][i]);
    //     T[1][i] = t0 + t1;
    //     T[2][i] = t0 - t1;
    //     T[3][i] = t2 + t3;
    //     T[4][i] = t2 - t3;
    //     T[5][i] = __fmaf_rn(I[1][i], 4.0f, t5);
    // }
    // #pragma unroll
    // for (int i = 0; i < 6; i++)
    // {
    //     float t0 = __fmaf_rn(T[i][2], 4.0f, -T[i][4]) * rcp6;
    //     float t1 = __fmaf_rn(T[i][1], 4.0f, -T[i][3]) * rcp6;
    //     float t2 = (T[i][4] - T[i][2]) * rcp24;
    //     float t3 = (T[i][3] - T[i][1]) * rcp12;
    //     float t4 = __fmaf_rn(T[i][2], -5.0f, T[i][4]);
    //     float t5 = __fmaf_rn(T[i][3], -5.0f, T[i][5]);
    //     Out[out_offset + 32*(i*6 + 0)] = %(cvt_out)s(__fmaf_rn(t4, rcp4, T[i][0]));
    //     Out[out_offset + 32*(i*6 + 1)] = %(cvt_out)s(t0 + t1);
    //     Out[out_offset + 32*(i*6 + 2)] = %(cvt_out)s(t0 - t1);
    //     Out[out_offset + 32*(i*6 + 3)] = %(cvt_out)s(t2 + t3);
    //     Out[out_offset + 32*(i*6 + 4)] = %(cvt_out)s(t2 - t3);
    //     Out[out_offset + 32*(i*6 + 5)] = %(cvt_out)s(__fmaf_rn(T[i][1], 4.0f, t5));
    // }

    float T[6][6];
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = __fmaf_rn(I[2][i], -4.0f, I[4][i]);
        float t1 = __fmaf_rn(I[1][i], -4.0f, I[3][i]);
        float t2 = I[4][i] - I[2][i];
        float t3 = I[3][i] - I[1][i];
        float t4 = __fmaf_rn(I[2][i], -5.0f, I[4][i]);
        float t5 = __fmaf_rn(I[3][i], -5.0f, I[5][i]);
        T[0][i] = __fmaf_rn(I[0][i], 4.0f, t4);
        T[1][i] = t0 + t1;
        T[2][i] = t0 - t1;
        T[3][i] = __fmaf_rn(t3,  2.0f, t2);
        T[4][i] = __fmaf_rn(t3, -2.0f, t2);
        T[5][i] = __fmaf_rn(I[1][i], 4.0f, t5);
    }
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = __fmaf_rn(T[i][2], -4.0f, T[i][4]);
        float t1 = __fmaf_rn(T[i][1], -4.0f, T[i][3]);
        float t2 = T[i][4] - T[i][2];
        float t3 = T[i][3] - T[i][1];
        float t4 = __fmaf_rn(T[i][2], -5.0f, T[i][4]);
        float t5 = __fmaf_rn(T[i][3], -5.0f, T[i][5]);
        Out[out_offset + 32*(i*6 + 0)] = %(cvt_out)s(__fmaf_rn(T[i][0], 4.0f, t4));
        Out[out_offset + 32*(i*6 + 1)] = %(cvt_out)s(t0 + t1);
        Out[out_offset + 32*(i*6 + 2)] = %(cvt_out)s(t0 - t1);
        Out[out_offset + 32*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(t3,  2.0f, t2));
        Out[out_offset + 32*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(t3, -2.0f, t2));
        Out[out_offset + 32*(i*6 + 5)] = %(cvt_out)s(__fmaf_rn(T[i][1], 4.0f, t5));
    }
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    code = code % {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "type4"   : _ew_types[dtype]["type4"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }
    # f = open("trans.cu", "w")
    # print >>f, code
    # f.close()
    # exit()

    module = SourceModule(code)
    kernel = module.get_function("xprop_image_trans_4x4")
    kernel.prepare("PPIIIIIIIIIIIIIIIIIIIIIII")
    return kernel

@context_dependent_memoize
def _get_fprop_filter_trans_4x4_kernel(dtype):

    code = r"""
%(common)s

__global__ void fprop_filter_trans_4x4(
    %(type)s* Out, const %(type)s* In,
    int RSK, int SK, int SK2, int K, int C1152)
{
    int tid  = threadIdx.x;
    int blkK = gridDim.x - blockIdx.x - 1;
    int c    = gridDim.y - blockIdx.y - 1;
    int k    = (blkK<<5) + tid;

    int out_offset = blkK*C1152 + c*1152 + tid;

    bool valid_k = k < K;

    int f_r0s0 = c*RSK  + k;
    int f_r0s1 = f_r0s0 + K;
    int f_r0s2 = f_r0s1 + K;

    int f_r1s0 = f_r0s0 + SK;
    int f_r1s1 = f_r0s1 + SK;
    int f_r1s2 = f_r0s2 + SK;

    int f_r2s0 = f_r0s0 + SK2;
    int f_r2s1 = f_r0s1 + SK2;
    int f_r2s2 = f_r0s2 + SK2;

    float I[3][3];

    I[0][0] = valid_k ? %(cvt_in)s(__ldg(In + f_r0s0)) : 0.0f;
    I[0][1] = valid_k ? %(cvt_in)s(__ldg(In + f_r0s1)) : 0.0f;
    I[0][2] = valid_k ? %(cvt_in)s(__ldg(In + f_r0s2)) : 0.0f;

    I[1][0] = valid_k ? %(cvt_in)s(__ldg(In + f_r1s0)) : 0.0f;
    I[1][1] = valid_k ? %(cvt_in)s(__ldg(In + f_r1s1)) : 0.0f;
    I[1][2] = valid_k ? %(cvt_in)s(__ldg(In + f_r1s2)) : 0.0f;

    I[2][0] = valid_k ? %(cvt_in)s(__ldg(In + f_r2s0)) : 0.0f;
    I[2][1] = valid_k ? %(cvt_in)s(__ldg(In + f_r2s1)) : 0.0f;
    I[2][2] = valid_k ? %(cvt_in)s(__ldg(In + f_r2s2)) : 0.0f;


    // float T[6][3];
    // #pragma unroll
    // for (int i = 0; i < 3; i++)
    // {
    //     float t0 = I[0][i] + I[2][i];
    //     float t1 = __fmaf_rn(I[2][i], 4.0f, I[0][i]);
    //     T[0][i] = I[0][i];
    //     T[1][i] = t0 + I[1][i];
    //     T[2][i] = t0 - I[1][i];
    //     T[3][i] = __fmaf_rn(I[1][i],  2.0f, t1);
    //     T[4][i] = __fmaf_rn(I[1][i], -2.0f, t1);
    //     T[5][i] = I[2][i];
    // }
    // #pragma unroll
    // for (int i = 0; i < 6; i++)
    // {
    //     float t0 = T[i][0] + T[i][2];
    //     float t1 = __fmaf_rn(T[i][2], 4.0f, T[i][0]);
    //     Out[out_offset + 32*(i*6 + 0)] = %(cvt_out)s(T[i][0]);
    //     Out[out_offset + 32*(i*6 + 1)] = %(cvt_out)s(t0 + T[i][1]);
    //     Out[out_offset + 32*(i*6 + 2)] = %(cvt_out)s(t0 - T[i][1]);
    //     Out[out_offset + 32*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(T[i][1],  2.0f, t1));
    //     Out[out_offset + 32*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(T[i][1], -2.0f, t1));
    //     Out[out_offset + 32*(i*6 + 5)] = %(cvt_out)s(T[i][2]);
    // }

    float rcp4  = 1.0f/4.0f;
    float rcp6  = 1.0f/6.0f;
    float rcp12 = 1.0f/12.0f;
    float rcp24 = 1.0f/24.0f;
    float T[6][3];
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float t0 = rcp6 * I[2][i];
        float t1 = __fmaf_rn(I[0][i], -rcp6, -t0);
        float t2 = __fmaf_rn(I[0][i], rcp24,  t0);
        T[0][i] = rcp4 * I[0][i];
        T[1][i] = __fmaf_rn(I[1][i], -rcp6,  t1);
        T[2][i] = __fmaf_rn(I[1][i],  rcp6,  t1);
        T[3][i] = __fmaf_rn(I[1][i],  rcp12, t2);
        T[4][i] = __fmaf_rn(I[1][i], -rcp12, t2);
        T[5][i] = I[2][i];
    }
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = rcp6 * T[i][2];
        float t1 = __fmaf_rn(T[i][0], -rcp6, -t0);
        float t2 = __fmaf_rn(T[i][0], rcp24,  t0);
        Out[out_offset + 32*(i*6 + 0)] = %(cvt_out)s(rcp4 * T[i][0]);
        Out[out_offset + 32*(i*6 + 1)] = %(cvt_out)s(__fmaf_rn(T[i][1], -rcp6,  t1));
        Out[out_offset + 32*(i*6 + 2)] = %(cvt_out)s(__fmaf_rn(T[i][1],  rcp6,  t1));
        Out[out_offset + 32*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(T[i][1],  rcp12, t2));
        Out[out_offset + 32*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(T[i][1], -rcp12, t2));
        Out[out_offset + 32*(i*6 + 5)] = %(cvt_out)s(T[i][2]);
    }

}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    code = code % {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "type4"   : _ew_types[dtype]["type4"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }

    module = SourceModule(code)
    kernel = module.get_function("fprop_filter_trans_4x4")
    kernel.prepare("PPIIIII")
    return kernel

@context_dependent_memoize
def _get_bprop_filter_trans_4x4_kernel(dtype):

    code = r"""
%(common)s

__global__ void bprop_filter_trans_4x4(
    %(type)s* O, const %(type)s* F,
    int RSK, int SK, int SK2, int K, int C, int K_1152)
{
    // Add padding to avoid all but 1 shared bank conflict on loads
    %(type)s __shared__ share[16][16*36 + 2];

    int tid  = threadIdx.x;
    int blkK = blockIdx.x;
    int blkC = blockIdx.y;

    int cs = tid >> 4;
    int ks = tid & 15;

    int c = blkC * 16 + cs;
    int k = blkK * 16 + ks;

    bool valid_ck = c < C && k < K;

    // Miror RS
    int f_r2s2 = c*RSK  + k;
    int f_r2s1 = f_r2s2 + K;
    int f_r2s0 = f_r2s1 + K;

    int f_r1s2 = f_r2s2 + SK;
    int f_r1s1 = f_r2s1 + SK;
    int f_r1s0 = f_r2s0 + SK;

    int f_r0s2 = f_r2s2 + SK2;
    int f_r0s1 = f_r2s1 + SK2;
    int f_r0s0 = f_r2s0 + SK2;

    float I[3][3];
    I[0][0] = valid_ck ? %(cvt_in)s(__ldg(F + f_r0s0)) : 0.0f;
    I[0][1] = valid_ck ? %(cvt_in)s(__ldg(F + f_r0s1)) : 0.0f;
    I[0][2] = valid_ck ? %(cvt_in)s(__ldg(F + f_r0s2)) : 0.0f;

    I[1][0] = valid_ck ? %(cvt_in)s(__ldg(F + f_r1s0)) : 0.0f;
    I[1][1] = valid_ck ? %(cvt_in)s(__ldg(F + f_r1s1)) : 0.0f;
    I[1][2] = valid_ck ? %(cvt_in)s(__ldg(F + f_r1s2)) : 0.0f;

    I[2][0] = valid_ck ? %(cvt_in)s(__ldg(F + f_r2s0)) : 0.0f;
    I[2][1] = valid_ck ? %(cvt_in)s(__ldg(F + f_r2s1)) : 0.0f;
    I[2][2] = valid_ck ? %(cvt_in)s(__ldg(F + f_r2s2)) : 0.0f;


    // float T[6][3];
    // #pragma unroll
    // for (int i = 0; i < 3; i++)
    // {
    //     float t0 = I[0][i] + I[2][i];
    //     float t1 = __fmaf_rn(I[2][i], 4.0f, I[0][i]);
    //     T[0][i] = I[0][i];
    //     T[1][i] = t0 + I[1][i];
    //     T[2][i] = t0 - I[1][i];
    //     T[3][i] = __fmaf_rn(I[1][i],  2.0f, t1);
    //     T[4][i] = __fmaf_rn(I[1][i], -2.0f, t1);
    //     T[5][i] = I[2][i];
    // }
    // #pragma unroll
    // for (int i = 0; i < 6; i++)
    // {
    //     float t0 = T[i][0] + T[i][2];
    //     float t1 = __fmaf_rn(T[i][2], 4.0f, T[i][0]);
    //     share[ks][cs + 16*(i*6 + 0)] = %(cvt_out)s(T[i][0]);
    //     share[ks][cs + 16*(i*6 + 1)] = %(cvt_out)s(t0 + T[i][1]);
    //     share[ks][cs + 16*(i*6 + 2)] = %(cvt_out)s(t0 - T[i][1]);
    //     share[ks][cs + 16*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(T[i][1],  2.0f, t1));
    //     share[ks][cs + 16*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(T[i][1], -2.0f, t1));
    //     share[ks][cs + 16*(i*6 + 5)] = %(cvt_out)s(T[i][2]);
    // }

    float rcp4  = 1.0f/4.0f;
    float rcp6  = 1.0f/6.0f;
    float rcp12 = 1.0f/12.0f;
    float rcp24 = 1.0f/24.0f;
    float T[6][3];
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float t0 = rcp6 * I[2][i];
        float t1 = __fmaf_rn(I[0][i], -rcp6, -t0);
        float t2 = __fmaf_rn(I[0][i], rcp24,  t0);
        T[0][i] = rcp4 * I[0][i];
        T[1][i] = __fmaf_rn(I[1][i], -rcp6,  t1);
        T[2][i] = __fmaf_rn(I[1][i],  rcp6,  t1);
        T[3][i] = __fmaf_rn(I[1][i],  rcp12, t2);
        T[4][i] = __fmaf_rn(I[1][i], -rcp12, t2);
        T[5][i] = I[2][i];
    }
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = rcp6 * T[i][2];
        float t1 = __fmaf_rn(T[i][0], -rcp6, -t0);
        float t2 = __fmaf_rn(T[i][0], rcp24,  t0);
        share[ks][cs + 16*(i*6 + 0)] = %(cvt_out)s(rcp4 * T[i][0]);
        share[ks][cs + 16*(i*6 + 1)] = %(cvt_out)s(__fmaf_rn(T[i][1], -rcp6,  t1));
        share[ks][cs + 16*(i*6 + 2)] = %(cvt_out)s(__fmaf_rn(T[i][1],  rcp6,  t1));
        share[ks][cs + 16*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(T[i][1],  rcp12, t2));
        share[ks][cs + 16*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(T[i][1], -rcp12, t2));
        share[ks][cs + 16*(i*6 + 5)] = %(cvt_out)s(T[i][2]);
    }

    __syncthreads();

    // now make c contiguous
    cs = tid & 15;
    ks = tid >> 4;

    k = blkK*16 + ks;

    if (k < K)
    {
        O += (blkC>>1)*K_1152 + k*1152 + (blkC&1)*16 + cs;

        #pragma unroll
        for (int i = 0; i < 36; i++)
            O[32*i] = share[ks][cs + 16*i];
    }
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    code = code % {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }
    #print code

    module = SourceModule(code)
    kernel = module.get_function("bprop_filter_trans_4x4")
    kernel.prepare("PPIIIIII")
    return kernel


@context_dependent_memoize
def _get_update_image_trans_4x4_kernel(dtype):

    code = r"""

%(common)s

__device__ __forceinline__ int div64(int value, int magic, int shift)
{
    int result;
    // if the divisor is a power of two the magic will be 1 and it's just a simple right shift
    if (magic == 1)
        result = value >> shift;
    // Otherwise multiply by magic and right shift just the high bits
    else
        asm(".reg .u64 res64;\n\t"
            ".reg .u32 lo32, hi32;\n\t"
            "mul.wide.u32 res64, %%1, %%2;\n\t"
            "mov.b64 {lo32, hi32}, res64;\n\t"
            "shr.b32 %%0, hi32, %%3;\n\t"
            : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}

__global__ void update_image_trans_4x4(
    %(type)s* O, const %(type)s* In,
    int C, int Y, int X, int N, int pad_y, int pad_x,
    int GY, int GX, int GXS, int GYS2,
    int X2cn, int magic_X2cn, int shift_X2cn, int shift_c, int shift_n,
    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,
    int YXN, int XN, int GYGXN1152, int GXN1152, int N1152)
{
    // Add padding to avoid all but 1 shared bank conflict on loads
    %(type)s __shared__ share[16][16*36 + 2];

    int tid        = threadIdx.x;
    int blk_YXcn   = blockIdx.x;
    int blk_N      = blockIdx.y;
    int blk_C      = blockIdx.z;

    // unpack y,x,c,n from blockIdx.x
    int gy2     = div64(blk_YXcn, magic_X2cn, shift_X2cn);
    int blk_Xcn = blk_YXcn - gy2*X2cn;

    int shift_cn = shift_c + shift_n;

    int gx2    = blk_Xcn >> shift_cn;
    int blk_cn = blk_Xcn - (gx2 << shift_cn);

    int blk_c = blk_cn >> shift_n;
    int blk_n = blk_cn - (blk_c << shift_n);

    int blkN = (blk_N << shift_n) + blk_n;
    int blkC = (blk_C << shift_c) + blk_c;

    // Implement a square wave block id remapping
    // (for all but last row (if odd number of rows))
    int gy = gy2 << 1;
    int gx = gx2;
    if (gy2 != GYS2)
    {
        gy += (gx2 & 1) ^ ((gx2 & 2) >> 1);
        gx  = gx2 >> 1;
    }
    // Scan backwards on odd rows
    if (gy2 & 1)
        gx = GXS - gx - 1;

    int ns = tid & 15;
    int cs = tid >> 4;

    // Super block YXN coordinates
    int y0 = (gy << shlY) + (((ns & maskY) >> shrY) << 2) - pad_y;
    int x0 = (gx << shlX) + (((ns & maskX) >> shrX) << 2) - pad_x;
    int n  = (blkN << shlN) + (ns & maskN);
    int c  = (blkC << 4) + cs;

    bool valid = c < C && n < N;

    bool xin[6], yin[6];
    float I[6][6];

    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        xin[i] = x0 + i >= 0 && x0 + i < X && valid;
        yin[i] = y0 + i >= 0 && y0 + i < Y;
    }

    int offset = c*YXN + y0*XN + x0*N + n;

    #pragma unroll
    for (int y = 0; y < 6; y++)
    {
        if (y) offset += XN;

        #pragma unroll
        for (int x = 0; x < 6; x++)
        {
            %(type)s val = 0;
            if (yin[y] && xin[x])
                val = __ldg(In + offset + x*N);
            I[y][x] = %(cvt_in)s(val);
        }
    }

    float T[6][6];
    float rcp4  = 1.0f/4.0f;
    float rcp6  = 1.0f/6.0f;
    float rcp12 = 1.0f/12.0f;
    float rcp24 = 1.0f/24.0f;

    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        //float t0 = __fmaf_rn(I[2][i], -4.0f, I[4][i]);
        //float t1 = __fmaf_rn(I[1][i], -4.0f, I[3][i]);
        //float t2 = I[4][i] - I[2][i];
        //float t3 = I[3][i] - I[1][i];
        //float t4 = __fmaf_rn(I[2][i], -5.0f, I[4][i]);
        //float t5 = __fmaf_rn(I[3][i], -5.0f, I[5][i]);
        //T[0][i] = __fmaf_rn(I[0][i], 4.0f, t4);
        //T[1][i] = t0 + t1;
        //T[2][i] = t0 - t1;
        //T[3][i] = __fmaf_rn(t3,  2.0f, t2);
        //T[4][i] = __fmaf_rn(t3, -2.0f, t2);
        //T[5][i] = __fmaf_rn(I[1][i], 4.0f, t5);

        float t0 = __fmaf_rn(I[2][i], 4.0f, -I[4][i]) * rcp6;
        float t1 = __fmaf_rn(I[1][i], 4.0f, -I[3][i]) * rcp6;
        float t2 = (I[4][i] - I[2][i]) * rcp24;
        float t3 = (I[3][i] - I[1][i]) * rcp12;
        float t4 = __fmaf_rn(I[2][i], -5.0f, I[4][i]);
        float t5 = __fmaf_rn(I[3][i], -5.0f, I[5][i]);
        T[0][i] = __fmaf_rn(t4, rcp4, I[0][i]);
        T[1][i] = t0 + t1;
        T[2][i] = t0 - t1;
        T[3][i] = t2 + t3;
        T[4][i] = t2 - t3;
        T[5][i] = __fmaf_rn(I[1][i], 4.0f, t5);
    }

    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        // float t0 = __fmaf_rn(T[i][2], -4.0f, T[i][4]);
        // float t1 = __fmaf_rn(T[i][1], -4.0f, T[i][3]);
        // float t2 = T[i][4] - T[i][2];
        // float t3 = T[i][3] - T[i][1];
        // float t4 = __fmaf_rn(T[i][2], -5.0f, T[i][4]);
        // float t5 = __fmaf_rn(T[i][3], -5.0f, T[i][5]);
        // share[ns][cs + 16*(i*6 + 0)] = %(cvt_out)s(__fmaf_rn(T[i][0], 4.0f, t4));
        // share[ns][cs + 16*(i*6 + 1)] = %(cvt_out)s(t0 + t1);
        // share[ns][cs + 16*(i*6 + 2)] = %(cvt_out)s(t0 - t1);
        // share[ns][cs + 16*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(t3,  2.0f, t2));
        // share[ns][cs + 16*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(t3, -2.0f, t2));
        // share[ns][cs + 16*(i*6 + 5)] = %(cvt_out)s(__fmaf_rn(T[i][1], 4.0f, t5));

        float t0 = __fmaf_rn(T[i][2], 4.0f, -T[i][4]) * rcp6;
        float t1 = __fmaf_rn(T[i][1], 4.0f, -T[i][3]) * rcp6;
        float t2 = (T[i][4] - T[i][2]) * rcp24;
        float t3 = (T[i][3] - T[i][1]) * rcp12;
        float t4 = __fmaf_rn(T[i][2], -5.0f, T[i][4]);
        float t5 = __fmaf_rn(T[i][3], -5.0f, T[i][5]);
        share[ns][cs + 16*(i*6 + 0)] = %(cvt_out)s(__fmaf_rn(t4, rcp4, T[i][0]));
        share[ns][cs + 16*(i*6 + 1)] = %(cvt_out)s(t0 + t1);
        share[ns][cs + 16*(i*6 + 2)] = %(cvt_out)s(t0 - t1);
        share[ns][cs + 16*(i*6 + 3)] = %(cvt_out)s(t2 + t3);
        share[ns][cs + 16*(i*6 + 4)] = %(cvt_out)s(t2 - t3);
        share[ns][cs + 16*(i*6 + 5)] = %(cvt_out)s(__fmaf_rn(T[i][1], 4.0f, t5));
    }

    __syncthreads();

    // now make c contiguous
    cs = tid & 15;
    ns = tid >> 4;

    // apply the super block to just grid coordinates this time
    gy = (gy << (shlY-2)) + ((ns & maskY) >> shrY);
    gx = (gx << (shlX-2)) + ((ns & maskX) >> shrX);
    n  = (blkN << shlN)   + (ns  & maskN);

    if (n < N && gy < GY && gx < GX)
    {
        // output dim: (C32,GY,GX,N,c32)
        // where c32 is the 1152 element transform data for 32 values of c
        // We group two blkC's to form each transform row

        O += (blkC >> 1)*GYGXN1152 + gy*GXN1152 + gx*N1152 + n*1152 + (blkC & 1)*16 + cs;

        #pragma unroll
        for (int i = 0; i < 36; i++)
            O[32*i] = share[ns][cs + 16*i];
    }
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    code = code % {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }
    # f = open("trans.cu", "w")
    # print >>f, code
    # f.close()

    module = SourceModule(code)
    kernel = module.get_function("update_image_trans_4x4")
    kernel.prepare("PPIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    return kernel


@context_dependent_memoize
def _get_update_delta_trans_4x4_kernel(dtype):

    code = r"""

%(common)s

__device__ __forceinline__ int div64(int value, int magic, int shift)
{
    int result;
    // if the divisor is a power of two the magic will be 1 and it's just a simple right shift
    if (magic == 1)
        result = value >> shift;
    // Otherwise multiply by magic and right shift just the high bits
    else
        asm(".reg .u64 res64;\n\t"
            ".reg .u32 lo32, hi32;\n\t"
            "mul.wide.u32 res64, %%1, %%2;\n\t"
            "mov.b64 {lo32, hi32}, res64;\n\t"
            "shr.b32 %%0, hi32, %%3;\n\t"
            : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}

__global__ void update_delta_trans_4x4(
    %(type)s* O, const %(type)s* In,
    int K, int Y, int X, int N, int GY, int GX,
    int Xkn, int magic_Xkn, int shift_Xkn, int shift_k, int shift_n,
    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,
    int YXN, int XN, int GYGXN1152, int GXN1152, int N1152)
{
    // Add padding to avoid all but 1 shared bank conflict on loads
    %(type)s __shared__ share[16][16*36 + 2];

    int tid        = threadIdx.x;
    int blk_YXkn   = blockIdx.x;
    int blk_N      = blockIdx.y;
    int blk_K      = blockIdx.z;

    // unpack y,x,k,n from blockIdx.x
    int gy      = div64(blk_YXkn, magic_Xkn, shift_Xkn);
    int blk_Xkn = blk_YXkn - gy*Xkn;

    int shift_kn = shift_k + shift_n;

    int gx     = blk_Xkn >> shift_kn;
    int blk_kn = blk_Xkn - (gx << shift_kn);

    int blk_k = blk_kn >> shift_n;
    int blk_n = blk_kn - (blk_k << shift_n);

    int blkN = (blk_N << shift_n) + blk_n;
    int blkK = (blk_K << shift_k) + blk_k;

    int ns = tid & 15;
    int ks = tid >> 4;

    // Super block YXN coordinates
    int y0 = (gy << shlY) + (((ns & maskY) >> shrY) << 2);
    int x0 = (gx << shlX) + (((ns & maskX) >> shrX) << 2);
    int n  = (blkN << shlN) + (ns & maskN);
    int k  = (blkK << 4) + ks;

    bool valid = k < K && n < N;

    bool xin[4], yin[4];
    float I[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        xin[i] = x0 + i >= 0 && x0 + i < X && valid;
        yin[i] = y0 + i >= 0 && y0 + i < Y;
    }

    const %(type)s* In0 = In + k*YXN + y0*XN + x0*N + n;

    #pragma unroll
    for (int y = 0; y < 4; y++)
    {
        if (y)
            In0 += XN;
        const %(type)s* In1 = In0 + N;
        const %(type)s* In2 = In1 + N;
        const %(type)s* In3 = In2 + N;

        I[y][0] = yin[y] && xin[0] ? %(cvt_in)s(__ldg(In0)) : 0.0f;
        I[y][1] = yin[y] && xin[1] ? %(cvt_in)s(__ldg(In1)) : 0.0f;
        I[y][2] = yin[y] && xin[2] ? %(cvt_in)s(__ldg(In2)) : 0.0f;
        I[y][3] = yin[y] && xin[3] ? %(cvt_in)s(__ldg(In3)) : 0.0f;
    }

    float T[6][4];
    // float rcp3  = 1.0f/3.0f;
    // float rcp4  = 1.0f/4.0f;
    // float rcp6  = 1.0f/6.0f;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float t0 = I[0][i] + I[2][i];
        float t1 = __fmaf_rn(I[2][i], 4.0f, I[0][i]);
        float t2 = I[1][i] + I[3][i];
        float t3 = __fmaf_rn(I[3][i], 4.0f, I[1][i]);
        T[0][i] = I[0][i];
        T[1][i] = t0 + t2;
        T[2][i] = t0 - t2;
        T[3][i] = __fmaf_rn(t3,  2.0f, t1);
        T[4][i] = __fmaf_rn(t3, -2.0f, t1);
        T[5][i] = I[3][i];

        // float t0 = (I[0][i] + I[2][i]) * -rcp6;
        // float t1 = __fmaf_rn(I[0][i], rcp4, I[2][i]) * rcp6;
        // float t2 = (I[1][i] + I[3][i]) * rcp6;
        // float t3 = __fmaf_rn(I[1][i], rcp4, I[3][i]) * rcp3;
        // T[0][i] = I[0][i]*rcp4;
        // T[1][i] = t0 - t2;
        // T[2][i] = t0 + t2;
        // T[3][i] = t1 + t3;
        // T[4][i] = t1 - t3;
        // T[5][i] = I[3][i];
    }

    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = T[i][0] + T[i][2];
        float t1 = __fmaf_rn(T[i][2], 4.0f, T[i][0]);
        float t2 = T[i][1] + T[i][3];
        float t3 = __fmaf_rn(T[i][3], 4.0f, T[i][1]);
        share[ns][ks + 16*(i*6 + 0)] = %(cvt_out)s(T[i][0]);
        share[ns][ks + 16*(i*6 + 1)] = %(cvt_out)s(t0 + t2);
        share[ns][ks + 16*(i*6 + 2)] = %(cvt_out)s(t0 - t2);
        share[ns][ks + 16*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(t3,  2.0f, t1));
        share[ns][ks + 16*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(t3, -2.0f, t1));
        share[ns][ks + 16*(i*6 + 5)] = %(cvt_out)s(T[i][3]);

        //float t0 = (T[i][0] + T[i][2]) * -rcp6;
        //float t1 = __fmaf_rn(T[i][0], rcp4, T[i][2]) * rcp6;
        //float t2 = (T[i][1] + T[i][3]) * rcp6;
        //float t3 = __fmaf_rn(T[i][1], rcp4, T[i][3]) * rcp3;
        //share[ns][ks + 16*(i*6 + 0)] = %(cvt_out)s(T[i][0]*rcp4);
        //share[ns][ks + 16*(i*6 + 1)] = %(cvt_out)s(t0 - t2);
        //share[ns][ks + 16*(i*6 + 2)] = %(cvt_out)s(t0 + t2);
        //share[ns][ks + 16*(i*6 + 3)] = %(cvt_out)s(t1 + t3);
        //share[ns][ks + 16*(i*6 + 4)] = %(cvt_out)s(t1 - t3);
        //share[ns][ks + 16*(i*6 + 5)] = %(cvt_out)s(T[i][3]);
    }

    __syncthreads();

    // now make k contiguous
    ks = tid & 15;
    ns = tid >> 4;

    // apply the super block to just grid coordinates this time
    gy = (gy << (shlY-2)) + ((ns & maskY) >> shrY);
    gx = (gx << (shlX-2)) + ((ns & maskX) >> shrX);
    n  = (blkN << shlN)   + (ns  & maskN);

    if (n < N && gy < GY && gx < GX)
    {
        // output dim: (K32,GY,GX,N,k32)
        // where k32 is the 1152 element transform data for 32 values of k
        // We group two blkK's to form each transform row

        O += (blkK >> 1)*GYGXN1152 + gy*GXN1152 + gx*N1152 + n*1152 + (blkK & 1)*16 + ks;

        #pragma unroll
        for (int i = 0; i < 36; i++)
            O[32*i] = share[ns][ks + 16*i];
    }
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    code = code % {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }
    #print code

    module = SourceModule(code)
    kernel = module.get_function("update_delta_trans_4x4")
    kernel.prepare("PPIIIIIIIIIIIIIIIIIIIIIIII")
    return kernel
