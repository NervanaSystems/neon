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
"""
Python code to wrap convolution kernels
"""
from __future__ import division
from builtins import object
from future.utils import native_str
import numpy as np
import pycuda.driver as drv
from pycuda.tools import context_dependent_memoize
from neon.backends import kernel_specs
from neon.backends.cuda_templates import _common_round, _ew_types, _common_fp16_to_fp32
from neon.backends.util.source_module import SourceModule
from math import ceil
from operator import mul
import sys
import os.path


from neon.util.shelver import atomic_shelve


if sys.version_info >= (3, 0):
    from functools import reduce


class KernelGroup(object):
    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 dil_d, dil_h, dil_w, bprop=False):

        self.params = (N, C, K, D, H, W, T, R, S, M, P, Q,
                       pad_d, pad_h, pad_w, str_d, str_h, str_w,
                       dil_d, dil_h, dil_w)

        self.bprop       = bprop
        self.lib         = lib
        self.dtype       = dtype
        self.dtype_str   = dtype.str[1:]
        self.vec_size    = 4 if dtype.itemsize == 4 else 8
        self.kernel_name = None
        self.kernel_opts = tuple()

        if dtype.type is np.float16:
            self.clss = "hconv"
        elif dtype.type is np.float32:
            self.clss = "sconv"
        elif dtype.type is np.int16:
            self.clss = "fconv"
        elif dtype.type is np.int8:
            self.clss = "bconv"
        else:
            raise TypeError("dtype not supported.")

    def bind_params(self, *args):
        raise TypeError("bind_params not implemented.")

    def execute(self, repeat=1, unbind=True):
        raise TypeError("execute not implemented.")

    def xprop_params(self, O, X, bias, bsum, beta, relu, brelu, slope):

        bsum_data = self.bsum.bind_params(bsum)
        x_data    = O.gpudata if X is None else X.gpudata

        # TODO: expose more compound operations
        self.kernel_options = self.kernel_opts
        if bias is not None:
            self.kernel_options += ("bias",)
            bsum_data = bias.gpudata
            assert bsum is None, "Cannot combine bias and bsum"
        if relu or brelu:
            if relu:
                self.kernel_options += ("prelu",)  if slope else ("relu",)
            else:
                self.kernel_options += ("bprelu",) if slope else ("brelu",)
                assert bias is None, "Cannot combine bias and brelu"
                assert X is not None, "X param required for bprop_relu"
        elif beta:
            self.kernel_options += ("beta",)
            assert bias is None, "Cannot combine bias and beta"
        if bsum is not None:
            self.kernel_options += ("bsum",)

        return (bsum_data, x_data)

    def __str__(self):
        (N, C, K, D, H, W, T, R, S, M, P, Q,
        pad_d, pad_h, pad_w, str_d, str_h, str_w,
        dil_d, dil_h, dil_w) = self.params
        kernel_name = self.kernel_name
        for opt in self.kernel_opts:
            if type(opt) is not tuple:
                kernel_name += "_" + opt
        if self.bprop:
            return "%s NCK:(%3d,%3d,%3d) DHW:(%3d,%3d,%3d) TRS:(%2d,%2d,%2d) str:(%d,%d,%d)" % (
                kernel_name, N, K, C, M, P, Q, T, R, S, str_d, str_h, str_w)
        else:
            return "%s NCK:(%3d,%3d,%3d) DHW:(%3d,%3d,%3d) TRS:(%2d,%2d,%2d) str:(%d,%d,%d)" % (
                kernel_name, N, C, K, D, H, W, T, R, S, str_d, str_h, str_w)


class FpropCuda(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 dil_d, dil_h, dil_w):

        super(FpropCuda, self).__init__(lib, dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                                        pad_d, pad_h, pad_w, str_d, str_h, str_w,
                                        dil_d, dil_h, dil_w)

        from neon.backends.kernels.cuda.convolution import _get_conv_kernel
        self.get_kernel = _get_conv_kernel
        self.kernel_name = "FpropCuda"

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % self.vec_size == 0, "K dim must be multiple of %d" % self.vec_size

        magic_PQ = _magic64(P * Q)
        magic_Q = _magic64(Q)
        magic_S = _magic32(R * S + 32, S)
        HWN = H * W * N
        RST = R * S * T
        KRST = K * RST
        PQ = P * Q
        PQN = PQ * N
        self.RS = R * S

        grid = (PQ * (-(-N // 32)), (-(-K // 32)), 1)
        block = (8, 8, 1)
        static_kernel_args = _flatten([C, D, H, W, N, T, R, S, K, M, P, Q,
                                       str_w, str_h, pad_w, pad_h,
                                       dil_w, dil_h,
                                       HWN // 4, KRST // 4, PQN // 4,
                                       PQ, 0, 0,
                                       magic_PQ, magic_Q, magic_S])
        self.launch_args = [grid, block] + [None] * 7 + static_kernel_args
        self.shared = RST * 4 * 2

        self.output_trans = CompoundOps(lib, dtype, K, PQN)
        lib.set_scratch_size(self.output_trans.size)

    def bind_params(self, I, F, O,
        X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
        relu=False, brelu=False, slope=0.0):

        assert I.dtype == F.dtype == O.dtype == self.dtype

        self.lib.scratch_buffer_init()
        output_data = self.output_trans.bind_params(O, X, bias, bsum, alpha, beta or slope, relu, brelu)
        self.launch_args[2:9] = (self.lib.stream, 1.0, 0.0, I.gpudata, F.gpudata, output_data, 0)

    def execute(self, repeat=1, unbind=True):

        kernel = self.get_kernel(self.dtype_str, self.RS, False, "fprop")

        for r in range(repeat):
            kernel.prepared_async_call(*self.launch_args, shared_size=self.shared)
            self.output_trans.execute()

        if unbind:
            self.output_trans.unbind()
            self.launch_args[2:9] = (None,) * 7


class BpropCuda(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 dil_d, dil_h, dil_w):

        super(BpropCuda, self).__init__(lib, dtype,
            N, C, K, D, H, W, T, R, S, M, P, Q,
            pad_d, pad_h, pad_w, str_d, str_h, str_w,
            dil_d, dil_h, dil_w, bprop=True)

        from neon.backends.kernels.cuda.convolution import _get_conv_kernel
        self.get_kernel = _get_conv_kernel
        self.kernel_name = "BpropCuda"

        assert N % 32 == 0, "N dim must be multiple of 32"

        magic_HW = _magic64(H * W)
        magic_W = _magic64(W)
        magic_RS = _magic32(R * S * T + 32, R * S)
        magic_S = _magic32(R * S + 32, S)
        HW = H * W
        HWN = HW * N
        RST = R * S * T
        CRST = C * RST
        PQ = P * Q
        PQN = PQ * N
        self.RS = R * S

        grid = (HW * (-(-N // 32)), -(-C // 32), 1)
        block = (8, 8, 1)
        static_kernel_args = _flatten([K, M, P, Q, N, T, R, S, C, D, H, W,
                                       str_w, str_h, pad_w, pad_h,
                                       dil_w, dil_h,
                                       PQN // 4, CRST // 4, HWN // 4,
                                       HW, 0, 0,
                                       magic_HW, magic_W, magic_S])
        self.launch_args = [grid, block] + [None] * 7 + static_kernel_args
        self.shared = R * S * T * 4 * 2

        self.filter_trans = FilterDimShuffle(lib, dtype, C, T, R, S, K)
        self.output_trans = CompoundOps(lib, dtype, C, HWN)
        lib.set_scratch_size(self.filter_trans.size, self.output_trans.size)

    def bind_params(self, I, F, O,
        X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
        relu=False, brelu=False, slope=0.0):

        assert I.dtype == F.dtype == O.dtype == self.dtype
        assert self.params[1] % 4 == 0, "C dim must be a multiple of 4 for Kepler bprop kernel"

        self.lib.scratch_buffer_init()
        filter_data = self.filter_trans.bind_params(F)
        output_data = self.output_trans.bind_params(O, X, bias, bsum, alpha, beta or slope, relu, brelu)
        self.launch_args[2:9] = (self.lib.stream, 1.0, 0.0, I.gpudata, filter_data, output_data, 0)

    def execute(self, repeat=1, unbind=True):

        kernel = self.get_kernel(self.dtype_str, self.RS, False, "bprop")

        for r in range(repeat):
            self.filter_trans.execute()
            kernel.prepared_async_call(*self.launch_args, shared_size=self.shared)
            self.output_trans.execute()

        if unbind:
            self.output_trans.unbind()
            self.filter_trans.unbind()
            self.launch_args[2:9] = (None,) * 7


class UpdateCuda(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 dil_d, dil_h, dil_w):

        super(UpdateCuda, self).__init__(lib, dtype,
            N, C, K, D, H, W, T, R, S, M, P, Q,
            pad_d, pad_h, pad_w, str_d, str_h, str_w,
            dil_d, dil_h, dil_w)

        from neon.backends.kernels.cuda.convolution import _get_conv_kernel
        self.get_kernel = _get_conv_kernel
        self.kernel_name = "UpdateCuda"

        assert N % 32 == 0, "N dim must be multiple of 32"

        HWN = H * W * N
        RS = R * S
        RST = RS * T
        KRST = K * RST
        CRSTK = KRST * C
        PQ = P * Q
        PQN = PQ * N
        magic_S = _magic32(R * S + 32, S)
        self.RS = R * S

        if lib.deterministic:
            grid_P = 1
            grid_Q = 1
            self.determ = CRSTK
        else:
            grid_P = P
            grid_Q = Q
            self.determ = 0

        pq_blocks = grid_P * grid_Q
        magic_PQ = _magic64(pq_blocks)
        magic_Q = _magic64(grid_Q)

        grid = (pq_blocks * (-(-K // 32)), (-(-(C * RS) // 32)), 1)
        block = (8, 32, 1)
        static_kernel_args = _flatten([C, D, H, W, N, T, R, S, K, M, P, Q,
                                       str_w, str_h, pad_w, pad_h,
                                       dil_w, dil_h,
                                       HWN // 4, KRST // 4, PQN // 4,
                                       pq_blocks, grid_P, grid_Q,
                                       magic_PQ, magic_Q, magic_S])
        self.launch_args = [grid, block] + [None] * 7 + static_kernel_args

        self.output_trans = UpdateConvReduce(lib, 1, CRSTK)
        lib.set_scratch_size(self.output_trans.size)

    def bind_params(self, I, E, O, alpha=1.0, beta=0.0, no_op=False):
        assert I.dtype == E.dtype == O.dtype == self.dtype

        self.lib.scratch_buffer_init()
        output_data = self.output_trans.bind_params(O, alpha, beta, no_op)
        self.zero_args = (output_data, 0, O.size, self.lib.stream)
        self.launch_args[2:9] = (self.lib.stream, 1.0, 0.0, I.gpudata, E.gpudata, output_data, 0)

    def execute(self, repeat=1, unbind=True):
        kernel = self.get_kernel(self.dtype_str, self.RS, False, "update")

        for r in range(repeat):
            drv.memset_d32_async(*self.zero_args)
            kernel.prepared_async_call(*self.launch_args)
            self.output_trans.execute()

        if unbind:
            self.output_trans.unbind()
            self.zero_args = None
            self.launch_args[2:9] = (None,) * 7


class XpropDirect(KernelGroup):

    def __init__(self, op, lib, dtype,
                 N, C, K, D, H, W, T, R, S, M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 dil_d, dil_h, dil_w, bprop=False):

        super(XpropDirect, self).__init__(lib, dtype,
            N, C, K, D, H, W, T, R, S, M, P, Q,
            pad_d, pad_h, pad_w, str_d, str_h, str_w,
            dil_d, dil_h, dil_w, bprop)

        if N % 64 == 0 and K % self.vec_size == 0:
            self.init_largeN(op)
        else:
            self.init_smallN(op)
        lib.set_scratch_size(self.filter_trans.size, self.bsum.size)

    def init_largeN(self, op):

        (N, C, K, D, H, W, T, R, S, M, P, Q,
        pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w) = self.params

        for blockN in (128, 64):
            if N % blockN == 0:
                break

        #TODO: build 32x64 kernels
        K_tiles = (128, 64, 32) if blockN == 128 else (128, 64)
        for blockK in K_tiles:
            mod = K % blockK
            if mod == 0 or mod > blockK - 32:
                break

        kname   = "%s_direct_%s_%dx%d" % (self.clss, op, blockK, blockN)
        threads = kernel_specs.kernels[kname]["threads"]

        gridK = _ceil_div(K, blockK)
        gridN = _ceil_div(N, blockN)
        RS    = R * S
        TRS   = T * RS
        TRSK  = K * TRS
        k     = _closest_divisor(gridK, 128 // blockK)
        P2    = P // 2
        Q2    = Q * 2
        Qk    = Q2 * k
        PQk   = P * Q * k

        magic_PQk = _magic64(PQk)
        magic_Qk  = _magic64(Qk)
        magic_k   = _magic32(Qk, k)
        magic_RS  = _magic32(TRS + 32, RS)
        magic_S   = _magic32(RS  + 32,  S)

        bsum_warps = blockN // 64
        gridNw     = gridN * bsum_warps
        gridQNw    = Q * gridNw
        gridPQNw   = P * gridQNw
        gridMPQNw  = M * gridPQNw
        gridMPQ    = M * P * Q
        grid       = (gridMPQ * k, gridK // k, gridN)

        self.kernel_opts = tuple()
        self.kernel_name = kname
        self.kernel_args = [grid, (threads,1,1), None, None, None, None, None, None, None, None, None]
        self.kernel_args.extend(_flatten([
            N, K, D, H, W, W * N, H * W * N, D * H * W * N,
            C, TRSK, TRS, RS, T, R, S, magic_RS, magic_S,
            pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w,
            P2, Q, PQk, Qk, k, magic_PQk, magic_Qk, magic_k,
            Q * N, P * Q * N, M * P * Q * N, gridNw, gridQNw, gridPQNw, gridMPQNw ]))

        self.shared = TRS * 4 * 2
        self.bsum   = BatchNormSum(self.lib, K, gridMPQNw)

    def init_smallN(self, op):

        (N, C, K, D, H, W, T, R, S, M, P, Q,
        pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w) = self.params

        assert N % 4 == 0 or N in (1,2), "N dim must be multiple of 4 or equal to 1 or 2"

        for blockN in (32,16,8,4,2,1):
            if N % blockN == 0:
                break

        if P == 1:
            # 1D conv
            sb_params_in = {
                #blkN: supM, shfM, supP, shfP, supQ, shfQ, supN, shfN
                32 : ( 0x000, 0,   0x000, 0,   0x000, 0,    7,   5  ), # 1x1  nnn(nn)
                16 : ( 0x000, 0,   0x000, 0,   0x102, 1,    3,   4  ), # 1x2  xnn(nn)
                8  : ( 0x000, 0,   0x000, 0,   0x201, 2,    1,   3  ), # 2x2  xxn(nn)
                4  : ( 0x000, 0,   0x000, 0,   0x300, 3,    0,   2  ), # 2x4  xxx(nn)
                2  : ( 0x000, 0,   0x000, 0,   0x300, 4,    0,   1  ), # 4x4  xxx(xn)
                1  : ( 0x000, 0,   0x000, 0,   0x300, 5,    0,   0  ), # 4x8  xxx(xx)
            }
            sb_params_out = {
                #blkN:  supM,  supP,  supQ, supN
                32 : ( 0x000, 0x000, 0x000, 31 ), # 1x1  nnnnn
                16 : ( 0x000, 0x000, 0x104, 15 ), # 1x2  xnnnn
                8  : ( 0x000, 0x000, 0x203,  7 ), # 2x2  xxnnn
                4  : ( 0x000, 0x000, 0x302,  3 ), # 2x4  xxxnn
                2  : ( 0x000, 0x000, 0x401,  1 ), # 4x4  xxxxn
                1  : ( 0x000, 0x000, 0x500,  0 ), # 4x8  xxxxx
            }
        else:
            sb_params_in = {
                #blkN: supM, shfM, supP, shfP, supQ, shfQ, supN, shfN
                32 : ( 0x000, 0,   0x000, 0,   0x000, 0,    7,   5  ), # 1x1  nnn(nn)
                16 : ( 0x000, 0,   0x000, 0,   0x102, 1,    3,   4  ), # 1x2  xnn(nn)
                8  : ( 0x000, 0,   0x102, 1,   0x101, 1,    1,   3  ), # 2x2  yxn(nn)
                4  : ( 0x000, 0,   0x102, 1,   0x200, 2,    0,   2  ), # 2x4  yxx(nn)
                2  : ( 0x000, 0,   0x201, 2,   0x100, 2,    0,   1  ), # 4x4  yyx(xn)
                1  : ( 0x000, 0,   0x201, 2,   0x100, 3,    0,   0  ), # 4x8  yyx(xx)
            }
            sb_params_out = {
                #blkN:  supM,  supP,  supQ, supN
                32 : ( 0x000, 0x000, 0x000, 31 ), # 1x1  nnnnn
                16 : ( 0x000, 0x000, 0x104, 15 ), # 1x2  xnnnn
                8  : ( 0x000, 0x104, 0x103,  7 ), # 2x2  yxnnn
                4  : ( 0x000, 0x104, 0x202,  3 ), # 2x4  yxxnn
                2  : ( 0x000, 0x203, 0x201,  1 ), # 4x4  yyxxn
                1  : ( 0x000, 0x203, 0x300,  0 ), # 4x8  yyxxx
            }
        superM, shiftM, superP, shiftP, superQ, shiftQ, superN, shiftN = sb_params_in.get(blockN)
        SuperM, SuperP, SuperQ, SuperN = sb_params_out.get(blockN)

        blockM  = 1 << shiftM
        blockP  = 1 << shiftP
        blockQ  = 1 << shiftQ
        gridM   = _ceil_div(M, blockM)
        gridP   = _ceil_div(P, blockP)
        gridQ   = _ceil_div(Q, blockQ)
        gridN   = _ceil_div(N, blockN)
        gridK   = _ceil_div(K, 64)
        gridP2  = gridP // 2
        gridQ2  = gridQ * 2

        RS       = R * S
        TRS      = T * RS
        TRSK     = K * TRS
        n        = _closest_divisor(gridN, 2)
        k        = _closest_divisor(gridK, 2)
        nk       = n * k
        Qnk      = gridQ2 * nk
        PQnk     = gridP * gridQ * nk

        magic_PQnk = _magic64(PQnk)
        magic_Qnk  = _magic64(Qnk)
        magic_nk   = _magic32(Qnk, nk)
        magic_k    = _magic32(nk,   k)
        magic_RS   = _magic32(TRS, RS)
        magic_S    = _magic32(RS,   S)

        gridMPQ = gridM * gridP * gridQ
        grid    = (gridMPQ * nk, gridK // k, gridN // n)

        options = list()
        if     N == 1: options.append("N1")
        elif   N == 2: options.append("N2")
        elif   N < 32: options.append("SN")
        if K % 4 != 0: options.append("K1")

        self.kernel_opts = tuple(options)
        self.kernel_name = "%s_direct_%s_64x32" % (self.clss, op)
        self.kernel_args = [grid, (128,1,1), None, None, None, None, None, None, None, None, None]
        self.kernel_args.extend(_flatten([
            C, D, H, W, N, K, M, P, Q,
            str_d, str_h, str_w, pad_d, pad_h, pad_w, dil_d, dil_h, dil_w,
            D * H * W * N, H * W * N, W * N, M * P * Q * N, P * Q * N, Q * N,
            PQnk, Qnk, nk, n, k, magic_PQnk, magic_Qnk, magic_nk, magic_k,
            max(K - 32,0), K * 32 * self.dtype.itemsize, TRSK, TRS, RS, S, magic_RS, magic_S,
            gridP2, gridQ, gridN, gridQ * gridN, gridP * gridQ * gridN, gridMPQ * gridN,
            superM, superP, superQ, superN,
            shiftM, shiftP, shiftQ, shiftN,
            SuperM, SuperP, SuperQ, SuperN ]))

        self.bsum = BatchNormSum(self.lib, K, gridMPQ * gridN)

        if N >= 32:
            self.shared = (T * R * S + 1) * 4 * 2
        else:
            self.shared = T * R * S * 4 * (32 >> shiftN)

    def bind_params(self, I, F, O,
        X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
        relu=False, brelu=False, slope=0.0, no_op=0):

        if not (I.dtype == O.dtype == self.dtype):
            ERR_STR = (
                'I.dtype: {}, O.dtype: {}, self.dtype: {} should'
                'all be the same'
            )
            raise TypeError(ERR_STR.format(I.dtype, O.dtype, self.dtype))

        self.lib.scratch_buffer_init()
        filter_data = self.filter_trans.bind_params(F)
        bsum_data, x_data = self.xprop_params(O, X, bias, bsum, beta, relu, brelu, slope)

        self.kernel_args[2:11] = (self.lib.stream, bsum_data, x_data,
                                  O.gpudata, I.gpudata, filter_data, alpha, beta or slope, no_op)

    def execute(self, repeat=1, unbind=True):

        kernel = kernel_specs.get_kernel(self.kernel_name, self.kernel_options)

        for r in range(repeat):
            self.filter_trans.execute()
            kernel.prepared_async_call(*self.kernel_args, shared_size=self.shared)
            self.bsum.execute()

        if unbind:
            self.filter_trans.unbind()
            self.bsum.unbind()
            self.kernel_args[2:11] = (None,) * 9

class FpropDirect(XpropDirect):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 dil_d, dil_h, dil_w):

        # The filters may still be in fp32 so we potentially need to dynamically quantize
        if dtype.itemsize != 4:
            self.filter_trans = ConvertDataType(lib, dtype, C * T * R * S * K, out_mode=False)
        # if kernel is sconv then no need for any transform in fprop.
        else:
            self.filter_trans = NoopTransform()

        super(FpropDirect, self).__init__("fprop", lib, dtype,
            N, C, K, D, H, W, T, R, S, M, P, Q,
            pad_d, pad_h, pad_w, str_d, str_h, str_w,
            dil_d, dil_h, dil_w)

class BpropDirect(XpropDirect):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 dil_d, dil_h, dil_w):

        self.filter_trans = FilterDimShuffle(lib, dtype, C, T, R, S, K)

        # invert padding
        pad_d = (T - 1) * dil_d - pad_d
        pad_h = (R - 1) * dil_h - pad_h
        pad_w = (S - 1) * dil_w - pad_w

        # Swap C<=>K and DHW<=>MPQ
        super(BpropDirect, self).__init__("bprop", lib, dtype,
            N, K, C, M, P, Q, T, R, S, D, H, W,
            pad_d, pad_h, pad_w, str_d, str_h, str_w,
            dil_d, dil_h, dil_w, bprop=True)

        self.kernel_args.extend(_flatten([
            _magic32(D + T, str_d),
            _magic32(H + R, str_h),
            _magic32(W + S, str_w) ]))


class UpdateDirect(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 dil_d, dil_h, dil_w):

        assert N % 4 == 0, "N dim must be multiple of 4"

        super(UpdateDirect, self).__init__(lib, dtype,
            N, C, K, D, H, W, T, R, S, M, P, Q,
            pad_d, pad_h, pad_w, str_d, str_h, str_w,
            dil_d, dil_h, dil_w)

        SMs = _get_sm_count()

        self.autotune_key = " ".join(native_str(x) for x in (
            "direct_updat_64x32", SMs, dtype.itemsize, lib.deterministic > 0,
            N, C, K, D, H, W, T, R, S, M, P, Q ))

        # insert Python version in filename to avoid Py2/Py3 incompatibilities in shelve
        self.autotune_db_file = os.path.join(lib.cache_dir, "autotune%d.db" % sys.version_info[0])
        self.init()

        lib.set_scratch_size(self.output_trans.size)

        # allow for .5 seconds worth of warmup when autotuning
        # assume 5 Tflops on 24 SMs
        self.warmup = min(max(int(2e12 / (M * P * Q * K * N * C * T * R * S * 2.0) * (SMs / 24.0)), 1), 5000)

    def init(self, autotune=False):

        (N, C, K, D, H, W, T, R, S, M, P, Q,
         pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w) = self.params

        for blockN in (32,16,8,4):
            if N % blockN == 0:
                break

        if P == 1:
            # 1D conv
            sb_params = {
                #blkN: supM, shfM, supP, shfP, supQ, shfQ, supN
                32 : ( 0x000, 0,   0x000, 0,   0x000, 0,   7 ), # 1x1  nnn
                16 : ( 0x000, 0,   0x000, 0,   0x102, 1,   3 ), # 1x2  xnn
                8  : ( 0x000, 0,   0x000, 0,   0x201, 2,   1 ), # 2x2  xxn
                4  : ( 0x000, 0,   0x000, 0,   0x300, 3,   0 ), # 2x4  xxx
            }
        else:
            sb_params = {
                #blkN: supM, shfM, supP, shfP, supQ, shfQ, supN
                32 : ( 0x000, 0,   0x000, 0,   0x000, 0,   7 ), # 1x1  nnn
                16 : ( 0x000, 0,   0x000, 0,   0x102, 1,   3 ), # 1x2  xnn
                4  : ( 0x000, 0,   0x102, 1,   0x200, 2,   0 ), # 2x4  yxx
                8  : ( 0x000, 0,   0x102, 1,   0x101, 1,   1 ), # 2x2  yxn
            }
        superM, shiftM, superP, shiftP, superQ, shiftQ, superN = sb_params.get(blockN)

        blockM  = 1 << shiftM
        blockP  = 1 << shiftP
        blockQ  = 1 << shiftQ
        GM      = _ceil_div(M, blockM)
        GP      = _ceil_div(P, blockP)
        GQ      = _ceil_div(Q, blockQ)
        self.GP = GP
        self.GQ = GQ

        if autotune:
            strideP, strideQ = autotune
        else:
            with atomic_shelve(self.autotune_db_file) as autotune_db:
                if self.autotune_key in autotune_db:
                    strideP, strideQ = autotune_db[self.autotune_key]
                    #print strideP, strideQ, self.autotune_key
                    self.initialized = True
                else:
                    # prior to autotuning set the maximum for scratch space
                    # memory allocation purposes
                    if GP * GQ > 192:
                        strideP  = 192
                        strideQ  = 1
                    else:
                        strideP  = GP
                        strideQ  = GQ
                    self.initialized = False

        itemsize = self.dtype.itemsize
        RS       = R * S
        TRS      = T * RS
        CTRS     = C * TRS
        CTRSK    = K * CTRS
        GK       = _ceil_div(K,    32)
        GC       = _ceil_div(CTRS, 64)
        k        = _closest_divisor(GK, 4)
        c        = _closest_divisor(GC, 2)
        kc       = k * c
        Qkc      = strideQ * kc
        PQkc     = strideP * Qkc

        self.blocksMCK = GM * GK * GC

        magic_TRS  = _magic32(GC * 64,  TRS)
        magic_RS   = _magic32(TRS, RS)
        magic_S    = _magic32(RS,   S)
        magic_PQkc = _magic64(PQkc)
        magic_Qkc  = _magic64(Qkc)
        magic_kc   = _magic32(Qkc, kc)
        magic_c    = _magic32(kc, c)

        loopQ = strideQ * blockQ
        loopX = loopQ * str_w

        options = list()
        if N > blockN:
            loopQp = N * (loopQ - 1) * itemsize
            loopXp = N * (loopX - 1) * itemsize
        else:
            loopQp = N * loopQ * itemsize
            loopXp = N * loopX * itemsize
            options.append("SN")

        gridMPQ = GM * strideP * strideQ

        # If output grid is 1, don't use atomics.  Kernel is deterministic by default
        if gridMPQ == 1 or self.lib.deterministic:
            self.output_trans = UpdateConvReduce(self.lib, gridMPQ, CTRSK)
            self.zero = False
            options.append("D")
        else:
            self.output_trans = UpdateConvReduce(self.lib, 1, CTRSK)
            self.zero = True

        grid = (gridMPQ * kc, GC // c, GK // k)

        self.kernel_opts = tuple(options)
        self.kernel_name = "%s_direct_updat_64x32" % self.clss
        self.kernel_args = [grid, (128,1,1), None, None, None, None, 1.0]
        self.kernel_args.extend(_flatten([
            C, D, H, W, N, K, M, P, Q,
            str_d, str_h, str_w, pad_d, pad_h, pad_w, dil_d, dil_h, dil_w,
            D * H * W * N, H * W * N, W * N, M * P * Q * N * 16 * itemsize, M * P * Q * N, P * Q * N, Q * N,
            PQkc, Qkc, kc, c, k, magic_PQkc, magic_Qkc, magic_kc, magic_c,
            CTRSK, CTRS, TRS, RS, S, magic_TRS, magic_RS, magic_S,
            superM, superP, superQ, superN, shiftM, shiftP, shiftQ,
            strideP, strideQ, strideP * strideQ, GP, GQ,
            loopX, loopXp, loopQ, loopQp, blockN, blockN * itemsize ]))

    def autotune(self, I, E, O):

        # print "autotune: ", self.autotune_key

        start, stop = self.lib.get_events()

        # Only need to do warmup once
        if not self.lib.warmup:
            self.lib.warmup = True
            # warmup  with a conservative set of params
            self.init(autotune=(min(self.GP,192), 1))
            self.bind_params(I, E, O, no_op=True)
            self.execute(repeat=self.warmup, unbind=False)

        # we want at least this many blocks
        block_slots = _get_sm_count()
        # loops for given size of N
        loopsN = max(self.params[0] // 32, 1)

        GP = float(self.GP)
        GQ = float(self.GQ)
        small_set = GP * GQ <= 512

        results = []
        sys.stdout.write("Autotune " + native_str(self))
        progress = 0
        for threshold in (True, False):
            for strideP in range(1, self.GP + 1):
                for strideQ in range(1, self.GQ + 1):
                    if progress % 32 == 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                    progress += 1

                    # CTRSK copies in determ mode
                    outputs = strideP * strideQ
                    # minimal occupancy filter
                    blocks = self.blocksMCK * outputs
                    # gemm loop size filter
                    depth = (GP / strideP) * (GQ / strideQ) * loopsN

                    filters = strideP >= strideQ and \
                              blocks  >= block_slots and \
                              depth   >= 8.0

                    # Never allow settings beyond our max scratch size allocation
                    if outputs <= 192:
                        # In case we filter out all settings, run though the loops again
                        # this time looking only at settings that didn't pass.
                        if small_set or (threshold and filters) or (not threshold and not filters):

                            settings = (strideP, strideQ)
                            self.init(autotune=settings)
                            self.bind_params(I, E, O, no_op=True)
                            start.record(stream=self.lib.stream)
                            self.execute(repeat=2, unbind=False)
                            stop.record(stream=self.lib.stream)
                            stop.synchronize()
                            msecs = stop.time_since(start) / 2.0
                            results.append((msecs, settings))
                        # else:
                        #     print strideP, strideQ, blocks, round(depth,1)

            # if we got any results, no need to disable the filter
            if len(results) > 0:
                break
        sys.stdout.write('\n')

        results.sort()
        settings = results[0][1]
        # for res in results[0:10]:
        #     print res
        with atomic_shelve(self.autotune_db_file) as autotune_db:
            autotune_db[self.autotune_key] = settings

        self.init(autotune=settings)

    def bind_params(self, I, E, O, alpha=1.0, beta=0.0, no_op=False):

        assert I.dtype == E.dtype == self.dtype

        if not self.initialized:
            self.initialized = True
            self.autotune(I, E, O)

        self.lib.scratch_buffer_init()

        output_data = self.output_trans.bind_params(O, alpha, beta, no_op)

        if self.zero:
            self.zero_args = ( output_data, 0, O.size, self.lib.stream )

        self.kernel_args[2:6] = (self.lib.stream, output_data, I.gpudata, E.gpudata)

    def execute(self, repeat=1, unbind=True):

        kernel = kernel_specs.get_kernel(self.kernel_name, self.kernel_opts)

        for r in range(repeat):
            if self.zero:
                drv.memset_d32_async(*self.zero_args)

            kernel.prepared_async_call(*self.kernel_args)
            self.output_trans.execute()

        if unbind:
            self.output_trans.unbind()
            self.zero_args = None
            self.kernel_args[2:6] = (None,) * 4


# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 32 bits
# Shamelessly pulled directly from:
# http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
def _magic32(nmax, d):
    nc = ((nmax + 1) // d) * d - 1
    nbits = len(bin(nmax)) - 2
    for p in range(0, 2 * nbits + 1):
        if 2 ** p > nc * (d - 1 - (2 ** p - 1) % d):
            m = (2 ** p + d - 1 - (2 ** p - 1) % d) // d
            return (m, p)
    raise ValueError("Can't find magic number for division")


# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 64 bits and the shift
# lops off the lower 32 bits
def _magic64(d):
    # 3 is a special case that only ends up in the high bits
    # if the nmax is 0xffffffff
    # we can't use 0xffffffff for all cases as some return a 33 bit
    # magic number
    nmax = 0xffffffff if d == 3 else 0x7fffffff
    magic, shift = _magic32(nmax, d)
    if magic != 1:
        shift -= 32
    return (magic, shift)

_div64 = r"""
__device__ __forceinline__ int div64(int value, int magic, int shift)
{
    // if the divisor is a power of 2 the magic will be 1 and it's just a simple right shift
    // Otherwise multiply by magic and right shift just the high bits
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}
"""

# flatten a nested list of lists or values
def _flatten(lst):
    return sum(([x] if not isinstance(x, (list, tuple))
                else _flatten(x) for x in lst), [])


def _ceil_div(x, y):
    return -(-x // y)


def _closest_divisor(val, div, maxdiv=8):
    return -sorted([(abs(i - div), -i) for i in range(1, maxdiv) if val % i == 0])[0][1]


@context_dependent_memoize
def _get_sm_count():
    attributes = drv.Context.get_device().get_attributes()
    return attributes[drv.device_attribute.MULTIPROCESSOR_COUNT]

# Pass through transform (for code simplification)
class NoopTransform(object):
    def __init__(self):
        self.size = 0
    def bind_params(self, A):
        return A.gpudata
    def execute(self):
        pass
    def unbind(self):
        pass

# fast axis=0 reduction kernel used for deterministic update
class UpdateConvReduce(object):
    def __init__(self, lib, gridMPQ, CTRSK):
        blocks     = _ceil_div(CTRSK, 32)
        PQCRSTK    = gridMPQ * CTRSK
        self.mpq   = gridMPQ > 1
        self.lib   = lib
        self.size  = PQCRSTK * 4
        self.args  = [(blocks, 1, 1), (32, 1, 1), None, None, None, 1, 0, CTRSK, PQCRSTK]

    def bind_params(self, U, alpha, beta, no_op):
        if self.mpq or alpha != 1.0 or beta != 0.0 or U.dtype.type != np.float32:
            update_data    = self.lib.scratch_buffer_offset(self.size)
            output_data    = update_data if no_op else U.gpudata
            self.args[2:7] = (self.lib.stream, output_data, update_data, alpha, beta)
            self.kernel    = _get_update_conv_reduce_kernel(U.dtype.str[1:], beta != 0)
            return update_data
        self.kernel = None
        if no_op:
            return self.lib.scratch_buffer_offset(self.size)
        return U.gpudata

    def execute(self):
        if self.kernel is not None:
            self.kernel.prepared_async_call(*self.args)

    def unbind(self):
        self.kernel    = None
        self.args[2:5] = (None,) * 3


@context_dependent_memoize
def _get_update_conv_reduce_kernel(dtype, beta):

    kernel_code = r"""
%(common)s

__global__ void update_conv_reduce(%(type)s* Out, const float* In, float alpha, float beta, int CRSTK, int PQCRSTK)
{
    int offset = blockIdx.x * 32 + threadIdx.x;

    if (offset < CRSTK)
    {
        float sum = 0.0f;
        int i0 = offset;
        while (i0 < PQCRSTK)
        {
            int i1 = i0 + CRSTK;
            int i2 = i1 + CRSTK;
            int i3 = i2 + CRSTK;

            sum += In[i0];
            sum += i1 < PQCRSTK ? In[i1] : 0.0f;
            sum += i2 < PQCRSTK ? In[i2] : 0.0f;
            sum += i3 < PQCRSTK ? In[i3] : 0.0f;

            i0 = i3 + CRSTK;
        }
        Out[offset] = %(output)s;
    }
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    template_vals = {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }
    if beta:
        output = "%(cvt_out)s(sum * alpha + %(cvt_in)s(Out[offset]) * beta)"
    else:
        output = "%(cvt_out)s(sum * alpha)"
    template_vals["output"] = output % template_vals

    code   = kernel_code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("update_conv_reduce")
    kernel.prepare("PPffII")
    return kernel

# fast axis=1 reduction kernel used for deterministic compounded batch norm mean
class BatchNormSum(object):
    def __init__(self, lib, K, gridMPQN):
        self.lib   = lib
        self.size  = K * gridMPQN * 4
        self.args  = [(K, 1, 1), (256, 1, 1), None, None, None, gridMPQN]
        #self.shape = (K, gridMPQN)

    def bind_params(self, bsum):
        if bsum is not None:
            bsum_data      = self.lib.scratch_buffer_offset(self.size)
            self.args[2:5] = (self.lib.stream, bsum.gpudata, bsum_data)
            self.kernel    = _get_batchnorm_sum_kernel()
            #drv.memset_d32_async( bsum_data, 0, self.size//4, None )
            return bsum_data
        self.kernel = None
        return 0

    def execute(self):
        if self.kernel is not None:
            self.kernel.prepared_async_call(*self.args)
            #self.ary = np.empty(self.shape, np.float32)
            #drv.memcpy_dtoh_async(self.ary, self.args[4], None)
            #np.savetxt("out.txt", ary, fmt='%10.5f')

    def unbind(self):
        self.kernel    = None
        self.args[2:5] = (None,) * 3


@context_dependent_memoize
def _get_batchnorm_sum_kernel():
    kernel_code = r"""
#define THREADS 256

__global__ void batchnorm_sum(float* Out, const float* In, int N)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    __shared__ float sPartials[THREADS];

    In += bid*N + tid;

    float sum = 0.0f;
    for (int i = tid; i < N; i += THREADS)
    {
        sum += *In;
        In  += THREADS;
    }
    sPartials[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int a = THREADS >> 1; a > 32; a >>= 1)
    {
        if ( tid < a )
            sPartials[tid] += sPartials[tid + a];
        __syncthreads();
    }
    if ( tid < 32 )
    {
        sum = sPartials[tid] + sPartials[tid + 32];

        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
            sum += __shfl_xor(sum, i);

        if ( tid == 0 )
            Out[bid] = sum;
    }
}
"""
    module = SourceModule(kernel_code)
    kernel = module.get_function("batchnorm_sum")
    kernel.prepare("PPI")
    return kernel


# for kernels that can't compound these ops internally, use an external kernel
class CompoundOps(object):
    def __init__(self, lib, dtype, K, N):
        for threads in (1024,512,256,128):
            if N >= threads * 8:
                break
        self.threads = threads
        self.dtype   = dtype.str[1:]
        self.lib     = lib
        self.size    = K * N * dtype.itemsize
        self.args    = [(K, 1, 1), (threads, 1, 1), None, None, None, None, None, None, 1.0, 0.0, N]

    def bind_params(self, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0, relu=False, brelu=False):
        if bsum is not None or bias is not None or relu or brelu or beta != 0.0 or alpha != 1.0:
            # beta is reused as slope param in prelu / bprop_prelu
            do_beta    = (beta != 0.0 or alpha != 1.0) and not relu and not brelu
            bias_data  = 0 if bias is None else bias.gpudata
            bsum_data  = 0 if bsum is None else bsum.gpudata
            x_data     = O.gpudata    if X is None else X.gpudata
            if bias is not None or relu or brelu or beta != 0.0 or alpha != 1.0:
                input_data = self.lib.scratch_buffer_offset(self.size)
            else:
                input_data = O.gpudata

            self.args[2:10] = (self.lib.stream, O.gpudata, bsum_data, bias_data, input_data, x_data, alpha, beta)
            self.kernel    = _get_compound_ops_kernel(self.dtype, self.threads,
                bias is not None, bsum is not None, do_beta, relu, brelu)
            return input_data
        self.kernel = None
        return O.gpudata

    def execute(self):
        if self.kernel is not None:
            self.kernel.prepared_async_call(*self.args)

    def unbind(self):
        self.kernel    = None
        self.args[2:8] = (None,) * 6

@context_dependent_memoize
def _get_compound_ops_kernel(dtype, threads, bias, bsum, beta, relu, brelu):

    kernel_name = "compound"
    if bias:  kernel_name += "_bias"
    if relu:  kernel_name += "_relu"
    if beta:  kernel_name += "_beta"
    if brelu: kernel_name += "_brelu"
    if bsum:  kernel_name += "_bsum"

    kernel_code = r"""
%(common)s

#define THREADS %(threads)s

__global__ void %(name)s(
    %(type)s* Out,
    float*    Bsum,
    const  float* __restrict__ Bias,
    const %(type)s* __restrict__ In,
    const %(type)s* __restrict__ X,
    float alpha, float beta, int N)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    In += bid*N + tid;
    %(inits)s

    for (int i = tid; i < N; i += THREADS)
    {
        float data = %(cvt_in)s(*In); In += THREADS;
        %(loads)s

        %(ops)s
    }
    %(bsum)s
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    vals = {
        "name"    : kernel_name,
        "threads" : threads,
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
        "bsum"    : "",
    }
    inits = []
    if beta or brelu:
        inits.append("X += bid*N + tid;")
    if bias:
        inits.append("Bias += bid;")
    if bias or relu or beta or brelu:
        inits.append("Out += bid*N + tid;")
    if bsum:
        inits.append("float sum = 0.0f;")
    vals["inits"] = "\n    ".join(inits)

    loads = []
    if beta or brelu:
        loads.append("float x = %(cvt_in)s(*X);  X += THREADS;" % vals)
    if bias:
        loads.append("float bias = *Bias;")
    vals["loads"] = "\n        ".join(loads)

    ops = []
    if bias:
        ops.append("data += bias;")
    if relu:
        ops.append("data = max(data, 0.0f) + min(data, 0.0f) * beta;")
    if beta:
        ops.append("data = data * alpha + x * beta;")
    if brelu:
        ops.append("data *= (x > 0.0f) + (x < 0.0f) * beta;")
    if bsum:
        ops.append("sum += data;")
    if bias or relu or beta or brelu:
        ops.append("*Out = %(cvt_out)s(data); Out += THREADS;" % vals)
    vals["ops"] = "\n        ".join(ops)

    if bsum:
        vals["bsum"] = r"""
    __shared__ float sPartials[THREADS];

    sPartials[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int a = THREADS >> 1; a > 32; a >>= 1)
    {
        if ( tid < a )
            sPartials[tid] += sPartials[tid + a];
        __syncthreads();
    }
    if ( tid < 32 )
    {
        sum = sPartials[tid] + sPartials[tid + 32];

        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
            sum += __shfl_xor(sum, i);

        if ( tid == 0 )
            Bsum[bid] = sum;
    }
"""
    code   = kernel_code % vals
    module = SourceModule(code)
    kernel = module.get_function(kernel_name)
    kernel.prepare("PPPPPffI")
    return kernel

# Convert scratch to output tensor or from input tensor to scratch
class ConvertDataType(object):
    def __init__(self, lib, scratch_dtype, size, out_mode=True):
        # find efficent kernel dims
        SMs = _get_sm_count() * 4
        for depth in (8,7,6,5):
            if size >> depth >= SMs:
                break
        blk_depth  = 1 << depth
        blocks     = _ceil_div(size, blk_depth)
        self.out   = out_mode
        self.lib   = lib
        self.dtype = scratch_dtype.str[1:]
        self.size  = size * scratch_dtype.itemsize
        self.args  = [(blocks, 1, 1), (32, 1, 1), None, None, None, blk_depth, size]

    def bind_params(self, A):
        a_dtype = A.dtype.str[1:]
        if a_dtype != self.dtype:
            cvt_data = self.lib.scratch_buffer_offset(self.size)
            if self.out:
                self.args[2:5] = (self.lib.stream, A.gpudata, cvt_data)
                self.kernel    = _get_convert_dtype_kernel(a_dtype, self.dtype)
            else:
                self.args[2:5] = (self.lib.stream, cvt_data, A.gpudata)
                self.kernel    = _get_convert_dtype_kernel(self.dtype, a_dtype)
            return cvt_data
        self.kernel = None
        return A.gpudata

    def execute(self):
        if self.kernel is not None:
            self.kernel.prepared_async_call(*self.args)

    def unbind(self):
        self.kernel    = None
        self.args[2:5] = (None,) * 3

@context_dependent_memoize
def _get_convert_dtype_kernel(otype, itype):
    kernel_code = r"""
#define THREADS 32

%(common)s

__global__ void convert_dtype(%(otype)s* Out, const %(itype)s* In, int blk_depth, int size)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    int offset = bid*blk_depth + tid;

    int end = min(offset + blk_depth, size);

    while (offset < end)
    {
        Out[offset] = %(cvt_out)s(%(cvt_in)s(In[offset]));
        offset += THREADS;
    }
}
"""
    common  = _common_round["nearest"].get(otype, "")
    if itype == "f2":
        common += _common_fp16_to_fp32

    code = kernel_code % {
        "common"  : common,
        "itype"   : _ew_types[itype]["type"],
        "otype"   : _ew_types[otype]["type"],
        "cvt_in"  : _ew_types[itype]["cvt"],
        "cvt_out" : _ew_types[otype]["cvt_out"],
    }
    module = SourceModule(code)
    kernel = module.get_function("convert_dtype")
    kernel.prepare("PPII")
    return kernel


class FilterDimShuffle(object):
    def __init__(self, lib, dtype, C, T, R, S, K):
        gridC      = _ceil_div(C, 32)
        gridK      = _ceil_div(K, 32)
        self.lib   = lib
        self.dim   = (C, T, R, S, K)
        self.size  = int(np.prod(self.dim)) * dtype.itemsize
        self.otype = dtype.str[1:]
        self.args  = [(gridK, gridC, T * R * S), (32, 8, 1), None, None, None]
        self.args.extend(_flatten([
            T * R * S * K, R * S * K, S * K, K, T * R * S * C, R * S * C, S * C, C,
            R * S, T, R, S, _magic32(T * R * S, R * S), _magic32(R * S, S)]))

    def bind_params(self, F):
        filter_data    = self.lib.scratch_buffer_offset(self.size)
        self.args[2:5] = (self.lib.stream, filter_data, F.gpudata)
        self.kernel    = _get_shuffle_kernel(self.otype, F.dtype.str[1:])
        return filter_data

    def execute(self):
        self.kernel.prepared_async_call(*self.args)

    def unbind(self):
        self.kernel    = None
        self.args[2:5] = (None,) * 3

@context_dependent_memoize
def _get_shuffle_kernel(otype, itype):

    _shuffle_kernel = r"""
%(common)s

__global__ void filter_dimshuffle(
    %(otype)s* out, const %(itype)s* in,
    int TRSK, int RSK, int SK, int K,
    int TRSC, int RSC, int SC, int C,
    int RS, int T, int R, int S,
    int magic_RS, int shift_RS,
    int magic_S,  int shift_S)
{
    __shared__ %(otype)s tile[32][33];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int bk  = blockIdx.x;
    int bc  = blockIdx.y;
    int trs = blockIdx.z;

    int k  = bk * 32 + tx;
    int c  = bc * 32 + ty;

    int t  = magic_RS * trs; t >>= shift_RS;
    int rs = trs - t*RS;

    int r = magic_S * rs; r >>= shift_S;
    int s = rs - r*S;

    for (int j = 0; j < 32; j += 8)
    {
        int cj = c + j;
        if (cj < C && k < K)
            tile[ty + j][tx] = %(cvt_out)s(in[ cj*TRSK + t*RSK + r*SK + s*K + k ]);
    }
    __syncthreads();

    k = bk * 32 + ty;
    c = bc * 32 + tx;

    // Mirror RST
    s = S - s - 1;
    r = R - r - 1;
    t = T - t - 1;

    for (int i = 0; i < 32; i += 8)
    {
        int ki = k + i;
        if (ki < K && c < C)
            out[ ki*TRSC + t*RSC + r*SC + s*C + c ] = tile[tx][ty + i];
    }
}
"""
    # Allow fp32 in and fp16 out
    cvt_out = ""
    if otype != itype:
        cvt_out = _ew_types[otype]["cvt_out"]

    code = _shuffle_kernel % {
        "common"  : _common_round["nearest"].get(otype, ""),
        "itype"   : _ew_types[itype]["type"],
        "otype"   : _ew_types[otype]["type"],
        "cvt_out" : cvt_out,
    }
    module = SourceModule(code)
    kernel = module.get_function("filter_dimshuffle")
    kernel.prepare("PPIIIIIIIIIIIIIIII")
    return kernel


# Older kernel not currently in use.. but will be
@context_dependent_memoize
def _get_transpose_kernel(dtype):

    _transpose_kernel = r"""
__global__ void filter_transpose(%(type)s* out, const %(type)s* in, int rows, int cols)
{
    __shared__ %(type)s tile[32][33];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int gx = bx * 32 + tx;
    int gy = by * 32 + ty;

    for (int j = 0; j < 32; j += 8)
    {
        int gy8 = gy + j;
        if (gy8 < rows && gx < cols)
            tile[ty + j][tx] = in[gy8*cols + gx];
    }
    __syncthreads();

    gx = by * 32 + tx;
    gy = bx * 32 + ty;

    for (int j = 0; j < 32; j += 8)
    {
        int gy8 = gy + j;
        if (gy8 < cols && gx < rows)
            out[gy8*rows + gx] = tile[tx][ty + j];
    }
}
"""
    code = _transpose_kernel % _ew_types[dtype]
    module = SourceModule(code)
    kernel = module.get_function("filter_transpose")
    kernel.prepare("PPII")
    return kernel


@context_dependent_memoize
def _get_copy_transpose_kernel(dtype, shape, axes=None):

    src = list(range(len(shape)))
    dst = list(axes)

    src_dim = src[-1]
    dst_dim = dst[-1]

    # If the inner dim is the same for both, no need for shared memory tile
    # Then map the outer source dim to the threadIdx.y values
    if src_dim == dst_dim:
        dst_dim = src[0]
        shared_tile = False
    else:
        shared_tile = True

    src_offset = []
    dst_offset = []
    params = []
    values = []
    magic = ""

    # add dims for bounds checking
    for dim in (src_dim, dst_dim):
        params.append("int dim_%s" % dim)
        values.append(shape[dim])

    # collapse src and dst shape by 32
    grid_shape = list(shape)
    grid_shape[src_dim] = _ceil_div(shape[src_dim], 32)
    grid_shape[dst_dim] = _ceil_div(shape[dst_dim], 32)

    # get a src list without dst dim
    src2 = [s for s in src if s != dst_dim]

    # get the name of the first compound index
    blkx_name = compound_idx = "".join(native_str(x) for x in src2)

    # generate the magic number math to extract all indeces
    while len(src2) > 1:

        idx1 = src2[0]
        del src2[0]
        idx2 = "".join(native_str(i) for i in src2)
        div = reduce(mul, (grid_shape[i] for i in src2), 1)

        params.extend(p % idx2 for p in ("int magic_%s", "int shift_%s", "int div_%s"))
        values.extend(_magic64(div))
        values.append(div)

        magic += r"""
    int idx_{1} = div64(idx_{0}, magic_{2}, shift_{2});
    int idx_{2} = idx_{0} - idx_{1}*div_{2};
""".format(compound_idx, idx1, idx2)

        compound_idx = idx2

    # Add params for src strides and generate src offset
    # The param values will be added externally
    for s in src:
        params.append("long long src_str_%d" % s)
        src_offset.append("src_str_%d*idx_%d" % (s, s))

    # Add params for dst strides and generate dst offset
    for d in dst:
        params.append("long long dst_str_%d" % d)
        dst_offset.append("dst_str_%d*idx_%d" % (d, d))

    num_strides = len(src) + len(dst)

    if shared_tile:
        copy_transpose = r"""
%(common)s

__global__ void copy_transpose(%(type)s* out, const %(type)s* in, %(params)s)
{
    __shared__ %(type)s tile[32][33];

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int idx_%(blk)s = blockIdx.x;
    int idx_%(dst)s = blockIdx.y;

    %(magic)s

    idx_%(src)s = (idx_%(src)s << 5) + tid_x;
    idx_%(dst)s = (idx_%(dst)s << 5) + tid_y;

    const %(type)s* in00 = in   + %(src_offset)s;
    const %(type)s* in08 = in00 + src_str_%(dst)s*8;
    const %(type)s* in16 = in08 + src_str_%(dst)s*8;
    const %(type)s* in24 = in16 + src_str_%(dst)s*8;

    bool b%(src)s = idx_%(src)s < dim_%(src)s;

    if (idx_%(dst)s +  0 < dim_%(dst)s && b%(src)s) tile[tid_y +  0][tid_x] = *in00;
    if (idx_%(dst)s +  8 < dim_%(dst)s && b%(src)s) tile[tid_y +  8][tid_x] = *in08;
    if (idx_%(dst)s + 16 < dim_%(dst)s && b%(src)s) tile[tid_y + 16][tid_x] = *in16;
    if (idx_%(dst)s + 24 < dim_%(dst)s && b%(src)s) tile[tid_y + 24][tid_x] = *in24;

    __syncthreads();

    %(type)s val00 = tile[tid_x][tid_y +  0];
    %(type)s val08 = tile[tid_x][tid_y +  8];
    %(type)s val16 = tile[tid_x][tid_y + 16];
    %(type)s val24 = tile[tid_x][tid_y + 24];

    idx_%(src)s += tid_y - tid_x;
    idx_%(dst)s += tid_x - tid_y;

    bool b%(dst)s = idx_%(dst)s < dim_%(dst)s;

    %(type)s* out00 = out   + %(dst_offset)s;
    %(type)s* out08 = out00 + dst_str_%(src)s*8;
    %(type)s* out16 = out08 + dst_str_%(src)s*8;
    %(type)s* out24 = out16 + dst_str_%(src)s*8;

    if (idx_%(src)s +  0 < dim_%(src)s && b%(dst)s) *out00 = val00;
    if (idx_%(src)s +  8 < dim_%(src)s && b%(dst)s) *out08 = val08;
    if (idx_%(src)s + 16 < dim_%(src)s && b%(dst)s) *out16 = val16;
    if (idx_%(src)s + 24 < dim_%(src)s && b%(dst)s) *out24 = val24;
}
"""
    else:
        copy_transpose = r"""
%(common)s

__global__ void copy_transpose(%(type)s* out, const %(type)s* in, %(params)s)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int idx_%(blk)s = blockIdx.x;
    int idx_%(dst)s = blockIdx.y;

    %(magic)s

    idx_%(src)s = (idx_%(src)s << 5) + tid_x;
    idx_%(dst)s = (idx_%(dst)s << 5) + tid_y;

    bool b%(src)s    = idx_%(src)s      < dim_%(src)s;
    bool b%(dst)s_00 = idx_%(dst)s +  0 < dim_%(dst)s && b%(src)s;
    bool b%(dst)s_08 = idx_%(dst)s +  8 < dim_%(dst)s && b%(src)s;
    bool b%(dst)s_16 = idx_%(dst)s + 16 < dim_%(dst)s && b%(src)s;
    bool b%(dst)s_24 = idx_%(dst)s + 24 < dim_%(dst)s && b%(src)s;

    %(type)s val00 = 0;
    %(type)s val08 = 0;
    %(type)s val16 = 0;
    %(type)s val24 = 0;

    const %(type)s* in00 = in   + %(src_offset)s;
    const %(type)s* in08 = in00 + src_str_%(dst)s*8;
    const %(type)s* in16 = in08 + src_str_%(dst)s*8;
    const %(type)s* in24 = in16 + src_str_%(dst)s*8;

    if (b%(dst)s_00) val00 = *in00;
    if (b%(dst)s_08) val08 = *in08;
    if (b%(dst)s_16) val16 = *in16;
    if (b%(dst)s_24) val24 = *in24;

    %(type)s* out00 = out   + %(dst_offset)s;
    %(type)s* out08 = out00 + dst_str_%(dst)s*8;
    %(type)s* out16 = out08 + dst_str_%(dst)s*8;
    %(type)s* out24 = out16 + dst_str_%(dst)s*8;

    if (b%(dst)s_00) *out00 = val00;
    if (b%(dst)s_08) *out08 = val08;
    if (b%(dst)s_16) *out16 = val16;
    if (b%(dst)s_24) *out24 = val24;
}
"""
    code = copy_transpose % dict(
        common=_div64,
        type=_ew_types[dtype[1:]]["type"],
        params=", ".join(params),
        blk=blkx_name,
        src=src_dim,
        dst=dst_dim,
        magic=magic,
        src_offset=" + ".join(src_offset),
        dst_offset=" + ".join(dst_offset)
    )
    # print code
    module = SourceModule(code)
    kernel = module.get_function("copy_transpose")
    kernel.prepare("PP" + ("I" * (len(params) - num_strides)) + "q" * num_strides)

    grid_x = grid_shape[src_dim]
    grid_y = grid_shape[dst_dim]
    for s in src:
        if s not in (src_dim, dst_dim):
            grid_x *= grid_shape[s]

    kernel.grid = (grid_x, grid_y, 1)
    kernel.block = (32, 8, 1)
    kernel.args = tuple(values)

    return kernel
