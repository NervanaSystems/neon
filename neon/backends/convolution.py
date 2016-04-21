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

import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize
from neon.backends import kernel_specs
from neon.backends.cuda_templates import _common_round, _ew_types
from math import ceil
from operator import mul
import sys
import os.path
import shelve

if sys.version_info >= (3, 0):
    from functools import reduce


class KernelGroup(object):
    def __init__(self, lib, dtype):

        self.lib = lib
        self.dtype = dtype
        self.dtype_str = dtype.str[1:]
        self.vec_size = 4 if dtype.itemsize == 4 else 8

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

    def __str__(self):
        raise TypeError("please implement __str__ to describe kernel params for logging.")

    def bind_params(self, *args):
        raise TypeError("bind_params not implemented.")

    def execute(self, repeat=1, unbind=True):
        raise TypeError("execute not implemented.")

    def init_bsum(self, bsum, flags):
        flags |= self.flags
        if bsum:
            bsum_gpudata = bsum.gpudata
            self.bsum_zero = [bsum_gpudata, 0, bsum.size, self.lib.stream]
            flags |= 4
        else:
            bsum_gpudata = 0
            self.bsum_zero = 0
            flags &= ~4
        return bsum_gpudata, flags

    def k_partitions(self, K, tiles):
        k = K
        partitions = []
        for tile_K in tiles:
            grid_K = (k + tiles[-1] - 1) // tile_K
            if grid_K > 0:
                partitions.append([tile_K, grid_K, K-k])
                k -= grid_K * tile_K
            if k <= 0:
                break
        return partitions

    def xprop_kernels(self, op, tile_dim, tile_N, grid_N, K, tiles, PQM, RST, args):

        self.kernels = []
        for tile_K, grid_K, offset_K in self.k_partitions(K, tiles):

            kernel_name = "%s_%s_%s%d_N%d" % (self.clss, op, tile_dim, tile_K, tile_N)

            block = (kernel_specs.kernels[kernel_name]["threads"], 1, 1)
            if RST > 1:
                grid = (PQM, grid_K, grid_N)
            else:
                grid = (grid_K, grid_N, PQM)

            params = [
                kernel_name, grid, block, None,
                None, None, None, None, None, None, None, offset_K]
            params.extend(args)

            self.kernels.append(params)

class FpropCuda(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 bsum):

        super(FpropCuda, self).__init__(lib, dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % self.vec_size == 0, "K dim must be multiple of %d" % self.vec_size

        magic_PQ = _magic64(P*Q)
        magic_Q = _magic64(Q)
        magic_S = _magic32(R*S+32, S)
        HWN = H * W * N
        RST = R * S * T
        KRST = K * RST
        PQ = P * Q
        PQN = PQ * N
        from neon.backends.kernels.cuda.convolution import _get_conv_kernel
        self.kernel = _get_conv_kernel(dtype=self.dtype.str[1:], filter_size=R*S,
                                       bsum=bsum, operation="fprop")
        grid = (PQ * (-(-N // 32)), (-(-K // 32)), 1)
        block = (8, 8, 1)
        static_kernel_args = _flatten([C, D, H, W, N, T, R, S, K, M, P, Q,
                                       str_w, str_h, pad_w, pad_h,
                                       HWN // 4, KRST // 4, PQN // 4,
                                       PQ, 0, 0,
                                       magic_PQ, magic_Q, magic_S])
        self.launch_args = [grid, block] + [None] * 7 + static_kernel_args

        self.shared = RST * 4 * 2
        self.flags = (bsum and 4)

    def bind_params(self, I, F, O, alpha, beta, bsum, flags=0):

        assert I.dtype == F.dtype == O.dtype

        bsum_gpudata, flags = self.init_bsum(bsum, flags)

        self.launch_args[2:9] = (self.lib.stream, alpha, beta,
                                 I.gpudata, F.gpudata, O.gpudata, bsum_gpudata)

    def execute(self, repeat=1, unbind=True):

        for r in range(repeat):

            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)

            self.kernel.prepared_async_call(*self.launch_args, shared_size=self.shared)

        if unbind:
            self.bsum_zero = None
            self.launch_args[2:9] = (None,) * 7

    def __str__(self):
        return "FpropCuda"


class BpropCuda(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 bsum):

        super(BpropCuda, self).__init__(lib, dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % self.vec_size == 0, "K dim must be multiple of %d" % self.vec_size

        magic_HW = _magic64(H*W)
        magic_W = _magic64(W)
        magic_RS = _magic32(R*S*T+32, R*S)
        magic_S = _magic32(R*S+32, S)
        HW = H * W
        HWN = HW * N
        RST = R * S * T
        CRST = C * RST
        PQ = P * Q
        PQN = PQ * N

        self.bsum = bsum
        from neon.backends.kernels.cuda.convolution import _get_conv_kernel
        self.kernel = _get_conv_kernel(dtype=self.dtype.str[1:], filter_size=R*S,
                                       bsum=bsum, operation="bprop")
        grid = (HW * (-(-N // 32)), -(-C // 32), 1)
        block = (8, 8, 1)
        static_kernel_args = _flatten([K, M, P, Q, N, T, R, S, C, D, H, W,
                                       str_w, str_h, pad_w, pad_h,
                                       PQN // 4, CRST // 4, HWN // 4,
                                       HW, 0, 0,
                                       magic_HW, magic_W, magic_S])
        self.launch_args = [grid, block] + [None] * 7 + static_kernel_args

        self.shared = R*S*T * 4 * 2
        self.flags = (bsum and 4)

        # generate the kernel args for dim shuffling CTRSK => KTRSC
        shuffle_grid = (_ceil_div(K, 32), _ceil_div(C, 32), R*S*T)
        self.shuffle_size = C*T*R*S*K*dtype.itemsize
        self.shuffle_args = [shuffle_grid, (32, 8, 1), None, None, None]
        self.shuffle_args.extend(_flatten([
            R*S*T*K, R*S*K, S*K, K,
            R*S*T*C, R*S*C, S*C, C,
            R*S, T, R, S, magic_RS, magic_S]))

        lib.set_scratch_size(self.shuffle_size)

    def bind_params(self, I, F, O, alpha, beta, bsum, flags=0):

        assert I.dtype == F.dtype == O.dtype
        if self.bsum:
            assert bsum is not None, "must use initialized bsum config"

        bsum_gpudata, flags = self.init_bsum(bsum, flags)

        filter_temp = self.lib.scratch_buffer(self.shuffle_size)

        self.shuffle_args[2:5] = (self.lib.stream, filter_temp, F.gpudata)
        self.launch_args[2:9] = (self.lib.stream, alpha, beta,
                                 I.gpudata, filter_temp, O.gpudata, bsum_gpudata)

    def execute(self, repeat=1, unbind=True):
        C = self.shuffle_args[12]
        assert C >= 4, "C dim must be 4 or greater for CUDA C backprop kernel"

        shuffle_kernel = _get_shuffle_kernel(self.dtype.str[1:])

        for r in range(repeat):

            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)

            shuffle_kernel.prepared_async_call(*self.shuffle_args)
            self.kernel.prepared_async_call(*self.launch_args, shared_size=self.shared)

        if unbind:
            self.bsum_zero = None
            self.shuffle_args[2:5] = (None,) * 3
            self.launch_args[2:9] = (None,) * 7

    def __str__(self):
        return "BpropCuda"


class UpdateCuda(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w):

        super(UpdateCuda, self).__init__(lib, dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"

        HWN = H * W * N
        RS = R * S
        RST = RS * T
        KRST = K * RST
        CRSTK = KRST * C
        PQ = P * Q
        PQN = PQ * N
        magic_S = _magic32(R*S+32, S)

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

        from neon.backends.kernels.cuda.convolution import _get_conv_kernel
        self.kernel = _get_conv_kernel(dtype=self.dtype.str[1:], filter_size=R*S,
                                       bsum=False, operation="update")
        grid = (pq_blocks * (-(-K // 32)), (-(-(C*RS) // 32)), 1)
        block = (8, 32, 1)
        static_kernel_args = _flatten([C, D, H, W, N, T, R, S, K, M, P, Q,
                                       str_w, str_h, pad_w, pad_h,
                                       HWN // 4, KRST // 4, PQN // 4,
                                       PQ, grid_P, grid_Q,
                                       magic_PQ, magic_Q, magic_S])
        self.launch_args = [grid, block] + [None] * 7 + static_kernel_args

        lib.set_scratch_size((self.determ or C*T*R*S*K)*4)

    def update_grid(self, kernel_name, base_blocks, P, Q, SM_count):

        threads = kernel_specs.kernels[kernel_name]["threads"]
        occupancy = kernel_specs.kernels[kernel_name]["occupancy"]

        # warps per scheduler for one block
        occ_per_block = threads / (32.0 * 4.0 * SM_count)

        grid = []
        for p in range(1, P+1):
            for q in range(1, Q+1):

                occup = p*q*base_blocks * occ_per_block
                groups = occup / occupancy
                slots = ceil(groups)

                # This is a heuristic that keeps the balance of work accross the SMs
                # while also maximizing the work that each block does
                heuristic = min(abs(x - slots) for x in range(4, 8)) + (slots - groups) / 100.0

                grid.append((p, q, heuristic))

        grid.sort(key=lambda x: x[-1])

        return (grid[0][0], grid[0][1], threads)

    def bind_params(self, I, E, O, alpha):

        assert I.dtype == E.dtype

        if O.dtype.type is not np.float32:

            update_temp = self.lib.scratch_buffer((self.determ or O.size)*4)

            self.convert_args = [update_temp, "f4", O, False]
        else:
            update_temp = O.gpudata
            self.convert_args = False

        self.zero_args = [update_temp, 0, O.size, self.lib.stream]

        beta = 0.0
        bsum_gpudata = 0
        self.launch_args[2:9] = (self.lib.stream, alpha, beta,
                                 I.gpudata, E.gpudata, O.gpudata, bsum_gpudata)

    def execute(self, repeat=1, unbind=True):

        for r in range(repeat):

            drv.memset_d32_async(*self.zero_args)

            self.kernel.prepared_async_call(*self.launch_args)

            if self.convert_args:
                _fp_convert(*self.convert_args)

        if unbind:
            self.zero_args = self.convert_args = None
            self.launch_args[2:9] = (None,) * 7

    def __str__(self):
        return "UpdateCuda"

class FpropDirect(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 relu, bsum):

        super(FpropDirect, self).__init__(lib, dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % self.vec_size == 0, "K dim must be multiple of %d" % self.vec_size

        tile_N = 128 if N > 64 else 64
        grid_N = _ceil_div(N, tile_N)
        tile_K = (128, 64, 32) if tile_N == 128 else (128, 64)

        magic_PQ = _magic64(P*Q)
        magic_Q = _magic64(Q)
        magic_RS = _magic32(R*S*T+32, R*S)
        magic_S = _magic32(R*S+32, S)

        self.xprop_kernels(
            "fprop", "K", tile_N, grid_N, K, tile_K, P*Q*M, R*S*T,
            _flatten([N, K, D, H, W, W*N, H*W*N, D*H*W*N,
                      C, K*R*S*T, R*S*T, R*S, magic_RS, S, magic_S,
                      pad_d, pad_h, pad_w, str_d, str_h, str_w,
                      Q, P*Q, Q*N, P*Q*N, M*P*Q*N, magic_Q, magic_PQ]))

        self.shared = R*S*T * 4 * 2
        self.flags = (relu and 2) + (bsum and 4)

    def bind_params(self, I, F, O, alpha, beta, bsum, flags=0):

        assert I.dtype == F.dtype == O.dtype

        bsum_gpudata, flags = self.init_bsum(bsum, flags)

        for kernel in self.kernels:
            kernel[3:11] = (self.lib.stream, bsum_gpudata, O.gpudata, I.gpudata, F.gpudata,
                            alpha, beta, flags)

    def execute(self, repeat=1, unbind=True):

        for r in range(repeat):

            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)

            for kernel_params in self.kernels:
                kernel = kernel_specs.get_kernel(kernel_params[0])
                kernel.prepared_async_call(*kernel_params[1:], shared_size=self.shared)

        if unbind:
            self.bsum_zero = None
            for kernel_params in self.kernels:
                kernel_params[3:11] = (None,) * 8

    def __str__(self):
        return "FpropDirect " + str([k[0] for k in self.kernels])

class BpropDirect(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 relu, bsum):

        super(BpropDirect, self).__init__(lib, dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert C % self.vec_size == 0, "C dim must be multiple of %d" % self.vec_size

        tile_N = 128 if N > 64 else 64
        grid_N = _ceil_div(N, tile_N)
        tile_C = (128, 64, 32) if tile_N == 128 else (128, 64)

        magic_HW = _magic64(H*W)
        magic_W = _magic64(W)
        magic_RS = _magic32(R*S*T+32, R*S)
        magic_S = _magic32(R*S+32, S)
        magic_str_w = _magic32(W + S, str_w)
        magic_str_h = _magic32(H + R, str_h)
        magic_str_d = _magic32(D + T, str_d)

        self.xprop_kernels(
            "bprop", "C", tile_N, grid_N, C, tile_C, D*H*W, R*S*T,
            _flatten([N, C, M, P, Q, Q*N, P*Q*N, M*P*Q*N,
                      K, C*R*S*T, R*S*T, R*S, magic_RS, S, magic_S,
                      pad_d, pad_h, pad_w, str_d, str_h, str_w,
                      W, H*W, W*N, H*W*N, D*H*W*N, magic_W, magic_HW,
                      R, T, magic_str_w, magic_str_h, magic_str_d]))

        self.shared = R*S*T * 4 * 2
        self.flags = (relu and 2) + (bsum and 4)

        # generate the kernel args for dim shuffling CTRSK => KTRSC
        shuffle_grid = (_ceil_div(K, 32), _ceil_div(C, 32), R*S*T)
        self.shuffle_size = C*T*R*S*K*dtype.itemsize
        self.shuffle_args = [shuffle_grid, (32, 8, 1), None, None, None]
        self.shuffle_args.extend(_flatten([
            R*S*T*K, R*S*K, S*K, K,
            R*S*T*C, R*S*C, S*C, C,
            R*S, T, R, S, magic_RS, magic_S]))

        lib.set_scratch_size(self.shuffle_size)

    def bind_params(self, I, F, O, alpha, beta, bsum, flags=0):

        assert I.dtype == F.dtype == O.dtype

        bsum_gpudata, flags = self.init_bsum(bsum, flags)

        filter_temp = self.lib.scratch_buffer(self.shuffle_size)

        self.shuffle_args[2:5] = (self.lib.stream, filter_temp, F.gpudata)

        for kernel in self.kernels:
            kernel[3:11] = (self.lib.stream, bsum_gpudata, O.gpudata, I.gpudata, filter_temp,
                            alpha, beta, flags)

    def execute(self, repeat=1, unbind=True):

        shuffle_kernel = _get_shuffle_kernel(self.dtype_str)

        for r in range(repeat):

            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)

            shuffle_kernel.prepared_async_call(*self.shuffle_args)

            for kernel_params in self.kernels:
                kernel = kernel_specs.get_kernel(kernel_params[0])
                kernel.prepared_async_call(*kernel_params[1:], shared_size=self.shared)

        if unbind:
            self.bsum_zero = None
            self.shuffle_args[2:5] = (None,) * 3
            for kernel_params in self.kernels:
                kernel_params[3:11] = (None,) * 8

    def __str__(self):
        return "BpropDirect " + str([k[0] for k in self.kernels])


class XpropDirect2(KernelGroup):

    def __init__(self, op, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 relu):

        super(XpropDirect2, self).__init__(lib, dtype)

        assert N % 4 == 0 or N in (1,2), "N dim must be multiple of 4 or equal to 1 or 2"

        for blockN in (32,16,8,4,2,1):
            if N % blockN == 0:
                break

        self.params = (N, C, K, D, H, W, T, R, S, M, P, Q,
                       pad_d, pad_h, pad_w, str_d, str_h, str_w)

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
        gridP2  = max(gridP // 2, 1)
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

        grid = (gridM*gridP*gridQ*nk, gridK//k, gridN//n)

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
            str_d, str_h, str_w, pad_d, pad_h, pad_w,
            D*H*W*N, H*W*N, W*N, M*P*Q*N, P*Q*N, Q*N,
            PQnk, Qnk, nk, n, k, magic_PQnk, magic_Qnk, magic_nk, magic_k,
            max(K-32,0), K*32*dtype.itemsize, TRSK, TRS, RS, S, magic_RS, magic_S, gridP2, gridQ,
            superM, superP, superQ, superN,
            shiftM, shiftP, shiftQ, shiftN,
            SuperM, SuperP, SuperQ, SuperN ]))

        if N >= 32:
            self.shared = (T*R*S + 1) * 4 * 2
        else:
            self.shared = T*R*S * 4 * (32 >> shiftN)
        self.relu = relu

    def bind_params(self, I, F, O, alpha, beta, bsum, no_op=0):

        # TODO: allow hybrid types
        assert I.dtype == F.dtype == O.dtype

        # TODO: expose more compound operations
        self.kernel_options = self.kernel_opts
        if bsum:
            bsum_gpudata = bsum.gpudata
            self.bsum_zero = (bsum_gpudata, 0, bsum.size, self.lib.stream)
            self.kernel_options += ("bsum",)
        else:
            bsum_gpudata = 0
            self.bsum_zero = None

        if beta:
            self.kernel_options += ("beta",)

        if self.relu:
            self.kernel_options += ("relu",)

        if self.trans_size:
            filter_temp = self.lib.scratch_buffer(self.trans_size)
            self.trans_args[2:5] = (self.lib.stream, filter_temp, F.gpudata)
        else:
            filter_temp = F.gpudata

        self.kernel_args[2:11] = (self.lib.stream, bsum_gpudata, O.gpudata, O.gpudata, I.gpudata, filter_temp,
                             alpha, beta, no_op)

    def execute(self, repeat=1, unbind=True):

        kernel = kernel_specs.get_kernel(self.kernel_name, self.kernel_options)

        if self.trans_size:
            trans_kernel = _get_shuffle_kernel(self.dtype_str)

        for r in range(repeat):

            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)

            if self.trans_size:
                trans_kernel.prepared_async_call(*self.trans_args)

            kernel.prepared_async_call(*self.kernel_args, shared_size=self.shared)

        if unbind:
            self.bsum_zero = None
            self.kernel_args[2:11] = (None,) * 9
            if self.trans_size:
                self.trans_args[2:5] = (None,) * 3

    def __str__(self):
        (N, C, K, D, H, W, T, R, S, M, P, Q,
        pad_d, pad_h, pad_w, str_d, str_h, str_w) = self.params
        return "%s NCK:(%3d,%3d,%3d) HW:(%3d,%3d) RS:(%d,%d) str:(%d,%d)" % (
            self.kernel_name, N, C, K, H, W, R, S, str_h, str_w)

class FpropDirect2(XpropDirect2):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 relu, bsum):

        self.trans_size = 0

        super(FpropDirect2, self).__init__("fprop", lib, dtype,
            N, C, K, D, H, W, T, R, S, M, P, Q,
            pad_d, pad_h, pad_w, str_d, str_h, str_w, relu)

class BpropDirect2(XpropDirect2):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 relu, bsum):

        # invert padding
        pad_d = T - pad_d - 1
        pad_h = R - pad_h - 1
        pad_w = S - pad_w - 1

        # Swap C<=>K and DHW<=>MPQ
        super(BpropDirect2, self).__init__("bprop", lib, dtype,
            N, K, C, M, P, Q, T, R, S, D, H, W,
            pad_d, pad_h, pad_w, str_d, str_h, str_w, relu)

        self.kernel_args.extend(_flatten([
            _magic32(D + T, str_d),
            _magic32(H + R, str_h),
            _magic32(W + S, str_w) ]))

        gridC = _ceil_div(C, 32)
        gridK = _ceil_div(K, 32)

        self.trans_size = C*T*R*S*K * dtype.itemsize
        self.trans_args = [(gridK, gridC, T*R*S), (32, 8, 1), None, None, None]
        self.trans_args.extend(_flatten([
            T*R*S*K, R*S*K, S*K, K,
            T*R*S*C, R*S*C, S*C, C,
            R*S, T, R, S, _magic32(T*R*S, R*S), _magic32(R*S, S)]))

        lib.set_scratch_size(self.trans_size)

class UpdateDirect2(KernelGroup):

    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w):

        assert N % 4 == 0, "N dim must be multiple of 4"

        super(UpdateDirect2, self).__init__(lib, dtype)

        SMs = _get_sm_count()

        self.autotune_key = " ".join(str(x) for x in (
            "direct_updat_64x32", SMs, dtype.itemsize, lib.deterministic > 0,
            N, C, K, D, H, W, T, R, S, M, P, Q ))

        self.autotune_db_file = os.path.join(lib.cache_dir, "autotune.db")

        self.params = (N, C, K, D, H, W, T, R, S, M, P, Q,
                       pad_d, pad_h, pad_w, str_d, str_h, str_w)
        self.init(self.params)

        lib.set_scratch_size(self.output_size)

        # allow for .5 seconds worth of warmup when autotuning
        # assume 5 Tflops on 24 SMs
        self.warmup = min(max(int(2e12 / (M*P*Q*K*N*C*T*R*S*2.0) * (SMs / 24.0)), 1), 1000)

    def init(self, params, autotune=False):

        (N, C, K, D, H, W, T, R, S, M, P, Q,
         pad_d, pad_h, pad_w, str_d, str_h, str_w) = self.params

        for blockN in (32,16,8,4):
            if N % blockN == 0:
                break

        sb_params = {
            #blkN: supM, shfM, supP, shfP, supQ, shfQ, supN
            32 : ( 0x000, 0,   0x000, 0,   0x000, 0,   7 ), # 1x1  nnn
            16 : ( 0x000, 0,   0x000, 0,   0x102, 1,   3 ), # 1x2  xnn
            8  : ( 0x000, 0,   0x102, 1,   0x101, 1,   1 ), # 2x2  yxn
            4  : ( 0x000, 0,   0x102, 1,   0x200, 2,   0 ), # 2x4  yxx
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
            autotune_db = shelve.open(self.autotune_db_file)

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

            autotune_db.close()

        # print P, Q, blockP, blockQ
        # print GP, GQ, strideP, strideQ

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

        # If output grid is 1, don't use atomics.  Kernel is deterministic by default
        if GM*strideP*strideQ == 1:
            self.determ_size  = 0
            self.determ_shape = False
            self.zero         = False
            options.append("D")

        elif self.lib.deterministic:
            self.determ_size  = GM*strideP*strideQ * CTRSK
            self.determ_shape = (GM*strideP*strideQ, CTRSK)
            self.zero         = False
            options.append("D")

        else:
            self.determ_size  = 0
            self.determ_shape = False
            self.zero         = True

        grid = (GM*strideP*strideQ*kc, GC//c, GK//k)
        #print grid, c, k

        self.kernel_opts = tuple(options)
        self.kernel_name = "%s_direct_updat_64x32" % self.clss
        self.kernel_args = [grid, (128,1,1), None, None, None, None, None]
        self.kernel_args.extend(_flatten([
            C, D, H, W, N, K, M, P, Q,
            str_d, str_h, str_w, pad_d, pad_h, pad_w,
            D*H*W*N, H*W*N, W*N, M*P*Q*N*16*itemsize, M*P*Q*N, P*Q*N, Q*N,
            PQkc, Qkc, kc, c, k, magic_PQkc, magic_Qkc, magic_kc, magic_c,
            CTRSK, CTRS, TRS, RS, S, magic_TRS, magic_RS, magic_S,
            superM, superP, superQ, superN, shiftM, shiftP, shiftQ,
            strideP, strideQ, strideP*strideQ, GP, GQ,
            loopX, loopXp, loopQ, loopQp, blockN, blockN*itemsize ]))

        self.output_size = (self.determ_size or (self.dtype.type != np.float32 and CTRSK)) * 4

    def autotune(self, I, E, O):

        # print "autotune: ", self.autotune_key

        start, stop = self.lib.get_events()

        # Only need to do warmup once
        if not self.lib.warmup:
            self.lib.warmup = True
            # warmup  with a conservative set of params
            self.init(self.params, autotune=(min(self.GP,192), 1))
            self.bind_params(I, E, O, 1.0)
            self.execute(repeat=self.warmup, unbind=False)

        # we want at least this many blocks
        block_slots = _get_sm_count()
        # loops for given size of N
        loopsN = max(self.params[0] // 32, 1)

        GP = float(self.GP)
        GQ = float(self.GQ)
        small_set = GP * GQ <= 512

        results = []
        sys.stdout.write("Autotune " + str(self))
        progress = 0
        for threshold in (True, False):
            for strideP in range(1, self.GP+1):
                for strideQ in range(1, self.GQ+1):
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
                        #     print strideP, strideQ, blocks, round(depth,1)

            # if we got any results, no need to disable the filter
            if len(results) > 0:
                break
        sys.stdout.write('\n')

        results.sort()
        settings = results[0][1]
        # for res in results[0:10]:
        #     print res
        autotune_db = shelve.open(self.autotune_db_file)
        autotune_db[self.autotune_key] = settings
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
        else:
            update_temp = O.gpudata
            self.convert_args = False

        if self.zero:
            self.zero_args = [update_temp, 0, O.size, self.lib.stream]

        self.kernel_args[2:7] = (self.lib.stream, update_temp, I.gpudata, E.gpudata, alpha)

    def execute(self, repeat=1, unbind=True):

        kernel = kernel_specs.get_kernel(self.kernel_name, self.kernel_opts)

        for r in range(repeat):

            if self.zero:
                drv.memset_d32_async(*self.zero_args)

            kernel.prepared_async_call(*self.kernel_args)

            if self.convert_args:
                _fp_convert(*self.convert_args)

        if unbind:
            self.zero_args = self.convert_args = None
            self.kernel_args[2:7] = (None,) * 5

    def __str__(self):
        (N, C, K, D, H, W, T, R, S, M, P, Q,
        pad_d, pad_h, pad_w, str_d, str_h, str_w) = self.params
        return "%s NCK:(%3d,%3d,%3d) HW:(%3d,%3d) RS:(%d,%d) str:(%d,%d)" % (
            self.kernel_name, N, C, K, H, W, R, S, str_h, str_w)




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
    return -sorted([(abs(i-div), -i) for i in range(1, maxdiv) if val % i == 0])[0][1]


@context_dependent_memoize
def _get_sm_count():
    attributes = drv.Context.get_device().get_attributes()
    return attributes[drv.device_attribute.MULTIPROCESSOR_COUNT]


def _fp_convert(src_data, src_type, dest_tensor, reduce_shape):

    if reduce_shape:

        kernel = _get_reduce_kernel(dest_tensor.dtype.str[1:])
        blocks = _ceil_div(reduce_shape[1], 32)
        kernel.prepared_async_call((blocks, 1, 1), (32, 1, 1),
                                   dest_tensor.backend.stream,
                                   dest_tensor.gpudata,
                                   src_data,
                                   reduce_shape[1],
                                   reduce_shape[0]*reduce_shape[1])

    else:
        from neon.backends.nervanagpu import GPUTensor
        from neon.backends.float_ew import _get_compound_kernel, _get_fast_ew_dims

        # quick wrapper to convert raw fp32 scratch data to a destination tensor
        shape, strides = _get_fast_ew_dims(dest_tensor.size)
        kernel_args = [0,
                       dest_tensor.gpudata, strides[0], strides[1],
                       src_data, strides[0], strides[1],
                       shape[1]]

        kernel = _get_compound_kernel((
            (GPUTensor, 0, dest_tensor.dtype.str[1:], 0, False),
            (GPUTensor, 1, src_type, 0, False),
            ('assign', 0, False, 32)),
            dest_tensor.backend.compute_capability)
        kernel.prepared_async_call((shape[0], 1, 1),
                                   (32, 1, 1),
                                   dest_tensor.backend.stream,
                                   *kernel_args)


# fast axis=0 reduction kernel used for deterministic update
@context_dependent_memoize
def _get_reduce_kernel(dtype):

    _reduce_kernel = r"""
%(common)s

__global__ void reduce(%(type)s* out, const float* in, int CRSTK, int PQCRSTK)
{
    int offset = blockIdx.x * 32 + threadIdx.x;

    if (offset < CRSTK)
    {
        float sum = 0.0f;
        for (int i = offset; i < PQCRSTK; i += CRSTK)
        {
            sum += __ldg(in + i);
        }
        out[offset] = %(cvt_out)s(sum);
    }
}
"""
    template_vals = {
        "common": _common_round["nearest"].get(dtype, ""),
        "type": _ew_types[dtype]["type"],
    }
    if dtype == "f2":
        template_vals["cvt_out"] = "fp32_to_fp16"
    elif dtype == "f4":
        template_vals["cvt_out"] = ""
    elif dtype == "x2":
        template_vals["cvt_out"] = "fp32_to_int16"
    else:
        raise TypeError("Missing reduction type")

    code = _reduce_kernel % template_vals
    module = SourceModule(code)
    kernel = module.get_function("reduce")
    kernel.prepare("PPII")
    return kernel


@context_dependent_memoize
def _get_transpose_kernel(dtype):

    _transpose_kernel = r"""
__global__ void transpose(%(type)s* out, const %(type)s* in, int rows, int cols)
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
    kernel = module.get_function("transpose")
    kernel.prepare("PPII")
    return kernel


@context_dependent_memoize
def _get_shuffle_kernel(dtype):

    _shuffle_kernel = r"""
__global__ void dimShuffle(
    %(type)s* out, const %(type)s* in,
    int TRSK, int RSK, int SK, int K,
    int TRSC, int RSC, int SC, int C,
    int RS, int T, int R, int S,
    int magic_RS, int shift_RS,
    int magic_S,  int shift_S)
{
    __shared__ %(type)s tile[32][33];

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
            tile[ty + j][tx] = in[ cj*TRSK + t*RSK + r*SK + s*K + k ];
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
    code = _shuffle_kernel % _ew_types[dtype]
    module = SourceModule(code)
    kernel = module.get_function("dimShuffle")
    kernel.prepare("PPIIIIIIIIIIIIIIII")
    return kernel


@context_dependent_memoize
def _get_copy_transpose_kernel(dtype, shape, axes=None):

    src = range(len(shape))
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
    blkx_name = compound_idx = "".join(str(x) for x in src2)

    # generate the magic number math to extract all indeces
    while len(src2) > 1:

        idx1 = src2[0]
        del src2[0]
        idx2 = "".join(str(i) for i in src2)
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
        params.append("int src_str_%d" % s)
        src_offset.append("src_str_%d*idx_%d" % (s, s))

    # Add params for dst strides and generate dst offset
    for d in dst:
        params.append("int dst_str_%d" % d)
        dst_offset.append("dst_str_%d*idx_%d" % (d, d))

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
    kernel.prepare("PP" + "I"*len(params))

    grid_x = grid_shape[src_dim]
    grid_y = grid_shape[dst_dim]
    for s in src:
        if s not in (src_dim, dst_dim):
            grid_x *= grid_shape[s]

    kernel.grid = (grid_x, grid_y, 1)
    kernel.block = (32, 8, 1)
    kernel.args = tuple(values)

    return kernel
