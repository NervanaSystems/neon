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
Definition of the GPU layers
These layers are mainly used for old benchmarking code,
but they also cache all the computed params for complex layers.
TODO: clean up merge with CPU layers
TODO: remove any non-param caching code, neon layers should replace benchmark code.
"""
import numpy as np
import pycuda.driver as drv
from neon.backends import kernel_specs
from pycuda.tools import context_dependent_memoize
from operator import mul
from math import ceil
import sys

if sys.version_info >= (3, 0):
    from functools import reduce


class Layer(object):

    """
    GPU Layer base class
    """

    def __init__(self, lib, dtype, N, dtypeU=None):

        if hasattr(dtype, 'type'):
            self.dtype = dtype
        else:
            self.dtype = np.dtype(dtype)

        self.N      = N
        self.dtypeU = dtype if dtypeU is None else dtypeU
        self.lib    = lib
        self.flops  = 0
        self.sizeI  = 0
        self.sizeO  = 0
        self.sizeF  = 0
        self.weights   = None
        self.fprop_in  = None
        self.fprop_out = None
        self.bprop_in  = None
        self.bprop_out = None

        self.learning_rate = 0.0

    def init_activations(self, fprop_out=None):

        if fprop_out is not None:
            self.fprop_out = fprop_out
        else:
            self.fprop_out = self.lib.empty(self.dimO, dtype=self.dtype)

        self.act_stats = self.lib.empty((self.dimO2[0], 1), dtype=np.float32)

    def init_deltas(self, shared=None):

        if shared is None:
            self.bprop_out = self.lib.empty(self.dimI, dtype=self.dtype)
        else:
            self.bprop_out = shared[0].share(self.dimI)
            shared.reverse()

        self.delta_stats = self.lib.empty((self.dimI2[0], 1), dtype=np.float32)

    def init_weights(self, loc=0.0, scale=0.1, shared=None, zeros=False):

        if self.sizeF > 0:
            if zeros:
                self.weights  = self.lib.zeros(self.dimF, dtype=self.dtype)
            else:
                weights       = np.random.normal(loc, scale, self.dimF)
                self.weights  = self.lib.array(weights, dtype=self.dtype)

            if shared is None:
                self.updat_out = self.lib.empty(self.dimF, dtype=self.dtypeU)
            else:
                self.updat_out = shared.share(self.dimF, dtype=self.dtypeU)

            self.weight_stats = self.lib.empty((self.dimF2[0], 1), dtype=np.float32)

    def scale_weights(self, scale):

        mean = self.get_activation_mean()
        self.weights[:] *= scale/mean

    def fprop(self, fprop_in, scale_weights=0):
        if self.fprop_in is None and fprop_in:
            self.fprop_in = fprop_in.reshape(self.dimI)
        return self.fprop_in

    def bprop(self, bprop_in, beta=0):
        return bprop_in

    # fprop relu happens inside of the conv and gemm kernels
    def bprop_relu(self, bprop_in):

        bprop_in[:] = bprop_in * (self.fprop_out > 0)
        return bprop_in

    def grad_descent(self):

        self.weights[:] += self.updat_out*self.learning_rate

    def get_activation_mean(self):
        return self._get_mean(self.fprop_out, self.act_stats, self.dimO2)

    def get_activation_max(self):
        return self._get_max(self.fprop_out, self.act_stats, self.dimO2)

    def get_delta_mean(self):
        return self._get_mean(self.bprop_out, self.delta_stats, self.dimI2)

    def get_delta_max(self):
        return self._get_max(self.bprop_out, self.delta_stats, self.dimI2)

    def get_update_mean(self):
        if self.sizeF > 0:
            return self._get_mean(self.updat_out, self.weight_stats, self.dimF2)
        return 0

    def get_update_max(self):
        if self.sizeF > 0:
            return self._get_max(self.updat_out, self.weight_stats, self.dimF2)
        return 0

    def get_weight_mean(self):
        if self.sizeF > 0:
            return self._get_mean(self.weights, self.weight_stats, self.dimF2)
        return 0

    def get_weight_max(self):
        if self.sizeF > 0:
            return self._get_max(self.weights, self.weight_stats, self.dimF2)
        return 0

    def _get_mean(self, ary, buf, shape):

        buf1    = buf[0:1, 0:1]
        buf[:]  = self.lib.sum(abs(ary.reshape(shape)), axis=1)
        buf1[:] = self.lib.sum(buf, axis=0) * (1.0/ary.size)
        return float(buf1.get()[0, 0])

    def _get_max(self, ary, buf, shape):

        buf1    = buf[0:1, 0:1]
        buf[:]  = self.lib.max(abs(ary.reshape(shape)), axis=1)
        buf1[:] = self.lib.max(buf, axis=0)
        return float(buf1.get()[0, 0])

    def fprop_stats(self):
        print("fprop:%10.5f mean %11.5f max %s"
              % (self.get_activation_mean(), self.get_activation_max(), self))

    def bprop_stats(self):
        if self.bprop_out is not None:
            print("bprop:%10.5f mean %11.5f max %s"
                  % (self.get_delta_mean(), self.get_delta_max(), self))

        if self.weights is not None:
            up_mean, up_max = (self.get_update_mean(), self.get_update_max())
            wt_mean, wt_max = (self.get_weight_mean(), self.get_weight_max())
            rt_mean, rt_max = (0.0001 * up_mean/wt_mean, 0.0001 * up_max/wt_max)
            print("updat:%10.5f mean %11.5f max %s" % (up_mean, up_max, self))
            print("weigh:%10.5f mean %11.5f max" % (wt_mean, wt_max))
            print("ratio:%10.5f mean %11.5f max" % (rt_mean, rt_max))

    @staticmethod
    def create(lib, conf, prev_layer, dtype):

        config     = dict(conf)
        layer_type = config.pop("layer")

        # merge dtype specific settings
        config["dtype"] = dtype

        # merge shared params
        config.update(config.pop("common", {}))

        # Propagate the fixed and calculated dimensions
        if prev_layer is not None:
            config["N"] = prev_layer.N

            if layer_type is FullLayer:
                config["nIn"] = prev_layer.nOut
            elif layer_type is PoolLayer and type(prev_layer) is FullLayer:
                config["C"] = prev_layer.nOut
            elif layer_type is BatchNorm and type(prev_layer) is FullLayer:
                config["nIn"] = prev_layer.nOut
            else:
                config["C"] = prev_layer.K
                config["D"] = prev_layer.M
                config["H"] = prev_layer.P
                config["W"] = prev_layer.Q

                if layer_type is Inception:
                    partitions  = config.pop("partitions")
                    config["K"] = 0

                    config["partitions"] = []
                    for part in partitions:
                        layer_sequence = []
                        part_prev_layer = prev_layer
                        for layer_conf in part:
                            part_prev_layer = Layer.create(lib, layer_conf, part_prev_layer, dtype)
                            layer_sequence.append(part_prev_layer)

                        last = layer_sequence[-1]
                        config["partitions"].append(layer_sequence)
                        config["K"] += last.K
                        if "P" in config:
                            assert config["P"] == last.P and config["Q"] == last.Q
                        else:
                            config["M"] = last.M
                            config["P"] = last.P
                            config["Q"] = last.Q

        # Instantiate the layer
        return layer_type(lib, **config)


class DataLayer(Layer):

    """
    Input data layer.
    """

    def __init__(self, lib, dtype, N, C, D=1, H=1, W=1):

        super(DataLayer, self).__init__(lib, dtype, N)

        self.C = C
        self.K = C
        self.M = D
        self.P = H
        self.Q = W
        self.DHW = (D, H, W)
        self.dimI   = (C, D, H, W, N)
        self.dimO   = (C, D, H, W, N)
        self.dimI2  = (C*D*H*W, N)
        self.dimO2  = (C*D*H*W, N)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.sizeI  = self.sizeO

    def init_data(self, ary=None):
        if ary is None:
            self.fprop_out.fill(0)
        else:
            self.fprop_out.set(ary)

    def init_deltas(self, shared=None):
        pass

    def init_weights(self, loc=0.0, scale=0.1, shared=None, zeros=False):
        pass

    def fprop(self, fprop_in, scale_weights=0):
        return self.fprop_out

    def __str__(self):
        return "DataLayer: NCK: (%d, %d, %d) DHW:%s" % (self.N, self.C, self.K, self.DHW)


class FullLayer(Layer):

    """
    Fully connnected layer.
    """

    def __init__(self, lib, dtype, N, nIn, nOut, relu=False):

        super(FullLayer, self).__init__(lib, dtype, N)

        self.nIn    = nIn
        self.nOut   = nOut
        self.flops  = N * nIn * nOut * 2.0
        self.dimI   = (nIn, N)
        self.dimI2  = (nIn, N)
        self.dimO   = (nOut, N)
        self.dimO2  = (nOut, N)
        self.dimF   = (nOut, nIn)
        self.dimF2  = (nOut, nIn)
        self.sizeI  = nIn  * N
        self.sizeO  = nOut * N
        self.sizeF  = nIn  * nOut
        self.relu   = relu

    def fprop(self, fprop_in, scale_weights=0):

        fprop_in = super(FullLayer, self).fprop(fprop_in)
        self.lib.compound_dot(self.weights, fprop_in, self.fprop_out, self.relu)

        if scale_weights:
            self.scale_weights(scale_weights)
            self.fprop(fprop_in)

        return self.fprop_out

    def bprop(self, bprop_in, beta=0):

        if self.relu:
            self.bprop_relu(bprop_in)
        self.lib.compound_dot(self.weights.T, bprop_in, self.bprop_out)

        self.lib.compound_dot(bprop_in, self.fprop_in.T, self.updat_out)
        self.grad_descent()

        return self.bprop_out

    def __str__(self):
        return "FullLayer: N, nIn, nOut: (%d, %d, %d)" % (self.N, self.nIn, self.nOut)


class ConvLayer(Layer):

    """
    ConvLayer parameter object.
    This then is passed as an argument to all the convolution operations.

    N: Number of images in mini-batch
    C: Number of input feature maps
    K: Number of output feature maps

    D: Depth  of input image
    H: Height of input image
    W: Width  of input image

    T: Depth  of filter kernel
    R: Height of filter kernel
    S: Width  of filter kernel

    padding: amount of zero-padding around the given edge
    strides: factor to step the filters by in a given direction
    """

    def __init__(self, lib, dtype,
                 N, C, K,
                 D=1, H=1, W=1,
                 T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1,
                 relu=False, bsum=False,
                 deterministic_update=False):

        super(ConvLayer, self).__init__(lib, dtype, N, np.float32)

        vec_size = 4 if self.dtype.itemsize == 4 else 8

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % vec_size == 0, "K dim must be multiple of %d" % vec_size

        if self.dtype.type is np.float16:
            clss = "hconv"
        elif self.dtype.type is np.float32:
            clss = "sconv"
        else:
            raise TypeError("Type not supported.")

        # Compute the output spatial dimensions
        M = lib.output_dim(D, T, pad_d, str_d)
        P = lib.output_dim(H, R, pad_h, str_h)
        Q = lib.output_dim(W, S, pad_w, str_w)

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N, C, K)
        self.TRS = (T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_d, pad_h, pad_w)
        self.strides = (str_d, str_h, str_w)
        self.relu = relu
        self.bsum = bsum

        self.all_params = (N, C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        self.dimI   = (C, D, H, W, N)
        self.dimF   = (C, T, R, S, K)
        self.dimFb  = (K, T, R, S, C)
        self.dimO   = (K, M, P, Q, N)
        self.dimI2  = (C*D*H*W, N)
        self.dimF2  = (C*T*R*S, K)
        self.dimF2t = (K, C*T*R*S)
        self.dimO2  = (K*M*P*Q, N)
        self.dimS   = (K, 1)
        self.sizeI  = reduce(mul, self.dimI, 1)
        self.sizeF  = reduce(mul, self.dimF, 1)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.nOut   = reduce(mul, self.MPQ, 1) * K

        # precompute some multiplications for fast constant memory access
        HW    = H*W
        DHW   = D*HW
        WN    = W*N
        HWN   = H*WN
        DHWN  = D*HWN
        RS    = R*S
        RST   = T*RS
        CRST  = C*RST
        CRSTK = K*CRST
        KRST  = K*RST
        PQ    = P*Q
        PQM   = M*PQ
        QN    = Q*N
        PQN   = P*QN
        MPQN  = M*PQN

        if CRST > 2**16:
            assert CRST  < 2**16, "Integer division is faster with 16bit numerators"

        # precompute the magic numbers and shift amounts for integer division
        magic_HW    = _magic64(HW)
        magic_W     = _magic64(W)
        magic_PQ    = _magic64(PQ)
        magic_Q     = _magic64(Q)
        magic_RST   = _magic32(CRST, RST)
        magic_RS    = _magic32(RST+32, RS)
        magic_S     = _magic32(RS+32, S)
        magic_str_w = _magic32(W + S, str_w)
        magic_str_h = _magic32(H + R, str_h)
        magic_str_d = _magic32(D + T, str_d)

        # flop count for benchmarking
        self.flops = PQM * K * N * CRST * 2.0

        tile_N   = 128 if N > 64 else 64
        grid_N   = _grid_dim(tile_N, N)
        tiles_CK = (128, 64, 32) if tile_N == 128 else (128, 64)

        ####### FPROP ###########
        self.fprop_kernels = kernel_specs.xprop_conv_kernels(
            clss, "fprop", "K", tile_N, grid_N, K, tiles_CK, PQM, RST,
            _flatten([N, K, D, H, W, WN, HWN, DHWN,
                      C, KRST, RST, RS, magic_RS, S, magic_S,
                      pad_d, pad_h, pad_w, str_d, str_h, str_w,
                      Q, PQ, QN, PQN, MPQN, magic_Q, magic_PQ]))

        # shared lookup table size
        self.fprop_lut_size = RST * 4 * 2

        ####### BPROP ###########
        if C < 16 or C % vec_size != 0:
            # special kernel for deconv into first layer
            kernel_name = "%s_bprop_C1_N64" % clss

            grid  = (PQM, _grid_dim(32, CRST), _grid_dim(64, N))
            block = (32, 1, 1)

            self.bprop_kernels = [[kernel_name, grid, block, 0, _flatten([
                N, K, D, H, W, WN, HWN, DHWN,
                C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S,
                pad_d, pad_h, pad_w, str_d, str_h, str_w,
                Q, PQ, QN, PQN, MPQN, magic_Q, magic_PQ,
                CRST*8*self.dtype.itemsize, MPQN*8*self.dtype.itemsize])]]

            # generate the kernel args for transpose CRST,K => K,CRST
            self.shuffle_args = [CRST, K]
            gridX   = (K    >> 5) + (K    & 31 != 0)
            gridY   = (CRST >> 5) + (CRST & 31 != 0)
            self.shuffle_grid   = (gridX, gridY, 1)
            self.shuffle_block  = (32, 8, 1)
            self.bprop_zero     = self.sizeI * self.dtype.itemsize
            self.bprop_lut_size = 0

        else:

            self.bprop_kernels = kernel_specs.xprop_conv_kernels(
                clss, "bprop", "C", tile_N, grid_N, C, tiles_CK, DHW, RST, _flatten([
                    N, C, M, P, Q, QN, PQN, MPQN,
                    K, CRST, RST, RS, magic_RS, S, magic_S,
                    pad_d, pad_h, pad_w, str_d, str_h, str_w,
                    W, HW, WN, HWN, DHWN, magic_W, magic_HW,
                    R, T, magic_str_w, magic_str_h, magic_str_d]))

            # generate the kernel args for dim shuffling CRSTK => KRSTC
            self.shuffle_args = _flatten([
                RST*K, RS*K, S*K, K,
                RST*C, RS*C, S*C, C,
                RS, magic_RS, S, magic_S])
            gridX = (K >> 5) + (K & 31 != 0)
            gridY = (C >> 5) + (C & 31 != 0)
            self.shuffle_grid   = (gridX, gridY, RST)
            self.shuffle_block  = (32, 8, 1)
            self.bprop_zero     = 0
            self.bprop_lut_size = RST * 4 * 2

        # for k in self.fprop_kernels: print k
        # for k in self.bprop_kernels: print k
        # exit()

        ####### UPDATE ###########

        grid_C   = _grid_dim(128, CRST)
        sm_count = _get_sm_count()

        # in float32 for big feature_map layers the smaller tile is actually faster
        # so restrict tile selection to just that.
        if self.dtype.type is np.float32 and PQ > 56*56:
            K_tiles = (64,)
        else:
            K_tiles = (128, 64)

        if deterministic_update:
            determ = "D"
            if K <= 64:
                K_tiles = (64,)
            else:
                K_tiles = K_tiles[0:1]
            self.determ = CRSTK
        else:
            determ = ""
            self.determ = 0


        self.updat_kernels = []
        for tile_K, grid_K, offset_K in kernel_specs.K_partitions(K, K_tiles):

            kernel_name = "%s_updat%s_C128_K%d" % (clss, determ, tile_K)
            base_blocks = M*grid_C*grid_K

            grid_P, grid_Q, threads = kernel_specs.update_grid(kernel_name, base_blocks, P, Q, sm_count)
            # print grid_P, grid_Q

            grid_PQ   = grid_P * grid_Q
            magic_PQu = _magic64(grid_PQ)
            magic_Qu  = _magic64(grid_Q)

            block = (threads, 1, 1)
            if RST > 1:
                grid = (M*grid_PQ, grid_C, grid_K)
            else:
                grid = (grid_C, grid_K, M*grid_PQ)

            self.determ *= M*grid_PQ
            self.determ_shape = (M*grid_PQ, CRSTK)

            self.updat_kernels.append([kernel_name, grid, block, offset_K, _flatten([
                N, K, D, H, W, WN, HWN, DHWN,
                C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S,
                pad_d, pad_h, pad_w, str_d, str_h, str_w,
                P, Q, PQ, QN, PQN, MPQN, magic_Qu, magic_PQu,
                grid_P, grid_Q, grid_PQ, CRSTK])])

        # for k in self.updat_kernels: print k
        # exit()

    def init_activations(self, fprop_out=None):

        super(ConvLayer, self).init_activations(fprop_out)

        if self.bsum:
            self.batch_sum = self.lib.empty(self.dimS, dtype=np.float32)
        else:
            self.batch_sum = None

    def fprop(self, fprop_in, scale_weights=0):
        """
        Conv Layer forward propagation.

        Arguments:
            fprop_in (Tensor): Inputs
            scale_weights (float): Scale weights by scale/mean if nonzero

        Returns:
            fprop_out (Tensor): Output activations
        or
            (self.fprop_out, self.batch_sum) (tuple): Tuple with batch_sum
                added as the second entry.
        """
        fprop_in = super(ConvLayer, self).fprop(fprop_in)
        self.lib.fprop_conv(self, fprop_in, self.weights, self.fprop_out, bsum=self.batch_sum)

        if scale_weights:
            self.scale_weights(scale_weights)
            self.fprop(fprop_in)

        if self.bsum:
            return (self.fprop_out, self.batch_sum)
        return self.fprop_out

    def bprop(self, bprop_in, beta=0):

        if self.relu:
            self.bprop_relu(bprop_in)
        if self.bprop_out is not None:
            self.lib.bprop_conv(self, self.weights, bprop_in, self.bprop_out, beta=beta)

        self.lib.update_conv(self, self.fprop_in, bprop_in, self.updat_out)
        self.grad_descent()

        return self.bprop_out

    def __str__(self):
        return ("ConvLayer: NCK: (%d, %d, %d) DHW:%s TRS:%s MPQ:%s" %
                (self.N, self.C, self.K, self.DHW, self.TRS, self.MPQ))


# Add Deconv class
class DeconvLayer(ConvLayer):

    """
    DeconvLayer parameter object.
    This then is passed as an argument to all the convolution operations.

    N: Number of images in mini-batch
    C: Number of output feature maps
    K: Number of input feature maps

    P: Height of input
    Q: Width of input

    D: Depth  of output image
    H: Height of output image
    W: Width  of output image

    T: Depth  of filter kernel
    R: Height of filter kernel
    S: Width  of filter kernel

    padding: amount of zero-padding around the given edge
    strides: factor to step the filters by in a given direction
    """

    def __init__(self, lib, dtype,
                 N, C, K,
                 P, Q,
                 R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1,
                 relu=False, bsum=False,
                 deterministic_update=False):

        # Set T and D to be consts.
        D = T = 1

        # Cannot get exact, e.g. because not unique
        H = (P-1) * str_h - 2 * pad_h + R
        W = (Q-1) * str_w - 2 * pad_w + S

        super(DeconvLayer, self).__init__(
            lib, dtype,
            N, C, K,
            D, H, W,
            T, R, S,
            pad_d, pad_h, pad_w,
            str_d, str_h, str_w,
            relu, bsum, deterministic_update)

        self.nOut = reduce(mul, self.DHW, 1) * C
        self.H = H
        self.W = W

    def __str__(self):
        return ("DeconvLayer: NCK: (%d, %d, %d) DHW:%s TRS:%s MPQ:%s" %
                (self.N, self.C, self.K, self.DHW, self.TRS, self.MPQ))


class PoolLayer(Layer):

    """
    PoolLayer parameter object.
    This then is passed as an argument to all pooling kernels.

    op: max, avg, l2 pooling
    N: Number of images in mini-batch

    C: Number of input feature maps
    D: Depth  of input image
    H: Height of input image
    W: Width  of input image

    J: Size of feature map pooling window (maxout n_pieces)
    T: Depth  of pooling window
    R: Height of pooling window
    S: Width  of pooling window

    padding: amount of zero-padding around the given image or feature map edge
    strides: factor to step the window by in a given direction (overlap allowed)

    Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.

    """

    def __init__(self, lib, dtype,
                 op, N, C,
                 D=1, H=1, W=1,
                 J=1, T=1, R=1, S=1,
                 pad_c=0, pad_d=0, pad_h=0, pad_w=0,
                 str_c=None, str_d=None, str_h=None, str_w=None):

        super(PoolLayer, self).__init__(lib, dtype, N)

        if self.dtype.type is np.float16:
            clss = "hpool"
        elif self.dtype.type is np.float32:
            clss = "spool"
        else:
            raise TypeError("Type not supported.")

        # default to non-overlapping
        if str_c is None:
            str_c = J
        if str_d is None:
            str_d = T
        if str_h is None:
            str_h = R
        if str_w is None:
            str_w = S

        if str_c < J or str_d < T or str_h < R or str_w < S:
            self.overlap = (ceil(float(J)/str_c) *
                            ceil(float(T)/str_d) *
                            ceil(float(R)/str_h) *
                            ceil(float(S)/str_w))
        else:
            self.overlap = 0.0

        # TODO: detect other forms of gaps
        if str_c > J or str_d > T or str_h > R or str_w > S:
            self.gaps = 1
        else:
            self.gaps = 0

        bprop_zero = self.overlap or self.gaps

        # Compute the output dimensions
        K = lib.output_dim(C, J, pad_c, str_c, pooling=True)
        M = lib.output_dim(D, T, pad_d, str_d, pooling=True)
        P = lib.output_dim(H, R, pad_h, str_h, pooling=True)
        Q = lib.output_dim(W, S, pad_w, str_w, pooling=True)

        self.op   = op
        self.C    = C
        self.K    = K
        self.M    = M
        self.P    = P
        self.Q    = Q
        self.JTRS = (J, T, R, S)
        self.DHW  = (D, H, W)
        self.MPQ  = (M, P, Q)
        self.padding = (pad_c, pad_d, pad_h, pad_w)
        self.strides = (str_c, str_d, str_h, str_w)

        self.dimI   = (C, D, H, W, N)
        self.dimO   = (K, M, P, Q, N)
        self.dimF2  = None
        self.dimI2  = (C*D*H*W, N)
        self.dimO2  = (K*M*P*Q, N)
        self.sizeI  = reduce(mul, self.dimI, 1)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.nOut   = reduce(mul, self.MPQ, 1) * K

        # precompute some multiplications for fast constant memory access
        WN   = W*N
        HWN  = H*WN
        DHWN = D*HWN
        DH   = D*H
        RS   = R*S
        RST  = T*RS
        JRST = J*RST
        QN   = Q*N
        PM   = P*M
        PQN  = P*QN
        MPQN = M*PQN

        assert JRST <= N or N >= 32, "Edge case not currently implemented"
        assert JRST+32 < 2**16, "Integer division is faster with 16bit numerators"

        # precompute the magic numbers and shift amounts for integer division
        magic_RST = _magic32(JRST+32, RST)
        magic_RS  = _magic32(RST+32, RS)
        magic_S   = _magic32(RS+32, S)
        magic_P   = _magic32(PM, P)

        fprop_name = "fprop_" + op
        bprop_name = "bprop_" + op

        self.fprop_kernel = [fprop_name, (Q, PM, K), (N, 1, 1), _flatten([
            N, W, H, D, C, WN, HWN, DHWN,
            P, Q, magic_P, QN, PQN, MPQN,
            pad_c, pad_d, pad_h, pad_w,
            str_c, str_d, str_h, str_w,
            S, RS, RST, JRST, magic_S, magic_RS, magic_RST])]

        lut_size = JRST
        if lut_size % 4 != 0:
            lut_size += 4 - lut_size % 4

        self.bprop_lut_size = self.fprop_lut_size = lut_size * 4

        if self.overlap > 0:

            # we have a special kernel to handle the overlapping avg pooling
            bprop_name += "_overlap"

            magic_H     = _magic32(DH, H)
            magic_str_w = _magic32(W + S, str_w)
            magic_str_h = _magic32(H + R, str_h)
            magic_str_d = _magic32(D + T, str_d)
            magic_str_c = _magic32(C + J, str_c)

            self.bprop_kernel = [bprop_name, (W, DH, C), (N, 1, 1), _flatten([
                N, W, H, D, C, WN, HWN, DHWN, magic_H,
                pad_w, pad_h, pad_d, pad_c,
                str_w, str_h, str_d, str_c,
                magic_str_w, magic_str_h, magic_str_d, magic_str_c,
                S, R, T, J, RS, RST, JRST, magic_S, magic_RS, magic_RST,
                Q, P, M, K, QN, PQN, MPQN])]

            self.bprop_lut_size *= 2
        else:
            self.bprop_kernel = [bprop_name, (Q, PM, K), (N, 1, 1), _flatten([
                N, W, H, D, C, WN, HWN, DHWN,
                P, Q, magic_P, QN, PQN, MPQN,
                pad_c, pad_d, pad_h, pad_w,
                str_c, str_d, str_h, str_w,
                S, RS, RST, JRST, magic_S, magic_RS, magic_RST])]

    def fprop(self, fprop_in, scale_weights=0):
        """ Used for benchmarking only"""
        fprop_in = super(PoolLayer, self).fprop(fprop_in)
        self.lib.fprop_pool(self, fprop_in, self.fprop_out)
        return self.fprop_out

    def bprop(self, bprop_in, beta=0):
        self.lib.bprop_pool(self, self.fprop_in, bprop_in, self.bprop_out)
        return self.bprop_out

    def __str__(self):
        return ("PoolLayer: NCK: (%d, %d, %d) DHW:%s JTRS:%s MPQ:%s op: %s " %
                (self.N, self.C, self.K, self.DHW, self.JTRS, self.MPQ, self.op))


class Inception(Layer):

    """
    GoogLeNet inception assembly layer.

    """

    def __init__(self, lib, dtype, partitions,
                 N, C, K,
                 D=1, H=1, W=1,
                 M=1, P=1, Q=1):

        super(Inception, self).__init__(lib, dtype, N)

        self.partitions = partitions

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N, C, K)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)

        self.dimI  = (C, D, H, W, N)
        self.dimO  = (K, M, P, Q, N)
        self.dimI2 = (C*D*H*W, N)
        self.dimO2 = (K*M*P*Q, N)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut  = reduce(mul, self.MPQ, 1) * K

        self.sizeF = 0
        self.flops = 0
        for part in partitions:
            for layer in part:
                self.flops += layer.flops
                self.sizeF  = max(self.sizeF, layer.sizeF)
                if self.sizeF == layer.sizeF:
                    self.dimF = layer.dimF

    def __str__(self):
        out = "Inception: NCK: (%d, %d, %d) DHW:%s MPQ:%s\n" % (self.N, self.C, self.K,
                                                                self.DHW, self.MPQ)
        for i, part in enumerate(self.partitions):
            out += "  Part%d:\n" % (i+1)
            for layer in part:
                out += "    %s\n" % layer
        return out.rstrip()

    def init_activations(self, fprop_out=None):

        super(Inception, self).init_activations(fprop_out)
        K = 0
        for part in self.partitions:
            for layer in part:
                if layer is part[-1]:
                    layer.init_activations(self.fprop_out[K:K+layer.K, ...])
                    K += layer.K
                else:
                    layer.init_activations()

    def init_deltas(self, shared=None):

        super(Inception, self).init_deltas(shared)

        shared_deltas = shared[1:3] if shared else None

        for part in self.partitions:
            for layer in part:
                if layer is part[0]:
                    layer.bprop_out   = self.bprop_out
                    layer.delta_stats = self.delta_stats
                else:
                    layer.init_deltas(shared=shared_deltas)

    def init_weights(self, loc=0.0, scale=0.1, shared=None, zeros=False):

        for part in self.partitions:
            for layer in part:
                layer.init_weights(loc, scale, shared, zeros)

    def fprop(self, fprop_in, scale_weights=0):

        fprop_in = super(Inception, self).fprop(fprop_in)

        for part in self.partitions:
            part_fprop_in = fprop_in
            for layer in part:
                part_fprop_in = layer.fprop(part_fprop_in, scale_weights)

        return self.fprop_out

    def bprop(self, bprop_in):

        K = self.K
        for part in self.partitions[::-1]:
            part_bprop_in = bprop_in[K-part[-1].K:K, ...]
            K -= part[-1].K
            for layer in part[::-1]:
                if part is not self.partitions[-1] and layer is part[0]:
                    beta = 1.0
                else:
                    beta = 0.0

                part_bprop_in = layer.bprop(part_bprop_in, beta)

        return self.bprop_out

    def fprop_stats(self):
        for part in self.partitions:
            for layer in part:
                layer.fprop_stats()

    def bprop_stats(self):
        for part in self.partitions[::-1]:
            for layer in part[::-1]:
                layer.bprop_stats()


class BatchNorm(Layer):

    """
    Batch Normalization Layer
    """

    def __init__(self, lib, dtype, N, C=None, D=1, H=1, W=1, nIn=None, rho=0.99, eps=1e-6,
                 relu=False, bsum=False):
        """
        Batch Normalization layer

        Arguments:
            lib (Class): NervanaGPU instance
            dtype (Dtype): Data type
            N (int): batch size
            C (int): Number of input feature maps
            D (int): Depth  of input feature maps
            H (int): Height of input feature maps
            W (int): Width  of input feature maps
            nIn (int): Number on inputs for fully connected layer
            rho (float): Exponential window averaging factor
            eps (float): Constant added for numerical stability
            relu (bool): Flag for rectified linear activation function
            bsum (bool): PQN sum precomputed in conv kernel
        """
        super(BatchNorm, self).__init__(lib, dtype, N)

        self.rho  = rho
        self.eps  = eps
        self.relu = relu
        self.bsum = bsum

        if C is not None:

            self.C     = C
            self.K     = C
            self.M     = D
            self.P     = H
            self.Q     = W
            self.dimI  = (C, D, H, W, N)
            self.dimO  = (C, D, H, W, N)
            self.dimO2 = (C*D*H*W, N)
            self.dim2  = (C, D*H*W*N)
            self.nOut  = C*D*H*W

        elif nIn is not None:

            self.nOut  = nIn
            self.K     = nIn
            self.dimI  = (nIn, N)
            self.dimO  = (nIn, N)
            self.dimO2 = (nIn, N)
            self.dim2  = (nIn, N)

        else:
            raise ValueError("missing C or nIn")

        self.rcp_depth = 1.0/self.dim2[1]

    def __str__(self):
        return ("BatchNorm: (%d, %d)" % self.dim2)

    def init_activations(self, fprop_out=None):

        if fprop_out is not None:
            self.fprop_out = fprop_out.reshape(self.dim2)
        else:
            self.fprop_out = self.lib.empty(self.dim2, dtype=self.dtype)

        self.xvar = self.lib.empty((self.K, 1), dtype=self.dtype)
        if not self.bsum:
            self.xsum = self.lib.empty((self.K, 1), dtype=np.float32)

    def init_deltas(self, shared=None):
        pass

    def init_weights(self, loc=0.0, scale=0.1, shared=None, zeros=False):

        lib = self.lib

        self.beta  = lib.zeros((self.K, 1), dtype=self.dtype)
        self.gamma = lib.ones((self.K, 1),  dtype=self.dtype)

        self.gmean = lib.zeros((self.K, 1), dtype=self.dtype)
        self.gvar  = lib.zeros((self.K, 1), dtype=self.dtype)

        self.grad_beta  = lib.zeros((self.K, 1), dtype=self.dtype)
        self.grad_gamma = lib.zeros((self.K, 1), dtype=self.dtype)

    def fprop(self, fprop_in, scale_weights=0):
        """
        Batch normalization forward pass. Uses a compound kernel call.
        """
        if type(fprop_in) is tuple:
            fprop_in, bsum = fprop_in
        else:
            bsum = None

        if self.fprop_in is None:
            self.fprop_in = fprop_in.reshape(self.dim2)

        if bsum is None:
            self.xsum[:] = self.lib.sum(self.fprop_in, axis=1)
        else:
            self.xsum = bsum

        self.lib.compound_fprop_bn(
            self.fprop_in, self.xsum, self.xvar, self.gmean, self.gvar,
            self.gamma, self.beta, self.fprop_out, self.eps, self.rho, self.relu)

        return self.fprop_out

    def bprop(self, bprop_in, beta=0):

        if self.bprop_in is None:
            self.bprop_in = bprop_in.reshape(self.dim2)

        if self.relu:
            self.bprop_relu(self.bprop_in)

        self.lib.compound_bprop_bn(
            self.bprop_in, self.grad_gamma, self.grad_beta, self.fprop_in,
            self.xsum, self.xvar, self.gamma, self.eps)

        return self.bprop_in

    def fprop_stats(self):
        pass

    def bprop_stats(self):
        pass


def _grid_dim(tile_size, dim_size):
    return dim_size // tile_size + (dim_size % tile_size != 0)


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

# flatten a nested list of lists or values


def _flatten(lst):
    return sum(([x] if not isinstance(x, (list, tuple))
                else _flatten(x) for x in lst), [])


@context_dependent_memoize
def _get_sm_count():
    attributes = drv.Context.get_device().get_attributes()
    return attributes[drv.device_attribute.MULTIPROCESSOR_COUNT]
