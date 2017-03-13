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
Definition of the GPU layers
These layers are mainly used for old benchmarking code,
but they also cache all the computed params for complex layers.
TODO: clean up merge with CPU layers
TODO: remove any non-param caching code, neon layers should replace benchmark code.
"""
from __future__ import division
from builtins import object, str
import logging
from math import ceil
import numpy as np
from operator import mul
import pycuda.driver as drv
from pycuda.tools import context_dependent_memoize
import sys
from neon import logger as neon_logger
from neon.backends import kernel_specs
from neon.backends import convolution

logger = logging.getLogger(__name__)


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
        self.weights[:] *= scale / mean

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
        self.weights[:] += self.updat_out * self.learning_rate

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
        buf1[:] = self.lib.sum(buf, axis=0) * (1.0 / ary.size)
        return float(buf1.get()[0, 0])

    def _get_max(self, ary, buf, shape):

        buf1    = buf[0:1, 0:1]
        buf[:]  = self.lib.max(abs(ary.reshape(shape)), axis=1)
        buf1[:] = self.lib.max(buf, axis=0)
        return float(buf1.get()[0, 0])

    def fprop_stats(self):
        neon_logger.display("fprop:%10.5f mean %11.5f max %s"
              % (self.get_activation_mean(), self.get_activation_max(), self))

    def bprop_stats(self):
        if self.bprop_out is not None:
            neon_logger.display("bprop:%10.5f mean %11.5f max %s"
                  % (self.get_delta_mean(), self.get_delta_max(), self))

        if self.weights is not None:
            up_mean, up_max = (self.get_update_mean(), self.get_update_max())
            wt_mean, wt_max = (self.get_weight_mean(), self.get_weight_max())
            rt_mean, rt_max = (0.0001 * up_mean / wt_mean, 0.0001 * up_max / wt_max)
            neon_logger.display("updat:%10.5f mean %11.5f max %s" % (up_mean, up_max, self))
            neon_logger.display("weigh:%10.5f mean %11.5f max" % (wt_mean, wt_max))
            neon_logger.display("ratio:%10.5f mean %11.5f max" % (rt_mean, rt_max))

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
        self.dimI2  = (C * D * H * W, N)
        self.dimO2  = (C * D * H * W, N)
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
    dilation: dilation factor for each dimension
    """

    def __init__(self, lib, dtype,
                 N, C, K,
                 D=1, H=1, W=1,
                 T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1,
                 dil_d=1, dil_h=1, dil_w=1):

        super(ConvLayer, self).__init__(lib, dtype, N, np.float32)

        # Compute the output spatial dimensions
        M = lib.output_dim(D, T, pad_d, str_d, pooling=False, dilation=dil_d)
        P = lib.output_dim(H, R, pad_h, str_h, pooling=False, dilation=dil_h)
        Q = lib.output_dim(W, S, pad_w, str_w, pooling=False, dilation=dil_w)

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

        self.all_params = (N, C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w,
                           dil_d, dil_h, dil_w)

        self.dimI   = (C, D, H, W, N)
        self.dimF   = (C, T, R, S, K)
        self.dimFb  = (K, T, R, S, C)
        self.dimO   = (K, M, P, Q, N)
        self.dimI2  = (C * D * H * W, N)
        self.dimF2  = (C * T * R * S, K)
        self.dimF2t = (K, C * T * R * S)
        self.dimO2  = (K * M * P * Q, N)
        self.dimS   = (K, 1)
        self.sizeI  = reduce(mul, self.dimI, 1)
        self.sizeF  = reduce(mul, self.dimF, 1)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.nOut   = reduce(mul, self.MPQ, 1) * K

        # flop count for benchmarking
        self.flops = P * Q * M * K * N * C * R * S * T * 2.0

        args = (lib, self.dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w)

        #lib.enable_winograd = 0

        dilated_conv = (dil_d != 1 or dil_h != 1 or dil_w != 1)
        if dilated_conv:
            assert (dil_w > 0 and dil_h > 0 and dil_w > 0)

        ####### Cuda C ###########
        if lib.use_cudac_kernels:

            #3D conv not supported yet
            if T > 1 or D > 1:
                raise ValueError("3D Convolution not supported by CUDA C kernels and pre-Maxwell GPUs")

            # TODO small C bprop?
            self.fprop_kernels = convolution.FpropCuda(*args)
            self.bprop_kernels = convolution.BpropCuda(*args)
            self.updat_kernels = convolution.UpdateCuda(*args)

        ####### Winograd ###########
        elif lib.enable_winograd and R == 3 and S == 3 and all(x == 1 for x in (D,M,T,str_w,str_h,str_d)) and not dilated_conv:
            from .winograd_conv import (FpropWinograd_2x2_3x3, BpropWinograd_2x2_3x3, UpdateWinograd_3x3_2x2,
                                        FpropWinograd_4x4_3x3, BpropWinograd_4x4_3x3, UpdateWinograd_3x3_4x4)

            # Temp for now till we can autotune
            # 2 is safer for fp16 without batchnorm
            if dtype == np.float32 and lib.enable_winograd == 4:
                winograd = 4
            else:
                winograd = 2

            if C < 8:
                self.fprop_kernels = convolution.FpropDirect(*args)
            elif winograd == 4 and H * W < 112 * 112:
                self.fprop_kernels = FpropWinograd_4x4_3x3(*args)
            else:
                self.fprop_kernels = FpropWinograd_2x2_3x3(*args)

            if winograd == 4 and H * W < 112 * 112:
                self.bprop_kernels = BpropWinograd_4x4_3x3(*args)
            else:
                self.bprop_kernels = BpropWinograd_2x2_3x3(*args)

            if N >= 4 and (C < 8 or H * W > 112 * 112):
                self.updat_kernels = convolution.UpdateDirect(*args)
            elif winograd == 4:
                self.updat_kernels = UpdateWinograd_3x3_4x4(*args)
            else:
                self.updat_kernels = UpdateWinograd_3x3_2x2(*args)

#        elif lib.enable_winograd and not lib.deterministic and N > 1 and \
#            R == 5 and S == 5 and all(x == 1 for x in (D,M,T,str_w,str_h,str_d)):
#
#            from .winograd_conv import (FpropWinograd_2x2_5x5, BpropWinograd_2x2_5x5)
#
#            self.fprop_kernels = FpropWinograd_2x2_5x5(*args)
#            self.bprop_kernels = BpropWinograd_2x2_5x5(*args)
#            if N >= 4:
#                self.updat_kernels = convolution.UpdateDirect(*args)

        ####### Direct ###########
        else:

            self.fprop_kernels = convolution.FpropDirect(*args)
            self.bprop_kernels = convolution.BpropDirect(*args)
            if N >= 4:
                self.updat_kernels = convolution.UpdateDirect(*args)

        #logger.debug("%s: %s, %s, %s", str(self), str(self.fprop_kernels), str(self.bprop_kernels), str(self.updat_kernels))


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
        return ("ConvLayer: NCK: (%3d, %3d, %3d) HW:%s" %
                (self.N, self.C, self.K, self.DHW[1:3]))

# Add Deconv class
class DeconvLayer(ConvLayer):

    """
    DeconvLayer parameter object.
    This then is passed as an argument to all the convolution operations.

    N: Number of images in mini-batch
    C: Number of output feature maps
    K: Number of input feature maps

    M: Depth of input
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
    dilation: dilation factor for each dimension
    """

    def __init__(self, lib, dtype,
                 N, C, K,
                 M, P, Q,
                 T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1,
                 dil_d=1, dil_h=1, dil_w=1):

        tt = dil_d * (T - 1) + 1
        rr = dil_h * (R - 1) + 1
        ss = dil_w * (S - 1) + 1
        # Cannot get exact, e.g. because not unique
        D = (M - 1) * str_d - 2 * pad_d + tt
        H = (P - 1) * str_h - 2 * pad_h + rr
        W = (Q - 1) * str_w - 2 * pad_w + ss

        super(DeconvLayer, self).__init__(
            lib, dtype,
            N, C, K,
            D, H, W,
            T, R, S,
            pad_d, pad_h, pad_w,
            str_d, str_h, str_w,
            dil_d, dil_h, dil_w)

        self.nOut = reduce(mul, self.DHW, 1) * C

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

#        if str_c < J or str_d < T or str_h < R or str_w < S:
#            self.overlap = (ceil(float(J) / str_c) *
#                            ceil(float(T) / str_d) *
#                            ceil(float(R) / str_h) *
#                            ceil(float(S) / str_w))
#        else:
#            self.overlap = 0.0

        self.overlap = 1.0

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
        self.dimI2  = (C * D * H * W, N)
        self.dimO2  = (K * M * P * Q, N)
        self.sizeI  = reduce(mul, self.dimI, 1)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.nOut   = reduce(mul, self.MPQ, 1) * K

        # precompute some multiplications for fast constant memory access
        WN   = W * N
        HWN  = H * WN
        DHWN = D * HWN
        DH   = D * H
        RS   = R * S
        RST  = T * RS
        JRST = J * RST
        QN   = Q * N
        PQN  = P * QN
        MPQN = M * PQN

        assert JRST + 32 < 2**16, "Integer division is faster with 16bit numerators"

        sb_large = {
            #SB  shlP maskP shrP shlQ maskQ shrQ maskN shrN
            1  : (0,   0x00, 0,   0,   0x00, 0,   0xfff, 32), # 1x1  nnnnn
            2  : (0,   0x00, 0,   1,   0x10, 4,   0x00f,  4), # 1x2  xnnnn
            4  : (0,   0x00, 0,   2,   0x18, 3,   0x007,  3), # 1x4  xxnnn
            8  : (0,   0x00, 0,   3,   0x1c, 2,   0x003,  2), # 1x8  xxxnn
            16 : (0,   0x00, 0,   4,   0x1e, 1,   0x001,  1), # 1x16 xxxxn
            32 : (0,   0x00, 0,   5,   0x1f, 0,   0x000,  0), # 1x32 xxxxx
        }
        sb_medium = {
            #SB  shlP maskP shrP shlQ maskQ shrQ maskN shrN
            8  : (1,   0x10, 4,   2,   0x0c, 2,   0x003,  2), # 2x4  yxxnn
            16 : (1,   0x10, 4,   3,   0x0e, 1,   0x001,  1), # 2x8  yxxxn
            32 : (1,   0x10, 4,   4,   0x0f, 0,   0x000,  0), # 2x16 yxxxx
        }
        sb_small = {
            #SB  shlP maskP shrP shlQ maskQ shrQ maskN shrN
            16 : (2,   0x18, 3,   2,   0x06, 1,   0x001,  1), # 4x4  yyxxn
            32 : (2,   0x18, 3,   3,   0x07, 0,   0x000,  0), # 4x8  yyxxx
        }

        if N == 1:
            super_block = 0
        elif N < 32:
            super_block = len(bin(N - 1)) - 2
        else:
            super_block = 5
        super_block = 1 << (5 - super_block)

        # try to minimize the zero overlap in the superblock
        # but maximize the x dim of the superblock for more contiguous memory access
        if super_block < 8 or Q > 64:
            sb_params = sb_large.get(super_block)
        elif super_block < 16 or Q > 32:
            sb_params = sb_medium.get(super_block)
        else:
            sb_params = sb_small.get(super_block)

        supP = _ceil_div(P, 1 << sb_params[0])
        supQ = _ceil_div(Q, 1 << sb_params[3])

        # precompute the magic numbers and shift amounts for integer division
        magic_RST = _magic32(JRST + 32, RST)
        magic_RS  = _magic32(RST + 32, RS)
        magic_S   = _magic32(RS + 32, S)
        magic_P   = _magic32(M * supP, supP)

        fprop_name = "fprop_" + op
        bprop_name = "bprop_" + op

        threads = 32 if super_block > 1 else N

        self.fprop_kernel = [fprop_name, (supQ, supP * M, K), (threads, 1, 1), _flatten([
            N, W, H, D, C, WN, HWN, DHWN,
            P, Q, magic_P, QN, PQN, MPQN,
            pad_c, pad_d, pad_h, pad_w,
            str_c, str_d, str_h, str_w,
            S, RS, RST, JRST, magic_S, magic_RS, magic_RST,
            supP, supQ, sb_params ])]

        lut_size = JRST
        if lut_size % 4 != 0:
            lut_size += 4 - lut_size % 4

        self.bprop_lut_size = self.fprop_lut_size = super_block * lut_size * 4

        if self.overlap > 0:

            # we have a special kernel to handle the overlapping avg pooling
            bprop_name += "_overlap"

            magic_str_w = _magic32(W + S, str_w)
            magic_str_h = _magic32(H + R, str_h)
            magic_str_d = _magic32(D + T, str_d)
            magic_str_c = _magic32(C + J, str_c)

            if super_block > 1:

                bprop_name += "_smallN"

                if super_block < 8 or W > 64:
                    sb_params = sb_large.get(super_block)
                elif super_block < 16 or W > 32:
                    sb_params = sb_medium.get(super_block)
                else:
                    sb_params = sb_small.get(super_block)

                supH = _ceil_div(H, 1 << sb_params[0])
                supW = _ceil_div(W, 1 << sb_params[3])

                magic_H = _magic32(D * supH, supH)

                maxLutSize = \
                    _ceil_div(S, str_w) * \
                    _ceil_div(R, str_h) * \
                    _ceil_div(T, str_d) * \
                    _ceil_div(J, str_c)

                #neon_logger.display((supW, D*supH, C), sb_params, maxLutSize)

                self.bprop_kernel = [bprop_name, (supW, D * supH, C), (threads, 1, 1), _flatten([
                    N, W, H, D, C, WN, HWN, DHWN, magic_H,
                    pad_w, pad_h, pad_d, pad_c,
                    str_w, str_h, str_d, str_c,
                    magic_str_w, magic_str_h, magic_str_d, magic_str_c,
                    S, R, T, J, RS, RST, JRST, magic_S, magic_RS, magic_RST,
                    Q, P, M, K, QN, PQN, MPQN,
                    supH, supW, sb_params, maxLutSize])]

                lut_size = maxLutSize
                if lut_size % 4 != 0:
                    lut_size += 4 - lut_size % 4

                self.bprop_lut_size = super_block * lut_size * 4 * 2

            else:

                # The overlap kernel can be much more efficient if we aren't doing superblocking
                magic_H = _magic32(DH, H)

                self.bprop_kernel = [bprop_name, (W, DH, C), (threads, 1, 1), _flatten([
                    N, W, H, D, C, WN, HWN, DHWN, magic_H,
                    pad_w, pad_h, pad_d, pad_c,
                    str_w, str_h, str_d, str_c,
                    magic_str_w, magic_str_h, magic_str_d, magic_str_c,
                    S, R, T, J, RS, RST, JRST, magic_S, magic_RS, magic_RST,
                    Q, P, M, K, QN, PQN, MPQN])]

                self.bprop_lut_size = lut_size * 4 * 2
        else:
            self.bprop_kernel = [bprop_name, (supQ, supP * M, K), (threads, 1, 1), _flatten([
                N, W, H, D, C, WN, HWN, DHWN,
                P, Q, magic_P, QN, PQN, MPQN,
                pad_c, pad_d, pad_h, pad_w,
                str_c, str_d, str_h, str_w,
                S, RS, RST, JRST, magic_S, magic_RS, magic_RST,
                supP, supQ, sb_params])]

    def init_activations(self, fprop_out=None):

        super(PoolLayer, self).init_activations(fprop_out)

        self.argmax = self.lib.empty(self.dimO, dtype=np.uint8)

    def fprop(self, fprop_in, scale_weights=0):
        """ Used for benchmarking only"""
        fprop_in = super(PoolLayer, self).fprop(fprop_in)
        self.lib.fprop_pool(self, fprop_in, self.fprop_out, argmax=self.argmax)
        return self.fprop_out

    def bprop(self, bprop_in, beta=0):
        self.lib.bprop_pool(self, bprop_in, self.bprop_out, argmax=self.argmax)
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
        self.dimI2 = (C * D * H * W, N)
        self.dimO2 = (K * M * P * Q, N)
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
            out += "  Part%d:\n" % (i + 1)
            for layer in part:
                out += "    %s\n" % layer
        return out.rstrip()

    def init_activations(self, fprop_out=None):

        super(Inception, self).init_activations(fprop_out)
        K = 0
        for part in self.partitions:
            for layer in part:
                if layer is part[-1]:
                    layer.init_activations(self.fprop_out[K:K + layer.K, ...])
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
            part_bprop_in = bprop_in[K - part[-1].K:K, ...]
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
            self.dimO2 = (C * D * H * W, N)
            self.dim2  = (C, D * H * W * N)
            self.nOut  = C * D * H * W

        elif nIn is not None:

            self.nOut  = nIn
            self.K     = nIn
            self.dimI  = (nIn, N)
            self.dimO  = (nIn, N)
            self.dimO2 = (nIn, N)
            self.dim2  = (nIn, N)

        else:
            raise ValueError("missing C or nIn")

        self.rcp_depth = 1.0 / self.dim2[1]

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

def _ceil_div(x, y):
    return -(-x // y)

@context_dependent_memoize
def _get_sm_count():
    attributes = drv.Context.get_device().get_attributes()
    return attributes[drv.device_attribute.MULTIPROCESSOR_COUNT]
