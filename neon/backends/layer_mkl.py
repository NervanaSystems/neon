# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
"""
MKL backend layers
"""
from __future__ import division
import numpy as np
from ctypes import c_longlong, c_float
from neon.backends.layer_cpu import ConvLayer, DeconvLayer, PoolLayer


class ConvLayerMKL(ConvLayer):

    """
    MKL based ConvLayer
    """
    def __init__(self, lib, dtype,
                 N, C, K,
                 D=1, H=1, W=1,
                 T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1,
                 dil_d=1, dil_h=1, dil_w=1):

        super(ConvLayerMKL, self).__init__(lib, dtype,
                                           N, C, K, D, H, W, T, R, S,
                                           pad_d, pad_h, pad_w, str_d, str_h, str_w,
                                           dil_d, dil_h, dil_w)
        self.dnnPrimitives = np.zeros((1, 50), dtype=np.uint64)
        self.init_f = 0  # forward init flag
        self.init_bd = 0  # backward data
        self.init_bw = 0  # backward weight
        self.dilated = any(d != 1 for d in self.dilation)
        self.is_mklop = True
        if D != 1 or T != 1 or self.dilated:
            self.is_mklop = False

    def xprop_conv_dilated(self, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
                           relu=False, brelu=False, slope=0.0, backward=False):

        O_ = O.get()
        F_ = F.get()
        I_ = I.get()

        if backward:
            raise ValueError()
            I = I._tensor.reshape(self.dimO)
            O = O._tensor.reshape(self.dimI)
        else:
            I_ = I_.reshape(self.dimI)
            O_ = O_.reshape(self.dimO)

        F_ = F_.reshape(self.dimF)
        if bias is not None:
            raise ValueError()
            bias = bias._tensor.reshape((O.shape[0], 1))
        if bsum is not None:
            raise ValueError()
            bsum = bsum._tensor.reshape((O.shape[0], 1))
        if self.dot:
            raise ValueError()
            C = F.shape[0]
            K = F.shape[-1]
            if backward:
                F = F.reshape((C, K))
                I = I.reshape((K, -1))
            else:
                F = F.reshape((C, K)).T
                I = I.reshape((C, -1))
            if beta:
                O[:] = alpha * np.dot(F, I).reshape(O.shape) + beta * X
            else:
                O[:] = np.dot(F, I).reshape(O.shape)
                self.compound_ops(O, X, bias, bsum, relu, brelu, slope)
            return
        if backward:
            raise ValueError()
            F = np.transpose(F[:, ::-1, ::-1, ::-1], (4, 1, 2, 3, 0)).copy()
            mSlice, pSlice, qSlice = self.dSlice, self.hSlice, self.wSlice
        else:
            mSlice, pSlice, qSlice = self.mSlice, self.pSlice, self.qSlice
        K, M, P, Q, N = O_.shape
        for m in range(M):
            sliceT, sliceD, _ = mSlice[m]
            for p in range(P):
                sliceR, sliceH, _ = pSlice[p]
                for q in range(Q):
                    sliceS, sliceW, _ = qSlice[q]
                    slicedF = F_[:, sliceT, sliceR, sliceS].reshape((-1, K))
                    slicedI = I_[:, sliceD, sliceH, sliceW].reshape((-1, N))
                    if beta:
                        raise ValueError()
                        O[:, m, p, q] = \
                            alpha * np.dot(slicedF.T, slicedI) + beta * X[:, m, p, q]
                    else:
                        O_[:, m, p, q] = np.dot(slicedF.T, slicedI)
        O[:] = O_.flatten()
        O.backend.convert_mkl(O)
        O.shape5D = self.dimO

    def xprop_conv(self, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
                   relu=False, brelu=False, slope=0.0, backward=False, layer_op=None):

        if layer_op is None:
            layer_op = self
        if not layer_op.get_is_mklop():
            I.backend.convert(I)
            F.backend.convert(F)
            I.clean_mkl()
            F.clean_mkl()
            super(ConvLayerMKL, self).xprop_conv(
                I, F, O, X, bias, bsum, alpha, beta, relu, brelu, slope, backward)
            return

        # hack for dealing with dilated conv
        if self.dilated:
            self.xprop_conv_dilated(I, F, O, X, bias, bsum, alpha,
                                    beta, relu, brelu, slope, backward)
            return

        if X is None:
            X = O

        # TODO, support bias
        C, D, H, W, N = self.dimI
        C, T, R, S, K = self.dimF
        K, M, P, Q, N = self.dimO
        pad_d, pad_h, pad_w = self.padding
        str_d, str_h, str_w = self.strides
        primitives = c_longlong(self.dnnPrimitives.ctypes.data)
        mkl_res = 0
        if not backward:
            mkl_res = I.backend.mklEngine.Conv_forward(
                I.get_prim(), O.get_prim(), F.get_prim(), primitives, self.init_f,
                N, C, H, W, R, S, str_h, str_w, pad_h, pad_w, K, P, Q)
            self.init_f = 1
            O.shape5D = self.dimO
        else:
            beta_ = c_float(beta)
            I.backend.mklEngine.Conv_bwdData(
                I.get_prim(), O.get_prim(), F.get_prim(), primitives,
                N, K, P, Q, self.init_bd, beta_)
            O.shape5D = self.dimI
            self.init_bd = 1
        if mkl_res != 0:
            super(ConvLayerMKL, self).xprop_conv(
                I, F, O, X, bias, bsum, alpha, beta, relu, brelu, slope, backward)
            I.clean_mkl()
            O.clean_mkl()
            layer_op.set_not_mklop()
            return

    def update_conv(self, I, E, U, alpha=1.0, beta=0.0, layer_op=None):

        if not self.get_is_mklop():
            I.backend.convert(I)
            I.clean_mkl()
            E.backend.convert(E)
            E.clean_mkl()
            super(ConvLayerMKL, self).update_conv(I, E, U, alpha, beta)
            return

        # not deal with alpha, beta yet
        K, M, P, Q, N = self.dimO
        primitives = c_longlong(self.dnnPrimitives.ctypes.data)
        I.backend.mklEngine.Conv_bwdFilter(
            I.get_prim(), E.get_prim(), U.get_prim(), primitives,
            N, K, P, Q, self.init_bw, self.init_bd)
        self.init_bw = 1


class DeconvLayerMKL(DeconvLayer):

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
    dilation: dilation factor for each dimension
    """

    def xprop_conv(self, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
                   relu=False, brelu=False, slope=0.0, backward=False, layer_op=None):

        I.backend.convert(I)
        I.clean_mkl()
        O.backend.convert(O)
        O.clean_mkl()
        super(DeconvLayerMKL, self).xprop_conv(I, F, O, X, bias, bsum, alpha, beta,
                                               relu, brelu, slope, backward, layer_op)

    def update_conv(self, I, E, U, alpha=1.0, beta=0.0, layer_op=None):

        I.backend.convert(I)
        I.clean_mkl()

        E.backend.convert(E)
        E.clean_mkl()

        super(DeconvLayerMKL, self).update_conv(I, E, U, alpha, beta, layer_op)


class PoolLayerMKL(PoolLayer):

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

        super(PoolLayerMKL, self).__init__(lib, dtype,
                                           op, N, C, D, H, W, J, T, R, S,
                                           pad_c, pad_d, pad_h, pad_w,
                                           str_c, str_d, str_h, str_w)
        self.dnnPrimitives = np.zeros((1, 20), dtype=np.uint64)
        self.initOk_f = 0
        self.initOk_b = 0
        self.is_mklop = True


class ReluLayerMKL(object):
    '''
    Relu layer for MKL backend
    Initialize primitive
    '''
    def __init__(self):
        self.dnnPrimitives = np.zeros((1, 12), dtype=np.uint64)
        self.initOk_b = 0
        self.initOk_f = 0
        self.inputMKL = True
        self.shape5D = None


class BatchNormLayerMKL(object):
    '''
    Relu layer for MKL backend
    Initialize primitive
    '''
    def __init__(self, in_shape):
        self.dnnPrimitives = np.zeros((1, 20), dtype=np.uint64)
        self.init_f = 0
        self.init_b = 0
        self.in_shape = in_shape
        self.shape5D = None


class MergeSumLayerMKL(object):
    '''
    MergeSum for MKL backend
    Initialize primitive
    '''
    def __init__(self, layer_num):

        # declare forward sum, backward sum and temperary tensors used in sum
        self.sum_prim_f = np.zeros((1), dtype=np.uint64)
        self.sum_prim_b = np.zeros((1), dtype=np.uint64)
        self.tensors = np.zeros((4 * layer_num), dtype=np.uint64)
        # self.in_shape = in_shape
        self.shape5D = None
        self.layer_num = layer_num


class MergeBroadcastLayerMKL(object):
    '''
    MergeBroadcast for MKL backend
    '''
    def __init__(self, layer_num):
        # mkl internal memory pointer
        self.primitive = np.zeros((15), dtype=np.uint64)
        self.sum_prim = np.zeros((1),  dtype=np.uint64)

        # temp tensors pointers
        self.tensors_temp = np.zeros((4 * layer_num), dtype=np.uint64)

        self.initOK_f = 0
        self.initOK_b = 0
        self.in_shape5D = None
        self.out_shape5D = None
        self.layer_num = layer_num

        # to record input channels
        self.channels = np.zeros(layer_num, dtype=np.uint64)
