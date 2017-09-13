# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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
CPU backend layers
"""
from __future__ import division
from builtins import object
import math
from operator import mul
import numpy as np
from functools import reduce


def ceil_div(x, y):
    """
    same as int(ceil(float(x)/y)), so no need to import math lib
    """
    return -(-x // y)


class ConvLayer(object):

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

        # Compute the output spatial dimensions
        M = lib.output_dim(D, T, pad_d, str_d, dilation=dil_d)
        P = lib.output_dim(H, R, pad_h, str_h, dilation=dil_h)
        Q = lib.output_dim(W, S, pad_w, str_w, dilation=dil_w)

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
        self.dilation = (dil_d, dil_h, dil_w)

        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)
        self.dimO = (K, M, P, Q, N)
        self.dimS = (K, 1)
        self.dimI2 = (C * D * H * W, N)
        self.dimF2 = (C * T * R * S, K)
        self.dimO2 = (K * M * P * Q, N)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeF = reduce(mul, self.dimF, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut = reduce(mul, self.MPQ, 1) * K

        if all(x == 1 for x in self.TRS) and \
           all(p == 0 for p in self.padding) and \
           all(s == 1 for s in self.strides) and \
           all(d == 1 for d in self.dilation):
            self.dot = True
        else:
            self.dot = False

            self.mSlice = [self.fprop_slice(m, T, D, pad_d, str_d, dil_d) for m in range(M)]
            self.pSlice = [self.fprop_slice(p, R, H, pad_h, str_h, dil_h) for p in range(P)]
            self.qSlice = [self.fprop_slice(q, S, W, pad_w, str_w, dil_w) for q in range(Q)]
            self.dSlice = [self.bprop_slice(d, T, M, pad_d, str_d, dil_d) for d in range(D)]
            self.hSlice = [self.bprop_slice(h, R, P, pad_h, str_h, dil_h) for h in range(H)]
            self.wSlice = [self.bprop_slice(w, S, Q, pad_w, str_w, dil_w) for w in range(W)]
        self.is_mklop = False

    def get_is_mklop(self):
        return self.is_mklop

    def set_is_mklop(self):
        self.is_mklop = True

    def set_not_mklop(self):
        self.is_mklop = False

    def fprop_slice(self, q, S, X, padding, stride, dilation):
        f1 = None
        qs = q * stride - padding
        for s in range(S):
            x = qs + s * dilation
            if f1 is None and x >= 0 and x < X:
                x1 = x
                f1 = s
            if x < X:
                x2 = x
                f2 = s
        if f1 is None:
            return (slice(0, 0, 1), slice(0, 0, 1), 0)
        return (slice(f1, f2 + 1), slice(x1, x2 + 1, dilation), f2 - f1 + 1)

    def bprop_slice(self, x, S, Q, padding, stride, dilation):
        qs = x - (dilation * (S - 1) - padding)
        f1 = None
        for s in range(S):
            q = qs + s * dilation
            if q % stride == 0:
                q //= stride
                if q >= 0 and q < Q:
                    if f1 is None:
                        f1 = s
                        x1 = q
                    f2 = s
                    x2 = q
        if f1 is None:
            return (slice(0, 0, 1), slice(0, 0, 1), 0)

        f_step = 1
        while ((f_step * dilation) % stride) != 0:
            f_step += 1
        x_step = (f_step * dilation) // stride
        return (slice(f1, f2 + 1, f_step), slice(x1, x2 + 1, x_step), 0)

    def compound_ops(self, O, X, bias, bsum, relu, brelu, slope):
        if bias is not None:
            O[:] = (O.reshape((O.shape[0], -1)) + bias).reshape(O.shape)
        if relu or brelu:
            if relu:
                if slope:
                    O[:] = np.maximum(O, 0.0) + slope * np.minimum(O, 0.0)
                else:
                    O[:] = np.maximum(O, 0.0)
            else:
                if slope:
                    O[:] = O * ((X > 0.0) + slope * (X < 0.0))
                else:
                    O[:] = O * (X > 0.0)
        if bsum is not None:
            bsum[:] = np.sum(O.reshape((O.shape[0], -1)), axis=1, keepdims=True)

    def xprop_conv(self, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
                   relu=False, brelu=False, slope=0.0, backward=False, layer_op=None):

        if X is None:
            X = O

        if backward:
            I = I._tensor.reshape(self.dimO)
            O = O._tensor.reshape(self.dimI)
            X = X._tensor.reshape(self.dimI)
        else:
            I = I._tensor.reshape(self.dimI)
            O = O._tensor.reshape(self.dimO)
            X = X._tensor.reshape(self.dimO)
        F = F._tensor.reshape(self.dimF)
        if bias is not None:
            bias = bias._tensor.reshape((O.shape[0], 1))
        if bsum is not None:
            bsum = bsum._tensor.reshape((O.shape[0], 1))

        # 1x1 conv can be cast as a simple dot operation
        if self.dot:
            C = F.shape[0]
            K = F.shape[-1]
            if backward:
                # CxHWN = CxK . KxHWN
                F = F.reshape((C, K))
                I = I.reshape((K, -1))
            else:
                # KxHWN = CxK.T . CxHWN
                F = F.reshape((C, K)).T
                I = I.reshape((C, -1))

            if beta:
                O[:] = alpha * np.dot(F, I).reshape(O.shape) + beta * X
            else:
                O[:] = np.dot(F, I).reshape(O.shape)
                self.compound_ops(O, X, bias, bsum, relu, brelu, slope)
            return

        if backward:
            # C <=> K and mirror T, R, S  (0, 1, 2, 3, 4) => (4, 1, 2, 3, 0)
            F = np.transpose(F[:, ::-1, ::-1, ::-1], (4, 1, 2, 3, 0)).copy()
            mSlice, pSlice, qSlice = self.dSlice, self.hSlice, self.wSlice
        else:
            mSlice, pSlice, qSlice = self.mSlice, self.pSlice, self.qSlice

        K, M, P, Q, N = O.shape

        for m in range(M):
            sliceT, sliceD, _ = mSlice[m]
            for p in range(P):
                sliceR, sliceH, _ = pSlice[p]
                for q in range(Q):
                    sliceS, sliceW, _ = qSlice[q]

                    slicedF = F[:, sliceT, sliceR, sliceS].reshape((-1, K))
                    slicedI = I[:, sliceD, sliceH, sliceW].reshape((-1, N))

                    if beta:
                        O[:, m, p, q] = alpha * np.dot(slicedF.T, slicedI) + \
                            beta * X[:, m, p, q]
                    else:
                        O[:, m, p, q] = np.dot(slicedF.T, slicedI)

        if not beta:
            self.compound_ops(O, X, bias, bsum, relu, brelu, slope)

    # grad_bias is added for convolution layer with bias
    def update_conv(self, I, E, U, alpha=1.0, beta=0.0, grad_bias=None, layer_op=None):

        C = self.C
        K, M, P, Q, N = self.dimO

        I = I._tensor.reshape(self.dimI)
        E = E._tensor.reshape(self.dimO)
        U = U._tensor.reshape(self.dimF)

        # 1x1 conv can be cast as a simple dot operation
        if self.dot:
            # CxK = CxHWN . KxHWN.T
            I = I.reshape((C, -1))
            E = E.reshape((K, -1)).T
            if beta:
                U[:] = alpha * np.dot(I, E).reshape(U.shape) + beta * U
            else:
                U[:] = alpha * np.dot(I, E).reshape(U.shape)
            return

        if beta:
            U *= beta
        else:
            U.fill(0.0)

        for m in range(M):
            sliceT, sliceD, tlen = self.mSlice[m]
            for p in range(P):
                sliceR, sliceH, rlen = self.pSlice[p]
                for q in range(Q):
                    sliceS, sliceW, slen = self.qSlice[q]

                    slicedI = I[:, sliceD, sliceH, sliceW].reshape((-1, N))
                    slicedE = E[:, m, p, q]
                    update = np.dot(slicedI, slicedE.T).reshape((C, tlen, rlen, slen, K))
                    if alpha == 1.0:
                        U[:, sliceT, sliceR, sliceS] += update
                    else:
                        U[:, sliceT, sliceR, sliceS] += alpha * update


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

        # Add below to get H and W tracked
        self.H = H
        self.W = W

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
        self.dilation = (dil_d, dil_h, dil_w)

        # Did not change the names of dimI, dimO, etc. even though dimI is now technically the
        # dimension of the output
        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)
        self.dimO = (K, M, P, Q, N)
        self.dimI2 = (C * D * H * W, N)
        self.dimF2 = (C * T * R * S, K)
        self.dimO2 = (K * M * P * Q, N)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeF = reduce(mul, self.dimF, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        # nOut has to change because P and Q are now the inputs
        self.nOut = reduce(mul, self.DHW, 1) * C

        if all(x == 1 for x in self.TRS) and \
           all(p == 0 for p in self.padding) and \
           all(s == 1 for s in self.strides) and \
           all(d == 1 for d in self.dilation):
            self.dot = True
        else:
            self.dot = False
            self.dSlice = [self.bprop_slice(d, T, M, pad_d, str_d, dil_d) for d in range(D)]
            self.hSlice = [self.bprop_slice(h, R, P, pad_h, str_h, dil_h) for h in range(H)]
            self.wSlice = [self.bprop_slice(w, S, Q, pad_w, str_w, dil_w) for w in range(W)]
            self.mSlice = [self.fprop_slice(m, T, D, pad_d, str_d, dil_d) for m in range(M)]
            self.pSlice = [self.fprop_slice(p, R, H, pad_h, str_h, dil_h) for p in range(P)]
            self.qSlice = [self.fprop_slice(q, S, W, pad_w, str_w, dil_w) for q in range(Q)]


class PoolLayer(object):

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
            self.overlap = (math.ceil(float(J) / str_c) *
                            math.ceil(float(T) / str_d) *
                            math.ceil(float(R) / str_h) *
                            math.ceil(float(S) / str_w))
        else:
            self.overlap = 0.0

        # Compute the output dimensions
        K = lib.output_dim(C, J, pad_c, str_c, pooling=True)
        M = lib.output_dim(D, T, pad_d, str_d, pooling=True)
        P = lib.output_dim(H, R, pad_h, str_h, pooling=True)
        Q = lib.output_dim(W, S, pad_w, str_w, pooling=True)

        self.op = op
        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.N = N
        self.JTRS = (J, T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_c, pad_d, pad_h, pad_w)
        self.strides = (str_c, str_d, str_h, str_w)

        self.dimI = (C, D, H, W, N)
        self.dimO = (K, M, P, Q, N)
        self.dimF2 = None
        self.dimI2 = (C * D * H * W, N)
        self.dimO2 = (K * M * P * Q, N)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut = reduce(mul, self.MPQ, 1) * K

        self.kSlice = [self.pool_slice(k, J, C, pad_c, str_c) for k in range(K)]
        self.mSlice = [self.pool_slice(m, T, D, pad_d, str_d) for m in range(M)]
        self.pSlice = [self.pool_slice(p, R, H, pad_h, str_h) for p in range(P)]
        self.qSlice = [self.pool_slice(q, S, W, pad_w, str_w) for q in range(Q)]

    def pool_slice(self, q, S, X, padding, strides):
        qs = q * strides - padding
        firstI = None
        for s in range(S):
            x = qs + s
            if x >= 0 and x < X:
                if firstI is None:
                    firstI = x
                lastI = x
        return (slice(firstI, lastI + 1), lastI - firstI + 1)
