# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
import math
from operator import mul


def ceil_div(x, y):
    """
    same as int(ceil(float(x)/y)), so no need to import math lib
    """
    return -(-x // y)


def output_dim(X, S, padding, strides):
    """
    compute along 1 dimension, with these sizes, what will be the output dimension

    Arguments:
        X (int): input data dimension
        S (int): filter dimension
        padding (int): padding on each side
        strides (int): striding
    """
    return (X - S + 2 * padding)/strides + 1


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
    """

    def __init__(self, lib, dtype,
                 N, C, K,
                 D=1, H=1, W=1,
                 T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1,
                 bsum=False):

        # Compute the output spatial dimensions
        M = output_dim(D, T, pad_d, str_d)
        P = output_dim(H, R, pad_h, str_h)
        Q = output_dim(W, S, pad_w, str_w)

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
        self.bsum = bsum

        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)
        self.dimO = (K, M, P, Q, N)
        self.dimI2 = (C * D * H * W, N)
        self.dimF2 = (C * T * R * S, K)
        self.dimO2 = (K * M * P * Q, N)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeF = reduce(mul, self.dimF, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut = reduce(mul, self.MPQ, 1) * K

        self.mSlice = [self.fprop_slice(m, T, D, pad_d, str_d) for m in range(M)]
        self.pSlice = [self.fprop_slice(p, R, H, pad_h, str_h) for p in range(P)]
        self.qSlice = [self.fprop_slice(q, S, W, pad_w, str_w) for q in range(Q)]
        self.dSlice = [self.bprop_slice(d, T, M, pad_d, str_d) for d in range(D)]
        self.hSlice = [self.bprop_slice(h, R, P, pad_h, str_h) for h in range(H)]
        self.wSlice = [self.bprop_slice(w, S, Q, pad_w, str_w) for w in range(W)]

    def fprop_slice(self, q, S, X, padding, strides):
        firstF = 0
        lastF = S - 1
        qs = q * strides - padding
        x2 = qs + lastF
        if qs < 0:
            firstF = -qs
            qs = 0
        if x2 >= X:
            dif = x2 - X + 1
            lastF -= dif
            x2 -= dif
        return (slice(firstF, lastF+1), slice(qs, x2+1), lastF-firstF+1)

    def bprop_slice(self, x, S, Q, padding, strides):
        qs = x - (S - padding - 1)
        sliceF = []
        sliceO = []
        for s in range(S):
            q = qs + s
            if q % strides == 0:
                q //= strides
                if q >= 0 and q < Q:
                    sliceF.append(S - s - 1)
                    sliceO.append(q)
        return sliceF, sliceO


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
                 str_d=1, str_h=1, str_w=1):

        # Set T, M and D to be consts.
        T = 1
        M = 1
        D = 1

        # Cannot get exact, e.g. because not unique
        H = (P - 1) * str_h - 2 * pad_h + R
        W = (Q - 1) * str_w - 2 * pad_w + S

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

        self.dSlice = [self.bprop_slice(d, T, M, pad_d, str_d) for d in range(D)]
        self.hSlice = [self.bprop_slice(h, R, P, pad_h, str_h) for h in range(H)]
        self.wSlice = [self.bprop_slice(w, S, Q, pad_w, str_w) for w in range(W)]
        self.mSlice = [self.fprop_slice(m, T, D, pad_d, str_d) for m in range(M)]
        self.pSlice = [self.fprop_slice(p, R, H, pad_h, str_h) for p in range(P)]
        self.qSlice = [self.fprop_slice(q, S, W, pad_w, str_w) for q in range(Q)]


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
        K = output_dim(C, J, pad_c, str_c)
        M = output_dim(D, T, pad_d, str_d)
        P = output_dim(H, R, pad_h, str_h)
        Q = output_dim(W, S, pad_w, str_w)

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
        return (slice(firstI, lastI+1), lastI-firstI+1)
