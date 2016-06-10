#!/usr/bin/python

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
import numpy as np
#from ipdb import set_trace
from struct import pack, unpack
from neon import logger as neon_logger

def ceil_div(x, y):
    return -(-x // y)

def out_dim(S, X, padding, strides):
    return ceil_div(X - S + 1 + 2 * padding, strides)

def strip_mantissa(val):
    i = unpack('I', pack('f', val))[0] & 0x7f800000
    f = unpack('f', pack('I', i))[0]
    return f

def immediate(val):
    i = unpack('I', pack('f', val))[0] & 0x7ffff000
    f = unpack('f', pack('I', i))[0]
    return f

def quantize(ary, bits, sign=1):
    maxval = float(np.max(np.absolute(ary)))
    scale  = strip_mantissa(maxval) / float(1 << (bits - sign - 1))
    ary    = np.around(ary * (1.0 / scale)).astype(np.int64)
    return ary, np.float64(scale)

######### Direct Convolution #########

def fconv_slice(q, S, X, padding, strides):
    f1 = 0
    f2 = S - 1
    x1 = q * strides - padding
    x2 = x1 + f2
    if x1 < 0:
        f1 = -x1
        x1 = 0
    if x2 >= X:
        dif = x2 - X + 1
        f2 -= dif
        x2 -= dif
    return (slice(f1, f2 + 1), slice(x1, x2 + 1), f2 - f1 + 1)

def bconv_slice(x, S, Q, padding, strides):
    qs = x - (S - padding - 1)
    firstF = None
    for s in range(S): #TODO remove loop logic here.
        q = qs + s
        if q % strides == 0:
            q //= strides
            if q >= 0 and q < Q:
                if firstF is None:
                    firstF = s
                    firstE = q
                lastF = s
                lastE = q
    if firstF is None:
        return (slice(0,0,1), slice(0,0,1), 0)
    return (slice(firstF,lastF + 1,strides), slice(firstE,lastE + 1,1), 0)

def xprop_direct(I, F, O, padding, strides, backward=False):

    if all(x == 1 for x in F.shape[1:3]):
        C = F.shape[0]
        K = F.shape[4]
        if backward:
            # CxHWN = CxK . KxHWN
            O[:] = np.dot( F.reshape((C, -1)),   I.reshape((K, -1)) ).reshape((O.shape))
        else:
            # KxHWN = CxK.T . CxHWN
            O[:] = np.dot( F.reshape((C, -1)).T, I.reshape((C, -1)) ).reshape((O.shape))
        return

    if backward:
        # C <=> K and mirror R,S
        F = np.transpose(F[:,::-1,::-1,:], (3,1,2,0)).copy()
        xconv_slice = bconv_slice
    else:
        xconv_slice = fconv_slice

    C, Y, X, N = I.shape
    C, R, S, K = F.shape
    K, P, Q, N = O.shape

    qSlice = [ xconv_slice(q, S, X, padding[0], strides[0]) for q in range(Q) ]

    for p in range(P):
        sliceR, sliceY, _ = xconv_slice(p, R, Y, padding[1], strides[1])

        for q in range(Q):
            sliceS, sliceX, _ = qSlice[q]

            slicedF = F[:,sliceR,sliceS,:].reshape((-1, K))
            slicedI = I[:,sliceY,sliceX,:].reshape((-1, N))

            O[:,p,q,:] = np.dot( slicedF.T,  slicedI )

def updat_direct(I, E, U, padding, strides):

    C, Y, X, N = I.shape
    K, P, Q, N = E.shape
    C, R, S, K = U.shape

    if all(x == 1 for x in (R, S)):
        # CxK = CxHWN . KxHWN.T
        U[:] = np.dot( I.reshape((C, -1)), E.reshape((K, -1)).T ).reshape((U.shape))
        return

    U.fill(0.0)

    qSlice = [ fconv_slice(q, S, X, padding[0], strides[0]) for q in range(Q) ]

    for p in range(P):
        sliceR, sliceY, rlen = fconv_slice(p, R, Y, padding[1], strides[1])

        for q in range(Q):
            sliceS, sliceX, slen = qSlice[q]

            slicedI = I[:,sliceY,sliceX,:].reshape((-1, N))
            slicedE = E[:,p,q,:]

            U[:,sliceR,sliceS,:] += np.dot( slicedI,  slicedE.T ).reshape((C, rlen, slen, K))

######### Winograd Convolution #########

I_2x2_5x5 = (
    np.array([
    [ 1.0,  0.00 ],
    [ 1.0,  0.75 ],
    [ 1.0, -0.75 ],
    [ 1.0,  1.50 ],
    [ 1.0, -1.50 ],
    [ 0.0,  1.00 ]]),
)

F_2x2_5x5 = (
    np.array([
    [    64.0 / 81.0,        0.0,       0.0,      0.0,      0.0 ],
    [ -128.0 / 243.0, -32.0 / 81.0, -8.0 / 27.0, -2.0 / 9.0, -1.0 / 6.0 ],
    [ -128.0 / 243.0,  32.0 / 81.0, -8.0 / 27.0,  2.0 / 9.0, -1.0 / 6.0 ],
    [   32.0 / 243.0,  16.0 / 81.0,  8.0 / 27.0,  4.0 / 9.0,  2.0 / 3.0 ],
    [   32.0 / 243.0, -16.0 / 81.0,  8.0 / 27.0, -4.0 / 9.0,  2.0 / 3.0 ],
    [          0.0,        0.0,       0.0,      0.0,      1.0 ]]),
)

O_2x2_5x5 = (
    np.array([
    [  1.265625,  0.0,     0.0,     0.0,     0.0,      0.0      ],
    [  0.0,      -1.6875,  1.6875, -0.84375, 0.84375,  1.265625 ],
    [ -2.8125,   -2.25,   -2.25,   -0.5625, -0.5625,   0.0      ],
    [  0.0,       0.75,   -0.75,    1.5,    -1.5,     -2.8125   ],
    [  1.0,       1.0,     1.0,     1.0,     1.0,      0.0      ],
    [  0.0,       0.0,     0.0,     0.0,     0.0,      1.0      ]]),
)

def trans_I_2x2_5x5(Iw, I, minimal=False, trans=False):
    if minimal:

        T0 = np.empty((6,2))
        T1 = np.empty((6,6))

        for O, I in ((T0, I), (T1, T0.T)):

            # 4*2 + 4*6 = 30

            O[0,:] = I[0,:]
            O[1,:] = I[0,:] + I[1,:] * 0.75
            O[2,:] = I[0,:] - I[1,:] * 0.75
            O[3,:] = I[0,:] + I[1,:] * 1.50
            O[4,:] = I[0,:] - I[1,:] * 1.50
            O[5,:] = I[1,:]

        Iw[:] = T1.T

    else:
        Iw[:] = np.dot( np.dot(I_2x2_5x5[trans[0]], I), I_2x2_5x5[trans[1]].T )


def trans_F_2x2_5x5(Fw, F, minimal=False, trans=False):
    if minimal:

        T0 = np.empty((6,5))
        T1 = np.empty((6,6))

        for O, I in ((T0, F[::-1,::-1]), (T1, T0.T)):

            # 14*5 + 14*6 = 154

            t0 = I[2,:] * 8.0 / 27.0
            t1 = I[1,:] * 32.0 / 81.0   + I[3,:] * 2.0 / 9.0
            t2 = I[1,:] * 16.0 / 81.0   + I[3,:] * 4.0 / 9.0
            t3 = I[0,:] * -128.0 / 243.0 - I[4,:] * 1.0 / 6.0 - t0
            t4 = I[0,:] * 32.0 / 243.0  + I[4,:] * 2.0 / 3.0 + t0

            O[0,:] = I[0,:] * 64.0 / 81.0
            O[1,:] = t3 - t1
            O[2,:] = t3 + t1
            O[3,:] = t4 + t2
            O[4,:] = t4 - t2
            O[5,:] = I[4,:]

        Fw[:] = T1.T

    else:
        Fw[:] = np.dot( np.dot(F_2x2_5x5[trans[0]], F[::-1,::-1]), F_2x2_5x5[trans[1]].T )

def trans_O_2x2_5x5(Mw, minimal=False, trans=False):
    if minimal:

        T0 = np.empty((6,6))
        T1 = np.empty((6,6))

        for O, I in ((T0, Mw), (T1, T0.T)):

            # 16*6 + 16*6 = 192

            t0 = I[1,:] + I[2,:]
            t1 = I[3,:] + I[4,:]
            t2 = I[1,:] - I[2,:]
            t3 = I[3,:] - I[4,:]

            O[0,:] = I[0,:] * 1.265625
            O[4,:] = I[0,:] + t0  + t1
            O[2,:] = t0 * -2.25   + t1 * -0.5625  + I[0,:] * -2.8125
            O[1,:] = t2 * -1.6875 + t3 * -0.84375 + I[5,:] *  1.265625
            O[3,:] = t2 *  0.75   + t3 *  1.5     + I[5,:] * -2.8125
            O[5,:] = I[5,:]

        return T1.T

    else:
        return np.dot( np.dot(O_2x2_5x5[trans[0]], Mw), O_2x2_5x5[trans[1]].T )


def image_slice(x, X, B):
    x0 = x * B
    x1 = x0 + B
    if x1 > X:
        return slice(x0,X,1), (0,1)
    return slice(x0,x1,1), (0,0)

def output_slice(x, P, B, D, pad):
    p0 = x * B + pad - 4
    p1 = p0 + D
    if p0 < 0:
        m0 = -p0
        p0 = 0
    else:
        m0 = 0
    if p1 > P:
        m1 = D - (p1 - P)
        p1 = P
    else:
        m1 = D
    return slice(p0,p1,1), slice(m0,m1,1)

def xprop_winograd(I, F, O, padding, minimal=False, trans=False, backward=False):

    if backward:
        # C <=> K and mirror R,S
        F = np.transpose(F[:,::-1,::-1,:], (3,1,2,0)).copy()
        # Invert padding
        padding = [4 - p for p in padding]

    C, Y, X, N = I.shape
    K, P, Q, N = O.shape

    B = 2
    D = 6
    Yw = ceil_div(Y, B)
    Xw = ceil_div(X, B)

    Fw = np.empty((D,D,C,K))
    Iw = np.empty((D,D,C,Yw,Xw,N))
    Mw = np.empty((D,D,K,Yw,Xw,N)) #, dtype=np.int64
    O.fill(0.0)

    # Transform Filters
    for c in range(C):
        for k in range(K):
            trans_F_2x2_5x5(Fw[:,:,c,k], F[c,:,:,k], minimal, trans)

    # Iterate over image transform dimensions and slice out tiles of the image
    for y in range(Yw):
        slice_y, pad_y = image_slice(y, Y, B)

        for x in range(Xw):
            slice_x, pad_x = image_slice(x, X, B)

            sliceI = I[:, slice_y, slice_x, :]

            # add any zero padding if needed
            if pad_y[1] or pad_x[1]:
                sliceI = np.pad(sliceI, ((0,0), pad_y, pad_x, (0,0)), 'constant')

            # Apply the Image transform
            for c in range(C):
                for n in range(N):
                    trans_I_2x2_5x5(Iw[:,:,c,y,x,n], sliceI[c,:,:,n], minimal, trans)

    #print Iw[:,:,0,0,0,0]
    # print Fw[:,:,0,0].reshape(36,1)
    #exit()

    # Fw, scaleF = quantize(Fw, 16)
    # Iw, scaleI = quantize(Iw, 16)

    # Fw = Fw.astype(np.float16).astype(np.float64)
    # Iw = Iw.astype(np.float32).astype(np.float64)

    # Batched gemm for the pointwise multiplication step
    for s in range(D):
        for t in range(D):
            # [K,Yw,Xw,N] = [C,K].T . [C,YwXwN]
            Mw[s,t] = np.dot( Fw[s,t].T, Iw[s,t].reshape(C, -1) ).reshape((K,Yw,Xw,N))

    #print Mw[:,:,0,0,0,0]
    #exit()

    # Mw = Mw.astype(np.float64) * scaleF * scaleI

    # Iterate over the convovled result in the pointwise space and apply inverse transform
    for y in range(Yw):
        slice_p, slice_y = output_slice(y, P, B, D, padding[0])
        for x in range(Xw):
            slice_q, slice_x = output_slice(x, Q, B, D, padding[1])
            for k in range(K):
                for n in range(N):
                    # Toss out any points that don't fit
                    Out = trans_O_2x2_5x5(Mw[:,:,k,y,x,n], minimal, trans)

                    O[k,slice_p,slice_q,n] += Out[slice_y,slice_x]


### Test Code ###

np.set_printoptions(threshold=8192 * 4, linewidth=600, formatter={'float':lambda x: "%4.0f" % x})

minimal = 1
trans = (0,0)
ones = 0
N    = 4
C, K = 4, 4
Y, X = 6, 6
R, S = 5, 5     # Fixed winograd dim
strides = 1, 1  # Fixed winograd dim
padding = 2, 2  # 0-2

P = out_dim(R, Y, padding[0], strides[0])
Q = out_dim(S, X, padding[1], strides[1])

neon_logger.display("{}".format(P, Q))

dimI = (C,Y,X,N)
dimF = (C,R,S,K)
dimO = (K,P,Q,N)

if ones:
    I  = np.ones(dimI)
    F  = np.ones(dimF)
    E  = np.ones(dimO)

    # for c in range(2):
    #     for n in range(32):
    #         I[c,:,:,n] = np.arange(0,36, dtype=np.float32).reshape(6,6)

    #I[0,:,:,0] = np.arange(0,36).reshape(6,6)
    F[0,:,:,0] = np.arange(0,25).reshape(5,5)

    # for p,q in np.ndindex((Y,X)):
    #     I[:,p,q,:] = np.identity(N)

    # for p,q in np.ndindex((P,Q)):
    #     for n in range(N):
    #         E[:,p,q,n] = range(K)

    # for c in range(C):
    #     for n in range(N):
    #         I[c,:,:,n] = c+1 #np.arange(1+c,37+c).reshape((6,6))

    # for k in range(K):
    #     for n in range(N):
    #         E[k,:,:,n] = k+1 #np.arange(1+k,17+k).reshape((4,4))

else:
    I  = np.random.uniform(-1.0, 1.0, dimI)
    #F  = np.random.normal(0.0, 0.1, dimF)
    F  = np.random.uniform(-1.0, 1.0, dimF)
    E  = np.random.uniform(-1.0, 1.0, dimO)

Od = np.empty(dimO)
Ow = np.empty(dimO) #, dtype=np.float32

Bd = np.empty(dimI)
Bw = np.empty(dimI) #, dtype=np.float32

xprop_direct(I, F, Od, padding, strides)
xprop_winograd(I, F, Ow, padding, minimal=minimal, trans=trans)

xprop_direct(E, F, Bd, padding, strides, backward=True)
xprop_winograd(E, F, Bw, padding, minimal=minimal, trans=trans, backward=True)

difO = Od - Ow
difB = Bd - Bw

neon_logger.display(abs(difO).max() / Od.max())
neon_logger.display(abs(difB).max() / Bd.max())

# print Od[0,:,:,0]
# print Ow[0,:,:,0]
# print difO[0,:,:,0]
