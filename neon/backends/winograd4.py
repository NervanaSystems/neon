#!/usr/bin/python

# Copyright 2015-2016 Nervana Systems Inc. All rights reserved.
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



I_4x4_3x3 = (
    np.array([
    [ 4.0,  0.0, -5.0,  0.0, 1.0, 0.0],
    [ 0.0, -4.0, -4.0,  1.0, 1.0, 0.0],
    [ 0.0,  4.0, -4.0, -1.0, 1.0, 0.0],
    [ 0.0, -2.0, -1.0,  2.0, 1.0, 0.0],
    [ 0.0,  2.0, -1.0, -2.0, 1.0, 0.0],
    [ 0.0,  4.0,  0.0, -5.0, 0.0, 1.0]]),

    np.array([
    [ 4.0,  0.0, -5.0,  0.0, 1.0, 0.0],
    [ 0.0, -4.0, -4.0,  1.0, 1.0, 0.0],
    [ 0.0,  4.0, -4.0, -1.0, 1.0, 0.0],
    [ 0.0, -2.0, -1.0,  2.0, 1.0, 0.0],
    [ 0.0,  2.0, -1.0, -2.0, 1.0, 0.0],
    [ 0.0,  4.0,  0.0, -5.0, 0.0, 1.0]]),

    np.array([
    [ 1.0,  0.0,     -5.0 / 4.0,  0.0,      1.0 / 4.0, 0.0],
    [ 0.0,  2.0 / 3.0,  2.0 / 3.0, -1.0 / 6.0, -1.0 / 6.0, 0.0],
    [ 0.0, -2.0 / 3.0,  2.0 / 3.0,  1.0 / 6.0, -1.0 / 6.0, 0.0],
    [ 0.0, -1.0 / 12., -1.0 / 24.,  1.0 / 12.,  1.0 / 24., 0.0],
    [ 0.0,  1.0 / 12., -1.0 / 24., -1.0 / 12.,  1.0 / 24., 0.0],
    [ 0.0,  4.0,      0.0,     -5.0,      0.0,     1.0]]),

    np.array([
    [ 441.0 / 400.0,         0.0, -137.0 / 50.00,         0.0, 1.0, 0.0],
    [ 0.0,         -63.0 / 40.00,   -9.0 / 4.000,    7.0 / 10.0, 1.0, 0.0],
    [ 0.0,          63.0 / 40.00,   -9.0 / 4.000,   -7.0 / 10.0, 1.0, 0.0],
    [ 0.0,        -147.0 / 200.0,  -49.0 / 100.0,    3.0 / 2.00, 1.0, 0.0],
    [ 0.0,         147.0 / 200.0,  -49.0 / 100.0,   -3.0 / 2.00, 1.0, 0.0],
    [ 0.0,         441.0 / 400.0,          0.0, -137.0 / 50.0, 0.0, 1.0]]),

    np.array([
    [ 0.87890625,  0,          -2.640625,  0,        1, 0 ],
    [ 0,          -1.40625,    -2.25,      0.625,    1, 0 ],
    [ 0,           1.40625,    -2.25,     -0.625,    1, 0 ],
    [ 0,          -0.5859375,  -0.390625,  1.5,      1, 0 ],
    [ 0,           0.5859375,  -0.390625, -1.5,      1, 0 ],
    [ 0,           0.87890625,  0,        -2.640625, 0, 1 ]]),

    np.array([
    [ 0.87890625,  0,          -2.640625,  0,        1, 0 ],
    [ 0,          -1.40625,    -2.25,      0.625,    1, 0 ],
    [ 0,           1.40625,    -2.25,     -0.625,    1, 0 ],
    [ 0,          -0.5859375,  -0.390625,  1.5,      1, 0 ],
    [ 0,           0.5859375,  -0.390625, -1.5,      1, 0 ],
    [ 0,           0.87890625,  0,        -2.640625, 0, 1 ]]),
)

F_4x4_3x3 = (
    np.array([
    [  1.0 / 4.0,  0.0,      0.0     ],
    [ -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0 ],
    [ -1.0 / 6.0,  1.0 / 6.0, -1.0 / 6.0 ],
    [  1.0 / 24.,  1.0 / 12.,  1.0 / 6.0 ],
    [  1.0 / 24., -1.0 / 12.,  1.0 / 6.0 ],
    [  0.0,      0.0,      1.0     ]]),

    np.array([
    [ 1.0,  0.0,  0.0 ],
    [ 1.0,  1.0,  1.0 ],
    [ 1.0, -1.0,  1.0 ],
    [ 1.0,  2.0,  4.0 ],
    [ 1.0, -2.0,  4.0 ],
    [ 0.0,  0.0,  1.0 ]]),

    np.array([
    [ 1.0,  0.0,  0.0 ],
    [ 1.0,  1.0,  1.0 ],
    [ 1.0, -1.0,  1.0 ],
    [ 1.0,  2.0,  4.0 ],
    [ 1.0, -2.0,  4.0 ],
    [ 0.0,  0.0,  1.0 ]]),

    np.array([
    [  400.0 / 441.00,          0.0,        0.0 ],
    [ -625.0 / 1078.0, -125.0 / 308.0, -25.0 / 88.0 ],
    [ -625.0 / 1078.0,  125.0 / 308.0, -25.0 / 88.0 ],
    [   25.0 / 198.00,   25.0 / 132.0,  25.0 / 88.0 ],
    [   25.0 / 198.00,  -25.0 / 132.0,  25.0 / 88.0 ],
    [           0.0,          0.0,        1.0 ]]),

    np.array([
    [ 1.13777777777778,   0,                  0                ],
    [-0.688403361344538, -0.430252100840336, -0.26890756302521 ],
    [-0.688403361344538,  0.430252100840336, -0.26890756302521 ],
    [ 0.119514472455649,  0.179271708683473,  0.26890756302521 ],
    [ 0.119514472455649, -0.179271708683473,  0.26890756302521 ],
    [ 0,                  0,                  1                ]]),

    np.array([
    [ 1, 1,         1 ,       1,     1,     0 ],
    [ 0, 0.625,    -0.625,    1.5,  -1.5,   0 ],
    [ 0, 0.390625,  0.390625, 2.25,  2.25,  1 ]]).T,
)

O_4x4_3x3 = (
    np.array([
    [ 1.0, 1.0,  1.0, 1.0,  1.0, 0.0 ],
    [ 0.0, 1.0, -1.0, 2.0, -2.0, 0.0 ],
    [ 0.0, 1.0,  1.0, 4.0,  4.0, 0.0 ],
    [ 0.0, 1.0, -1.0, 8.0, -8.0, 1.0 ]]),

    np.array([
    [ 1.0 / 4.0, -1.0 / 6.0, -1.0 / 6.0, 1.0 / 24.,  1.0 / 24., 0.0 ],
    [ 0.0,     -1.0 / 6.0,  1.0 / 6.0, 1.0 / 12., -1.0 / 12., 0.0 ],
    [ 0.0,     -1.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0,  1.0 / 6.0, 0.0 ],
    [ 0.0,     -1.0 / 6.0,  1.0 / 6.0, 1.0 / 3.0, -1.0 / 3.0, 1.0 ]]),

    np.array([
    [ 1.0, 1.0,  1.0, 1.0,  1.0, 0.0 ],
    [ 0.0, 1.0, -1.0, 2.0, -2.0, 0.0 ],
    [ 0.0, 1.0,  1.0, 4.0,  4.0, 0.0 ],
    [ 0.0, 1.0, -1.0, 8.0, -8.0, 1.0 ]]),

    np.array([
    [ 1.0,       1.0000,        1.0000,      1.0,       1.0, 0.0 ],
    [ 0.0,   7.0 / 10.000,   -7.0 / 10.000,  3.0 / 2.0,  -3.0 / 2.0, 0.0 ],
    [ 0.0,  49.0 / 100.00,   49.0 / 100.00,  9.0 / 4.0,   9.0 / 4.0, 0.0 ],
    [ 0.0, 343.0 / 1000.0, -343.0 / 1000.0, 27.0 / 8.0, -27.0 / 8.0, 1.0 ]]),

    np.array([
    [ 1, 1,            1,           1,      1,     0 ],
    [ 0, 0.625,       -0.625,       1.5,   -1.5,   0 ],
    [ 0, 0.390625,     0.390625,    2.25,   2.25,  0 ],
    [ 0, 0.244140625, -0.244140625, 3.375, -3.375, 1 ]]),

    np.array([
    [ 1.13777777777778,   0,                  0,                 0,                ],
    [-0.688403361344538, -0.430252100840336, -0.26890756302521, -0.168067226890756 ],
    [-0.688403361344538,  0.430252100840336, -0.26890756302521,  0.168067226890756 ],
    [ 0.119514472455649,  0.179271708683473,  0.26890756302521,  0.403361344537815 ],
    [ 0.119514472455649, -0.179271708683473,  0.26890756302521, -0.403361344537815 ],
    [ 0,                  0,                  0,                 1,                ]]).T,
)

rcp3  = 1.0 / 3.0
rcp4  = 1.0 / 4.0
rcp6  = 1.0 / 6.0
rcp12 = 1.0 / 12.0
rcp24 = 1.0 / 24.0

def trans_I_4x4_3x3(Iw, I, minimal=False, trans=False):
    if minimal:

        T0 = np.empty((6,6))
        T1 = np.empty((6,6))

        for O, I in ((T0, I), (T1, T0.T)):

    # np.array([
    # [ 0.87890625,  0,           -2.640625,       0,        1, 0 ],
    # [ 0,          -1.40625,     -2.25,           0.625,    1, 0 ],
    # [ 0,           1.40625,     -2.25,          -0.625,    1, 0 ],
    # [ 0,          -0.5859375,   -0.390625,       1.5,      1, 0 ],
    # [ 0,           0.5859375,   -0.390625,      -1.5,      1, 0 ],
    # [ 0,           0.87890625,   0,             -2.640625, 0, 1 ]]),


    # np.array([
    # [ 441.0/400.0,         0.0, -137.0/50.00,         0.0, 1.0, 0.0],
    # [ 0.0,         -63.0/40.00,   -9.0/4.000,    7.0/10.0, 1.0, 0.0],
    # [ 0.0,          63.0/40.00,   -9.0/4.000,   -7.0/10.0, 1.0, 0.0],
    # [ 0.0,        -147.0/200.0,  -49.0/100.0,    3.0/2.00, 1.0, 0.0],
    # [ 0.0,         147.0/200.0,  -49.0/100.0,   -3.0/2.00, 1.0, 0.0],
    # [ 0.0,         441.0/400.0,          0.0, -137.0/50.0, 0.0, 1.0]]),

            t0 = I[4,:] - I[2,:] * 2.25
            t1 = I[3,:] - I[1,:] * 2.25
            t2 = I[4,:] - I[2,:] * 0.390625
            t3 = I[3,:] - I[1,:] * 0.390625

            O[0,:] = I[0,:] * 0.87890625 - I[2,:] * 2.640625 + I[4,:]
            O[1,:] = t0 + t1 * 0.625
            O[2,:] = t0 - t1 * 0.625
            O[3,:] = t2 + t3 * 1.5
            O[4,:] = t2 - t3 * 1.5
            O[5,:] = I[1,:] * 0.87890625 - I[3,:] * 2.640625 + I[5,:]



            # t0 = I[4,:] - I[2,:]*2.25
            # t1 = I[3,:] - I[1,:]*2.25
            # t2 = I[4,:] - I[2,:]*0.49
            # t3 = I[3,:] - I[1,:]*0.49

            # O[0,:] = I[0,:]*1.1025 - I[2,:]*2.74 + I[4,:]
            # O[1,:] = t0 + t1*0.7
            # O[2,:] = t0 - t1*0.7
            # O[3,:] = t2 + t3*1.5
            # O[4,:] = t2 - t3*1.5
            # O[5,:] = I[1,:]*1.1025 - I[3,:]*2.74 + I[5,:]

            # t0 = I[4,:] - I[2,:]*4.0
            # t1 = I[3,:] - I[1,:]*4.0
            # t2 = I[4,:] - I[2,:]
            # t3 = I[3,:] - I[1,:]
            # O[0,:] = I[0,:]*4.0 - I[2,:]*5.0 + I[4,:]
            # O[1,:] = t0 + t1
            # O[2,:] = t0 - t1
            # O[3,:] = t2 + t3*2.0
            # O[4,:] = t2 - t3*2.0
            # O[5,:] = I[1,:]*4.0 - I[3,:]*5.0 + I[5,:]

            #t0 = (I[2,:]*4.0 - I[4,:])*rcp6
            #t1 = (I[1,:]*4.0 - I[3,:])*rcp6
            #t2 = (I[4,:] - I[2,:])*rcp24
            #t3 = (I[3,:] - I[1,:])*rcp12
            #O[0,:] = I[0,:] + (I[2,:]*-5.0 + I[4,:])*rcp4
            #O[1,:] = t0 + t1
            #O[2,:] = t0 - t1
            #O[3,:] = t2 + t3
            #O[4,:] = t2 - t3
            #O[5,:] = I[1,:]*4.0 - I[3,:]*5.0 + I[5,:]

        Iw[:] = T1.T

    else:
        Iw[:] = np.dot( np.dot(I_4x4_3x3[trans[0]], I), I_4x4_3x3[trans[1]].T )

def trans_F_4x4_3x3(Fw, F, minimal=False, trans=False):
    if minimal:

        T0 = np.empty((6,3))
        T1 = np.empty((6,6))

        for O, I in ((T0, F), (T1, T0.T)):

            t0 =  25.0 / 88.0    * I[2,:]
            t1 = -625.0 / 1078.0 * I[0,:] - t0
            t2 =  25.0 / 198.0   * I[0,:] + t0

            O[0,:] = I[0,:] * 400.0 / 441.00
            O[1,:] = t1 - 125.0 / 308.0 * I[1,:]
            O[2,:] = t1 + 125.0 / 308.0 * I[1,:]
            O[3,:] = t2 + 25.0 / 132.0  * I[1,:]
            O[4,:] = t2 - 25.0 / 132.0  * I[1,:]
            O[5,:] = I[2,:]

            # t0 =  rcp6  * I[2,:]
            # t1 = -rcp6  * I[0,:] - t0
            # t2 =  rcp24 * I[0,:] + t0

            # O[0,:] = rcp4 * I[0,:]
            # O[1,:] = t1 - rcp6  * I[1,:]
            # O[2,:] = t1 + rcp6  * I[1,:]
            # O[3,:] = t2 + rcp12 * I[1,:]
            # O[4,:] = t2 - rcp12 * I[1,:]
            # O[5,:] = I[2,:]

            # t0 = I[0,:] + I[2,:]
            # t1 = I[0,:] + I[2,:]*4.0
            # O[0,:] = I[0,:]
            # O[1,:] = t0 + I[1,:]
            # O[2,:] = t0 - I[1,:]
            # O[3,:] = t1 + I[1,:]*2.0
            # O[4,:] = t1 - I[1,:]*2.0
            # O[5,:] = I[2,:]

        Fw[:] = T1.T

    else:
        Fw[:] = np.dot( np.dot(F_4x4_3x3[trans[0]], F), F_4x4_3x3[trans[1]].T )

def trans_O_4x4_3x3(Mw, minimal=False, trans=False):
    if minimal:

        T0 = np.empty((4,6))
        T1 = np.empty((4,4))

        for O, I in ((T0, Mw), (T1, T0.T)):

            t0 = I[1,:] + I[2,:]
            t1 = I[3,:] + I[4,:]
            t2 = I[1,:] - I[2,:]
            t3 = I[3,:] - I[4,:]

            O[0,:] = t0 + t1  + I[0,:]
            O[1,:] = t2 * 0.625       + t3 * 1.500
            O[2,:] = t0 * 0.390625    + t1 * 2.250
            O[3,:] = t2 * 0.244140625 + t3 * 3.375 + I[5,:]

            # t0 = I[1,:] + I[2,:]
            # t1 = I[3,:] + I[4,:]
            # t2 = I[1,:] - I[2,:]
            # t3 = I[3,:] - I[4,:]

            # O[0,:] = t0 + t1  + I[0,:]
            # O[1,:] = t2*0.700 + t3*1.500
            # O[2,:] = t0*0.490 + t1*2.250
            # O[3,:] = t2*0.343 + t3*3.375 + I[5,:]

            #t0 =  I[1,:] + I[2,:]
            #t1 =  I[3,:] + I[4,:]
            #t2 =  I[1,:] - I[2,:]
            #t3 =  I[3,:] - I[4,:]
            #O[0,:] = t0 + t1     + I[0,:]
            #O[1,:] = t2 + t3*2.0
            #O[2,:] = t0 + t1*4.0
            #O[3,:] = t2 + t3*8.0 + I[5,:]

        return T1.T

    else:
        return np.dot( np.dot(O_4x4_3x3[trans[0]], Mw), O_4x4_3x3[trans[1]].T )


def trans_F_3x3_4x4(Fw, F, minimal=False, trans=False):
    if minimal:

        T0 = np.empty((6,4))
        T1 = np.empty((6,6))

        for O, I in ((T0, F), (T1, T0.T)):

            # np.array([
            # [ 1.13777777777778,   0,                  0,                 0,                ],
            # [-0.688403361344538, -0.430252100840336, -0.26890756302521, -0.168067226890756 ],
            # [-0.688403361344538,  0.430252100840336, -0.26890756302521,  0.168067226890756 ],
            # [ 0.119514472455649,  0.179271708683473,  0.26890756302521,  0.403361344537815 ],
            # [ 0.119514472455649, -0.179271708683473,  0.26890756302521, -0.403361344537815 ],
            # [ 0,                  0,                  0,                 1,                ]]).T,

            t0 = I[2,:] * 0.26890756302521
            t1 = I[0,:] * -0.688403361344538 - t0
            t2 = I[0,:] * 0.119514472455649 + t0
            t3 = I[1,:] * 0.430252100840336 + I[3,:] * 0.168067226890756
            t4 = I[1,:] * 0.179271708683473 + I[3,:] * 0.403361344537815

            O[0,:] = I[0,:] * 1.13777777777778
            O[1,:] = t1 - t3
            O[2,:] = t1 + t3
            O[3,:] = t2 + t4
            O[4,:] = t2 - t4
            O[5,:] = I[3,:]

            # 1.0,  0.0,        0.0,          0.0,
            # 1.0,  7.0/10.000, 49.0/100.00,  343.0/1000.0,
            # 1.0, -7.0/10.000, 49.0/100.00, -343.0/1000.0,
            # 1.0,  3.0/2.0,    9.0/4.0,      27.0/8.0,
            # 1.0, -3.0/2.0,    9.0/4.0,     -27.0/8.0,
            # 0.0,  0.0,        0.0,          1.0

            # t0 =  I[0,:] + I[2,:]*49.0/100.0
            # t1 =  I[0,:] + I[2,:]*9.0/4.0
            # t2 =  I[1,:] + I[3,:]*49.0/100.0
            # t3 =  I[1,:] + I[3,:]*9.0/4.0

            # O[0,:] = I[0,:]
            # O[1,:] = t0 + t2*7.0/10.0
            # O[2,:] = t0 - t2*7.0/10.0
            # O[3,:] = t1 + t3*3.0/2.0
            # O[4,:] = t1 - t3*3.0/2.0
            # O[5,:] = I[3,:]

            # t0 =  I[0,:] + I[2,:]
            # t1 =  I[0,:] + I[2,:]*4.0
            # t2 =  I[1,:] + I[3,:]
            # t3 =  I[1,:] + I[3,:]*4.0

            # O[0,:] = I[0,:]
            # O[1,:] = t0 + t2
            # O[2,:] = t0 - t2
            # O[3,:] = t1 + t3*2.0
            # O[4,:] = t1 - t3*2.0
            # O[5,:] = I[3,:]

            # t0 = (I[0,:] + I[2,:])*-rcp6
            # t1 = (I[0,:]*rcp4 + I[2,:])*rcp6
            # t2 = (I[1,:] + I[3,:])*rcp6
            # t3 = (I[1,:]*rcp4 + I[3,:])*rcp3

            # O[0,:] = I[0,:]*rcp4
            # O[1,:] = t0 - t2
            # O[2,:] = t0 + t2
            # O[3,:] = t1 + t3
            # O[4,:] = t1 - t3
            # O[5,:] = I[3,:]

        Fw[:] = T1.T

    else:
        Fw[:] = np.dot( np.dot(O_4x4_3x3[trans[0]].T, F), O_4x4_3x3[trans[1]] )

f400_441  = immediate(400.0 / 441.0)
f625_1078 = immediate(625.0 / 1078.0)
f25_198   = immediate(25.0 / 198.0)
f125_308  = immediate(125.0 / 308.0)
f25_132   = immediate(25.0 / 132.0)
f25_88    = immediate(25.0 / 88.0)

def trans_O_3x3_4x4(Mw, minimal=False, trans=False):
    if minimal:

        T0 = np.empty((3,6))
        T1 = np.empty((3,3))

        for O, I in ((T0, Mw), (T1, T0.T)):

            # np.array([
            # [ 1, 1,         1 ,       1,     1,     0 ],
            # [ 0, 0.625,    -0.625,    1.5,  -1.5,   0 ],
            # [ 0, 0.390625,  0.390625, 2.25,  2.25,  1 ]]).T,

            t0 = I[1,:] + I[2,:]
            t1 = I[3,:] + I[4,:]

            O[0,:] = I[0,:] + t0 + t1
            O[1,:] = 0.625 * (I[1,:] - I[2,:]) + 1.5 * (I[3,:] - I[4,:])
            O[2,:] = 0.390625 * t0 + 2.25 * t1 + I[5,:]

            # {400.0/441.00, -625.0/1078.0, -625.0/1078.0, 25.0/198.00, 25.0/198.00, 0.},
            # {0.,           -125.0/308.0,   125.0/308.0,  25.0/132.0, -25.0/132.0,  0.},
            # {0.,           -25.0/88.0,    -25.0/88.0,    25.0/88.0,   25.0/88.0,   1.}

            # {400/441, -25*25/98*11, -25*25/98*11, 1*25/18*11,  1*25/18*11,  0 },
            # {0,        -5*25/28*11,   5*25/28*11, 1*25/12*11, -1*25/12*11,  0 },
            # {0,        -1*25/ 8*11,  -1*25/ 8*11, 1*25/ 8*11,  1*25/ 8*11,  1 }

            # t0 = I[1,:] + I[2,:]
            # t1 = I[3,:] + I[4,:]

            # O[0,:] = I[0,:]*f400_441 - t0*f625_1078 + t1*f25_198
            # O[1,:] = f125_308 * (I[2,:] - I[1,:]) + f25_132 * (I[3,:] - I[4,:])
            # O[2,:] = (t1 - t0)*f25_88 + I[5,:]

            # t0 = rcp6 * I[1,:]
            # t1 = rcp6 * I[2,:]
            # t2 = rcp6 * (I[3,:] + I[4,:])
            # t3 = t0 + t1

            # O[0,:] = rcp4 * (I[0,:] + t2) - t3
            # O[1,:] = t1 - t0 + rcp12 * (I[3,:] - I[4,:])
            # O[2,:] = t2 - t3 + I[5,:]

            # t0 = -rcp6 * (I[1,:] + I[2,:])
            # t1 =  rcp6 * (I[3,:] + I[4,:])

            # O[0,:] = rcp4 * (I[0,:] + t1) + t0
            # O[1,:] = rcp6 * (I[2,:] - I[1,:]) + rcp12 * (I[3,:] - I[4,:])
            # O[2,:] = t0 + t1 + I[5,:]

            # t0 = I[1,:] + I[2,:]
            # t1 = I[3,:] + I[4,:]

            # O[0,:] = I[0,:] + t0 + t1
            # O[1,:] = I[1,:] - I[2,:] + 2*(I[3,:] - I[4,:])
            # O[2,:] = t0 + 4*t1 + I[5,:]

        return T1.T

    else:
        return np.dot( np.dot(F_4x4_3x3[trans[0]].T, Mw), F_4x4_3x3[trans[1]] )

def image_slice(x, X, B, D, pad=0):
    start = x * B - pad
    stop  = start + D
    pad = [0,0]
    if start < 0:
        pad[0] = -start
        start = 0
    if stop - 1 >= X:
        pad[1] = stop - X
    return start, stop, pad

def output_slice(p, P, B):
    p0 = p * B
    p1 = p0 + B
    if p1 > P:
        p1 = P
    return p0, p1, p1 - p0

def xprop_winograd(I, F, O, padding, minimal=False, trans=False, backward=False):

    if backward:
        # C <=> K and mirror R,S
        F = np.transpose(F[:,::-1,::-1,:], (3,1,2,0)).copy()
        # Invert padding
        padding = [2 - p for p in padding]

    C, Y, X, N = I.shape
    K, P, Q, N = O.shape

    B = 4
    D = B + 2
    Yw = ceil_div(P, B)
    Xw = ceil_div(Q, B)

    Fw = np.empty((D,D,C,K))
    Iw = np.empty((D,D,C,Yw,Xw,N))
    Mw = np.empty((D,D,K,Yw,Xw,N)) #, dtype=np.int64

    # Transform Filters
    for c in range(C):
        for k in range(K):
            trans_F_4x4_3x3(Fw[:,:,c,k], F[c,:,:,k], minimal, trans)

    # Iterate over image transform dimensions and slice out tiles of the image
    for y in range(Yw):
        start_y, stop_y, pad_y = image_slice(y, Y, B, D, padding[0])

        for x in range(Xw):
            start_x, stop_x, pad_x = image_slice(x, X, B, D, padding[1])

            sliceI = I[:, start_y:stop_y, start_x:stop_x, :]

            # add any zero padding if needed
            if any(pad_y) or any(pad_x):
                sliceI = np.pad(sliceI, ((0,0), pad_y, pad_x, (0,0)), 'constant')

            # Apply the Image transform
            for c in range(C):
                for n in range(N):
                    trans_I_4x4_3x3(Iw[:,:,c,y,x,n], sliceI[c,:,:,n], minimal, trans)

    #print Iw[:,:,0,0,0,0]
    # print Fw[:,:,0,0].reshape(36,1)
    #exit()

    # Fw, scaleF = quantize(Fw, 16)
    # Iw, scaleI = quantize(Iw, 16)

    Fw = Fw.astype(np.float16).astype(np.float64)
    Iw = Iw.astype(np.float16).astype(np.float64)

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
        p0, p1, plen = output_slice(y, P, B)
        for x in range(Xw):
            q0, q1, qlen = output_slice(x, Q, B)
            for k in range(K):
                for n in range(N):
                    # Toss out any points that don't fit
                    #print y, x, plen, qlen
                    Out = trans_O_4x4_3x3(Mw[:,:,k,y,x,n], minimal, trans)
                    #print Out
                    #print Out[0:plen,0:qlen]
                    O[k,p0:p1,q0:q1,n] = Out[0:plen,0:qlen]


def updat_winograd(I, E, U, padding, minimal=False, trans=False, inner=True):

    C, Y, X, N = I.shape
    K, P, Q, N = E.shape

    B = 4
    D = B + 2
    Yw = ceil_div(P, B)
    Xw = ceil_div(Q, B)

    Iw = np.empty((D,D,N,C))
    Ew = np.empty((D,D,N,K))
    if inner:
        Mw = np.empty((D,D,C,K))
        U.fill(0.0)
    else:
        Mw = np.zeros((D,D,C,K))

    for y in range(Yw):
        start_y, stop_y, pad_y = image_slice(y, Y, B, D, padding[0])
        start_p, stop_p, pad_p = image_slice(y, P, B, B)

        for x in range(Xw):
            start_x, stop_x, pad_x = image_slice(x, X, B, D, padding[1])
            start_q, stop_q, pad_q = image_slice(x, Q, B, B)

            sliceI = I[:, start_y:stop_y, start_x:stop_x, :]
            sliceE = E[:, start_p:stop_p, start_q:stop_q, :]

            if any(pad_y) or any(pad_x):
                sliceI = np.pad(sliceI, ((0,0), pad_y, pad_x, (0,0)), 'constant')

            if any(pad_p) or any(pad_q):
                sliceE = np.pad(sliceE, ((0,0), pad_p, pad_q, (0,0)), 'constant')

            for c in range(C):
                for n in range(N):
                    trans_I_4x4_3x3(Iw[:,:,n,c], sliceI[c,:,:,n], minimal, trans)

            for k in range(K):
                for n in range(N):
                    trans_F_3x3_4x4(Ew[:,:,n,k], sliceE[k,:,:,n], minimal, trans)

            # print Iw[:,:,0,0]
            # print Ew[:,:,0,0]
            # exit()

            Ew = Ew.astype(np.float16).astype(np.float64)
            Iw = Iw.astype(np.float16).astype(np.float64)

            for s in range(D):
                for t in range(D):
                    # [C,K] += [N,C].T . [N,K]
                    if inner:
                        Mw[s,t]  = np.dot( Iw[s,t].T, Ew[s,t] )
                    else:
                        Mw[s,t] += np.dot( Iw[s,t].T, Ew[s,t] )

            # Transform can be applied in inner or outer loop
            if inner:
                for c in range(C):
                    for k in range(K):
                        U[c,:,:,k] += trans_O_3x3_4x4(Mw[:,:,c,k], minimal, trans)

    # outer loop transform
    if not inner:
        for c in range(C):
            for k in range(K):
                U[c,:,:,k] = trans_O_3x3_4x4(Mw[:,:,c,k], minimal, trans)



### Test Code ###

np.set_printoptions(threshold=8192 * 4, linewidth=600, formatter={'float':lambda x: "%6.3f" % x})

minimal = 0
trans = (4,4)
ones = 0
N    = 32
C, K = 32, 32
Y, X = 6, 6
R, S = 3, 3     # Fixed winograd dim
strides = 1, 1  # Fixed winograd dim
padding = 0, 0  # 0-2

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


    # for p,q in np.ndindex((Y,X)):
    #     I[:,p,q,:] = np.identity(N)

    # for p,q in np.ndindex((P,Q)):
    #     for n in range(N):
    #         E[:,p,q,n] = list(range(K))

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

Ud = np.empty(dimF)
Uw = np.empty(dimF)


xprop_direct(I, F, Od, padding, strides)
xprop_winograd(I, F, Ow, padding, minimal=minimal, trans=trans)

xprop_direct(E, F, Bd, padding, strides, backward=True)
xprop_winograd(E, F, Bw, padding, minimal=minimal, trans=trans, backward=True)

updat_direct(I, E, Ud, padding, strides)
updat_winograd(I, E, Uw, padding, minimal=minimal, trans=trans)

difO = Od - Ow
difB = Bd - Bw
difU = Ud - Uw

neon_logger.display(abs(difO).max() / Od.max())
neon_logger.display(abs(difB).max() / Bd.max())
neon_logger.display(abs(difU).max() / Ud.max())

# print Bd[0,:,:,0]
# print Bw[0,:,:,0]

# print Ud[0,:,:,0]
# print Uw[0,:,:,0]
# print difU[0,:,:,0]

