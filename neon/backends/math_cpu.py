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
import sys
import numpy as np
from cffi import FFI
from neon import NervanaObject
from neon.transforms import Rectlinclip

ffi = FFI()
omp_threshold = 100000


def isccontiguous(x):
    return x.flags['C_CONTIGUOUS'] is True


def isfcontiguous(x):
    return x.flags['F_CONTIGUOUS'] is True


def iscontiguous(x):
    return isccontiguous(x) or isfcontiguous(x)


def isscalar(x):
    return isinstance(x, np.float) or isinstance(x, np.int) \
           or isinstance(x, np.float32)


def issame_buffer(x1, x2):
    return x1.__array_interface__['data'][0] == x2.__array_interface__['data'][0]


def isfloat_tensor(x):
    return isinstance(x, np.ndarray) and x.dtype == np.float32


def istensor1d(x):
    return (x.ndim == 1) or (x.ndim == 2 and x.shape[1] == 1)


def istensor2d(x):
    return x.ndim == 2 and x.shape[1] != 1


def blas_copy(x1, x2):
    if issame_buffer(x1, x2):
        x1[:] = x2
    elif isfloat_tensor(x1) and isfloat_tensor(x2) \
            and isccontiguous(x1) and isccontiguous(x2) and x1.shape == x2.shape \
            and x1.size > omp_threshold:
        _x1 = ffi.cast("float *", ffi.from_buffer(x1))
        _x2 = ffi.cast("float *", ffi.from_buffer(x2))
        NervanaObject.be.mathlib.cmath_copy(_x2, x1.size, _x1)
    else:
        x1[:] = x2


def neg(x):
    if isfloat_tensor(x) and isccontiguous(x) \
            and x.size > omp_threshold:
        y = np.empty_like(x)
        _x = ffi.cast("float *", ffi.from_buffer(x))
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_neg(_x, x.size, _y)
        return y
    else:
        return -x


def sqrt(x):
    if isfloat_tensor(x) and isccontiguous(x) \
            and x.size > omp_threshold:
        y = np.empty_like(x)
        _x = ffi.cast("float *", ffi.from_buffer(x))
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_sqrt(_x, x.size, _y)
        return y
    else:
        return np.sqrt(x)


def square(x):
    if isfloat_tensor(x) and isccontiguous(x) \
            and x.size > omp_threshold:
        y = np.empty_like(x)
        _x = ffi.cast("float *", ffi.from_buffer(x))
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_square(_x, x.size, _y)
        return y
    else:
        return np.square(x)


def exp(x):
    if isfloat_tensor(x) and isccontiguous(x) \
            and x.size > omp_threshold:
        y = np.empty_like(x)
        _x = ffi.cast("float *", ffi.from_buffer(x))
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_exp(_x, x.size, _y)
        return y
    else:
        return np.exp(x)


def log(x):
    if isfloat_tensor(x) and isccontiguous(x) \
            and x.size > omp_threshold:
        y = np.empty_like(x)
        _x = ffi.cast("float *", ffi.from_buffer(x))
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_log(_x, x.size, _y)
        return y
    else:
        return np.log(x)


def safelog(x):
    if isfloat_tensor(x) and isccontiguous(x) \
            and x.size > omp_threshold:
        y = np.empty_like(x)
        _x = ffi.cast("float *", ffi.from_buffer(x))
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_safelog(_x, x.size, _y)
        return y
    else:
        return np.log(np.maximum(x, np.exp(-50.)))


def add(x1, x2):
    if isfloat_tensor(x1) and isfloat_tensor(x2) \
            and isccontiguous(x1) and isccontiguous(x2):
        if x1.shape == x2.shape and x1.size > omp_threshold:
            y = np.empty_like(x1)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_add(_x1, _x2, x1.size, _y)
            return y
        elif istensor2d(x1) and istensor1d(x2) \
                and x1.shape[0] == x2.shape[0] and x1.size > omp_threshold:
            y = np.empty_like(x1)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_addmv(_x1, _x2, x1.shape[0], x1.shape[1], _y)
            return y
        elif istensor1d(x1) and istensor2d(x2) \
                and x1.shape[0] == x2.shape[0] and x2.size > omp_threshold:
            y = np.empty_like(x2)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_addmv(_x2, _x1, x2.shape[0], x2.shape[1], _y)
            return y
        else:
            return np.add(x1, x2)
    else:
        return np.add(x1, x2)


def sub(x1, x2):
    if isfloat_tensor(x1) and isfloat_tensor(x2) \
            and isccontiguous(x1) and isccontiguous(x2):
        if x1.shape == x2.shape and x1.size > omp_threshold:
            y = np.empty_like(x1)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_sub(_x1, _x2, x1.size, _y)
            return y
        elif istensor2d(x1) and istensor1d(x2) \
                and x1.shape[0] and x2.shape[0] and x1.size > omp_threshold:
            y = np.empty_like(x1)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_submv(_x1, _x2, x1.shape[0], x1.shape[1], _y)
            return y
        else:
            return np.subtract(x1, x2)
    else:
        return np.subtract(x1, x2)


def mul(x1, x2):
    if isfloat_tensor(x1) and isscalar(x2) \
            and isccontiguous(x1) and x1.size > omp_threshold:
        y = np.empty_like(x1)
        _x1 = ffi.cast("float *", ffi.from_buffer(x1))
        _x2 = ffi.cast("double", x2)
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_mul(_x1, x1.size, _x2, _y)
        return y
    elif isscalar(x1) and isfloat_tensor(x2) \
            and isccontiguous(x2) and x2.size > omp_threshold:
        y = np.empty_like(x2)
        _x1 = ffi.cast("double", x1)
        _x2 = ffi.cast("float *", ffi.from_buffer(x2))
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_mul(_x2, x2.size, _x1, _y)
        return y
    elif isfloat_tensor(x1) and isfloat_tensor(x2)\
            and isccontiguous(x1) and isccontiguous(x2):
        if x1.shape == x2.shape and x1.size > omp_threshold:
            y = np.empty_like(x1)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_mulmm(_x1, _x2, x1.size, _y)
            return y
        elif istensor2d(x1) and istensor1d(x2)\
                and x1.shape[0] == x2.shape[0] and x1.size > omp_threshold:
            y = np.empty_like(x1)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_mulmv(_x1, _x2, x1.shape[0], x1.shape[1], _y)
            return y
        elif istensor1d(x1) and istensor2d(x2)\
                and x1.shape[0] == x2.shape[0] and x2.size > omp_threshold:
            y = np.empty_like(x2)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_mulmv(_x2, _x1, x2.shape[0], x2.shape[1], _y)
            return y
        else:
            return np.multiply(x1, x2)
    else:
        return np.multiply(x1, x2)


def div(x1, x2):
    if isfloat_tensor(x1) and isscalar(x2) \
            and isccontiguous(x1) and x1.size > omp_threshold:
        y = np.empty_like(x1)
        _x1 = ffi.cast("float *", ffi.from_buffer(x1))
        _x2 = ffi.cast("double", x2)
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_div(_x1, x1.size, _x2, _y)
        return y
    elif isfloat_tensor(x1) and isfloat_tensor(x2)\
            and isccontiguous(x1) and isccontiguous(x2):
        if x1.shape == x2.shape and x1.size > omp_threshold:
            y = np.empty_like(x1)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_divmm(_x1, _x2, x1.size, _y)
            return y
        elif istensor2d(x1) and istensor1d(x2)\
                and x1.shape[0] == x2.shape[0] and x1.size > omp_threshold:
            y = np.empty_like(x1)
            _x1 = ffi.cast("float *", ffi.from_buffer(x1))
            _x2 = ffi.cast("float *", ffi.from_buffer(x2))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_divmv(_x1, _x2, x1.shape[0], x1.shape[1], _y)
            return y
        else:
            return np.divide(x1, x2)
    else:
        return np.divide(x1, x2)


def _force_corder(x):
    if x.flags.f_contiguous:
        return (x.T, 't')
    else:
        return (x, 'n')


def blas_dot(a, b, c=None, alpha=1.0, beta=0):
    # convert non-contiguous tensor into contiguous to speed up
    if not iscontiguous(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
    if not iscontiguous(b):
        b = np.ascontiguousarray(b, dtype=np.float32)
    if (iscontiguous(a) and iscontiguous(b) and c is None) \
            or (iscontiguous(a) and iscontiguous(b) and iscontiguous(c)) \
            and sys.version_info < (3, 4):
        m = a.shape[0]
        n = b.shape[1]
        k = a.shape[1]
        a_corder, transa = _force_corder(a)
        b_corder, transb = _force_corder(b)

        if c is None:
            c = np.zeros((m, n), dtype=np.float32)

        lda = k if transa is 'n' else m
        ldb = n if transb is 'n' else k
        ldc = n

        # TODO: cffi cast not compatible with f_contiguous array with py3.4
        _a = ffi.cast("float *", ffi.from_buffer(a))
        _b = ffi.cast("float *", ffi.from_buffer(b))
        _c = ffi.cast("float *", ffi.from_buffer(c))
        _transa = ffi.cast("char", transa)
        _transb = ffi.cast("char", transb)
        NervanaObject.be.mathlib.cmath_gemm(
            _transa, _transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)
    else:
        if beta == 0:
            np.dot(a, b, c)
            if alpha != 1:
                np.multiply(c, alpha, c)
            return c
        if beta != 1:
            np.multiply(c, beta, c)
        tmp = np.empty(c.shape, dtype=c.dtype)
        np.dot(a, b, tmp)
        if alpha != 1:
            np.multiply(tmp, alpha, tmp)
        np.add(c, tmp, c)
    return c


def sum(x, axis=None, keepdims=True):
    if isfloat_tensor(x) and isccontiguous(x) and istensor2d(x) \
            and x.size > omp_threshold and axis is not None and keepdims is True:
        if axis == 1:
            y = np.zeros((x.shape[0], 1), dtype=x.dtype)
            _x = ffi.cast("float *", ffi.from_buffer(x))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_sum(_x, 1, x.shape[0], x.shape[1], _y)
            return y
        elif axis == 0:
            y = np.zeros((1, x.shape[1]), dtype=x.dtype)
            _x = ffi.cast("float *", ffi.from_buffer(x))
            _y = ffi.cast("float *", ffi.from_buffer(y))
            NervanaObject.be.mathlib.cmath_sum(_x, 0, x.shape[0], x.shape[1], _y)
            return y
        else:
            raise AttributeError('axis of a 2d tensor should only be 0 or 1')
    else:
        return np.sum(x, axis=axis, keepdims=keepdims)


def blas_axpby(a, x, b, y):
    if isfloat_tensor(x) and isfloat_tensor(y) and isccontiguous(x) \
       and isccontiguous(y) and x.shape == y.shape:
        _x = ffi.cast("float *", ffi.from_buffer(x))
        _y = ffi.cast("float *", ffi.from_buffer(y))
        NervanaObject.be.mathlib.cmath_axpby(x.size, a, _x, b, _y)
    else:
        y = a*x + b*y


def change_data_store_order(a, a_row, a_col, a_len, b, axis=1):
    if iscontiguous(a._tensor):
        _a = ffi.cast("float *", ffi.from_buffer(a._tensor))
        _b = ffi.cast("float *", ffi.from_buffer(b._tensor))
        NervanaObject.be.mathlib.cmath_change_data_store_order(_a, axis, a_row, a_col, a_len, _b)

        return b
    else:
        raise AttributeError('a should be contiguous')


def add_and_act(h, h_ff, bias, activation):
    if iscontiguous(h) and iscontiguous(h_ff) and iscontiguous(bias):
        if isinstance(activation, Rectlinclip) and (activation.slope == 0):
            _h = ffi.cast("float *", ffi.from_buffer(h))
            _h_ff = ffi.cast("float *", ffi.from_buffer(h_ff))
            _bias = ffi.cast("float *", ffi.from_buffer(bias))
            NervanaObject.be.mathlib.cmath_add_and_act(_h, _h_ff, _bias, h.shape[0], h.shape[1],
                                                       activation.xcut)
            return True
        else:
            return False
    else:
        return False


def trans2d(W_recur_f, W_recur_b, W_recur_f_contiguous, W_recur_b_contiguous):
    if iscontiguous(W_recur_f._tensor) and iscontiguous(W_recur_b._tensor) \
       and W_recur_f.shape == W_recur_b.shape:
        _in_0 = ffi.cast("float *", ffi.from_buffer(W_recur_f._tensor))
        _in_1 = ffi.cast("float *", ffi.from_buffer(W_recur_b._tensor))
        _out_0 = ffi.cast("float *", ffi.from_buffer(W_recur_f_contiguous._tensor))
        _out_1 = ffi.cast("float *", ffi.from_buffer(W_recur_b_contiguous._tensor))
        NervanaObject.be.mathlib.cmath_trans2d(_in_0, _in_1, W_recur_f.shape[0],
                                               W_recur_f.shape[1], _out_0, _out_1)


def act_and_mul(in_deltas, hs, activation):
    if iscontiguous(in_deltas._tensor) and iscontiguous(hs._tensor):
        if isinstance(activation, Rectlinclip) and (activation.slope == 0):
            _in_deltas = ffi.cast("float *", ffi.from_buffer(in_deltas._tensor))
            _hs = ffi.cast("float *", ffi.from_buffer(hs._tensor))
            NervanaObject.be.mathlib.cmath_act_and_mul(_in_deltas, _hs, in_deltas.size,
                                                       activation.xcut)
            return True
        else:
            return False
    else:
        return False
