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
MKL based backend interface and tensor data structure.
"""
from __future__ import division
from builtins import zip
import logging

from neon.backends import layer_mkl
from neon.backends.util.check_mkl import get_mkl_lib
from neon.backends.nervanacpu import CPUTensor, NervanaCPU, CustomNumpy
import ctypes
from cffi import FFI
from ctypes import c_longlong, c_float, c_double, c_int
import numpy as np
from neon.backends import math_cpu
from neon.backends.backend import OpTreeNode
import os
import sys

_none_slice = slice(None, None, None)

logger = logging.getLogger(__name__)


# TODO: enable this flag to find numerical problems
# np.seterr(all='raise')


class MKLTensor(CPUTensor):
    """
    MKLTensor, special for MKL Backend
    """
    _tensor = None

    def __init__(self,
                 backend,
                 shape=None,
                 dtype=np.float32,
                 ary=None,
                 name=None,
                 persist_values=True,
                 base=None):

        super(MKLTensor, self).__init__(backend, shape, dtype, ary, name,
                                        persist_values, base)

        # add 4 address for cpu layout and buffer, mkl layout and buffer
        self.primitive = np.zeros((4), dtype=np.uint64)
        if ary is not None:
            self.primitive[1] = self._tensor.ctypes.data
        self.shape5D = None

    def get_prim(self):
        return c_longlong(self.primitive.ctypes.data)

    def clean_mkl(self):
        self.primitive[2] = 0
        self.primitive[3] = 0

    def set_mkl(self, tensor):
        self.primitive[0] = tensor.primitive[0]
        self.primitive[2] = tensor.primitive[2]
        self.primitive[3] = tensor.primitive[3]
        self.shape5D = tensor.shape5D

    def __str__(self):
        """
        Returns a string representation of this Tensor.

        Returns:
            str: the representation.
        """
        if self._tensor.base is not None:
            base_id = id(self._tensor.base)
        else:
            base_id = id(self._tensor)
        return ("MKLTensor(base 0x%x) name:%s shape:%s dtype:%s strides:%s"
                " is_c_contiguous:%s" % (base_id, self.name, self.shape,
                                         self.dtype, self._tensor.strides,
                                         self._tensor.flags.c_contiguous))

    def get(self):
        """
        Return the array.
        """
        self.backend.convert(self)
        return self._tensor.copy()

    def reshape(self, *shape):
        """
        Return a reshaped view.
        """
        newTensor = super(MKLTensor, self).reshape(*shape)
        newTensor.set_mkl(self)
        return newTensor


def _assign_right_to_left(left, right):
    math_cpu.blas_copy(left, right)


# how to overlaod numpy_call_dict?
numpy_call_dict_mkl = {
    # assign
    "assign": _assign_right_to_left,
    # zero_operand ops
    # unary ops
    "neg": lambda left: math_cpu.neg(left),
    "abs": lambda left: np.abs(left),
    "sgn": lambda left: np.sign(left),
    "sqrt": lambda left: math_cpu.sqrt(left),
    "sqr": lambda left: math_cpu.square(left),
    "exp": lambda left: math_cpu.exp(left),
    "log": lambda left: math_cpu.log(left),
    "safelog": lambda left: math_cpu.safelog(left),
    "exp2": lambda left: np.exp2(left),
    "log2": lambda left: np.log2(left),
    "sig": lambda left: 1. / (1. + np.exp(-left)),
    "sig2": lambda left: 1. / (1. + np.exp2(-left)),
    "tanh": lambda left: np.tanh(left),
    "tanh2": lambda left: (np.exp2(2. * left) - 1.) / (np.exp2(2. * left) + 1.),
    "transpose": lambda left: np.transpose(left),
    "rint": lambda left: np.rint(left),
    # binary ops
    "add": lambda left, right: math_cpu.add(left, right),
    "sub": lambda left, right: math_cpu.sub(left, right),
    "mul": lambda left, right: math_cpu.mul(left, right),
    "div": lambda left, right: math_cpu.div(left, right),
    "eq": lambda left, right: left == right,
    "ne": lambda left, right: left != right,
    "lt": lambda left, right: left < right,
    "le": lambda left, right: left <= right,
    "gt": lambda left, right: left > right,
    "ge": lambda left, right: left >= right,
    "pow": lambda left, right: np.power(left, right),
    "minimum": lambda left, right: np.minimum(left, right),
    "maximum": lambda left, right: np.maximum(left, right),
    "dot": lambda left, right: np.dot(left, right),
    # reduction ops
    "sum": lambda op_dict, left: math_cpu.sum(left, axis=op_dict['axis'], keepdims=True),
    "max": lambda op_dict, left: np.max(left, axis=op_dict['axis'], keepdims=True),
    "min": lambda op_dict, left: np.min(left, axis=op_dict['axis'], keepdims=True),
    "argmax": lambda op_dict, left: CustomNumpy.argmax(left, axis=op_dict['axis'], keepdims=True),
    "argmin": lambda op_dict, left: CustomNumpy.argmin(left, axis=op_dict['axis'], keepdims=True),
}


class NervanaMKL(NervanaCPU):
    """
    MKL Backend
    """
    backend_name = 'mkl'

    def __init__(self,
                 rng_seed=None,
                 default_dtype=np.float32,
                 hist_bins=64,
                 hist_offset=-48,
                 compat_mode=None,
                 # Ignored
                 num_devices=None,
                 stochastic_round=None,
                 device_id=None,
                 deterministic=None
                 ):
        super(NervanaMKL, self).__init__(rng_seed, default_dtype,
                                         hist_bins, hist_offset, compat_mode=compat_mode)
        self.tensor_cls = MKLTensor
        logger.info("Initialized NervanaMKL")
        assert get_mkl_lib(), "MKL is not installed correctly"

        path = os.path.dirname(os.path.realpath(__file__))
        header_path = os.path.join(os.path.dirname(__file__), 'mklEngine', 'src',
                                   'math_cpu.header')

        if sys.platform == 'win32':
            mkl_ml_path = os.path.join(path, os.pardir, 'backends', 'mklEngine', 'mklml.dll')
            ctypes.windll.LoadLibrary(mkl_ml_path)
            mkl_engine_path = os.path.join(path, os.pardir, 'backends', 'mklEngine',
                                           'mklEngine.dll')
            self.mklEngine = ctypes.windll.LoadLibrary(mkl_engine_path)
            math_engine_path = os.path.join(os.path.dirname(__file__), 'mklEngine', 'cmath.dll')
        else:
            mkl_engine_path = os.path.join(path, os.pardir, 'backends', 'mklEngine',
                                           'mklEngine.so')
            self.mklEngine = ctypes.cdll.LoadLibrary(mkl_engine_path)
            math_engine_path = os.path.join(os.path.dirname(__file__), 'mklEngine', 'cmath.so')

        ffi = FFI()
        with open(header_path) as header:
            ffi.cdef(header.read())
        self.mathlib = ffi.dlopen(math_engine_path)

    def is_mkl(self):
        return True

    def convert(self, a):
        if a.primitive[2] == 0:
            return a
        C, D, H, W, N = a.shape5D
        self.mklEngine.ConvertBack(a.get_prim(), N, C, H, W)
        return a

    def convert_mkl(self, a):
        self.mklEngine.ConvertToMKL(a.get_prim())
        return a

    def convert_data(self, tensor, layer_mkl):
        if not layer_mkl and tensor is not None:
            if type(tensor) == MKLTensor:
                self.convert(tensor)
                tensor.clean_mkl()
            elif type(tensor) is list or type(tensor) is tuple:
                for i in tensor:
                    self.convert_data(i, layer_mkl)
            elif type(tensor) is OpTreeNode or type(tensor) is np.ndarray:
                return
            else:
                assert False, 'unsupported input for convert ' + str(type(tensor))

    def clean_data(self, tensor, layer_mkl):
        if layer_mkl and tensor is not None and type(tensor) == MKLTensor:
            tensor.clean_mkl()

    def get_numpy(self, a):
        # transfer mkl tensor into a new numpy
        numpy_a = a._tensor.copy()
        b = a.primitive[1]
        a.primitive[1] = numpy_a.ctypes.data
        self.convert(a)
        a.primitive[1] = b
        return numpy_a

    def execute(self, optree):
        return super(NervanaMKL, self).execute(optree, numpy_call_dict=numpy_call_dict_mkl)

    def copy_transpose(self, a, out, axes=None, repeat=1):
        """
        use MKL transposition to speed up
        """
        if axes is None and a._tensor.ctypes.data != out._tensor.ctypes.data and len(a.shape) == 2:
            inp = c_longlong(a._tensor.ctypes.data)
            outp = c_longlong(out._tensor.ctypes.data)
            m, n = a.shape
            self.mklEngine.MatTrans(inp, outp, c_longlong(m), c_longlong(n))
        else:
            out._tensor[:] = np.transpose(a._tensor, axes).copy()

    def conv_layer(self, dtype,
                   N, C, K,
                   D=1, H=1, W=1,
                   T=1, R=1, S=1,
                   pad_d=0, pad_h=0, pad_w=0,
                   str_d=1, str_h=1, str_w=1,
                   dil_d=1, dil_h=1, dil_w=1):
        """
        Create a new ConvLayer parameter object.
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

        dtype: need to know dtype to setup proper kernels and params.

        bsum: calculate the sum along the batchnorm axis for fprop or bprop
              outputs an fp32 tensor of size Kx1

        """
        return layer_mkl.ConvLayerMKL(
            self, dtype, N, C, K, D, H, W, T, R, S,
            pad_d, pad_h, pad_w, str_d, str_h, str_w,
            dil_d, dil_h, dil_w)

    def deconv_layer(self, dtype,
                     N, C, K,
                     M, P, Q,
                     T=1, R=1, S=1,
                     pad_d=0, pad_h=0, pad_w=0,
                     str_d=1, str_h=1, str_w=1,
                     dil_d=1, dil_h=1, dil_w=1):
        """
        Create a new DeconvLayer parameter object.
        This then is passed as an argument to all the convolution operations.

        N: Number of images in mini-batch
        C: Number of output feature maps
        K: Number of input feature maps

        M: Depth  of input
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

        dtype: need to know dtype to setup proper kernels and params.
        """
        return layer_mkl.DeconvLayerMKL(self, dtype, N, C, K, M, P, Q, T, R, S,
                                        pad_d, pad_h, pad_w, str_d, str_h, str_w,
                                        dil_d, dil_h, dil_w)

    def pool_layer(self, dtype,
                   op, N, C,
                   D=1, H=1, W=1,
                   J=1, T=1, R=1, S=1,
                   pad_c=0, pad_d=0, pad_h=0, pad_w=0,
                   str_c=None, str_d=None, str_h=None, str_w=None):
        """
        Create a new PoolLayer parameter object.
        This then is passed as an argument to all pooling kernels.

        op: "max", "avg", "l2" pooling (currently bprop only supports max, but not avg and l2)
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
        # default to non-overlapping
        if str_c is None:
            str_c = J
        if str_d is None:
            str_d = T
        if str_h is None:
            str_h = R
        if str_w is None:
            str_w = S

        return layer_mkl.PoolLayerMKL(self, dtype, op, N, C, D, H, W, J, T, R, S,
                                      pad_c, pad_d, pad_h, pad_w, str_c, str_d, str_h, str_w)

    def fprop_pool(self, layer, I, O, argmax=None, beta=0.0):
        """
        Forward propagate pooling layer.

        Arguments:
            layer (PoolLayer): The pool layer object, different backends have
                               different pool layers.
            I (Tensor): Input tensor.
            O (Tensor): output tensor.
            argmax (Tensor): tensor to store location of the maximum
        """
        assert layer.sizeI == I.size
        assert layer.sizeO == O.size
        assert layer.op == "max" or layer.op == 'avg'
        if layer.op == "max":
            assert layer.sizeO == argmax.size

        J, T, R, S = layer.JTRS
        C, D, H, W, N = layer.dimI
        K, M, P, Q, N = layer.dimO
        pad_c, pad_d, pad_h, pad_w = layer.padding
        str_c, str_d, str_h, str_w = layer.strides

        # unsupported fall back to cpu
        if J > 1 or T > 1 or D > 1:
            super(NervanaMKL, self).fprop_pool(layer, I, O, argmax, beta)
            return

        if layer.op == "max":
            bMax = 1
        elif layer.op == 'avg':
            bMax = 0
        if self.check_caffe_compat():
            bCeil = 1
        else:
            bCeil = 0

        primitives = c_longlong(layer.dnnPrimitives.ctypes.data)
        self.mklEngine.MaxPooling_fprop(
            I.get_prim(), O.get_prim(), primitives, layer.initOk_f, bMax,
            N, C, H, W, R, S, str_h, str_w, pad_h, pad_w, K, P, Q, bCeil)
        layer.initOk_f = 1
        O.shape5D = layer.dimO

    def bprop_pool(self, layer, I, O, argmax=None, alpha=1.0, beta=0.0):
        """
        Backward propagate pooling layer.

        Arguments:
            layer (PoolLayer): The pool layer object. Different backends have
                               different pool layers.
            I (Tensor): Input (error) tensor.
            O (Tensor): Output (delta) tensor.
            argmax (Tensor): tensor to store location of the maximum
            alpha (float): linear scaling (does not work for l2 pooling)
            beta (float): accumulation value into grad_I
        """
        assert layer.sizeI == O.size
        assert layer.sizeO == I.size
        if layer.op == "max":
            assert layer.sizeO == argmax.size

        J, T, R, S = layer.JTRS
        C, D, H, W, N = layer.dimI
        K, M, P, Q, N = layer.dimO
        pad_c, pad_d, pad_h, pad_w = layer.padding
        str_c, str_d, str_h, str_w = layer.strides

        # unsupported fall back to cpu
        if J > 1 or T > 1 or D > 1:
            super(NervanaMKL, self).bprop_pool(layer, I, O, argmax, alpha, beta)
            return

        primitives = c_longlong(layer.dnnPrimitives.ctypes.data)
        self.mklEngine.MaxPooling_bprop(I.get_prim(), O.get_prim(),
                                        primitives, layer.initOk_b)
        layer.initOk_b = 1
        O.shape5D = layer.dimI

    def compound_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, bsum=None):
        """
        Doing following operations (* is dot product)
        C = alpha * A * B   + beta * C
        C = alpha * A.T * B + beta * C
        C = alpha * A * B.T + beta * C.

        relu: if true applied before output (and prior to beta addition)

        The operation will be short-circuited to: out <- alpha * left * right
        if beta has value 0 (the default).

        Arguments:
            A, B (CPUTensor): input operands
            C (MCPUTensor): output
            alpha (float): scale A*B term
            beta (float): scale C term before sum
            relu (bool): whether to apply ReLu before output
        """

        # checking type and shape
        assert A.dtype == B.dtype == C.dtype
        assert A.shape[0] == C.shape[0]
        assert B.shape[1] == C.shape[1]
        assert A.shape[1] == B.shape[0]

        # cleaner implementation, shall be equivalent to the one below
        # if relu:
        #     C[:] = self.log(1. + self.exp(alpha * self.dot(A, B))) + beta * C
        # else:
        #     C[:] = alpha * self.dot(A, B) + beta * C

        if not relu:
            if C._tensor.flags['C_CONTIGUOUS'] is not True:
                tmp = np.empty(C.shape, dtype=C.dtype)
                if beta != 0:
                    tmp[:] = C._tensor
                math_cpu.blas_dot(A._tensor, B._tensor, tmp, alpha, beta)
                C._tensor[:] = tmp
            else:
                math_cpu.blas_dot(A._tensor, B._tensor, C._tensor, alpha, beta)
        else:
            # mfma: change np.multiply to mul
            if beta != 1:
                np.multiply(C._tensor, beta, C._tensor)
            tmp = np.empty(C.shape, dtype=C.dtype)
            np.dot(A._tensor, B._tensor, tmp)
            # mfma: change np.multiply to mul
            if alpha != 1:
                np.multiply(tmp, alpha, tmp)
            if relu:
                self.Relu(tmp, tmp)
            np.add(C._tensor, tmp, C._tensor)
        if bsum is not None:
            bsum[:] = self.sum(C, 1)

        return C

    def relu_layer(self):
        return layer_mkl.ReluLayerMKL()

    def fprop_relu(self, layer, x, slope):
        if layer is None:
            layer = layer_mkl.ReluLayerMKL()
        if not hasattr(x, 'shape5D'):
            return self.maximum(x, 0) + slope * self.minimum(0, x)
        if slope != 0:
            self.convert(x)
            x.clean_mkl()
            return self.maximum(x, 0) + slope * self.minimum(0, x)

        if x.shape5D is not None:
            C, D, H, W, N = x.shape5D
        else:
            C, N = x._tensor.shape
            D, H, W = 1, 1, 1
            x.shape5D = C, D, H, W, N
        layer.shape5D = C, D, H, W, N

        primitives = c_longlong(layer.dnnPrimitives.ctypes.data)
        if x.primitive[3] == 0:
            layer.inputMKL = False
        self.mklEngine.Relu_f(x.get_prim(), primitives, layer.initOk_f, N, C, H, W)
        layer.initOk_f = 1
        return x

    def bprop_relu(self, layer, x, error, deltas, slope):
        if layer is None:
            layer = layer_mkl.ReluLayerMKL()
        if slope != 0 or error is None:
            if error is not None:
                self.convert(error)
                error.clean_mkl()
            return self.greater(x, 0) + slope * self.less(x, 0)

        # to be moved to C code
        if not layer.inputMKL:
            self.convert(error)
            error.clean_mkl()
        primitives = c_longlong(layer.dnnPrimitives.ctypes.data)
        self.mklEngine.Relu_b(x.get_prim(), error.get_prim(), primitives, layer.initOk_b)
        layer.initOk_b = 1
        deltas.set_mkl(error)
        deltas.shape5D = layer.shape5D
        if deltas.primitive[3] == 0:
            deltas[:] = error

    def fprop_transform(self, nglayer, transform, inputs, outputs, relu=False):
        if relu:
            transform(inputs, nglayer)
        else:
            return super(NervanaMKL, self).fprop_transform(
                nglayer, transform, inputs, outputs, relu)

    def bprop_transform(self, nglayer, transform, outputs, error, deltas, relu):
        if relu:
            transform.bprop(outputs, nglayer, error, deltas)
        else:
            super(NervanaMKL, self).bprop_transform(
                nglayer, transform, outputs, error, deltas, relu)

    def compound_fprop_bn(self, x, xsum, xvar, gmean, gvar, gamma, beta, y,
                          eps, rho, compute_batch_sum,
                          accumbeta=0.0, relu=False, binary=False,
                          inference=False, outputs=None, layer=None):
        if layer is None or outputs is None or not isinstance(layer.in_shape, tuple):
            super(NervanaMKL, self).compound_fprop_bn(x, xsum, xvar, gmean, gvar,
                                                      gamma, beta, y, eps, rho,
                                                      compute_batch_sum,
                                                      accumbeta, relu, binary,
                                                      inference, outputs, layer
                                                      )
            return

        primitives = c_longlong(layer.dnnPrimitives.ctypes.data)
        if len(layer.in_shape) == 3:
            C = layer.in_shape[0]
            H = layer.in_shape[1]
            W = layer.in_shape[2]
            D = 1
        elif len(layer.in_shape) == 2:
            C = layer.in_shape[0]
            H = layer.in_shape[1]
            W = layer.in_shape[1]
            D = 1
        elif len(layer.in_shape) == 4:
            C = layer.in_shape[0]
            H = layer.in_shape[1]
            W = layer.in_shape[2]
            D = layer.in_shape[3]

        N = int(x.shape[-1]) // H // W // D  # this is/corresponds to the batch size
        gmean = c_longlong(gmean._tensor.ctypes.data)
        gvar = c_longlong(gvar._tensor.ctypes.data)
        self.mklEngine.BatchNormFprop(x.get_prim(), outputs.get_prim(),
                                      gamma.get_prim(), beta.get_prim(), gmean, gvar,
                                      c_float(rho), N, C, H, W * D, c_double(eps),
                                      primitives, layer.init_f, c_int(inference))
        layer.init_f = 1
        layer.shape5D = outputs.shape5D = C, D, H, W, N
        if inference:
            self.convert(outputs)
            y[:] = y * gamma + beta
            self.convert_mkl(outputs)

    def batchnorm_layer(self, in_shape):
        return layer_mkl.BatchNormLayerMKL(in_shape)

    def compound_bprop_bn(self, deltas, grad_gamma, grad_beta,
                          error, inputs, xsum, xvar, gamma,
                          eps, binary=False, layer=None):
        if not layer or not isinstance(layer.in_shape, tuple):
            super(NervanaMKL, self).compound_bprop_bn(deltas, grad_gamma,
                                                      grad_beta, error,
                                                      inputs, xsum, xvar,
                                                      gamma, eps, binary,
                                                      layer)
            return

        primitives = c_longlong(layer.dnnPrimitives.ctypes.data)

        self.mklEngine.BatchNormBackp(inputs.get_prim(), error.get_prim(),
                                      deltas.get_prim(), grad_gamma.get_prim(),
                                      grad_beta.get_prim(), layer.in_shape[0],
                                      primitives, layer.init_b)
        layer.init_b = 1
        deltas.shape5D = layer.shape5D

    def fprop_skipnode(self, x, y, beta):
        y.set_mkl(x)
        # skipnode will do nothing for mkltensor but copy for CPU tensor
        if not x.primitive[2]:
            super(NervanaMKL, self).fprop_skipnode(x, y, beta)

    def bprop_skipnode(self, error, delta, alpha, beta):
        delta.set_mkl(error)
        if not error.primitive[2]:
            super(NervanaMKL, self).bprop_skipnode(error, delta, alpha, beta)

    def mergesum_layer(self, layer_num):
        return layer_mkl.MergeSumLayerMKL(layer_num)

    def sum_tensor(self, sum, layer_num, tensors, output):
        inp = c_longlong(tensors.ctypes.data)
        size = c_longlong(np.prod(output.shape))
        prim = c_longlong(sum.ctypes.data)
        self.mklEngine.MklSumTensor(layer_num, inp, size, output.get_prim(), prim)

    def fprop_mergesum(self, ngLayer, inputs, inference, layers, outputs, out_shape):
        ngLayer.shape5D = inputs.shape5D
        for i, l in enumerate(layers):
            l.fprop(inputs, inference)
            alloc_layers = [ll for ll in l.layers if ll.owns_output]
            ngLayer.tensors[(i * 4):(i * 4 + 4)] = alloc_layers[-1].outputs.primitive[0:4]

        C, H, W = out_shape
        outputs.shape5D = (C, 1, H, W, outputs.shape[-1])
        self.sum_tensor(ngLayer.sum_prim_f, ngLayer.layer_num, ngLayer.tensors, outputs)

    def bprop_mergesum(self, ngLayer, alpha, beta, layers, error, deltas):
        for i, l in enumerate(reversed(layers)):
            e = l.bprop(error)
            ngLayer.tensors[(i * 4):(i * 4 + 4)] = e.primitive[0:4]

        self.sum_tensor(ngLayer.sum_prim_b, ngLayer.layer_num, ngLayer.tensors, deltas)
        deltas.shape5D = ngLayer.shape5D

    def mergebroadcast_layer(self, layer_num):
        return layer_mkl.MergeBroadcastLayerMKL(layer_num)

    def fprop_mergebroadcast(self, ngLayer, inputs, inference, outputs, layers, out_shape):
        for l in layers:
            l.fprop(inputs, inference)

        if ngLayer.initOK_f == 0:
            C, H, W = layers[0].out_shape
            N = outputs.shape[-1]
            C = out_shape[0]
            ngLayer.in_shape5D = inputs.shape5D
            ngLayer.out_shape5D = N, C, H, W
            for i, layer in enumerate(layers):
                ngLayer.channels[i] = layer.out_shape[0]

        N, C, H, W = ngLayer.out_shape5D
        outputs.shape5D = C, 1, H, W, N

        for i, layer in enumerate(layers):
            alloc_layers = [l for l in layer.layers if l.owns_output]
            ngLayer.tensors_temp[(i*4):(i*4 + 4)] = alloc_layers[-1].outputs.primitive[0:4]

        channel = c_longlong(ngLayer.channels.ctypes.data)
        inp = c_longlong(ngLayer.tensors_temp.ctypes.data)
        out = outputs.get_prim()
        prim = c_longlong(ngLayer.primitive.ctypes.data)
        self.mklEngine.Concat_f(inp, ngLayer.layer_num,
                                out, prim, channel, ngLayer.initOK_f, N, C, H, W)
        ngLayer.initOK_f = 1

    def bprop_mergebroadcast(self, ngLayer, layers, error_views,
                             error, deltas, out_shape, alpha, beta, alphas, betas):
        C, D, H, W, N = ngLayer.in_shape5D
        i = 0
        for l, e in zip(layers, error_views):
            ngLayer.tensors_temp[(i * 4):(i * 4 + 4)] = e.primitive[0:4]
            i += 1

        channel = c_longlong(ngLayer.channels.ctypes.data)
        tensors = c_longlong(ngLayer.tensors_temp.ctypes.data)
        prim = c_longlong(ngLayer.primitive.ctypes.data)
        self.mklEngine.Concat_b(tensors, ngLayer.layer_num, error.get_prim(), prim, channel,
                                ngLayer.initOK_b, N, H, W)

        ngLayer.initOK_b = 1

        i = 0
        for l, e in list(zip(layers, error_views)):
            e.primitive[0:4] = ngLayer.tensors_temp[(i * 4):(i * 4 + 4)]
            e.shape5D = l.layers[-1].outputs.shape5D
            err = l.bprop(e)
            ngLayer.tensors_temp[(i * 4):(i * 4 + 4)] = err.primitive[0:4]
            i += 1

        if deltas is None:
            return

        size = c_longlong(np.prod(ngLayer.in_shape5D))
        prim = c_longlong(ngLayer.sum_prim.ctypes.data)
        tensors = c_longlong(ngLayer.tensors_temp.ctypes.data)
        self.mklEngine.MklSumTensor(ngLayer.layer_num, tensors, size, deltas.get_prim(), prim)

        deltas.shape5D = ngLayer.in_shape5D

    def compound_rnn_unroll_fprop(self, W_recur, h_prev_s, h_ff_s, h_s, bias,
                                  nout, num_steps, num_used_steps, activation,
                                  reverse=False):
        """
        Time step unrolling portion of recurrent layer fprop.

        Arguments:
            W_recur (Tensor): Recurrent weight matrix.
            h_prev_s (Array): Array of per time step hidden state tensors. Each
                element in the array is a single time step view into one tensor
                containing all of the time steps in sequence.
            h_ff_s (Array): Array of per time step hidden state tensors. Each
                element in the array is a single time step view into one tensor
                containing all of the time steps in sequence.
            h_s (Array): Array of per time step hidden state tensors. Each
                element in the array is a single time step view into one tensor
                containing all of the time steps in sequence.
            bias (Tensor): Bias tensor to add at each time step.
            nout (integer): Number of output units for the layer.
            num_steps (integer): Total number of time steps in the buffer.
            num_used_steps (integer): Number of time steps being used for real
                data.
            activation (Transform): Activation function for the layer.
            reverse (boolean): When true, unrolling will iterate over time steps
                in reverse (for BiRNN).
        """
        if num_used_steps is not None and num_used_steps < num_steps:
            h_s = h_s[:num_used_steps]
            h_prev_s = h_prev_s[:num_used_steps]
            h_ff_s = h_ff_s[:num_used_steps]

        if reverse:
            steps = reversed(list(zip(h_s, h_prev_s, h_ff_s)))
        else:
            steps = zip(h_s, h_prev_s, h_ff_s)

        for (h, h_prev, h_ff) in steps:
            if h_ff is h:
                self.compound_dot(W_recur, h_prev, h, beta=1.0)
                h[:] = activation(h + bias)
            else:
                self.compound_dot(W_recur, h_prev, h)
                if not math_cpu.add_and_act(h._tensor, h_ff._tensor, bias._tensor, activation):
                    h[:] = activation(h + h_ff + bias)

    def compound_rnn_unroll_bprop(self, W_recur, delta_prev_s, delta_s, h_s,
                                  nout, num_steps, num_used_steps, activation,
                                  reverse=True):
        """
        Time step unrolling portion of recurrent layer bprop.

        Arguments:
            W_recur (Tensor): Recurrent weight matrix.
            delta_prev_s (Array): Array of per time step input delta tensors.
                Each element in the array is a single time step view into one
                tensor containing all of the time steps in sequence.
            delta_s (Array): Array of per time step input delta tensors.
                Each element in the array is a single time step view into one
                tensor containing all of the time steps in sequence.
            h_s (Tensor): Array of per time step hidden state tensors. Each
                element in the array is a single time step view into one tensor
                containing all of the time steps in sequence.
            nout (integer): Number of output units for the layer.
            num_steps (integer): Total number of time steps in the buffer.
            num_used_steps (integer): Number of time steps being used for real
                data.
            activation (Transform): Activation function for the layer.
            reverse (boolean): When true, unrolling will iterate over time steps
                in reverse (default case for RNN).
        """
        if num_used_steps is not None and num_used_steps < num_steps:
            h_s = h_s[:num_used_steps]

        if reverse:
            steps = reversed(list(zip(h_s, delta_s, delta_prev_s)))
        else:
            steps = zip(h_s, delta_s, delta_prev_s)

        for (hs, in_deltas, prev_in_deltas) in steps:
            if not math_cpu.act_and_mul(in_deltas, hs, activation):
                in_deltas[:] = activation.bprop(hs) * in_deltas
            self.compound_dot(W_recur, in_deltas, prev_in_deltas, beta=1.0)

    def allocate_new_deltas(self, delta, in_shape, parallelism):
        """
        For MKL backends, allocate new deltas for broadcast
        """
        return self.iobuf(in_shape, parallelism=parallelism)

    def allocate_new_outputs(self, layer, share_output):
        layer.allocate()

    def change_data_store_order(self, a, a_row, a_col, a_len, axis=1, b=None):
        return math_cpu.change_data_store_order(a, a_row, a_col, a_len, axis=1, b=b)

    def trans2d(self, W_recur_f, W_recur_b, W_recur_f_contiguous, W_recur_b_contiguous):
        math_cpu.trans2d(W_recur_f, W_recur_b, W_recur_f_contiguous, W_recur_b_contiguous)

    def bibnrnn_layer(self, h_buffer_all, h_ff_buffer, W_recur_f, W_recur_b, nsteps, nout):
        """
        Create a new BiBNRNN parameter object. To change the data storage type
        This then is passed as an argument to all the BiBNRNN operations.

        N: Number of images in mini-batch
        C: Number of output feature maps
        K: Number of input feature maps


        """
        return layer_mkl.BiBNRNNLayerMKL(h_buffer_all, h_ff_buffer, W_recur_f, W_recur_b, nsteps,
                                         nout)

    def compound_rnn_unroll_fprop_bibnrnn(self, ngLayer, h_buffer_all, h_ff_buffer, W_recur_f,
                                          h_prev_not_used_in_mkl, h_ff_f_not_used_in_mkl,
                                          h_f_not_used_in_mkl, b_f, W_recur_b,
                                          h_next_not_used_in_mkl, h_ff_b_not_used_in_mkl,
                                          h_b_not_used_in_mkl, b_b, nout, nsteps, nsteps_used,
                                          activation):
        self.change_data_store_order(h_buffer_all, h_buffer_all.shape[0], nsteps + 2,
                                     h_buffer_all.shape[1] // (nsteps + 2),
                                     b=ngLayer.h_all_contiguous)
        self.change_data_store_order(h_ff_buffer, h_ff_buffer.shape[0], nsteps,
                                     h_ff_buffer.shape[1] // nsteps,
                                     b=ngLayer.h_ff_buffer_contiguous)
        self.compound_rnn_unroll_fprop(W_recur_f, ngLayer.h_prev_contiguous,
                                       ngLayer.h_ff_f_contiguous, ngLayer.h_f_contiguous, b_f,
                                       nout, nsteps, nsteps_used, activation, False)

        self.compound_rnn_unroll_fprop(W_recur_b, ngLayer.h_next_contiguous,
                                       ngLayer.h_ff_b_contiguous, ngLayer.h_b_contiguous, b_b,
                                       nout, nsteps, nsteps_used, activation, True)

        self.change_data_store_order(ngLayer.h_all_contiguous, nsteps + 2,
                                     ngLayer.h_all_contiguous.shape[0],
                                     ngLayer.h_all_contiguous.shape[1] // (nsteps + 2),
                                     b=h_buffer_all)
        self.change_data_store_order(ngLayer.h_ff_buffer_contiguous, nsteps,
                                     ngLayer.h_ff_buffer_contiguous.shape[0],
                                     ngLayer.h_ff_buffer_contiguous.shape[1] // nsteps,
                                     b=h_ff_buffer)

    def compound_rnn_unroll_bprop_bibnrnn(self, ngLayer, error, in_deltas_f_not_used_in_mkl,
                                          prev_in_deltas_not_used_in_mkl,
                                          in_deltas_b_not_used_in_mkl,
                                          next_in_deltas_not_used_in_mkl, W_recur_f, W_recur_b,
                                          h_f_not_used_in_mkl,
                                          h_b_not_used_in_mkl, nout, nsteps, nsteps_used,
                                          activation, h_buffer_all):
        self.change_data_store_order(error, error.shape[0], nsteps, error.shape[1] // nsteps,
                                     b=ngLayer.error_contiguous)
        self.trans2d(W_recur_f, W_recur_b, ngLayer.W_recur_f_T_contiguous,
                     ngLayer.W_recur_b_T_contiguous)
        self.compound_rnn_unroll_bprop(ngLayer.W_recur_f_T_contiguous, ngLayer.prev_in_deltas,
                                       ngLayer.in_deltas_f, ngLayer.h_f_contiguous,
                                       nout, nsteps, nsteps_used, activation, True)

        self.compound_rnn_unroll_bprop(ngLayer.W_recur_b_T_contiguous, ngLayer.next_in_deltas,
                                       ngLayer.in_deltas_b, ngLayer.h_b_contiguous,
                                       nout, nsteps, nsteps_used, activation, False)

        self.change_data_store_order(ngLayer.h_all_contiguous, nsteps + 2,
                                     ngLayer.h_all_contiguous.shape[0],
                                     ngLayer.h_all_contiguous.shape[1] // (nsteps + 2),
                                     b=h_buffer_all)
        self.change_data_store_order(ngLayer.error_contiguous, nsteps,
                                     ngLayer.error_contiguous.shape[0],
                                     ngLayer.error_contiguous.shape[1] // nsteps, b=error)
