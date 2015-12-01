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
Our CPU based backend interface and tensor data structure. Our implementation
wraps :mod:`numpy` ndarray and related operations
"""

import numpy as np
import logging
import time
from neon.backends.backend import Tensor, Backend, OpTreeNode, OpCollection
from neon.backends.layer_cpu import ConvLayer, DeconvLayer, PoolLayer

_none_slice = slice(None, None, None)

logger = logging.getLogger(__name__)

# TODO: enable this flag to find numerical problems
# np.seterr(all='raise')


class CPUTensor(Tensor):

    """
    The n-dimensional array data structure that resides in host memory,
    and is meant to be manipulated on the CPU.  wrapped `numpy.ndarray` tensor.

    Arguments:
        dtype (numpy.ndtype, optional): underlying data type of the elements.
        ary   (data array, optional): optionally it can be Instantiated with
                                        a data array
        persist_values (bool, optional): If set to True (the default), the
                                         values assigned to this Tensor will
                                         persist across multiple begin and end
                                         calls.  Setting to False may provide a
                                         performance increase if values do
                                         not need to be maintained across such
                                         calls

    See also:
        NervanaCPU class
    """
    _tensor = None

    def __init__(self,
                 backend,
                 shape=None,
                 dtype=np.float32,
                 ary=None,
                 name=None,
                 persist_values=True):

        super(CPUTensor, self).__init__(backend, shape, dtype, name,
                                        persist_values)

        # supported dtypes
        assert dtype in (np.float16, np.float32, np.float64, np.uint8, np.int8,
                         np.uint16, np.int16, np.uint32, np.int32)

        dtype = np.dtype(dtype)

        if type(ary) != np.ndarray:
            self._tensor = np.array(ary, dtype)
        elif ary.dtype != dtype:
            self._tensor = ary.astype(dtype)
        else:
            self._tensor = ary
        while self._tensor.ndim < self._min_dims:
            self._tensor = self._tensor.reshape(self._tensor.shape + (1, ))

        if shape is not None and len(shape) < self._min_dims:
            self.shape = shape + (1, )
        else:
            self.shape = self._tensor.shape

        try:
            size = 1
            for dim in self.shape:
                size *= dim
        except TypeError:
            assert isinstance(self.shape, (int, long, np.integer))
            size = self.shape
            self.shape = (self.shape,)

        self.size = size

    @property
    def base(self):
        """The base of a tensor is none if it's an original tensor, or a reference to the parent
           tensor if it's a slice or view.
        """
        return self._tensor.base

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
        return ("CPUTensor(base 0x%x) name:%s shape:%s dtype:%s strides:%s"
                " is_c_contiguous:%s" % (base_id, self.name, self.shape,
                                         self.dtype, self._tensor.strides,
                                         self._tensor.flags.c_contiguous))

    def __repr__(self):
        """
        Returns a more unambiguous string representation of the Tensor.

        Returns:
            str: the representation.
        """
        return self.__str__()

    def __len__(self):
        """
        Return the size of the leading dimension of self.
        """
        if len(self.shape):
            return self.shape[0]
        else:
            return 0

    def __setitem__(self, key, value):
        """
        Assign the specified value to a subset of elements found via slice
        style indexing along each dimension. e.g. A[5:10, :] = 4.5.
        Each slice consists of start_idx:stop_idx:step_size triplets.  If
        step_size isn't specified it defaults to 1.  If start_idx isn't
        specified it defaults to 0.  If stop_idx isn't specified it defaults
        to the total number of elements along that dimension.  As such a slice
        value of ':' allows one to select all elements along that dimension.

        Arguments:
            key (int, slice, tuple): indices of each dimension's slice.
            value (numeric array, CPUTensor): values to be assigned to the
                                              extracted element subset.  If an
                                              array it should be the same shape
                                              as what key indexes (or be
                                              broadcastable as such).
        """

        self.__getitem__(key)._assign(value)
        return self

    def __getitem__(self, key):
        """
        Extract a subset view of the items via slice style indexing
        along each dimension. e.g. A[5:10, :].  Each slice consists of
        start_idx:stop_idx:step_size triplets.  If step_size isn't specified it
        defaults to 1.  If start_idx isn't specified it defaults to 0.  If
        stop_idx isn't specified it defaults to the total number of elements
        along that dimension.  As such a slice value of ':' allows one to
        select all elements along that dimension

        Arguments:
            key (int, slice, tuple): indices of each dimension's slice.

        Returns:
            CPUTensor: view of self corresponding to the subset items.

        """
        # speed up common case of [:]
        if not isinstance(key, tuple):
            if key == _none_slice:
                return self
            key = (key,)

        # ensure we return a view
        # exact same behavior as cpu
        # let a.shape = (3,4)
        # a[1,1] = 10 # cpu, gpu and numpy
        # type(a[1,1]) # for cpu and gpu type is Tensor; for numpy type is float
        first_int_idx = None
        is_all_int = True
        for idx, k in enumerate(key):
            if type(k) is int:
                if first_int_idx is None:
                    first_int_idx = idx
            else:
                is_all_int = False
                break
        if is_all_int:
            key_list = list(key)
            idx = key_list[first_int_idx]
            key_list[first_int_idx] = slice(idx, idx + 1, None)
            key = tuple(key_list)

        # return a view of the tensor
        return self.__class__(
            backend=self.backend,
            ary=self._tensor[key],
            dtype=self._tensor.dtype)

    def _assign(self, value):
        """
        Assign an input value to the CPU tensor. The NervanaCPU does clipping
        for int and uint types, when overflow happens

        Arguments:
            value (GPUTennsor, OpTreNode, numeric): the value to be assigned.

        """
        if isinstance(value, (CPUTensor, OpTreeNode)):
            OpTreeNode.build("assign", self, value)
        elif isinstance(value, (int, float, np.ndarray)):
            self.set(value)
        else:
            raise TypeError("Invalid type for assignment: %s" % type(value))

        return self

    def set(self, value):
        """
        Wrap the value into NervanaCPU tensor.

        Arguments:
            value: Array or single input. If it is array, check and Convert
                   the dtype and shape. If it is single value, broadcast to
                   the memory

        Returns:
            self
        """
        if isinstance(value, np.ndarray):
            if value.dtype is not self.dtype:
                value = value.astype(self.dtype)
            assert value.size == self.size
            if value.ndim < self._min_dims:
                value = value.reshape(self.shape)

        self._tensor[:] = value
        return self

    def get(self):
        """
        return the array
        """
        return self._tensor

    def asnumpyarray(self):
        """
        Convert the CPUTensor to an in host memory `numpy.ndarray`.  A copy of
        the data may be made depending on where the CPUTensor normally resides.

        Returns:
            numpy.ndarray view or copy of the CPUTensor data.
        """
        return self._tensor

    def take(self, indices, axis=None):
        """
        Select a subset of elements from an array across an axis

        Arguments:
            indices (Tensor, numpy ndarray): indicies of elements to select
            axis (int): axis across which to select the values

        Returns:
            Tensor: Tensor with selected values

        """
        if type(indices) == self.__class__:
            indices = indices._tensor
        # if indices are nx1 or 1xn, much of our code assumes these dims are
        # collapsed, hence the squeeze call.
        if type(indices) == np.ndarray:
            indices = indices.squeeze()
        return self.__class__(
            backend=self.backend,
            ary=self._tensor.take(indices, axis),
            dtype=self._tensor.dtype)

    def fill(self, value):
        """
        Assign specified value to each element of this CPUTensor.

        Arguments:
            value (numeric): The value to be assigned to each element.

        Return:
            CPUTensor: updated view of the data.
        """
        self._tensor.fill(value)
        return self

    def copy(self, a):
        """
        Construct and return a deep copy of the Tensor passed.

         Arguments:
            a (Tensor): the object to copy

        Returns:
            Tensor: new array object with the same values as input tensor
        """
        return self._assign(a)

    def copy_from(self, a):
        """ alias of copy """
        return self._assign(a)

    def reshape(self, *shape):
        """
        return a reshaped view
        """
        if isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        if shape == self.shape:
            return self

        return self.__class__(
            backend=self.backend,
            ary=self._tensor.reshape(shape),
            dtype=self._tensor.dtype)

    @property
    def T(self):
        """
        Return a transposed view

        For 2D tensor, will do a normal transpose
        For 3D tensor, will keep the 0 dim, swap the 1 and 2 dimensions

        """
        if len(self.shape) <= 2:
            ary = self._tensor.transpose()
        else:
            # support for batched dot.
            # perserve outer dimension but reverse inner dims
            # shape = np.concatenate((shape[-1:], shape[:-1])
            ary = self._tensor.swapaxes(1, 2)

        return self.__class__(
            backend=self.backend,
            ary=ary,
            dtype=self._tensor.dtype)

    def transpose(self, out=None):
        """
        Return a transposed view of the data.  Alias of .T property
        """
        if out:
            return OpTreeNode.build("assign", out, self.T)
        return self.T

    def share(self, shape, dtype=None, name=None):
        """
        return a view: ary, where ary.size <= self.size
        Allows easy sharing of temporary memory
        This is mostly provided for compatibility, -- dtype is ignored
        """
        size = np.prod(shape)
        if size > self.size:
            raise ValueError("total size of new array must <= size of parent")

        ary = self._tensor.ravel()[:size].reshape(shape)

        return self.__class__(
            backend=self.backend,
            ary=ary,
            dtype=self._tensor.dtype)

    def hist(self, tag):
        """
        Compute a histogram of the current tensor values.

        Arguments:
            tag (string): Tag to identify the current state of the tensor,
                          useful for disambiguating multiple histograms of the
                          same tensor at different points in time.

        Returns:
            Tensor containing the histogram data.

        """
        nbins = self.backend.hist_bins
        offset = self.backend.hist_offset
        bins = np.arange(nbins + 1) + float(offset)
        bins[0] = -float('Inf')
        np_inp_log_abs = np.rint(
            np.log2(np.abs(self.get().astype(np.float32))))
        np_hist, edges = np.histogram(np_inp_log_abs, density=False, bins=bins)
        nc_hist = self.backend._hist_tensor(tag)._assign(np_hist)
        return nc_hist

    # def repeat(self, repeats, axis):
    #     return self.__class__(
    #         backend=self.backend,
    #         ary=self._tensor.repeat(repeats, axis))


class CustomNumpy:

    @staticmethod
    def argmax(x, axis=1, keepdims=True):
        """
        calls numpy argmax with keepdims
        """
        new_shape = list(x.shape)
        new_shape[axis] = 1
        new_shape = tuple(new_shape)
        return np.argmax(x, axis=axis).reshape(new_shape)

    @staticmethod
    def argmin(x, axis=1, keepdims=True):
        """
        calls numpy argmin with keepdims
        """
        new_shape = list(x.shape)
        new_shape[axis] = 1
        new_shape = tuple(new_shape)
        return np.argmin(x, axis=axis).reshape(new_shape)


def _assign_right_to_left(left, right):
    left[:] = right

numpy_call_dict = {
    # assign
    "assign": _assign_right_to_left,
    # zero_operand ops
    # unary ops
    "neg": lambda left: -left,
    "abs": lambda left: np.abs(left),
    "sgn": lambda left: np.sign(left),
    "sqrt": lambda left: np.sqrt(left),
    "sqr": lambda left: np.square(left),
    "exp": lambda left: np.exp(left),
    "log": lambda left: np.log(left),
    "safelog": lambda left: np.log(np.maximum(left, np.exp(-50.))),
    "exp2": lambda left: np.exp2(left),
    "log2": lambda left: np.log2(left),
    "sig": lambda left: 1. / (1. + np.exp(-left)),
    "sig2": lambda left: 1. / (1. + np.exp2(-left)),
    "tanh": lambda left: np.tanh(left),
    "tanh2": lambda left: (np.exp2(2. * left) - 1.) / (np.exp2(2. * left) + 1.),
    "transpose": lambda left: np.transpose(left),
    # binary ops
    "add": lambda left, right: left + right,
    "sub": lambda left, right: left - right,
    "mul": lambda left, right: left * right,
    "div": lambda left, right: left / right,
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
    "sum": lambda op_dict, left: np.sum(left, axis=op_dict['axis'], keepdims=True),
    "max": lambda op_dict, left: np.max(left, axis=op_dict['axis'], keepdims=True),
    "min": lambda op_dict, left: np.min(left, axis=op_dict['axis'], keepdims=True),
    "argmax": lambda op_dict, left: CustomNumpy.argmax(left, axis=op_dict['axis'], keepdims=True),
    "argmin": lambda op_dict, left: CustomNumpy.argmin(left, axis=op_dict['axis'], keepdims=True),
}


class NervanaCPU(Backend):

    """
    Sets up a :mod:`numpy` based backend for matrix ops.  By default, we use
    32-bit element data types for any arrays constructed.

    Attributes:
        default_dtype (dtype): default element data type.
        tensor_cls: underlying Tensor type. For CPU backend, it will be CPU tensor

    See also:
        CPUTensor
    """

    def __init__(self,
                 rng_seed=None,
                 default_dtype=np.float32,
                 hist_bins=64,
                 hist_offset=-48,
                 compat_mode=None):

        if default_dtype not in [np.float16, np.float32, np.float64]:
            logger.error('Default data type for nervanagpu '
                         'backend must be float16, 32 or 64')
            raise ValueError

        super(NervanaCPU, self).__init__(rng_seed, default_dtype, compat_mode=compat_mode)

        self.device_type = 0
        self.device_id = 0
        self.tensor_cls = CPUTensor

        # log
        logger.info("Initialized NervanaCPU")

        self.hist_bins = hist_bins
        self.hist_offset = hist_offset
        self.hist_max = 4096
        self.hist_buf = self.empty((self.hist_max, hist_bins), dtype=np.int32)
        self.hist_idx = 0
        self.hist_map = dict()

    def rng_reset(self):
        """
        Reset the random state to the state where the Backend is first
        initialized.
        """
        self.rng.set_state(self.init_rng_state)

    def execute(self, optree):
        """
        Arguments:
            optree: (OpTreeNode): the OpTreeNode object that represents all
                                    the operations
        """
        # deal with onehot specially for now
        if (len(optree) == 3 and isinstance(optree[2], OpTreeNode) and
                optree[2][0]['op'] == 'onehot'):
            assert optree[0]['op'] == 'assign'
            assert isinstance(optree[1], Tensor)
            # get the output buffer
            array_output = optree[1].get()

            # get the output shape and onehot representation length will be on
            # this axis
            numpy_axis = optree[2][0]['axis']
            numpy_ind0 = optree[2][0]['idx'].get().squeeze()

            numpy_ind_len = numpy_ind0.size
            numpy_ind1 = range(numpy_ind_len)

            # ind for indexing
            numpy_ind = np.zeros((2, numpy_ind_len), dtype=np.int32)
            numpy_ind[numpy_axis] = numpy_ind0
            numpy_ind[1 - numpy_axis] = numpy_ind1
            array_output[:] = 0
            array_output[numpy_ind.tolist()] = 1

            return array_output

        # get post order stack
        postfix_stack = optree.traverse(list())

        # init compute stack
        compute_stack = []

        # iterate through postfix stack to compute result
        for p in postfix_stack:
            if isinstance(p, dict):
                # TODO add rand and onehot here
                if p['op'] in OpCollection.unary_ops:
                    left = compute_stack.pop()
                    compute_stack.append(numpy_call_dict[p['op']](left))
                elif p['op'] in OpCollection.binary_ops:
                    right = compute_stack.pop()
                    left = compute_stack.pop()
                    compute_stack.append(numpy_call_dict[p['op']](left, right))
                elif p['op'] in OpCollection.reduction_ops:
                    left = compute_stack.pop()
                    compute_stack.append(numpy_call_dict[p['op']](p, left))
                elif p['op'] in OpCollection.zero_operand_ops:
                    compute_stack.append(numpy_call_dict[p['op']](None))
                else:
                    raise NotImplementedError
            elif isinstance(p, CPUTensor):
                compute_stack.append(p._tensor)
            else:
                compute_stack.append(p)

        assert len(compute_stack) == 1
        return postfix_stack[0]

    def empty(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False):
        """
        Instantiate a new instance of the CPUTensor class without initializing
        individual element values.

        Arguments:
            shape (int, list): The size of each dimension of the Tensor.

            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value

            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls

        Returns:
            CPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(
            backend=self,
            ary=np.zeros(shape, dtype),
            dtype=dtype,
            name=name,
            persist_values=persist_values)

    def array(self, ary, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False):
        """
        Instantiate a new instance of the CPUTensor class setting each element
        value to what is specified in ary.

        Arguments:
            ary (numpy.ndarray): The data structure containing element values
                                 spread across a number of dimensions.  Python
                                 built-in types like ints and lists are
                                 supported.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value ('float32'
                                     unless overridden).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls

        Returns:
            CPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(
            backend=self,
            ary=np.array(ary, dtype),
            dtype=dtype,
            name=name,
            persist_values=persist_values)

    def zeros(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False):
        """
        Instantiate a new instance of the CPUTensor class setting each element
        value to 0.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value ('float32'
                                     unless overridden).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls

        Returns:
            CPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(
            backend=self,
            ary=np.zeros(shape, dtype),
            dtype=dtype,
            name=name,
            persist_values=persist_values)

    def ones(self, shape, dtype=None, name=None, persist_values=True,
             parallel=False, distributed=False):
        """
        Instantiate a new instance of the CPUTensor class setting each element
        value to 1.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value ('float32'
                                     unless overridden).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls

        Returns:
            CPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(
            backend=self,
            ary=np.ones(shape, dtype),
            dtype=dtype,
            name=name,
            persist_values=persist_values)

    def empty_like(self, ary, dtype=None, name=None, persist_values=True):
        """
        Instantiate a new instance of this backend's Tensor class, with the
        shape taken from ary.

        Arguments:
            ary (tensor object): Tensor to inherit the dimensions of.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls
        Returns:
            Tensor: array object
        """
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(
            backend=self,
            ary=np.zeros(ary.shape, dtype),
            dtype=dtype,
            name=name,
            persist_values=persist_values)

    def zeros_like(self, ary, dtype=None, name=None, persist_values=True):
        """
        Instantiate a new instance of this backend's Tensor class, with the
        shape taken from ary and populating each element with a value of 0.

        Arguments:
            ary (tensor object): Tensor to inherit the dimensions of.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls
        Returns:
            Tensor: array object
        """
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(
            backend=self,
            ary=np.zeros(ary.shape, dtype),
            dtype=dtype,
            name=name,
            persist_values=persist_values)

    def compound_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, bsum=None):
        """
        Doing following operations (* is dot product)
        C = alpha * A * B   + beta * C
        C = alpha * A.T * B + beta * C
        C = alpha * A * B.T + beta * C

        relu: if true applied before output (and prior to beta addition)

        The operation will be short-circuited to: out <- alpha * left * right
        if beta has value 0 (the default).

        Arguments:
            A, B (CPUTensor): input operands
            C (CPUTensor): output
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

        if beta == 0:
            if C._tensor.flags['C_CONTIGUOUS'] is not True:
                tmp = np.empty(C.shape, dtype=C.dtype)
                np.dot(A._tensor, B._tensor, tmp)
                C._tensor[:] = tmp.copy()
            else:
                np.dot(A._tensor, B._tensor, C._tensor)

            if relu:
                self.Relu(C._tensor, C._tensor)
        else:
            np.multiply(C._tensor, beta, C._tensor)
            tmp = np.empty(C.shape, dtype=C.dtype)
            np.dot(A._tensor, B._tensor, tmp)
            np.multiply(tmp, alpha, tmp)
            if relu:
                self.Relu(tmp, tmp)
            np.add(C._tensor, tmp, C._tensor)
        if bsum is not None:
            bsum[:] = self.sum(C, 1)

        return C

    def batched_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False):
        """
        Doing following operations:
        1. For fprop: A(K, C), B(X,C,N), C(X,K,N) --> call batched_dot(A, B, C)
        2. For bprop: A(K, C), B(X,K,N), C(X,C,N) --> call batched_dot(A.T, B, C)
        3. For update: A(X,K,N), B(X,C,N), C(K,C) --> call batched_dot(A, B.T, C)

        Arguments:
            A, B (CPUTensor): input operands
            C (CPUTensor): output
            alpha, beta, relu: see usage in dot()
        """
        assert A.dtype == B.dtype == C.dtype

        dima, dimb, dimc = 0, 0, 0
        # ldaz, ldbz, ldcz = 0, 0, 0 # commented for stylecheck
        batch_grid, batch_loops = 1, 1

        if len(A.shape) == 3:
            dima = 1

        if len(B.shape) == 3:
            dimb = 1

        assert dima or dimb, "Tensor A or B must have 3 dims to use batched_dot"

        if len(C.shape) == 3:
            dimc = 1
            batch_grid = C.shape[0]
            assert not dima or A.shape[0] == batch_grid
            assert not dimb or B.shape[0] == batch_grid

        if dima:
            batch_loops = A.shape[0]
            assert not dimb or B.shape[0] == batch_loops

        elif dimb:
            batch_loops = B.shape[0]
            assert not dima or A.shape[0] == batch_loops

        assert A.shape[0 + dima] == C.shape[0 + dimc]
        assert B.shape[1 + dimb] == C.shape[1 + dimc]
        assert A.shape[1 + dima] == B.shape[0 + dimb]

        tmp = np.zeros(C.shape)

        for i in range(batch_loops):
            if dima:
                tmp += np.dot(A._tensor[i], B._tensor[i])
            else:
                tmp[i] = np.dot(A._tensor, B._tensor[i])

        np.multiply(tmp, alpha, tmp)
        if relu:
            self.Relu(tmp, tmp)
        np.add(C._tensor * beta, tmp, C._tensor)

        return C

    def make_binary_mask(self, out, keepthresh=0.5):
        """
        Create a binary mask for dropout layers.

        Arguments:
            out (CPUTensor): Output tensor
            keepthresh (float): fraction of ones
        """
        out._tensor[:] = np.array(
            self.rng.uniform(size=out._tensor.shape) < keepthresh,
            dtype=out._tensor.dtype)

    def conv_layer(self, dtype,
                   N, C, K,
                   D=1, H=1, W=1,
                   T=1, R=1, S=1,
                   pad_d=0, pad_h=0, pad_w=0,
                   str_d=1, str_h=1, str_w=1,
                   bsum=False):
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

        dtype: need to know dtype to setup proper kernels and params.

        bsum: calculate the sum along the batchnorm axis for fprop or bprop
              outputs an fp32 tensor of size Kx1
        """
        return ConvLayer(self, dtype, N, C, K, D, H, W, T, R, S,
                         pad_d, pad_h, pad_w, str_d, str_h, str_w)

    def fprop_conv(self, layer, I, F, O, alpha=1.0, relu=False, bsum=None, beta=0.0):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output

        Arguments:
            layer: the conv layer as a parameter object
            I (CPUTensor): inputs
            F (CPUTensor): the weights (filters)
            O (CPUTensor): outputs
            alpha (float): linear scaling
            relu (boolean): apply ReLu or not before output
                            (currently not implemented)
            beta (float): accumulation value into O
        """
        assert layer.sizeI == I.size
        assert layer.sizeF == F.size
        assert layer.sizeO == O.size

        M, P, Q = layer.MPQ
        C, D, H, W, N = layer.dimI
        C, T, R, S, K = layer.dimF
        K, M, P, Q, N = layer.dimO

        pad_d, pad_h, pad_w = layer.padding
        str_d, str_h, str_w = layer.strides

        array_I = I.get().reshape(layer.dimI)
        array_F = F.get().reshape(layer.dimF)
        array_O = O.get().reshape(layer.dimO)

        for m in range(M):
            sliceT, sliceD, _ = layer.mSlice[m]

            for p in range(P):
                sliceR, sliceH, _ = layer.pSlice[p]

                for q in range(Q):
                    sliceS, sliceW, _ = layer.qSlice[q]

                    slicedF = array_F[:, sliceT, sliceR, sliceS, :].reshape((-1, K))
                    slicedI = array_I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))

                    array_O[:, m, p, q, :] = beta * array_O[:, m, p, q, :] + alpha * \
                        np.dot(slicedF.T,  slicedI)

        if bsum is not None:
            bsum[:] = array_O.sum((1, 2, 3, 4))

    def bprop_conv(self, layer, F, E, grad_I, alpha=1.0, relu=False, bsum=None, beta=0.0):
        """
        Backward propagate the error through a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            F (CPUTensor): the weights (filters)
            E (CPUTensor): errors
            grad_I (CPUTensor): gradient to inputs (output delta)
            alpha (float): linear scaling
            beta (float): accumulation value into grad_I
            relu (boolean): apply ReLu or not before output
                            (currently not implemented)
        """
        assert layer.sizeF == F.size
        assert layer.sizeO == E.size
        assert layer.sizeI == grad_I.size

        M, P, Q = layer.MPQ
        C, D, H, W, N = layer.dimI
        C, T, R, S, K = layer.dimF
        K, M, P, Q, N = layer.dimO

        pad_d, pad_h, pad_w = layer.padding
        str_d, str_h, str_w = layer.strides

        array_F = F.get().reshape(layer.dimF)
        array_E = E.get().reshape(layer.dimO)
        array_grad_I = grad_I.get().reshape(layer.dimI)

        array_F = np.transpose(array_F, (4, 1, 2, 3, 0)).copy()

        for d in range(D):
            sliceT, sliceM = layer.dSlice[d]

            for h in range(H):
                sliceR, sliceP = layer.hSlice[h]

                for w in range(W):
                    sliceS, sliceQ = layer.wSlice[w]

                    sliceTRS = np.array([
                        t * R * S + r * S + s
                        for t in sliceT
                        for r in sliceR
                        for s in sliceS], dtype=np.intp)

                    sliceMPQ = np.array([
                        m * P * Q + p * Q + q
                        for m in sliceM
                        for p in sliceP
                        for q in sliceQ], dtype=np.intp)

                    slicedF = array_F.reshape(
                        (K, -1, C))[:, sliceTRS, :].reshape((-1, C))
                    slicedE = array_E.reshape(
                        (K, -1, N))[:, sliceMPQ, :].reshape((-1, N))

                    array_grad_I[:, d, h, w, :] = beta * array_grad_I[:, d, h, w, :] + alpha * \
                        np.dot(slicedF.T, slicedE)
        # If this is the forward pass for deconv, compute bsum here
        if bsum is not None:
            bsum[:] = self.sum(grad_I.reshape(C, -1), 1)

    def update_conv(self, layer, I, E, U, alpha=1.0):
        """
        Compute the updated gradient for a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            I (CPUTensor): the inputs
            E (CPUTensor): the errors
            U (CPUTensor): the updates
            alpha (float): linear scaling
        """
        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeF == U.size

        C, D, H, W, N = layer.dimI
        C, T, R, S, K = layer.dimF
        K, M, P, Q, N = layer.dimO

        pad_d, pad_h, pad_w = layer.padding
        str_d, str_h, str_w = layer.strides

        array_I = I.get().reshape(layer.dimI)
        array_E = E.get().reshape(layer.dimO)
        array_U = U.get().reshape(layer.dimF)
        array_U.fill(0.)

        for m in range(M):
            sliceT, sliceD, tlen = layer.mSlice[m]

            for p in range(P):
                sliceR, sliceH, rlen = layer.pSlice[p]

                for q in range(Q):
                    sliceS, sliceW, slen = layer.qSlice[q]

                    slicedI = array_I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
                    slicedE = array_E[:, m, p, q, :]
                    array_U[:, sliceT, sliceR, sliceS, :] += alpha * np.dot(
                        slicedI, slicedE.T).reshape((C, tlen, rlen, slen, K))

    def deconv_layer(self, dtype,
                     N, C, K,
                     P, Q,
                     R=1, S=1,
                     pad_d=0, pad_h=0, pad_w=0,
                     str_d=1, str_h=1, str_w=1,
                     bsum=False):
        """
        Create a new PoolLayer parameter object.
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
        return DeconvLayer(self, dtype, N, C, K, P, Q, R, S,
                           pad_d, pad_h, pad_w, str_d, str_h, str_w)

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

        return PoolLayer(self, dtype, op, N, C, D, H, W, J, T, R, S,
                         pad_c, pad_d, pad_h, pad_w, str_c, str_d, str_h, str_w)

    def fprop_pool(self, layer, I, O, argmax=None):
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
        if layer.op == "max":
            assert layer.sizeO == argmax.size
        op = layer.op

        J, T, R, S = layer.JTRS
        C, D, H, W, N = layer.dimI
        K, M, P, Q, N = layer.dimO
        pad_c, pad_d, pad_h, pad_w = layer.padding
        str_c, str_d, str_h, str_w = layer.strides

        array_I = I.get().reshape(layer.dimI)
        array_O = O.get().reshape(layer.dimO)
        if op == "max":
            array_argmax = argmax.get().reshape(layer.dimO)

        for k in range(K):
            sliceC, _ = layer.kSlice[k]

            for m in range(M):
                sliceD, _ = layer.mSlice[m]

                for p in range(P):
                    sliceH, _ = layer.pSlice[p]

                    for q in range(Q):
                        sliceW, _ = layer.qSlice[q]

                        sliceI = array_I[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        if op == "max":
                            array_argmax[k, m, p, q, :] = np.argmax(sliceI, axis=0)
                            array_O[k, m, p, q, :] = np.max(sliceI, axis=0)
                        elif op == "avg":
                            array_O[k, m, p, q, :] = np.mean(sliceI, axis=0)
                        elif op == "l2":
                            array_O[k, m, p, q, :] = np.sqrt(np.sum(
                                np.square(sliceI), axis=0))

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
        op = layer.op

        J, T, R, S = layer.JTRS
        C, D, H, W, N = layer.dimI
        K, M, P, Q, N = layer.dimO
        pad_c, pad_d, pad_h, pad_w = layer.padding
        str_c, str_d, str_h, str_w = layer.strides

        array_E = I.get().reshape(layer.dimO)
        array_E[:] = array_E * alpha
        array_delta = O.get().reshape(layer.dimI)
        array_delta[:] = array_delta * beta
        if op == "max":
            array_argmax = argmax.get().reshape(layer.dimO)

        for k in range(K):
            sliceC, clen = layer.kSlice[k]

            for m in range(M):
                sliceD, dlen = layer.mSlice[m]

                for p in range(P):
                    sliceH, hlen = layer.pSlice[p]

                    for q in range(Q):
                        sliceW, wlen = layer.qSlice[q]

                        patch_in = (sliceC, sliceD, sliceH, sliceW, slice(None))
                        patch_out = (k, m, p, q, slice(None))
                        sliceB = array_delta[patch_in].reshape((-1, N))
                        if op == "max":
                            max_n = array_argmax[patch_out]
                            sliceB[max_n, range(N)] += array_E[patch_out]
                        elif op == "avg":
                            sliceB += array_E[patch_out] * (1.0 / sliceB.shape[0])
                        else:
                            raise NotImplementedError
                        array_delta[patch_in] = sliceB.reshape((clen, dlen, hlen, wlen, N))

    def compound_fprop_bn(self, x, xsum, xvar, gmean, gvar, gamma, beta, y, eps, rho, relu):
        """
        Function to perform batch normalization forward pass. Included
        for API compatibility with GPU compound kernel call.

        Arguments:
            x (Tensor): Input from previous layer
            xsum (Tensor): Precomputed batch sum over PQN dimension
            xvar (Tensor): Buffer for variance (computed in kernel)
            gmean (Tensor): global mean ()
            gvar (Tensor): global variance
            gamma (Tensor): scale parameter
            beta (Tensor): location paramter
            y (Tensor): normalized output
            eps (float): constant for numerical stability
            rho (float): exponential window averaging constant
        """
        xvar[:] = self.var(x, axis=1)
        xsum[:] = xsum / x.shape[1]  # reuse xsum instead of computing xmean
        xhat = (x - xsum) / self.sqrt(xvar + eps)

        gmean[:] = gmean * rho + (1.0 - rho) * xsum
        gvar[:] = gvar * rho + (1.0 - rho) * xvar

        outputs = y.reshape(xhat.shape)
        outputs[:] = xhat * gamma + beta

    def compound_bprop_bn(self, delta, grad_gamma, grad_beta, x, xsum, xvar,
                          gamma, eps):
        """
        Function to perform batch normalization backward pass. Included
        for API compatibility with GPU compound kernel call.

        Arguments:
            delta (Tensor): Delta buffer
            grad_gamma (Tensor): Gradient w.r.t. gamma
            grad_beta (Tensor): Gradient w.r.t. beta
            x (Tensor): feedforward input
            xsum (Tensor): Batch sum over PQN dimension
            xvar (Tensor): Batch variance
            gamma (Tensor): scale parameter
            eps (float): constant for numerical stability
        """
        xhat = (x - xsum) / self.sqrt(xvar + eps)
        grad_gamma[:] = self.sum(xhat * delta, axis=1)
        grad_beta[:] = self.sum(delta, axis=1)
        xtmp = (xhat * grad_gamma + grad_beta) / float(x.shape[1])
        delta[:] = gamma * (delta - xtmp) / self.sqrt(xvar + eps)

    def _hist_tensor(self, tag):
        """
        Create a tensor the right size for histogram data, with memory allocated
        in the contiguous histogram buffer. Track it by tag for later reference.
        """
        assert self.hist_idx < self.hist_max
        self.hist_map[tag] = (self.hist_idx)
        hist_buf = self.hist_buf[self.hist_idx]
        self.hist_idx += 1
        return hist_buf

    def dump_hist_data(self):

        hist_data = self.hist_buf
        hist_map = self.hist_map
        self.hist_map = dict()
        self.hist_idx = 0
        self.hist_buf = self.empty(
            (self.hist_max, self.hist_bins), dtype=np.int32)
        return hist_data, hist_map

    def Relu(self, ary, out=None):
        """
        Calculates the ReLu transformation for input array

        Arguments:
            ary: numpy array
            out: reference to output
        """
        if out is not None:
            return np.maximum(ary, 0, out)
        else:
            return np.maximum(ary, 0)

    def init_mark(self):
        """
        Generate a timing mark object

        Returns:
            timing mark (dict)
        """
        return {'time': 0}

    def record_mark(self, marker):
        """
        Mark the current time

        Arguments:
            marker (time mark): timing mark generated by init_mark()
        """
        marker['time'] = time.time()

    def get_time(self, start, end):
        """
        Return time between start and end marks

        Arguments:
            start (time maker): start time mark

            end (time marker): end time mark

        Returns:
            time elapsed between start and end time marks in milliseconds
        """
        return (end['time'] - start['time'])*1000.0
