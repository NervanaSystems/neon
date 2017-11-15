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
Our CPU based backend interface and tensor data structure. Our implementation
wraps :mod:`numpy` ndarray and related operations
"""
from __future__ import division
from builtins import object, round, str, zip
import numpy as np
import logging
import time
import functools
from neon.backends.backend import Tensor, Backend, OpTreeNode, OpCollection
from neon.backends.layer_cpu import ConvLayer, DeconvLayer, PoolLayer
from neon.util.compat import xrange

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
        ary (data array, optional): optionally it can be instantiated with
                                    a data array
        persist_values (bool, optional): If set to True (the default), the
                                         values assigned to this Tensor will
                                         persist across multiple begin and end
                                         calls.  Setting to False may provide a
                                         performance increase if values do
                                         not need to be maintained across such
                                         calls

    See also:
        :class:`NervanaCPU` class
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
            self.shape = shape + (1, )*(self._min_dims - len(shape))
        else:
            self.shape = self._tensor.shape

        shape_ = []
        size = 1
        for dim in self.shape:
            if int(dim) != dim:
                raise TypeError('shape dims must be integer values [%s]' % str(dim))
            dim = int(dim)
            shape_.append(dim)
            size *= dim
        self.shape = tuple(shape_)

        self.size = size
        self.base = base
        self.dtype = dtype

        self.is_contiguous = self._tensor.flags.c_contiguous

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
        select all elements along that dimension. To be consistent with GPU
        Tensors, CPU Tensors remove the axis that has size 1 unless it needs to
        maintain 2D.

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
        key_list = list(key)
        for idx, k in enumerate(key):
            if type(k) is int:
                k = self.shape[idx] + k if k < 0 else k
                key_list[idx] = slice(k, k + 1, None)
        key = tuple(key_list)

        new_shape = list(self._tensor[key].shape)
        for idx, k in enumerate(new_shape):
            if len(new_shape) > 2 and k is 1:
                new_shape.remove(k)

        # return a view of the tensor
        return self.__class__(
            backend=self.backend,
            ary=self._tensor[key].reshape(new_shape),
            dtype=self._tensor.dtype,
            base=self)

    def _assign(self, value):
        """
        Assign an input value to the CPU tensor. The NervanaCPU does clipping
        for int and uint types, when overflow happens

        Arguments:
            value (CPUTensor, OpTreeNode, numeric): the value to be assigned.

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
        Return the array.
        """
        return self._tensor.copy()

    def raw(self):
        """
        Access the raw buffer.

        Returns:
            pointer: A device specific pointer
        """
        return self._tensor.ctypes.data

    def asnumpyarray(self):
        """
        Deprecated.
        Scheduled to be removed in 2.0.
        Use get() instead.
        """
        return self._tensor

    def take(self, indices, axis=None):
        """
        Select a subset of elements from an array across an axis.

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
        new_shape = list(self.shape)
        new_shape[axis] = indices.size
        return self.__class__(
            backend=self.backend,
            ary=self._tensor.take(indices, axis).reshape(new_shape),
            dtype=self._tensor.dtype,
            base=self)

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
        """
        Alias of copy.

        Arguments:
            a (Tensor): the object to copy

        Returns:
            Tensor: new array object with the same values as input tensor
        """
        return self._assign(a)

    def reshape(self, *shape):
        """
        Return a reshaped view.
        """
        if isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        if shape == self.shape:
            return self

        try:
            ary = self._tensor.reshape(shape)
        except ValueError:
            def product(vec):
                return functools.reduce(lambda x, y: x * y, vec)

            raise ValueError((
                'The total size of a reshaped tensor must be the same as its '
                'existing size. Tensor is currently shape {current_shape} '
                'and size {current_size}. Attempted to reshape to '
                '{reshape_shape} which would be size {reshape_size}.'
            ).format(
                current_shape=self._tensor.shape,
                current_size=product(self._tensor.shape),
                reshape_shape=shape,
                reshape_size=product(shape),
            ))

        return self.__class__(
            backend=self.backend,
            ary=ary,
            dtype=self._tensor.dtype,
            base=self)

    @property
    def T(self):
        """
        Return a transposed view.

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
            dtype=self._tensor.dtype,
            base=self)

    def transpose(self, out=None):
        """
        Return a transposed view of the data.  Alias of .T property
        """
        if out:
            return OpTreeNode.build("assign", out, self.T)
        return self.T

    def share(self, shape, dtype=None, name=None):
        """
        Return a view: ary, where ary.size <= self.size.
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
            dtype=self._tensor.dtype,
            base=self)

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
            np.log2(np.abs(self._tensor.astype(np.float32))))
        np_hist, edges = np.histogram(np_inp_log_abs, density=False, bins=bins)
        nc_hist = self.backend._hist_tensor(tag)._assign(np_hist)
        return nc_hist

    # def repeat(self, repeats, axis):
    #     return self.__class__(
    #         backend=self.backend,
    #         ary=self._tensor.repeat(repeats, axis))


class CustomNumpy(object):

    @staticmethod
    def argmax(x, axis=1, keepdims=True):
        """
        Calls numpy argmax with keepdims.
        """
        new_shape = list(x.shape)
        new_shape[axis] = 1
        new_shape = tuple(new_shape)
        return np.argmax(x, axis=axis).reshape(new_shape)

    @staticmethod
    def argmin(x, axis=1, keepdims=True):
        """
        Calls numpy argmin with keepdims.
        """
        new_shape = list(x.shape)
        new_shape[axis] = 1
        new_shape = tuple(new_shape)
        return np.argmin(x, axis=axis).reshape(new_shape)


def _assign_right_to_left(left, right):
    left[:] = right


numpy_call_dict_cpu = {
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
    "rint": lambda left: np.rint(left),
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
    Sets up a :mod:`numpy` baseyd backend for matrix ops.  By default, we use
    32-bit element data types for any arrays constructed.

    Attributes:
        default_dtype (dtype): default element data type.
        tensor_cls: underlying Tensor type. For CPU backend, it will be CPU tensor

    See also:
        :class:`CPUTensor`
    """
    backend_name = 'cpu'

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

        if default_dtype not in [np.float16, np.float32, np.float64]:
            logger.error('Default data type for nervanagpu '
                         'backend must be float16, 32 or 64')
            raise ValueError

        super(NervanaCPU, self).__init__(rng_seed, default_dtype, compat_mode=compat_mode)

        # ensure an optimized BLAS is present and warn if not
        try:
            if not any(x in str(np.__config__.blas_opt_info['libraries']).lower()
                       for x in ['openblas', 'atlas', 'mkl', 'accelerate']):
                logger.warn("No accelerated BLAS libraries found, CPU "
                            "performance may suffer.  Consider installing "
                            "one of openblas, Atlas, MKL, or vecLib")
        except (AttributeError, KeyError):
            logger.warn("Problems inferring BLAS info, CPU performance may "
                        "be suboptimal")

        self.device_type = 0
        self.device_id = 0
        self.tensor_cls = CPUTensor

        logger.info("Initialized NervanaCPU")

        self.hist_bins, self.hist_offset = None, None
        self.set_hist_buffers(hist_bins, hist_offset)

        self.use_pinned_mem = False

    def consume(self, buf_index, hostlist, devlist):
        assert 0 <= buf_index < 2, 'Can only double buffer'

        if devlist[buf_index] is None:
            devlist[buf_index] = self.empty_like(
                hostlist[buf_index].T, dtype=hostlist[buf_index].dtype
            )

        devlist[buf_index][:] = hostlist[buf_index].T

    def set_hist_buffers(self, hist_bins, hist_offset):
        if (hist_bins != self.hist_bins or hist_offset != self.hist_offset):
            self.hist_bins = hist_bins
            self.hist_offset = hist_offset
            self.hist_max = 4096
            self.hist_buf = self.empty((self.hist_max, hist_bins), dtype=np.int32)
            self.hist_idx = 0
            self.hist_map = dict()

    def gen_rng(self, seed=None):
        """
        Generate the random number generator on host.

        Arguments:
            seed (int): random number generator seed

        Returns:
            seeded numpy RNG
        """
        self.rng = np.random.RandomState(seed)
        self.init_rng_state = self.rng_get_state()
        return self.rng

    def rng_set_state(self, state):
        """
        Set the RNG state for host RNG.

        Arguments:
            state (np.array): numpy random number state vector
        """
        self.rng.set_state(state)

    def rng_get_state(self):
        """
        Return the current state of the on-host RNG.

        Returns:
            np.array: the on-host RNG state vectors
        """
        return self.rng.get_state()

    def rng_reset(self):
        """
        Reset the random state to the state where the Backend is first
        initialized.
        """
        self.rng_set_state(self.init_rng_state)

    def fill_normal(self, ary, mean=0, stdv=1):
        """
        Fill ary with normally distributed random numbers.

        Arguments:
            ary (Tensor): Tensor to fill with random values
            mean (float): Mean value. Default 0
            stdv (float): standard deviation value.  Default 1
        """
        ary[:] = np.random.standard_normal(ary.shape) * stdv + mean

    def execute(self, optree, numpy_call_dict=numpy_call_dict_cpu):
        """
        Execute the optree. Break optree into sub-optrees if necessary.

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
            array_output = optree[1]._tensor

            # get the output shape and onehot representation length will be on
            # this axis
            numpy_axis = optree[2][0]['axis']
            numpy_ind0 = optree[2][0]['idx']._tensor.squeeze()

            numpy_ind_len = numpy_ind0.size
            numpy_ind1 = list(range(numpy_ind_len))

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

        try:
            ary = np.zeros(shape, dtype)
        except ValueError:
            raise ValueError(
                'Invalid shape or dtype. shape: {shape} dtype: {dtype}'.format(
                    shape=shape,
                    dtype=dtype,
                )
            )

        return self.tensor_cls(
            backend=self,
            ary=ary,
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
        C = alpha * A * B.T + beta * C.

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
        1 For fprop: A(K, C), B(X,C,N), C(X,K,N) --> call batched_dot(A, B, C)
        2 For bprop: A(K, C), B(X,K,N), C(X,C,N) --> call batched_dot(A.T, B, C)
        3 For update: A(X,K,N), B(X,C,N), C(K,C) --> call batched_dot(A, B.T, C).

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

    def xnor_compound_dot(self, A, B, C, beta=0.0, bsum=None):
        """
        Performs XNOR GEMM
        C = A * B

        Arguments:
            A (Tensor): left-hand side operand.
            B (Tensor): right-hand side operand.
            C (Tensor): output operand
        """

        # checking type and shape
        assert A.dtype == B.dtype == C.dtype

        assert A.shape[0] == C.shape[0]
        assert B.shape[1] == C.shape[1]
        assert A.shape[1] == B.shape[0]

        np.dot(A._tensor, B._tensor, C._tensor)

        if bsum is not None:
            bsum[:] = self.sum(C, 1)

        return C

    def copy_transpose(self, a, out, axes=None, repeat=1):
        """
        Function to perform a fast copy transpose/dimshuffle operation.
        Works just like numpy.transpose, but requires an output tensor argument.
        """
        out._tensor[:] = np.transpose(a._tensor, axes).copy()

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
        return ConvLayer(self, dtype, N, C, K, D, H, W, T, R, S,
                         pad_d, pad_h, pad_w, str_d, str_h, str_w,
                         dil_d, dil_h, dil_w)

    def fprop_conv(self, layer, I, F, O,
                   X=None, bias=None, bsum=None,
                   alpha=1.0, beta=0.0,
                   relu=False, brelu=False, slope=0.0, layer_op=None):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output.

        Arguments:
            layer: the conv layer as a parameter object
            I (CPUTensor): inputs
            F (CPUTensor): the weights (filters)
            O (CPUTensor): outputs

        Compounding Options:
            X: tensor to use in bprop_relu or beta
                can be same as O for beta accumulate (this is default when None)
                should be same shape as O
            bias: (K,1) tensor to use for adding bias to output
                O += bias
            bsum: (K,1) tensor to accumulate batch sum over (used in batchnorm or bprop_bias)
                bsum = sum(O.reshape(K,-1), axis=1)
                the sum operation is fully deterministic
            alpha, beta:
                O = alpha*O + beta*X
                O = alpha*O + beta*O   (if X==O)
            relu, slope: boolean flag to apply:
                O = max(O, 0) + beta*min(O, 0)
                can be combined with bias (where bias is added first)
            brelu, slope: boolean flag to apply:
                O *= (X > 0) + beta*(X < 0)
                can be combined with bsum tensor to output bprop_bias
        """
        layer.xprop_conv(I, F, O, X, bias, bsum, alpha,
                         beta, relu, brelu, slope, layer_op=layer)

    def bprop_conv(self, layer, F, E, grad_I,
                   X=None, bias=None, bsum=None,
                   alpha=1.0, beta=0.0,
                   relu=False, brelu=False, slope=0.0, layer_op=None):
        """
        Backward propagate the error through a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            F (CPUTensor): the weights (filters)
            E (CPUTensor): errors
            grad_I (CPUTensor): gradient to inputs (output delta)

        Compounding Options:
            X: tensor to use in bprop_relu or beta
                can be same as grad_I for beta accumulate (this is default when None)
                should be same shape as grad_I
            bias: (K,1) tensor to use for adding bias to output
                grad_I += bias
            bsum: (K,1) tensor to accumulate batch sum over (used in batchnorm or bprop_bias)
                bsum = sum(grad_I.reshape(K,-1), axis=1)
                the sum operation is fully deterministic
            alpha, beta:
                grad_I = alpha*grad_I + beta*X
                grad_I = alpha*grad_I + beta*grad_I   (if X==grad_I)
            relu, slope: boolean flag to apply:
                grad_I = max(grad_I, 0) + slope*min(grad_I, 0)
                can be combined with bias (where bias is added first)
            brelu, slope: boolean flag to apply:
                grad_I *= (X > 0) + slope*(X < 0)
                can be combined with bsum tensor to output bprop_bias
        """
        layer.xprop_conv(E, F, grad_I, X, bias, bsum, alpha, beta, relu, brelu, slope,
                         backward=True, layer_op=layer)

    def update_conv(self, layer, I, E, U, alpha=1.0, beta=0.0, grad_bias=None, layer_op=None):
        """
        Compute the updated gradient for a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            I (CPUTensor): the inputs
            E (CPUTensor): the errors
            U (CPUTensor): the updates
            alpha (float): linear scaling
            beta  (float): scaled accumulation
        """
        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeF == U.size
        layer.update_conv(I, E, U, alpha, beta, grad_bias=grad_bias, layer_op=layer_op)

    def deconv_layer(self, dtype,
                     N, C, K,
                     M, P, Q,
                     T=1, R=1, S=1,
                     pad_d=0, pad_h=0, pad_w=0,
                     str_d=1, str_h=1, str_w=1,
                     dil_d=1, dil_h=1, dil_w=1
                     ):
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
        return DeconvLayer(self, dtype, N, C, K, M, P, Q, T, R, S,
                           pad_d, pad_h, pad_w, str_d, str_h, str_w,
                           dil_d, dil_h, dil_w)

    def lrn_layer(self, dtype, N, C, D=1, H=1, W=1, J=1):
        """
        Create a new PoolLayer parameter object.
        This then is passed as an argument to all pooling kernels.

        N: Number of images in mini-batch

        C: Number of input feature maps
        H: Height of input image
        W: Width  of input image

        J: Size of feature map pooling window (maxout n_pieces)

        padding: amount of zero-padding around the given image or feature map edge
        strides: factor to step the window by in a given direction (overlap allowed)

        Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.
        """
        assert J % 2 == 1, "Only support odd LRN window size"
        pad_c = J // 2
        op = 'lrn'
        # Bunch of defaults since we're only interested in the k-axis
        lrn_opts = dict(T=1, R=1, S=1,
                        pad_c=pad_c,
                        pad_d=0, pad_h=0, pad_w=0,
                        str_c=1, str_d=1, str_h=1, str_w=1)

        return PoolLayer(self, dtype, op, N, C, D, H, W, J, **lrn_opts)

    def fprop_lrn(self, layer, I, O, denom, alpha=None, beta=None, ascale=1, bpower=1):
        """
        Forward propagate pooling layer.

        Arguments:
            layer (PoolLayer): The pool layer object, different backends have
                               different pool layers.
            I (Tensor): Input tensor.
            O (Tensor): output tensor.
            denom (Tensor): denominator tensor, stores the result of the squared pooling/contrast
            ascale (float): scaling parameter (alpha) to multiply the pooled sum (1.25e-5 in AK)
            bpower (float): exponential parameter (beta) to raise denominator by (0.75 in AK)
        """

        assert layer.sizeI == I.size
        assert layer.sizeO == O.size

        J, T, R, S = layer.JTRS
        C, D, H, W, N = layer.dimI
        K, M, P, Q, N = layer.dimO
        pad_c, pad_d, pad_h, pad_w = layer.padding
        str_c, str_d, str_h, str_w = layer.strides

        array_I = I._tensor.reshape(layer.dimI)
        array_O = O._tensor.reshape(layer.dimO)  # _tensor to write to
        # although we can calculate directly into O, keeping denom around is useful for bprop
        array_d = denom._tensor.reshape(layer.dimO)  # _tensor to write to

        for k in range(K):
            sliceC, _ = layer.kSlice[k]
            _ascale = ascale / J
            for m in range(M):
                sliceD, _ = layer.mSlice[m]
                for p in range(P):
                    sliceH, _ = layer.pSlice[p]
                    for q in range(Q):
                        sliceW, _ = layer.qSlice[q]
                        sliceI = array_I[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        array_d[k, m, p, q, :] = 1 + _ascale * np.sum(np.square(sliceI), axis=0)

        array_O[:] = array_I * np.power(array_d, -bpower)  # elementwise divide by denominator

    def bprop_lrn(self, layer, I, O, E, delta, denom, alpha=None, beta=None, ascale=1, bpower=1):
        """
        Backward propagate pooling layer.

        Arguments:
            layer (PoolLayer): The pool layer object. Different backends have
                               different pool layers.
            I (Tensor): Input tensor.
            E (Tensor): Error tensor.
            delta (Tensor): Gradient tensor (delta)
            denom (Tensor): denominator tensor computed during bprop
            ascale (float): scaling parameter (alpha) to multiply the pooled sum (1.25e-5 in AK)
            bpower (float): exponential parameter (beta) to raise denominator by (0.75 in AK)
        """

        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeI == delta.size

        J, T, R, S = layer.JTRS
        C, D, H, W, N = layer.dimI
        K, M, P, Q, N = layer.dimO
        pad_c, pad_d, pad_h, pad_w = layer.padding
        str_c, str_d, str_h, str_w = layer.strides

        array_I = I._tensor.reshape(layer.dimI)
        array_E = E._tensor.reshape(layer.dimO)
        array_O = O._tensor.reshape(layer.dimO)
        array_delta = delta._tensor.reshape(layer.dimI)  # write to
        array_denom = denom._tensor.reshape(layer.dimO)

        for k in range(K):
            sliceC, _ = layer.kSlice[k]
            for m in range(M):
                sliceD, _ = layer.mSlice[m]
                for p in range(P):
                    sliceH, _ = layer.pSlice[p]
                    for q in range(Q):
                        sliceW, _ = layer.qSlice[q]

                        _O = array_O[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        _E = array_E[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        _den = array_denom[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        # temporarily store part of the derivative in here
                        array_delta[k, m, p, q, :] = np.sum(_O * _E / _den, axis=0)

        array_delta[:] = -2 * bpower * (ascale / float(J)) * array_delta * array_I + (
            array_E * np.power(array_denom, -bpower))

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
        if layer.op == "max":
            assert layer.sizeO == argmax.size
        op = layer.op

        J, T, R, S = layer.JTRS
        C, D, H, W, N = layer.dimI
        K, M, P, Q, N = layer.dimO
        pad_c, pad_d, pad_h, pad_w = layer.padding
        str_c, str_d, str_h, str_w = layer.strides

        array_I = I._tensor.reshape(layer.dimI)
        array_O = O._tensor.reshape(layer.dimO)
        if op == "max":
            array_argmax = argmax._tensor.reshape(layer.dimO)

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
                            array_O[k, m, p, q, :] = array_O[k, m, p, q, :] * beta + \
                                np.max(sliceI, axis=0)
                        elif op == "avg":
                            array_O[k, m, p, q, :] = array_O[k, m, p, q, :] * beta + \
                                np.mean(sliceI, axis=0)
                        elif op == "l2":
                            array_O[k, m, p, q, :] = array_O[k, m, p, q, :] * beta + \
                                np.sqrt(np.sum(np.square(sliceI), axis=0))

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

        array_E = I._tensor.reshape(layer.dimO)
        array_E[:] = array_E * alpha
        array_delta = O._tensor.reshape(layer.dimI)
        array_delta[:] = array_delta * beta
        if op == "max":
            array_argmax = argmax._tensor.reshape(layer.dimO)

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
                            sliceB[max_n, list(range(N))] += array_E[patch_out]
                        elif op == "avg":
                            sliceB += array_E[patch_out] * (1.0 / sliceB.shape[0])
                        else:
                            raise NotImplementedError
                        array_delta[patch_in] = sliceB.reshape((clen, dlen, hlen, wlen, N))

    def _roipooling_slice(self, h, stride, H, roi_offset):
        """
        Slicing for ROIPooling along one dimension.
        h: is the index on the pooled map (output index)
        stride:
        H: the max of the input map
        roi_offset: how far hstart is from 0
        """
        hstart = int(np.floor(float(h) * stride))
        hend = int(np.ceil(float(h + 1) * stride))

        hstart = min(max(hstart + roi_offset, 0), H)
        hend = min(max(hend + roi_offset, 0), H)

        return slice(hstart, hend), hend - hstart

    def roipooling_fprop(self, I, rois, O, argmax, roi_count, C, H, W,
                         pooled_height, pooled_width, spatial_scale):
        """
        Function to perform fprop of ROIPooling

        Arguments:
            I (Tensor): (C, H, W, N)
            rois (Tensor): (ROIs, 5)
            O (Tensor): (C, pooled_height, pooled_width, roi_count)
            argmax (Tensor): (C, pooled_height, pooled_width, roi_count)
        """
        assert I.size == C * H * W * self.bsz,\
            "ROIPooling input feature map size do not match"
        assert O.size == argmax.size == C * pooled_height * pooled_width * roi_count,\
            "ROIPooling output shape do not match"

        assert rois.shape[1] == 5, "ROIs should be on the row dimension"
        assert rois.shape[0] == roi_count, "ROIs do not match with roi count"

        array_fm = I._tensor.reshape(C, H, W, self.bsz)
        array_rois = rois._tensor
        array_O = O._tensor.reshape(C, pooled_height, pooled_width, roi_count)

        array_argmax = argmax._tensor.reshape(C, pooled_height, pooled_width, roi_count)
        array_O[:] = 0
        array_argmax[:] = -1

        # combine the feature map with ROIs
        for b_id in xrange(roi_count):
            [idx, xmin, ymin, xmax, ymax] = array_rois[b_id]
            xmin = int(round(xmin * spatial_scale))
            xmax = int(round(xmax * spatial_scale))
            ymin = int(round(ymin * spatial_scale))
            ymax = int(round(ymax * spatial_scale))
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)

            stride_h = float(roi_height) / float(pooled_height)
            stride_w = float(roi_width) / float(pooled_width)

            for h_out in xrange(pooled_height):
                sliceh, lenh = self._roipooling_slice(h_out, stride_h, H, ymin)
                if sliceh.stop <= sliceh.start:
                    continue
                for w_out in xrange(pooled_width):
                    slicew, lenw = self._roipooling_slice(w_out, stride_w, W, xmin)
                    if slicew.stop <= slicew.start:
                        continue
                    else:
                        array_I = array_fm[:, sliceh, slicew, int(idx)].reshape(C, -1)
                        array_O[:, h_out, w_out, b_id] = np.max(array_I, axis=1)

                        # get the max idx respect to feature_maps coordinates
                        max_idx_slice = np.unravel_index(np.argmax(array_I, axis=1), (lenh, lenw))
                        max_idx_slice_h = max_idx_slice[0] + sliceh.start
                        max_idx_slice_w = max_idx_slice[1] + slicew.start
                        max_idx_slice = max_idx_slice_h * W + max_idx_slice_w
                        array_argmax[:, h_out, w_out, b_id] = max_idx_slice

    def roipooling_bprop(self, I, rois, O, argmax, roi_count, C, H, W,
                         pooled_height, pooled_width, spatial_scale):
        """
        Function to perform bprop of ROIPooling.

        Arguments:
            I (Tensor): input errors (C, pooled_height, pooled_width, roi_count)
            argmax (Tensor): max args from the fprp (C, pooled_height, pooled_width, roi_count)
            rois (Tensor): (ROIs, 5)
            O (Tensor): output deltas (C, H, W, N)
        """
        assert I.size == argmax.size == C * pooled_height * pooled_width * roi_count,\
            "ROIPooling bprop input size do not match"
        assert O.size == C * H * W * self.bsz,\
            "ROIPooling bprop output size do not match"

        assert rois.shape[1] == 5, "ROIs should be on the row dimension"
        assert rois.shape[0] == roi_count, "ROIs do not match with roi count"

        array_E = I._tensor.reshape(C, pooled_height, pooled_width, roi_count)
        array_rois = rois._tensor
        array_delta = O._tensor.reshape(C, H, W, self.bsz)
        array_argmax = argmax._tensor.reshape(C, pooled_height, pooled_width, roi_count)
        array_delta[:] = 0

        for b_id in xrange(roi_count):
            [idx, xmin, ymin, xmax, ymax] = array_rois[b_id]
            xmin = int(round(xmin * spatial_scale))
            xmax = int(round(xmax * spatial_scale))
            ymin = int(round(ymin * spatial_scale))
            ymax = int(round(ymax * spatial_scale))
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)

            stride_h = float(roi_height) / float(pooled_height)
            stride_w = float(roi_width) / float(pooled_width)

            # iterate all the w, h (from feature map) that fall into this ROIs
            for w in range(xmin, xmax + 1):
                for h in range(ymin, ymax + 1):
                    phstart = int(np.floor(float(h - ymin) / stride_h))
                    phend = int(np.ceil(float(h - ymin + 1) / stride_h))
                    pwstart = int(np.floor(float(w - xmin) / stride_w))
                    pwend = int(np.ceil(float(w - xmin + 1) / stride_w))

                    phstart = min(max(phstart, 0), pooled_height)
                    phend = min(max(phend, 0), pooled_height)
                    pwstart = min(max(pwstart, 0), pooled_width)
                    pwend = min(max(pwend, 0), pooled_width)

                    for ph in range(phstart, phend):
                        for pw in range(pwstart, pwend):
                            max_idx_tmp = array_argmax[:, ph, pw, b_id]
                            for c in range(C):
                                if max_idx_tmp[c] == (h * W + w):
                                    array_delta[c, h, w, int(idx)] += array_E[c, ph, pw, b_id]

    def nms(self, detections, threshold, normalized=False):
        """
        Function to perform non-maximal supression.

        Arguments:
            detections (Tensor): detection boxes (box_count, 5), each row has
                                 (x1, y1, x2, y2, score). Assume the boxes have already
                                 been sorted based on score in descending order
            output_mask (Tensor): pre-allocated buffer for mask output from the kernel
            threshold (float): box overlap threshold, boxes with smaller overlaps will be kept
            normalized (bool): whether box coordinates are normalized to image dimensions

        Outputs:
            keep_ind (list): list of indices
        """
        # for boxes in pixel space, we calculate size as x_max - x_min+1. However,
        # when the boxes are normalized to be between 0 and 1, we calculate
        # the size as x_max - x_min. offset controls this behavior.
        if normalized is True:
            offset = 0
        else:
            offset = 1

        dets = detections.get()

        # remove zero score entries
        keep = np.where(dets[:, 4] != 0)[0]

        x1 = dets[keep, 0]
        y1 = dets[keep, 1]
        x2 = dets[keep, 2]
        y2 = dets[keep, 3]
        scores = dets[keep, 4]

        areas = (x2 - x1 + offset) * (y2 - y1 + offset)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + offset)
            h = np.maximum(0.0, yy2 - yy1 + offset)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]

            order = order[inds + 1]

        return keep

    def compound_fprop_bn(self, x, xsum, xvar, gmean, gvar, gamma, beta, y,
                          eps, rho, compute_batch_sum, accumbeta=0.0, relu=False,
                          binary=False, inference=False, outputs=None, layer=None):
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
            beta (Tensor): location parameter
            y (Tensor): normalized output
            eps (float): constant for numerical stability
            rho (float): exponential window averaging constant
        """
        if inference:
            xhat = (x - gmean) / self.sqrt(gvar + eps)  # Op-tree only
            y[:] = y * accumbeta + xhat * gamma + beta
            return

        if compute_batch_sum:
            xsum[:] = self.sum(x, axis=1)

        xvar[:] = self.var(x, axis=1, binary=binary)
        xsum[:] = xsum / x.shape[1]  # reuse xsum instead of computing xmean

        gmean[:] = gmean * rho + (1.0 - rho) * xsum
        gvar[:] = gvar * rho + (1.0 - rho) * xvar

        if binary:
            xhat = self.shift(x - xsum, 1.0 / self.sqrt(xvar + eps))
            outputs = y.reshape(xhat.shape)
            outputs[:] = self.shift(xhat, gamma) + beta + accumbeta * outputs
        else:
            xhat = (x - xsum) / self.sqrt(xvar + eps)
            outputs = y.reshape(xhat.shape)
            outputs[:] = xhat * gamma + beta + accumbeta * outputs

    def compound_bprop_bn(self, delta_out, grad_gamma, grad_beta, delta_in, x, xsum, xvar,
                          gamma, eps, binary=False, layer=None):
        """
        Function to perform batch normalization backward pass. Included
        for API compatibility with GPU compound kernel call.

        Arguments:
            delta_out (Tensor): Delta buffer to write out to
            grad_gamma (Tensor): Gradient w.r.t. gamma
            grad_beta (Tensor): Gradient w.r.t. beta
            delta_in (Tensor): Delta buffer to read from (incoming errors)
            x (Tensor): feedforward input
            xsum (Tensor): Batch sum over PQN dimension
            xvar (Tensor): Batch variance
            gamma (Tensor): scale parameter
            eps (float): constant for numerical stability
            binary (bool): Binary shift based computations
        """
        if binary:
            op = self.shift
        else:
            def multiply(left, right):
                return left * right
            op = multiply

        inv_v = 1.0 / self.sqrt(xvar + eps)
        xhat = op(x - xsum, inv_v)
        grad_gamma[:] = self.sum(xhat * delta_in, axis=1)
        grad_beta[:] = self.sum(delta_in, axis=1)
        xtmp = (op(xhat, grad_gamma) + grad_beta) / float(x.shape[1])
        delta_out.reshape(delta_in.shape)[:] = op(op(delta_in - xtmp, gamma), inv_v)

    def compound_bprop_lut(self, nin, inputs, error, error_t, dW, pad_idx, alpha=1.0, beta=0):
        """
        Backward propagate lookup table layer.

        Arguments:
            nin (int): Number of input word_ids.
            inputs (Tensor): Input tensor.
            error (Tensor): Error tensor.
            error_t (Tensor): Transposed error tensor.
            dW (Tensor): Gradient tensor (delta).
            pad_idx (int):
            alpha (float):
            beta (float):
        """
        wrd_ids = inputs._tensor[0]
        unqidx, inv = np.unique(wrd_ids, return_inverse=True)
        groups = [np.where(inv == i) for i in range(len(unqidx))]

        for (wrd_id, group) in zip(unqidx, groups):
            if wrd_id != pad_idx:
                dW[wrd_id, :] = self.sum(error.take(group[0], axis=1), axis=1)
        """
        alternative bprop
        for (j, wrd_id) in enumerate(wrd_ids):
            dW[:, wrd_id] = dW[:, wrd_id] + error[:, j]
        """

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
        Calculates the ReLu transformation for input array.

        Arguments:
            ary: numpy array
            out: reference to output
        """
        if out is not None:
            return np.maximum(ary, 0, out)
        else:
            return np.maximum(ary, 0)

    def binarize(self, ary, out, stochastic=True):
        """
        Binarizes input array

        Arguments:
            ary: tensor
            out: reference to output
            stochastic: stochastic or deterministic
        """
        if stochastic:
            out[:] = (ary + 1)/2.0
            self.clip(out, 0, 1, out)
            prob = self.array(np.random.uniform(0, 1, size=ary.shape))
            self.less_equal(prob, out, out)
        else:
            self.greater_equal(ary, 0, out)
        out[:] = 2 * out - 1
        return out

    def shift(self, ary, shift_ary, value=True, out=None):
        """
        Shifts input array

        Arguments:
            ary: tensor
            shift_ary: tensor of shift amount
            out: reference to output
        """
        if value:
            exp = self.rint(self.safelog(self.absolute(shift_ary))/self.log(2))
            ap2 = self.multiply(self.sgn(shift_ary), self.exp2(exp))
        else:
            ap2 = self.exp2(shift_ary)

        if out is None:
            if hasattr(ary, 'shape'):
                out = self.empty_like(ary)
            else:
                out = self.empty((1, 1))
        out[:] = self.multiply(ary, ap2)
        return out

    def init_mark(self):
        """
        Generate a timing mark object.

        Returns:
            timing mark (dict)
        """
        return {'time': 0}

    def record_mark(self, marker):
        """
        Mark the current time.

        Arguments:
            marker (time mark): timing mark generated by init_mark()
        """
        marker['time'] = time.time()

    def synchronize_mark(self, marker):
        """
        Synchronize on the given marker.

        Arguments:
            marker (time mark): timing mark generated by init_mark()
        """
        # No-op on cpu
        return

    def get_time(self, start, end):
        """
        Return time between start and end marks.

        Arguments:
            start (time maker): start time mark

            end (time marker): end time mark

        Returns:
            time elapsed between start and end time marks in milliseconds
        """
        return (end['time'] - start['time']) * 1000.0

    def relu_layer(self):
        return None

    def fprop_relu(self, layer, x, slope):
        return self.maximum(x, 0) + slope * self.minimum(0, x)

    def bprop_relu(self, layer, x, error, deltas, slope):
        return self.greater(x, 0) + slope * self.less(x, 0)

    def fprop_softmax(self, x, axis):
        return (self.reciprocal(self.sum(
                self.exp(x - self.max(x, axis=axis)), axis=axis)) *
                self.exp(x - self.max(x, axis=axis)))

    def batchnorm_layer(self, in_shape):
        return None

    def fprop_transform(self, ngLayer, transform, inputs, outputs, relu=False):
        outputs[:] = transform(inputs)

    def bprop_transform(self, ngLayer, transform, outputs, error, deltas, relu):
        deltas[:] = transform.bprop(outputs) * error

    def fprop_skipnode(self, x, y, beta):
        y[:] = y * beta + x

    def bprop_skipnode(self, error, deltas, alpha, beta):
        deltas[:] = deltas * beta + alpha * error

    def mergesum_layer(self, layer_num):
        return None

    def fprop_mergesum(self, ngLayer, inputs, inference, layers, outputs, out_shape):
        for l in layers:
            beta = 0 if l is layers[0] else 1
            l.fprop(inputs, inference, beta=beta)

    def bprop_mergesum(self, ngLayer, alpha, beta, layers, error, deltas):
        for l in reversed(layers):
            b = beta if l is layers[-1] else 1
            l.bprop(error, alpha=alpha, beta=b)

    def mergebroadcast_layer(self, layer_num):
        return None

    def fprop_mergebroadcast(self, ngLayer, inputs, inference, outputs, layers, out_shape):
        for l in layers:
            l.fprop(inputs, inference)

    def bprop_mergebroadcast(self, ngLayer, layers, error_views, error,
                             delta, out_shape, alpha, beta, alphas, betas):
        betas[-1] = beta
        for l, e, a, b in reversed(list(zip(layers, error_views, alphas, betas))):
            l.bprop(e, alpha=a * alpha, beta=b)
