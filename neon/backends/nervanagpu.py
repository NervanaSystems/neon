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
Our GPU based backend interface and tensor data structure.
"""

import sys
import numpy as np
import pycuda.driver as drv
import logging
from pycuda.tools import context_dependent_memoize
from struct import unpack_from
from pytools import memoize_method
from functools import wraps
from math import log

from neon.backends import kernel_specs
from neon.backends.backend import Tensor, Backend, OpTreeNode, OpCollection
from neon.backends.layer_gpu import ConvLayer, DeconvLayer, PoolLayer, _get_sm_count
from neon.backends.kernels.cuda import pooling

_none_slice = slice(None, None, None)

logger = logging.getLogger(__name__)


class GPUTensor(Tensor):

    """
    The n-dimensional array data structure that resides in GPU memory,
    and is meant to be manipulated on the GPU.

    Arguments:
        dtype (numpy.ndtype, optional): Underlying data type of the elements.
        allocator (function, optional): Memory allocator.
        base (GPUTensor, optional): The base of the tensor. A tensor can have
                                    different views, this keep tracks of the
                                    original tensor.
        gpudata (pycuda._driver.DeviceAllocation, optional): The actual gpu
                                                             memory that stores
                                                             the tensor.
        strides (tuple, optional): Tuple of bytes to step in each dimension when traversing an
                                   array.
        take_array: The indices of the values to extract.
        is_trans (bool, optional): Whether the tensor is transposed or not.
        rounding (int, optional): Set to desired number of mantissa bits to
                                  stochasicaly round, to set to zero to disable
                                  stochastic rouding.

    See also:
        NervanaGPU class

    Notes:
        Unlike numpy, in this implementation we never collapse dimensions, and
        the minimal number of dimensions will be _min_dims (currently set to 2
        to match cudanet GPU implementation).  So a wrapped scalar will have
        dimension 1x1.
    """

    def __init__(self,
                 backend,
                 shape=None,
                 dtype=np.float32,
                 name=None,
                 persist_values=True,
                 allocator=drv.mem_alloc,
                 base=None,
                 gpudata=None,
                 strides=None,
                 take_array=None,
                 is_trans=False,
                 rounding=0):

        super(GPUTensor, self).__init__(backend, shape, dtype, name,
                                        persist_values)

        # supported dtypes
        assert dtype in (np.float16, np.float32, np.uint8, np.int8, np.uint16,
                         np.int16, np.uint32, np.int32)

        dtype = np.dtype(dtype)

        if isinstance(shape, (tuple, list)) and len(shape) < self._min_dims:
            shape = shape + (1, )

        try:
            size = 1
            for dim in shape:
                size *= dim
        except TypeError:
            assert isinstance(shape, (int, long, np.integer))
            size = shape
            shape = (shape, 1)

        if isinstance(size, np.integer):
            size = np.asscalar(size)

        # only support C ordering for now.
        if strides is None:
            self.strides = _contiguous_strides(shape)
        else:
            self.strides = tuple(strides)

        self.base = base
        self.shape = shape
        self.size = size
        self.dtype = dtype
        self.nbytes = dtype.itemsize * size
        self.allocator = allocator
        self.take_array = take_array
        self.is_trans = is_trans
        self.rounding = rounding
        self.kahan_count = 0
        self.kahan_reset = 0

        if gpudata is None:
            # print "allocate!"
            if size:
                # print(drv.mem_get_info())
                self.gpudata = allocator(self.nbytes)
            else:
                self.gpudata = None

            assert base is None
        else:
            self.gpudata = gpudata

    def __str__(self):
        """
        Returns a string representation of this Tensor.

        Returns:
            str: the representation.
        """
        return ("GPUTensor(base 0x%x) name:%s shape:%s dtype:%s strides:%s "
                "is_trans:%s is_contiguous:%s" % (self.gpudata, self.name,
                                                  self.shape, self.dtype,
                                                  self.strides, self.is_trans,
                                                  self.is_contiguous))

    def __repr__(self):
        """
        Returns a more unambiguous string representation of the Tensor.

        Returns:
            str: The representation.
        """
        return self.__str__()

    def __len__(self):
        """
        Returns the size of the leading dimension of self.

        Returns:
            int: The size of the leading dimension.
        """
        if len(self.shape):
            return self.shape[0]
        else:
            return 0

    def __setitem__(self, index, value):

        self.__getitem__(index)._assign(value)

    def __getitem__(self, index):
        """
        Return a sliced view of an array
        """
        if not isinstance(index, tuple):
            # speed up common case of [:]
            if index == _none_slice:
                return self
            index = (index,)

        new_shape = []
        new_offset = 0
        new_strides = []

        seen_ellipsis = False
        take_array = None

        index_axis = 0
        array_axis = 0

        while index_axis < len(index):

            index_entry = index[index_axis]

            if array_axis > len(self.shape):
                raise IndexError("too many axes in index")

            # Standard slicing (start:stop:step)
            if isinstance(index_entry, slice):
                start, stop, idx_strides = index_entry.indices(
                    self.shape[array_axis])

                array_strides = self.strides[array_axis]

                # def ceil_div(x, y): return -(-x // y)
                new_shape.append(-((start - stop) // idx_strides))
                new_strides.append(idx_strides * array_strides)
                new_offset += array_strides * start * self.dtype.itemsize

                index_axis += 1
                array_axis += 1

            # Fancy indexing
            elif isinstance(index_entry, (GPUTensor, np.ndarray, list, tuple)):

                if isinstance(index_entry, (list, tuple)):
                    index_entry = np.array(index_entry, dtype=np.int32)

                if isinstance(index_entry, np.ndarray):
                    index_entry = self.__class__(
                        self.backend, index_entry.shape, dtype=np.int32).set(index_entry)

                size = max(index_entry.shape)
                if size != index_entry.size:
                    raise IndexError(
                        "Fancy indexing only currently supported dim > 1 in a single dimension.")

                if take_array is not None:
                    raise IndexError(
                        "Fancy indexing only currently supported one axis at a time.")

                if index_entry.dtype.type is not np.int32:
                    # TODO: this should now work for all int types, but need to
                    # test
                    raise IndexError(
                        "Fancy indexing only currently supported with int32 types.")

                take_array = (index_entry, array_axis)

                new_shape.append(size)
                new_strides.append(self.strides[array_axis])

                index_axis += 1
                array_axis += 1

            elif isinstance(index_entry, (int, np.integer)):
                array_shape = self.shape[array_axis]
                if index_entry < 0:
                    index_entry += array_shape

                if not (0 <= index_entry < array_shape):
                    raise IndexError(
                        "subindex in axis %d out of range" % index_axis)

                new_offset += self.strides[array_axis] * \
                    index_entry * self.dtype.itemsize

                if len(self.shape) < 3:
                    new_shape.append(1)
                    new_strides.append(self.strides[array_axis])

                index_axis += 1
                array_axis += 1

            elif index_entry is Ellipsis:
                index_axis += 1

                remaining_index_count = len(index) - index_axis
                new_array_axis = len(self.shape) - remaining_index_count
                if new_array_axis < array_axis:
                    raise IndexError("invalid use of ellipsis in index")
                while array_axis < new_array_axis:
                    new_shape.append(self.shape[array_axis])
                    new_strides.append(self.strides[array_axis])
                    array_axis += 1

                if seen_ellipsis:
                    raise IndexError(
                        "more than one ellipsis not allowed in index")
                seen_ellipsis = True

            else:
                raise IndexError("invalid subindex in axis %d" % index_axis)

        while array_axis < len(self.shape):
            new_shape.append(self.shape[array_axis])
            new_strides.append(self.strides[array_axis])

            array_axis += 1

        return self.__class__(
            backend=self.backend,
            shape=tuple(new_shape),
            dtype=self.dtype,
            allocator=self.allocator,
            base=self,
            gpudata=int(self.gpudata) + new_offset,
            strides=new_strides,
            take_array=take_array,
            name=self.name,
            rounding=self.rounding)

    def __int__(self):
        """
        Returns an integer representation of the underlying gpu memory buffer.

        Returns:
            int: The int representation
        """
        return int(self.gpudata)

    def _assign(self, value):
        """
        Assign value to the tensor.

        Arguments:
            value (int, float, GPUTensor, OpTreeNode): The value to be assigned.
        """

        stream = self.backend.stream
        if isinstance(value, (int, float)):

            # if we have a contiguous array, then use the speedy driver kernel
            if self.is_contiguous:

                value = self.dtype.type(value)

                if self.dtype.itemsize == 1:
                    drv.memset_d8_async(
                        self.gpudata, unpack_from('B', value)[0], self.size, stream)
                elif self.dtype.itemsize == 2:
                    drv.memset_d16_async(
                        self.gpudata, unpack_from('H', value)[0], self.size, stream)
                else:
                    drv.memset_d32_async(
                        self.gpudata, unpack_from('I', value)[0], self.size, stream)

            # otherwise use our copy kerel
            else:
                OpTreeNode.build("assign", self, value)

        elif isinstance(value, GPUTensor):
            # TODO: add an is_binary_compat like function
            if self.is_contiguous and value.is_contiguous and self.dtype == value.dtype:
                drv.memcpy_dtod_async(
                    self.gpudata, value.gpudata, self.nbytes, stream)
            else:
                OpTreeNode.build("assign", self, value)

        # collapse and execute an op tree as a kernel
        elif isinstance(value, OpTreeNode):
            OpTreeNode.build("assign", self, value)

        # assign to numpy array (same as set())
        elif isinstance(value, np.ndarray):
            self.set(value)

        else:
            raise TypeError("Invalid type for assignment: %s" % type(value))

        return self

    def set(self, ary):
        """
        Copy host array to device.

        Arguments:
            ary: host array, needs to be contiguous

        Returns:
            GPUTensor: self
        """
        stream = self.backend.stream
        assert ary.size == self.size
        assert self.is_contiguous, "Array in set() must be contiguous"
        if ary.dtype is not self.dtype:
            ary = ary.astype(self.dtype)
        if ary.ndim < self._min_dims:
            ary = ary.reshape(ary.size, 1)
        assert ary.strides == tuple(
            self.dtype.itemsize * s for s in self.strides)

        drv.memcpy_htod_async(self.gpudata, ary, stream)

        return self

    def get(self, stream=None):
        """
        Copy device array to host.

        Returns:
            numpy.ndarray: A host numpy array
        """

        if self.is_contiguous:
            ary = np.empty(self.shape, self.dtype)
            drv.memcpy_dtoh_async(ary, self.gpudata, stream)
        else:
            # if it is not contiguous, need to copy it over to new device mem
            ary_d = self.backend.empty(self.shape, self.dtype)
            ary_d.copy(self)
            ary = np.empty(self.shape, self.dtype)
            drv.memcpy_dtoh_async(ary, ary_d.gpudata, stream)
        return ary

    def asnumpyarray(self):
        """
        asnumpyarray is an alias of get(), needed for MOP compatibility

        Returns:
            numpy.ndarray: A host numpy array
        """
        return self.get()

    def asbuffer(self):
        """
        asbuffer returns buffer interface to gpu data
        """
        return self.gpudata.as_buffer(self.nbytes)

    def take(self, indices, axis, out=None):
        """
        Take elements from an array along an axis.
        """
        if axis == 1:
            view = self.__getitem__((_none_slice, indices))
        else:
            view = self.__getitem__((indices, _none_slice))

        if out:
            return out._assign(view)
        return view

    def fill(self, value):
        return self._assign(value)

    def copy(self, a):
        return self._assign(a)

    def copy_from(self, a):
        """ alias of copy"""
        return self.set(a)

    def reshape(self, *shape):
        """
        return a reshaped view
        """
        if isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        if len(shape) < self._min_dims:
            shape = shape + (1, )

        if -1 in shape:
            missing_dim = -self.size / np.prod(shape)
            shape = tuple([missing_dim if x == -1 else x for x in shape])

        if shape == self.shape:
            return self

        size = np.prod(shape)

        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        if not self.is_contiguous:
            raise TypeError("reshaping of non-contiguous arrays is not yet supported")

        return self.__class__(
            backend=self.backend,
            shape=shape,
            dtype=self.dtype,
            allocator=self.allocator,
            base=self,
            gpudata=self.gpudata,
            strides=_contiguous_strides(shape),
            name=self.name,
            rounding=self.rounding)

    @property
    def T(self):
        """
        return a transposed view
        """
        if len(self.shape) <= 2:
            shape = self.shape[::-1]
            strides = self.strides[::-1]
        else:
            # support for batched dot.
            # perserve outer dimension but reverse inner dims
            shape = list(self.shape[::-1])
            strides = list(self.strides[::-1])
            shape = tuple(shape[-1:] + shape[:-1])
            strides = tuple(strides[-1:] + strides[:-1])

        return self.__class__(
            backend=self.backend,
            shape=shape,
            dtype=self.dtype,
            allocator=self.allocator,
            base=self,
            gpudata=self.gpudata,
            strides=strides,
            is_trans=not self.is_trans,
            name=self.name,
            rounding=self.rounding)

    def transpose(self, out=None):
        """
        Return a transposed view of the data.  Alias of .T property needed for
        MOP compatibility.
        """
        if out:
            return OpTreeNode.build("assign", out, self.T)
        return self.T

    def share(self, shape, dtype=None, name=None):
        """
        return a view: ary, where ary.size <= self.size
        Allows easy sharing of temporary memory
        """
        size = np.prod(shape)
        if size > self.size:
            raise ValueError("total size of new array must <= size of parent")

        if not self.is_contiguous:
            raise TypeError("sharing of non-contigous "
                            "arrays is not yet supported")

        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)

        new_base = self if self.base is None else self.base

        return self.__class__(
            backend=self.backend,
            shape=shape,
            dtype=dtype,
            allocator=self.allocator,
            base=new_base,
            gpudata=self.gpudata,
            strides=_contiguous_strides(shape),
            name=name,
            rounding=self.rounding)

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
        from neon.backends.float_ew import _compute_hist
        hist_tensor = self.backend._hist_tensor(tag)
        _compute_hist(self, hist_tensor.gpudata, nbins, offset)
        return hist_tensor

    @property
    def ptr(self):
        """
        Returns an integer representation of the underlying gpu memory buffer.

        Returns:
            int: The int representation
        """
        return self.gpudata.__int__()

    @property
    @memoize_method
    def is_contiguous(self):
        """
        Returns whether the memory of the tensor is contiguous.

        Return
            bool: Whether the memory of the tensor is contiguous.
        """
        return not self.take_array and self.strides == _contiguous_strides(self.shape)


def memoize_stacks(func):
    """
    memoize the stacks using intrinsic_key_maps
    """
    cache = {}

    @wraps(func)
    def memoizer(be, optree):
        optree_key, tensor_index_map, index_tensor_map = optree.intrinsic_key_maps()
        # make sure it's the same backend
        optree_key = (optree_key, id(be))
        if optree_key in cache:
            # replace tensors
            stacks, cached_tensor_index_map = cache[optree_key]
            for stack in stacks:
                for i in range(len(stack)):
                    if isinstance(stack[i], Tensor):
                        if stack[i] in cached_tensor_index_map:
                            stack[i] = index_tensor_map[
                                cached_tensor_index_map[stack[i]]]
            # update the cached_tensor_index_map
            cache[optree_key] = (stacks, tensor_index_map)
        else:
            # cache stacks and tensor_index_map
            # print ('created memoize stack')
            stacks = func(be, optree)
            cache[optree_key] = (stacks, tensor_index_map)
        return stacks

    return memoizer


class NervanaGPU(Backend):
    """
    The primary interface class and factory for GPUTensors

    Arguments:
        stochastic_round (int or bool, optional): set to desired number of mantissa
                                                    bits to stochasically round to.
                                                    Set to 0 or False to disable
                                                    stochastic rounding (the default).
                                                    Set to True to use default
                                                    rounding bit width.
        bench (bool, optional): set to True to print out performance data for
                                    most kernel calls.  If False (default) no
                                    performance data is printed.
        compat_mode (str, optional): set flag to match implementation of other libraries
                                     for compatibility.  currently only 'caffe' is supported

        TODO: define other keyword parameters!
        """

    def __init__(self,
                 rng_seed=None,
                 default_dtype=np.float32,
                 stochastic_round=False,
                 scratch_size=9 * 1024 * 1024,
                 device_id=0,
                 bench=False,
                 hist_bins=64,
                 hist_offset=-48,
                 compat_mode=None):

        if default_dtype not in [np.float16, np.float32]:
            raise ValueError('Default data type for nervanagpu '
                             'backend must be float16 or 32')

        if default_dtype is np.float32:
            if stochastic_round:
                if stochastic_round is True:
                    raise ValueError('Default rounding bit width is not '
                                     'supported for fp32.  Please specify '
                                     'number of bits to round to.')
                logger.warn('Using 32 bit floating point and setting stochastic '
                            'rounding to %d bits' % stochastic_round)

        # super class init
        super(NervanaGPU, self).__init__(rng_seed, default_dtype, compat_mode=compat_mode)

        # context
        drv.init()
        self.device_type = 1
        self.device_id = device_id if device_id is not None else 0
        self.ctx = drv.Device(device_id).make_context()

        # log
        logger.info("Initialized NervanaGPU")

        # stochastic_round
        assert stochastic_round is False, "Are you sure about using SR globally in the backend?"
        if stochastic_round:
            if stochastic_round is True:
                stochastic_round = 10
        else:
            stochastic_round = 0

        # attributes
        self.scratch_size = scratch_size
        self.round_mode = stochastic_round
        self.bench = bench
        self.stream = None
        self.buf = {}
        self.buf_active = {}

        # store histograms for batched memcpy
        self.hist_bins = hist_bins
        self.hist_offset = hist_offset
        self.hist_map = dict()
        self.hist_idx = 0
        self.hist_max = 4*4096
        self.hist_base = drv.mem_alloc(self.hist_bins * self.hist_max * 4)
        drv.memset_d32(self.hist_base, 0, self.hist_bins * self.hist_max)

        # store the rand pool for each context
        self.context_rand_state_map = {}  # stores gpu memory reference
        self.context_rand_state_alive = {}  # set whether randstate is fresh

    def scratch_buffer(self, size, offset=0):

        if offset & 255 != 0:
            offset += 256 - (offset & 255)

        if size + offset > self.scratch_size:
            raise RuntimeError("nervanagpu.scratch_size is too small for this operation.")

        return int(_get_scratch_data(self.scratch_size)) + offset * 4

    def __del__(self):
        try:
            self.ctx.detach()
        except:
            pass

    def rng_reset(self):
        """
        Reset the random state to the state where the Backend is first
        initialized.
        """
        self.rng.set_state(self.init_rng_state)
        for ctx in self.context_rand_state_alive:
            self.context_rand_state_alive[ctx] = False

    def _get_rand_state(self):
        """
        similar to @context_dependent_memoize, with additional ability to reset
        the random pool by `rng_reset`

        initialize our common pool of randomness (1/4 MB):
        MAX_THREADS_PER_MULTIPROCESSOR * 32 SMs (32 to be somewhat future proof
        and power of two). This size is currently hardcoded in the kernels,
        to be parameterized ...
        """
        ctx = drv.Context.get_current()
        if ctx in self.context_rand_state_map and self.context_rand_state_alive[ctx]:
            return self.context_rand_state_map[ctx]
        else:
            # generate random pool from numpy
            rand_init = self.rng.random_integers(
                0, 2 ** 32 - 1, (3 * 2048 * 32, 1)).astype(np.uint32)
            # copy to device
            if ctx in self.context_rand_state_map:
                rand_state = self.context_rand_state_map[ctx]
            else:
                rand_state = drv.mem_alloc(rand_init.nbytes)
                self.context_rand_state_map[ctx] = rand_state
            drv.memcpy_htod(rand_state, rand_init)
            self.context_rand_state_alive[ctx] = True
            return rand_state

    def _buf_malloc(self, shape):
        """
        returns a buffer of size shape, equivalent of calling be.empty(shape)
        """
        # create a list of buffers of the shape
        if shape not in self.buf:
            self.buf[shape] = []
        if shape not in self.buf_active:
            self.buf_active[shape] = []
        # allocate buffer if needed
        if len(self.buf[shape]) == 0:
            self.buf[shape].append(self.empty(shape, dtype=self.default_dtype))
        # get buf and put it in buf_active
        buf = self.buf[shape].pop()
        self.buf_active[shape].append(buf)
        return buf

    def _buf_free(self):
        """
        move all tensors from self.buffer_active to self.buffer
        the idea is to reuse those tensors for other optrees
        """
        for shape in self.buf_active:
            self.buf[shape].extend(self.buf_active[shape])
            self.buf_active[shape] = []

    def _hist_tensor(self, tag):
        """
        Create a tensor the right size for histogram data, with memory allocated
        in the contiguous histogram buffer. Track it by tag for later reference.
        """
        assert self.hist_idx < self.hist_max
        hist_buf = int(self.hist_base) + self.hist_idx * self.hist_bins * 4
        self.hist_map[tag] = (self.hist_idx)
        self.hist_idx += 1
        return GPUTensor(self, shape=(1, self.hist_bins), dtype=np.int32,
                         gpudata=hist_buf, name=tag)

    def dump_hist_data(self):
        hist_data = GPUTensor(self,
                              shape=(self.hist_idx, self.hist_bins),
                              dtype=np.int32,
                              gpudata=int(self.hist_base))
        hist_map = self.hist_map
        self.hist_map = dict()
        self.hist_idx = 0
        return hist_data, hist_map

    @memoize_stacks
    def _split_to_stacks(self, optree):
        """
        split an optree to stacks
        """
        # post-order traversal
        whole_stack = optree.traverse(list())

        # build stages, each stage contains a sub optree
        stages = []
        main_stage = []
        main_stage_axis = []

        # get minority axis for binary operation default, suports axis 0 and 1
        axis_count = [0, 0]
        for s in whole_stack:
            if isinstance(s, dict) and s['op'] in OpCollection.reduction_ops:
                assert s['axis'] == 0 or s['axis'] == 1
                axis_count[s['axis']] += 1
        minority_axis = 0 if axis_count[0] <= axis_count[1] else 1

        # traverse stack and split stages
        for s in whole_stack:
            if isinstance(s, dict):
                if s['op'] == 'dot':
                    # convert left and right child to tensor when it was not
                    right = main_stage.pop()
                    main_stage_axis.pop()  # don't care the value
                    left = main_stage.pop()
                    main_stage_axis.pop()  # don't care the value
                    if isinstance(left, OpTreeNode):
                        left_buf = self._buf_malloc(left.shape)
                        stages.append(OpTreeNode({"op": "assign"}, left_buf,
                                                 left))
                        left = left_buf
                    if isinstance(right, OpTreeNode):
                        right_buf = self._buf_malloc(right.shape)
                        stages.append(OpTreeNode({"op": "assign"}, right_buf,
                                                 right))
                        right = right_buf
                    # buffer to store the result of dot
                    buf = self._buf_malloc((left.shape[0], right.shape[1]))
                    # save to stages
                    stages.append(OpTreeNode({"op": "assign"}, buf,
                                             OpTreeNode(s, left, right)))
                    # push buf to main_stage
                    main_stage.append(buf)
                    main_stage_axis.append(None)
                elif s['op'] == 'transpose':
                    # the object being transposed must be optree here
                    operand = main_stage.pop()
                    main_stage_axis.pop()  # don't care the value
                    # allocate buf for the operand shape
                    buf = self._buf_malloc(operand.shape)
                    # evaluate to buf
                    stages.append(OpTreeNode({"op": "assign"}, buf, operand))
                    # put the buf back to main_stage
                    main_stage.append(buf.T)
                    main_stage_axis.append(None)
                elif s['op'] in OpCollection.reduction_ops:
                    # since 2d reduction is converted
                    assert s['axis'] is not None
                    operand = main_stage.pop()
                    prev_axis = main_stage_axis.pop()
                    if prev_axis is not None and prev_axis != s['axis']:
                        # put everything under previous reduction to buf
                        buf = self._buf_malloc(operand.shape)
                        stages.append(
                            OpTreeNode({"op": "assign"}, buf, operand))
                        # put the buf with current reduction to main stage
                        main_stage.append(OpTreeNode(s, buf, None))
                        main_stage_axis.append(s['axis'])
                    else:
                        # do standary OpCollection.unary_ops
                        main_stage.append(OpTreeNode(s, operand, None))
                        main_stage_axis.append(s['axis'])
                elif s['op'] in OpCollection.unary_ops:
                    # will not run into multiple-axis reduction problem
                    # just pop, build optree and put back
                    operand = main_stage.pop()
                    axis = main_stage_axis.pop()
                    main_stage.append(OpTreeNode(s, operand, None))
                    main_stage_axis.append(axis)  # cancelled out
                elif s['op'] in OpCollection.binary_ops:  # not dot
                    # binary ops might run into multiple-axis reduction
                    right = main_stage.pop()
                    prev_axis_right = main_stage_axis.pop()
                    left = main_stage.pop()
                    prev_axis_left = main_stage_axis.pop()
                    if (prev_axis_right is not None and
                            prev_axis_left is not None and
                            prev_axis_left != prev_axis_right):
                        # do reduction on minority axis
                        if prev_axis_left == minority_axis:
                            buf = self._buf_malloc(left.shape)
                            stages.append(
                                OpTreeNode({"op": "assign"}, buf, left))
                            left = buf
                            axis = prev_axis_right
                        else:
                            buf = self._buf_malloc(right.shape)
                            stages.append(
                                OpTreeNode({"op": "assign"}, buf, right))
                            right = buf
                            axis = prev_axis_left
                        # append to main stage
                        main_stage.append(OpTreeNode(s, left, right))
                        main_stage_axis.append(axis)
                    else:
                        # no multiple-axis reduction, perform standard process
                        main_stage.append(OpTreeNode(s, left, right))
                        axis = None
                        if prev_axis_left is not None:
                            axis = prev_axis_left
                        else:
                            axis = prev_axis_right
                        main_stage_axis.append(axis)
                else:
                    return NotImplemented
            else:
                # tensor or scalars, just push to main_stage
                main_stage.append(s)
                main_stage_axis.append(None)

        # append the the laste stage
        stages.append(main_stage[0])

        # build stacks for call_compound_kernel
        stacks = []
        for stage in stages:
            # now all stages is exact one simple optree
            assert(isinstance(stage, OpTreeNode))
            # create stack
            stacks.append(stage.traverse(list()))

        # free buffer from buf_active to buf, without loosing the reference
        self._buf_free()

        return stacks

    def _is_simple_stack(self, stack):
        """
        TODO move this to _split_to_stacks, deal with memoize better
        TODO add test to this func
        """
        reduction_axes = [False, False]
        for s in stack:
            if isinstance(s, dict):
                if s['op'] == 'dot' or s['op'] == 'transpose':
                    return False
                elif s['op'] in OpCollection.reduction_ops:
                    reduction_axes[s['axis']] = True
                    if reduction_axes[1 - s['axis']]:
                        return False
        return True

    def execute(self, optree):
        """
        Execute the optree. Break optree into sub-optrees if necessary.
        """
        from neon.backends.float_ew import call_compound_kernel

        # get post order stack
        stack = optree.traverse(list())

        # bypass stage creation
        if self._is_simple_stack(stack):
            return call_compound_kernel(self._get_rand_state(), *stack)

        # create stages and evaluate
        stacks = self._split_to_stacks(optree)

        for stack in stacks:
            if (len(stack) == 5 and isinstance(stack[3], dict) and
                    stack[3]['op'] == 'dot'):
                # evaluate the simple dot
                self.compound_dot(stack[1], stack[2], stack[0])
            else:
                call_compound_kernel(self._get_rand_state(), *stack)

        return stacks[-1][0]  # TODO: to be removed, used in partial

    def empty(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Allocate the space for a GPUTensor
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)

    def array(self, ary, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        converts a numpy array to a GPUTensor
        """
        dtype = self.default_dtype if dtype is None else dtype
        if ary.ndim < self._min_dims:
            ary = ary.reshape(ary.size, 1)
        return GPUTensor(self, ary.shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode).set(ary)

    def zeros(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Returns an array of the given shape and dtype filled with 0's.
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)._assign(0)

    def ones(self, shape, dtype=None, name=None, persist_values=True,
             parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Returns an array of the given shape and dtype filled with 1's.
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)._assign(1)

    def empty_like(self, other_ary, name=None):
        """
        Returns an array with the same params as another
        """
        return GPUTensor(self, other_ary.shape, dtype=other_ary.dtype,
                         name=name, persist_values=other_ary.persist_values,
                         allocator=other_ary.allocator, rounding=self.round_mode)

    def zeros_like(self, other_ary, name=None):
        """
        Returns an array with the same params as another
        """
        return GPUTensor(self, other_ary.shape, dtype=other_ary.dtype,
                         name=name, persist_values=other_ary.persist_values,
                         allocator=other_ary.allocator,
                         rounding=self.round_mode)._assign(0)

    def compound_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, bsum=None, repeat=1, size=None):
        """
        C = alpha * A * B   + beta * C
        C = alpha * A.T * B + beta * C
        C = alpha * A * B.T + beta * C

        relu: if true applied before output (and prior to beta addition)

        size: one of 32x128, 128x32, 64x128, 128x64, 128x128.  Sometimes the
              fastest tiling isn't chosen for you.
        """
        assert A.dtype.type == B.dtype.type == C.dtype.type

        # one dimention must be contiguous
        assert min(A.strides) == 1
        assert min(B.strides) == 1
        assert min(C.strides) == 1

        lda = max(A.strides)
        ldb = max(B.strides)
        ldc = max(C.strides)

        if A.is_trans:
            opA = 't'
            lda *= 8 * A.dtype.itemsize  # saves a kernel register
        else:
            opA = 'n'

        if B.is_trans:
            opB = 't'
        else:
            opB = 'n'
            ldb *= 8 * B.dtype.itemsize  # saves a kernel register

        op = opA + opB
        assert op != "tt"

        m = A.shape[0]
        n = B.shape[1]
        k = A.shape[1]

        assert m == C.shape[0]
        assert n == C.shape[1]
        assert k == B.shape[0]

        # Some basic tile size selection.
        # Your best bet is to benchmark your code with all 3 sizes
        # and manually fine tune the selection for each layer.
        # TODO: Perhaps I'll add an autotuning mode.
        if size is None:
            # find the shorter side
            short = min(m, n)
            # anything bigger than this just use 128
            if short < 384 - 16:
                # compute remainder of 128
                short128 = short % 128
                # if remainder is more than 112 just use 128
                if 0 < short128 < 112:
                    # to figure out when to use 64 over 32 we need to calc
                    # occupancy at 64
                    if 48 < short128 <= 64:
                        occupancy64 = short // 64
                        wide = max(m, n)
                        occupancy64 *= (wide // 128 + (wide %
                                                       128 != 0)) // _get_sm_count()
                        # 64 is only faster than 32 when occupancy is more than
                        # 1 warp per scheduler.
                        if occupancy64 > 1:
                            size = 64
                        else:
                            size = 32
                    else:
                        size = 32
                else:
                    size = 128
            # There's a large regime where 64 is faster, but it's hard to
            # characterize
            else:
                size = 128

            # match the kernel to the optimal short size but avoid not
            # implemented kernels
            if m >= n:
                if op == "nt":
                    size = 128
                sizeA, sizeB = (128, size)
            else:
                if op == "tn":
                    size = 128
                # temp till I can write these kernels (coming soon)
                elif size == 64:
                    size = 32
                sizeA, sizeB = (size, 128)

            size = "%dx%d" % (sizeA, sizeB)

        else:
            sizeA, sizeB = (int(s) for s in size.split('x'))

        gridA = m // sizeA + (m % sizeA != 0)
        gridB = n // sizeB + (n % sizeB != 0)
        threads = 256 if size == "128x128" else 128

        k_vec = 4 if sizeA == 32 or sizeB == 32 else 16

        if (op == "tn" and m % 4 == 0 and n % 4 == 0 or
                op == "nn" and k % k_vec == 0 and n % 4 == 0 or
                op == "nt" and k % k_vec == 0):
            op += "_vec"

        # nt and nn are more efficient with k%16==0
        if C.dtype.type is np.float16:
            clss = "hgemm"
        elif C.dtype.type is np.float32:
            clss = "sgemm"
        else:
            raise TypeError("Only floating point dot currently supported.")

        flags = 0
        if relu:
            flags |= 2

        kernel = kernel_specs.get_kernel("_".join((clss, op, size)))
        params = [
            (1, gridA, gridB), (threads, 1, 1), self.stream,
            C.gpudata, A.gpudata, B.gpudata, alpha, beta, flags,
            lda, ldb, ldc, m, n, k,
            0, 0, 0, 0]

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(*params)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(self.stream)

        for r in range(repeat):
            kernel.prepared_async_call(*params)

        if self.bench or repeat > 1:
            end.record(self.stream)
            end.synchronize()
            msecs = end.time_since(start) / repeat
            gflops = (m * n * k * 2.0) / (msecs * 1000000.0)
            print("%7.3f msecs %4.0f gflops (%s_%s: %d,%d,%d) size:%s grid:(%d,%d)" %
                  (msecs, gflops, clss, op, m, n, k, size, gridA, gridB))
            if repeat > 1:
                return gflops
        if bsum is not None:
            bsum[:] = self.sum(C, 1)
        return C

    def batched_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, repeat=1, size=None):
        assert A.dtype.type == B.dtype.type == C.dtype.type

        flags = 0
        if relu:
            flags |= 2

        dima, dimb, dimc = 0, 0, 0
        ldaz, ldbz, ldcz = 0, 0, 0
        batch_grid, batch_loops = 1, 1

        if len(A.shape) == 3:
            dima = 1
            ldaz = A.strides[0]

        if len(B.shape) == 3:
            dimb = 1
            ldbz = B.strides[0]

        assert dima or dimb, "Tensor A or B must have 3 dims to use batched_dot"

        if len(C.shape) == 3:
            dimc = 1
            ldcz = C.strides[0]
            batch_grid = C.shape[0]
            assert not dima or A.shape[0] == batch_grid
            assert not dimb or B.shape[0] == batch_grid

        elif dima:
            batch_loops = A.shape[0]
            assert not dimb or B.shape[0] == batch_loops

        elif dimb:
            batch_loops = B.shape[0]
            assert not dima or A.shape[0] == batch_loops

        m = A.shape[0 + dima]
        n = B.shape[1 + dimb]
        k = A.shape[1 + dima]

        assert m == C.shape[0 + dimc]
        assert n == C.shape[1 + dimc]
        assert k == B.shape[0 + dimb]

        lda = max(A.strides[dima:])
        ldb = max(B.strides[dimb:])
        ldc = max(C.strides[dimc:])

        if A.is_trans:
            opA = 't'
            lda *= 8 * A.dtype.itemsize  # saves a kernel register
        else:
            opA = 'n'

        if B.is_trans:
            opB = 't'
        else:
            opB = 'n'
            ldb *= 8 * B.dtype.itemsize  # saves a kernel register

        op = opA + opB
        assert op != "tt"

        short = min(m, n)
        if batch_loops > 1:
            size = 128
        elif size is None:
            if short % 128 == 0:
                size = 128
            elif short > 32 and short == n:  # temp
                size = 64
            else:
                size = 32

        if m >= n:
            if op == "nt":
                size = 128
            sizeA, sizeB = (128, size)
        else:
            if op == "tn":
                size = 128
            # temp till I can write these kernels (coming soon)
            elif size == 64:
                size = 32
            sizeA, sizeB = (size, 128)

        gridA = m // sizeA + (m % sizeA != 0)
        gridB = n // sizeB + (n % sizeB != 0)
        threads = 256 if size == 128 else 128
        size = "%dx%d" % (sizeA, sizeB)

        k_vec = 4 if sizeA == 32 or sizeB == 32 else 16

        if (op == "tn" and m % 4 == 0 and n % 4 == 0 or
                op == "nn" and k % k_vec == 0 and n % 4 == 0 or
                op == "nt" and k % k_vec == 0):
            op += "_vec"

        # nt and nn are more efficient with k%16==0
        if C.dtype.type is np.float16:
            clss = "hgemm"
        elif C.dtype.type is np.float32:
            clss = "sgemm"
        else:
            raise TypeError("Only floating point dot currently supported.")

        kernel = kernel_specs.get_kernel("_".join((clss, op, size)))
        params = [
            (batch_grid, gridA, gridB), (threads, 1, 1), self.stream,
            C.gpudata, A.gpudata, B.gpudata, alpha, beta, flags,
            lda, ldb, ldc, m, n, k,
            ldaz, ldbz, ldcz, batch_loops]

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(*params)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(self.stream)

        for r in range(repeat):
            kernel.prepared_async_call(*params)

        if self.bench or repeat > 1:
            end.record(self.stream)
            end.synchronize()
            msecs = end.time_since(start) / repeat
            gflops = (batch_loops * batch_grid * m * n * k * 2.0) / \
                (msecs * 1000000.0)
            print("%7.3f msecs %4.0f gflops (%s_%s: %d,%d,%d) size:%s grid:(%d,%d,%d) loops:%d" %
                  (msecs, gflops, clss, op, m, n, k, size, batch_grid, gridA, gridB, batch_loops))
            if repeat > 1:
                return gflops

        return C

    def make_binary_mask(self, out, keepthresh=0.5):
        """
        Create a binary mask for dropout layers.

        Arguments:
            out (GPUTensor): Output tensor
            keepthresh (float): fraction of ones
        """
        self.dropout(keep=keepthresh, out=out)

    def rand(self, out=None):
        """
        Generate random number uniformly distributed between 0 and 1.

        Arguments:
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("rand", None, None, out=out)

    def dropout(self, keep=0.5, out=None):
        """
        Returns a keep mask for dropout.

        Arguments:
            keep (int, optional): the keep threshold. Values smaller than keep
                                  will be set to 0, otherwise set to 1.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return self.less_equal(self.rand(), keep, out=out)

    def compensated_sum(self, sum_tensor, cmp_tensor, add_tensor, cmp_scale=1.0, add_scale=1.0):
        from neon.backends.float_ew import _get_compensated_sum_kernel, _get_fast_ew_dims

        if cmp_tensor.kahan_reset and cmp_tensor.kahan_count > cmp_tensor.kahan_reset:
            cmp_scale = 0
            cmp_tensor.kahan_count = 0

        assert sum_tensor.dtype.type == cmp_tensor.dtype.type == add_tensor.dtype.type

        cmp_tensor.kahan_count += 1

        shape, strides = _get_fast_ew_dims(sum_tensor.size)

        kernel = _get_compensated_sum_kernel(
            sum_tensor.dtype.str[1:], sum_tensor.rounding > 0)

        kernel.prepared_async_call(
            (shape[0], 1, 1), (32, 1, 1), self.stream, self._get_rand_state(),
            sum_tensor.gpudata, cmp_tensor.gpudata, add_tensor.gpudata,
            cmp_scale, add_scale,
            strides[0], strides[1],
            shape[1], sum_tensor.rounding)

    def conv_layer(self, dtype,
                   N, C, K,
                   D=1, H=1, W=1,
                   T=1, R=1, S=1,
                   pad_d=0, pad_h=0, pad_w=0,
                   str_d=1, str_h=1, str_w=1,
                   relu=False, bsum=False,
                   deterministic_update=False):
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

        relu: apply a relu to the output for fprop or bprop

        bsum: calculate the sum along the batchnorm axis for fprop or bprop
              outputs an fp32 tensor of size Kx1

        deterministic_update: eleminate atom adds in the update operation
                              can slow the kernel down

        """
        return ConvLayer(self, dtype, N, C, K, D, H, W, T, R, S,
                         pad_d, pad_h, pad_w, str_d, str_h, str_w,
                         relu, bsum, deterministic_update)

    def fprop_conv(self, layer, I, F, O, alpha=1.0, beta=0.0, bsum=None, repeat=1):
        assert layer.sizeI == I.size
        assert layer.sizeF == F.size
        assert layer.sizeO == O.size
        return self._execute_conv(
            layer, "fprop", layer.fprop_kernels, layer.fprop_lut_size,
            I, F, O, alpha, beta, bsum, 0, repeat)

    def bprop_conv(self, layer, F, E, grad_I, alpha=1.0, beta=0.0, bsum=None, repeat=1):
        assert layer.sizeF == F.size
        assert layer.sizeO == E.size
        assert layer.sizeI == grad_I.size
        return self._execute_conv(
            layer, "bprop", layer.bprop_kernels, layer.bprop_lut_size,
            E, F, grad_I, alpha, beta, bsum, layer.bprop_zero, repeat)

    def update_conv(self, layer, I, E, grad_F, alpha=1.0, repeat=1):
        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeF == grad_F.size

        return self._execute_conv(
            layer, "updat", layer.updat_kernels, 0,
            I, E, grad_F, alpha, 0.0, None, layer.sizeF*4, repeat)

    def _execute_conv(self, layer, op, kernel_list, shared,
                      A, B, C, alpha, beta, bsum, zero, repeat):

        assert A.dtype == B.dtype

        flags = 0
        if layer.relu:
            flags |= 2
        if layer.bsum and bsum is not None:
            flags |= 4
            batch_sum_data = bsum.gpudata
        else:
            batch_sum_data = 0

        A_gpudata      = A.gpudata
        B_gpudata      = B.gpudata
        C_gpudata      = C.gpudata
        shuffle_kernel = None
        convert_type   = False

        from neon.backends.float_ew import _get_transpose_kernel, _get_shuffle_kernel, _fp_convert

        if op == "bprop":
            B_gpudata = self.scratch_buffer(B.size)
            if zero:
                shuffle_kernel = _get_transpose_kernel(B.dtype.str[1:])
                if beta:
                    zero = False # let atomic adds accumulate on top
                    if beta != 1.0:
                        C[:] = C * beta # pre-apply beta
            else:
                shuffle_kernel = _get_shuffle_kernel(B.dtype.str[1:])
            shuffle_args = [layer.shuffle_grid, layer.shuffle_block, self.stream,
                            B_gpudata, B.gpudata] + layer.shuffle_args

        elif op == "updat" and C.dtype.type is not np.float32:
            C_gpudata    = self.scratch_buffer(C.size)
            convert_type = "f4"

        kernels = []
        for kernel in kernel_list:
            kernels.append([
                kernel_specs.get_kernel(kernel[0]), kernel[1], kernel[2], self.stream,
                batch_sum_data, C_gpudata, A_gpudata, B_gpudata, alpha, beta, flags,
                kernel[3]] + kernel[4])

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                for kernel in kernels:
                    kernel[0].prepared_async_call(*kernel[1:], shared_size=shared)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(stream=self.stream)

        for r in range(repeat):
            if zero:
                drv.memset_d8_async(C_gpudata, 0, zero, self.stream)

            if batch_sum_data:
                drv.memset_d32_async(batch_sum_data, 0, bsum.size, self.stream)

            if shuffle_kernel:
                shuffle_kernel.prepared_async_call(*shuffle_args)

            for kernel in kernels:
                kernel[0].prepared_async_call(*kernel[1:], shared_size=shared)

            if convert_type:
                _fp_convert(C_gpudata, convert_type, C)

        if self.bench or repeat > 1:
            end.record(stream=self.stream)
            end.synchronize()
            msecs  = end.time_since(start) / repeat
            gflops = layer.flops / (msecs * 1000000.0)
            print("%7.3f msecs %4.0f gflops %6.0f (%s: %s)" %
                  (msecs, gflops, layer.flops/1000000.0, op, layer))
            return msecs, gflops
        return 0, 0

    def deconv_layer(self, dtype,
                     N, C, K,
                     P, Q,
                     R=1, S=1,
                     pad_d=0, pad_h=0, pad_w=0,
                     str_d=1, str_h=1, str_w=1,
                     relu=False, bsum=False,
                     deterministic_update=False):
        """
        Create a new DeconvLayer parameter object.
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

        dtype: need to know dtype to setup proper kernels and params.

        relu: apply a relu to the output for fprop or bprop

        bsum: calculate the sum along the batchnorm axis for fprop or bprop
              outputs an fp32 tensor of size Kx1

        deterministic_update: eleminate atom adds in the update operation
                              can slow the kernel down
        """
        return DeconvLayer(self, dtype, N, C, K, P, Q, R, S,
                           pad_d, pad_h, pad_w, str_d, str_h, str_w,
                           relu, bsum, deterministic_update)

    def pool_layer(self, dtype,
                   op, N, C,
                   D=1, H=1, W=1,
                   J=1, T=1, R=1, S=1,
                   pad_c=0, pad_d=0, pad_h=0, pad_w=0,
                   str_c=None, str_d=None, str_h=None, str_w=None):
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

    def fprop_pool(self, layer, I, O, argmax=None, alpha=1.0, beta=0.0, repeat=1):

        assert layer.sizeI == I.size
        assert layer.sizeO == O.size
        if layer.op == "max":
            assert argmax is not None, "max pooling requires argmax buffer"
        return self._execute_pool(layer, I, O, argmax, alpha, beta, layer.fprop_kernel,
                                  layer.fprop_lut_size, repeat)

    def bprop_pool(self, layer, I, O, argmax=None, alpha=1.0, beta=0.0, repeat=1):

        assert layer.sizeI == O.size, "missmatch between sizeI %d and O %d" % (layer.sizeI, O.size)
        assert layer.sizeO == I.size, "missmatch between sizeO %d and I %d" % (layer.sizeO, I.size)
        if layer.op == "max":
            assert argmax is not None, "max pooling requires argmax buffer"
        if argmax is not None:
            assert layer.sizeO == argmax.size, "Pooling argmax size does not match input size!"
        assert I.dtype == O.dtype
        return self._execute_pool(layer, I, O, argmax, alpha, beta, layer.bprop_kernel,
                                  layer.bprop_lut_size, repeat)

    def _execute_pool(self, layer, I, O, argmax, alpha, beta, kernel_args, shared, repeat):

        assert I.dtype == O.dtype
        A_data = argmax.gpudata if argmax is not None else 0
        kernel = pooling.map_string2func(kernel_args[0], layer.dtype.str[1:])
        flags = 0
        params = [kernel_args[1], kernel_args[2], self.stream,
                  I.gpudata, O.gpudata, A_data, alpha, beta, flags]
        params.extend(kernel_args[3])

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(*params, shared_size=shared)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(self.stream)

        for r in range(repeat):
            kernel.prepared_async_call(*params, shared_size=shared)

        if self.bench or repeat > 1:
            end.record(self.stream)
            end.synchronize()
            msecs = end.time_since(start) / repeat
            print("%7.3f msecs (%s) grid:%s" % (msecs, layer, kernel_args[1]))

    def compound_fprop_bn(self, x, xsum, xvar, gmean, gvar, gamma, beta, y, eps, rho,
                          relu=False, threads=None, repeat=1):
        """
        Function to perform compound kernel call for batch normalization
        forward pass.

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
            relu (bool): Compuound ReLU activation in kernel
            threads (int): Number of GPU threads
            repeat (int): Repeats for benchmarking
        """
        assert xsum.dtype.type is np.float32

        K = x.shape[0]
        N = x.shape[1]

        if threads is None:
            if N <= 8192:
                threads = 1 << max(5, int(round(log(N, 2))) - 3)
            else:
                occup = K / (128.0 * _get_sm_count())
                for t in (32, 64, 128, 256, 512, 1024):
                    if occup * t > 5.0:
                        threads = t
                        break

        params = [(K, 1, 1), (threads, 1, 1), x.backend.stream,
                  y.gpudata, xvar.gpudata, gmean.gpudata, gvar.gpudata,
                  x.gpudata, xsum.gpudata, gmean.gpudata, gvar.gpudata,
                  gamma.gpudata, beta.gpudata, eps, rho, N, relu]

        from neon.backends.float_ew import _get_bn_fprop_kernel

        kernel = _get_bn_fprop_kernel(x.dtype.str[1:], threads)

        self._execute_bn(kernel, params, repeat, x.nbytes*2, N)

    def compound_bprop_bn(self, delta, grad_gamma, grad_beta, x, xsum, xvar,
                          gamma, eps, threads=None, repeat=1):
        """
        delta (Tensor): Delta buffer
        grad_gamma (Tensor): Gradient w.r.t. gamma
        grad_beta (Tensor): Gradient w.r.t. beta
        x (Tensor): feedforward input
        xsum (Tensor): Batch sum over PQN dimension
        xvar (Tensor): Batch variance
        gamma (Tensor): scale parameter
        eps (float): constant for numerical stability
        threads (int): Number of GPU threads
        repeat (int): Repeats for benchmarking
        """
        assert xsum.dtype.type is np.float32, "xsum should be fp32"

        K = x.shape[0]
        N = x.shape[1]

        if threads is None:
            if N <= 8192:
                threads = 1 << max(5, int(round(log(N, 2))) - 3)
            else:
                threads = 128 if K < 192 else 64

        params = [(K, 1, 1), (threads, 1, 1), x.backend.stream,
                  delta.gpudata, grad_gamma.gpudata, grad_beta.gpudata, delta.gpudata,
                  x.gpudata, xsum.gpudata, xvar.gpudata, gamma.gpudata, eps, N]

        from neon.backends.float_ew import _get_bn_bprop_kernel

        kernel = _get_bn_bprop_kernel(x.dtype.str[1:], threads)

        self._execute_bn(kernel, params, repeat, x.nbytes*4, N)

    def _execute_bn(self, kernel, params, repeat, size, N):

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(*params)

        if self.bench > 1 or repeat > 1:
            start, end = _get_events()
            start.record(self.stream)

        for r in range(repeat):
            kernel.prepared_async_call(*params)

        if self.bench > 1 or repeat > 1:
            end.record(self.stream)
            end.synchronize()
            msecs = end.time_since(start) / repeat
            bandwidth = size / (msecs * 1024 * 1024)
            blocks = params[0][0]
            threads = params[1][0]
            occup = blocks * threads / (128.0 * _get_sm_count())
            print("%7.3f msecs %4.0f GBps %s(%d,%d,%d) %.1f" %
                  (msecs, bandwidth, kernel.name, blocks, N, threads, occup))

    def init_mark(self):
        """
        Generate a timing mark object

        Returns:
            timing mark (pycude driver event)
        """
        return drv.Event()

    def record_mark(self, marker):
        """
        Mark the current time

        Arguments:
            marker (time mark): timing mark generated by init_mark()
        """
        marker.record()

    def get_time(self, start, end):
        """
        Return time between start and end marks

        Arguments:
            start (time maker): start time mark

            end (time marker): end time mark

        Returns:
            time elapsed between start and end time marks in milliseconds
        """
        end.synchronize()
        return end.time_since(start)


# Note the strides computed here do not include the dtype.itemsize
def _contiguous_strides(shape):
    if shape:
        strides = [1]
        for s in shape[:0:-1]:
            strides.append(strides[-1] * s)
        return tuple(strides[::-1])
    else:
        return ()


@context_dependent_memoize
def _get_scratch_data(scratch_size):
    return drv.mem_alloc(scratch_size * 4)


@context_dependent_memoize
def _get_events():
    return (drv.Event(), drv.Event())

# debugging tool
# import re
# import traceback as tb

# nrv_re = re.compile(r'nervanagpu\.py$')
# def print_trace():
#     caller = None
#     for frame in tb.extract_stack():
#         if GPUTensor.nrv_re.search(frame[0]):
#             break
#         caller = (frame[0],frame[1])
#     print caller
