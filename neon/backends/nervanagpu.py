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
Our GPU based backend interface and tensor data structure.
"""
from __future__ import division
from builtins import round
import os
import sys
import numpy as np
import math
import pycuda.driver as drv
import logging
from pycuda.tools import context_dependent_memoize
from pycuda.curandom import MRG32k3aRandomNumberGenerator as rng_mrg
from pycuda.gpuarray import GPUArray as p_gpuarray
from struct import unpack_from
from pytools import memoize_method
from functools import wraps
from math import log
from neon import logger as neon_logger
from neon.backends import kernel_specs
from neon.backends.backend import Tensor, Backend, OpTreeNode, OpCollection
from neon.backends.layer_gpu import ConvLayer, DeconvLayer, PoolLayer, _get_sm_count
from neon.backends.kernels.cuda import pooling, roipooling, binary, nms
from neon.backends.util.check_gpu import get_compute_capability
from neon.util.persist import get_cache_dir
from scikits.cuda import cublas

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
        :class:`NervanaGPU` class

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

        if not isinstance(shape, (tuple, list)):
            assert isinstance(shape, (int, np.integer))
            shape = (int(shape), )

        if isinstance(shape, (tuple, list)) and len(shape) < self._min_dims:
            shape = shape + (1, )*(self._min_dims - len(shape))

        shape_ = []
        size = 1
        for dim in shape:
            if int(dim) != dim:
                raise TypeError('shape dims must be integer values [%s]' % str(dim))
            dim = int(dim)
            shape_.append(dim)
            size *= dim
        # shape_ will only have int elements (e.g. instead of np.int64)
        shape = tuple(shape_)

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
                # print drv.mem_get_info()
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
        Return a sliced view of an array.
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
            if self.is_contiguous and value.is_contiguous and self.dtype == value.dtype and self.shape == value.shape:
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
        if ary.size != self.size:
            raise TypeError(
                'ary.size {} != self.size {}'.format(ary.size, self.size)
            )
        assert self.is_contiguous, "Array in set() must be contiguous"
        if ary.dtype is not self.dtype:
            ary = ary.astype(self.dtype)
        if ary.ndim < self._min_dims:
            ary = ary.reshape(ary.size, 1)

        strides = tuple(self.dtype.itemsize * s for s in self.strides)
        if (ary.strides != strides):
            raise TypeError(
                'ary.strides != self.strides * self.dtype.itemsize : {} != {} * {}'.format(
                    ary.strides, self.strides, self.dtype.itemsize
                )
            )

        drv.memcpy_htod_async(int(self.gpudata), ary, stream)

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

    def raw(self):
        """
        Access the raw buffer.

        Returns:
            pointer: A device specific pointer
        """
        return self.gpudata

    def asnumpyarray(self):
        """
        Deprecated.
        Scheduled to be removed in 2.0.
        Use get() instead.
        """
        return self.get()

    def asbuffer(self):
        """
        Returns buffer interface to gpu data.
        """
        return self.gpudata.as_buffer(self.nbytes)

    def take(self, indices, axis, out=None):
        """
        Take elements from an array along an axis.

        Arguments:
            indices (Tensor, numpy ndarray): indicies of elements to select
            axis (int): axis across which to select the values
            out (Tensor): Output Tensor to fill with selected values
        """
        if axis == 1:
            view = self.__getitem__((_none_slice, indices))
        else:
            view = self.__getitem__((indices, _none_slice))

        if out:
            return out._assign(view)
        return view

    def fill(self, value):
        """
        Assign specified value to each element of this GPUTensor.

        Arguments:
            value (numeric): The value to be assigned to each element.

        Return:
            GPUTensor: updated view of the data.
        """
        return self._assign(value)

    def copy(self, a):
        """
        Construct and return a deep copy of the Tensor passed.

         Arguments:
            a (Tensor): the object to copy

        Returns:
            GPUTensor: updated view of the data.
        """
        return self._assign(a)

    def copy_from(self, a):
        """
        Alias of copy.

        Arguments:
            a (Tensor): the object to copy

        Returns:
            GPUTensor: updated view of the data.
        """
        return self.set(a)

    def reshape(self, *shape):
        """
        Return a reshaped view.
        """
        if isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        if len(shape) < self._min_dims:
            shape = shape + (1, )

        if -1 in shape:
            missing_dim = -self.size // int(np.prod(shape))
            shape = tuple([missing_dim if x == -1 else x for x in shape])

        if shape == self.shape:
            return self

        size = np.prod(shape)

        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        if self.take_array:
            raise TypeError("reshaping of non-contiguous arrays is not yet supported")

        new_strides = _reshape_strides(self.strides, self.shape, shape)

        return self.__class__(
            backend=self.backend,
            shape=shape,
            dtype=self.dtype,
            allocator=self.allocator,
            base=self,
            gpudata=self.gpudata,
            strides=new_strides,
            name=self.name,
            rounding=self.rounding)

    @property
    def T(self):
        """
        Return a transposed view.
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
        Return a view: ary, where ary.size <= self.size.
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
    Memoize the stacks using intrinsic_key_maps.
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
    The primary interface class and factory for GPUTensors.

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
    backend_name = 'gpu'
    # size of the RNG pool on device
    # currently this is hard wired
    _RNG_POOL_SIZE = (3 * 2048 * 32, 1)
    def __init__(self,
                 rng_seed=None,
                 default_dtype=np.float32,
                 stochastic_round=False,
                 deterministic=None,
                 device_id=0,
                 bench=False,
                 scratch_size=0,
                 hist_bins=64,
                 hist_offset=-48,
                 compat_mode=None,
                 enable_winograd=True,
                 # Ignored
                 num_devices=None
                 ):
        from neon.backends.util import check_gpu
        check_gpu.ensure_gpu_capability(device_id)

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

        # context
        drv.init()
        self.device_type = 1
        self.device_id = device_id if device_id is not None else 0
        self.ctx = drv.Device(device_id).make_context()

        # store the rand pool for each context
        self.context_rand_state_map = {}  # stores gpu memory reference
        self.context_rand_state_alive = {}  # set whether randstate is fresh

        # super class init
        super(NervanaGPU, self).__init__(rng_seed,
                                         default_dtype,
                                         compat_mode=compat_mode,
                                         deterministic=deterministic)

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
        self.scratch_offset = 0
        self.round_mode = stochastic_round
        self.bench = bench
        self.stream = None
        self.buf = {}
        self.buf_active = {}
        self.warmup = False
        self.sm_count = _get_sm_count()

        # store histograms for batched memcpy
        self.hist_bins, self.hist_offset = None, None
        self.set_hist_buffers(hist_bins, hist_offset)

        # store GPU memory size in bytes
        self.gpu_memory_size = drv.mem_get_info()[1]

        # Fall back to CUDA C kernels on older (pre-Maxwell) GPU generations
        self.compute_capability = drv.Device(self.device_id).compute_capability()
        if self.compute_capability[0] < 5:
            self.use_cudac_kernels = True

            logger.warn("Neon is highly optimized for Maxwell GPUs. Although "
                        "you might get speedups over CPUs, note that you are "
                        "running on a pre-Maxwell GPU and you might not "
                        "experience the fastest performance. For faster "
                        "performance using the Nervana Cloud contact "
                        "info@nervanasys.com")
        else:
            self.use_cudac_kernels = False
        self.cublas_handle = cublas.cublasCreate()

        self.enable_winograd = enable_winograd
        self.cache_dir = get_cache_dir()

        self.use_pinned_mem = True

    def consume(self, buf_index, hostlist, devlist):
        assert 0 <= buf_index < 2, 'Can only double buffer'
        self.ctx.push()

        if devlist[buf_index] is None:
            devlist[buf_index] = self.empty_like(hostlist[buf_index].T)

        # TODO: do the transpose as a DataLoaderTransformer instead of in host
        # memory like this
        devlist[buf_index].set(hostlist[buf_index].T.copy())

        self.ctx.pop()

    def set_hist_buffers(self, hist_bins, hist_offset):
        if (hist_bins != self.hist_bins or hist_offset != self.hist_offset):
            self.hist_bins = hist_bins
            self.hist_offset = hist_offset
            self.hist_map = dict()
            self.hist_idx = 0
            self.hist_max = 4 * 4096
            self.hist_base = drv.mem_alloc(self.hist_bins * self.hist_max * 4)
            drv.memset_d32(self.hist_base, 0, self.hist_bins * self.hist_max)

    def scratch_buffer_reset(self):
        self.scratch_size = 0
        self.scratch_offset = 0
        _reset_scratch_data()

    def scratch_buffer_init(self):
        self.scratch_offset = 0

    def cleanup_backend(self):
        super(NervanaGPU, self).cleanup_backend()
        try:
            self.ctx.pop()
            self.ctx.detach()
        except drv.Error:
            pass

    def scratch_buffer(self, size):

        if size & 127 != 0:
            size += 128 - (size & 127)

        if size > self.scratch_size:
            raise RuntimeError(
                "nervanagpu.scratch_size(%d) is too small for this operation(%d)" % (
                    self.scratch_size, size))

        self.scratch_offset = size

        return int(_get_scratch_data(self.scratch_size))

    def scratch_buffer_offset(self, size):

        if size & 127 != 0:
            size += 128 - (size & 127)

        if size + self.scratch_offset > self.scratch_size:
            raise RuntimeError(
                "nervanagpu.scratch_size(%d) is too small for this operation(%d, %d)" % (
                    self.scratch_size, size, self.scratch_offset))

        data = int(_get_scratch_data(self.scratch_size)) + self.scratch_offset
        self.scratch_offset += size

        return data

    def set_scratch_size(self, *args):

        total_size = 0
        for size in args:
            if size & 127 != 0:
                size += 128 - (size & 127)
            total_size += size

        if total_size > self.scratch_size:
            self.scratch_size = total_size

    def __del__(self):
        try:
            self.ctx.detach()
        except drv.Error:
            pass

    def get_events(self):
        return _get_events()

    def gen_rng(self, seed=None):
        """
        Generate the random number generator on device and on host.

        Arguments:
            seed (int): random number generator seed

        Returns:
            seeded numpy RNG
        """
        # generate on host rng
        self.rng = np.random.RandomState(seed)

        # this RNG is for handling normally distributed numbers on device
        self.pcg = rng_mrg()
        # save the initial state of host rng
        self.init_rng_state = self.rng.get_state()

        # generate random integers to seed the LSFR
        # RNGs on the device
        self.init_rng_state_dev = self._gen_dev_randstate()

        # call below is mainly to set on device RNG states
        # to self.init_rng_state_dev
        self.rng_reset()

        # if the current context already has an rng clear it
        ctx = drv.Context.get_current()
        if ctx in self.context_rand_state_alive:
            self.context_rand_state_alive[ctx] = False

        # generate the on device RNG
        self._set_rand_state_dev(state=self.init_rng_state_dev)
        return self.rng

    def _gen_dev_randstate(self):
        """
        Generate a list of random uint32 numbers to seed the LFSR
        states on device.

        Returns:
            np.array: return a vector of uint32 numbers
        """
        # will use the numpy rng to generate the states
        # but want to reset it after this is done
        state_save = self.rng.get_state()

        # smaller number for 32bit systems
        maxexp = 32 if sys.maxsize > 2**32 else 30

        # draw _RNG_POOL_SIZE 32 bit ints to seed LFSR on device
        # lower bound 1 to avoid seeding LFSR with 0
        rand_init = self.rng.random_integers(1, 2**maxexp - 1, NervanaGPU._RNG_POOL_SIZE)
        rand_init = rand_init.astype(np.uint32)

        # put the numpy (on host) RNG back to its state before
        self.rng.set_state(state_save)

        return rand_init

    def rng_reset(self):
        """
        Reset the RNG to the initial state stored in
        self.init_rng_state and self.init_rng_state_dev
        for the host and device RNG, respectively.
        """
        self.rng_set_state( (self.init_rng_state, self.init_rng_state_dev) )

    def rng_set_state(self, rng_states):
        """
        Set the RNG state for both the on device and on host RNGs.

        Arguments:
            rng_states (tuple of np.arrays): tuple with 2 elements
                                                1) numpy random number state vector
                                                2) array of uint32 specifying on dev RNG state
        """
        assert type(rng_states) is tuple and len(rng_states) == 2
        self._set_rand_state_dev(state=rng_states[1])
        self.rng.set_state(rng_states[0])

    def rng_get_state(self):
        """
        Return the current state of the on-host and on-device RNGs.

        Returns:
            (np.array, np.array): the on-host and on-device RNG state vectors,
                                  respectively
        """
        dev_state = self._get_rand_state_dev()
        dev_state_local = np.zeros(NervanaGPU._RNG_POOL_SIZE).astype(np.uint32)
        drv.memcpy_dtoh(dev_state_local, dev_state)
        return (self.rng.get_state(), dev_state_local)

    def _set_rand_state_dev(self, state=None):
        """
        Set on device RNG states to values given by "state" input.

        Arguments:
            state (np.array or None): an array of uint32 values used to
                                      set the state of the on device LFSRs.
                                      if set to None, the state will be created
                                      randomly
        """
        ctx = drv.Context.get_current()
        if state is None:
            state = self._gen_dev_randstate()
        if ctx in self.context_rand_state_map:
            rand_state = self.context_rand_state_map[ctx]
        else:
            rand_state = drv.mem_alloc(state.nbytes)
            self.context_rand_state_map[ctx] = rand_state
        drv.memcpy_htod(rand_state, state)
        self.context_rand_state_alive[ctx] = True
        return

    def fill_normal(self, ary, mean=0, stdv=1):
        """
        Fill ary with normally distributed random numbers.

        Arguments:
            ary (Tensor): Tensor to fill with random values
            mean (float): Mean value. Default 0
            stdv (float): standard deviation value.  Default 1
        """
        self.pcg.fill_normal(p_gpuarray(ary.shape, ary.dtype, gpudata=ary.gpudata))
        if not all([mean == 0, stdv == 1]):
            ary[:] = ary * stdv + mean

    def _get_rand_state_dev(self):
        """
        Similar to @context_dependent_memoize, with additional ability to reset
        the random pool by `rng_reset`.

        initialize our common pool of randomness (1/4 MB):
        MAX_THREADS_PER_MULTIPROCESSOR * 32 SMs (32 to be somewhat future proof
        and power of two). This size is currently hardcoded in the kernels,
        to be parameterized ...
        """
        ctx = drv.Context.get_current()
        if not (ctx in self.context_rand_state_map and self.context_rand_state_alive[ctx]):
            self._set_rand_state_dev()
        return self.context_rand_state_map[ctx]

    def _buf_malloc(self, shape):
        """
        Returns a buffer of size shape, equivalent of calling be.empty(shape).
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
        Move all tensors from self.buffer_active to self.buffer
        the idea is to reuse those tensors for other optrees.
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
        Split an optree to stacks.
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
        Execute the optree.

        Arguments:
            optree: (OpTreeNode): the OpTreeNode object that represents all
                                  the operations
        """
        from neon.backends.float_ew import call_compound_kernel

        # get post order stack
        stack = optree.traverse(list())

        # bypass stage creation
        if self._is_simple_stack(stack):
            return call_compound_kernel(self._get_rand_state_dev(), self.compute_capability, *stack)

        # create stages and evaluate
        stacks = self._split_to_stacks(optree)

        for stack in stacks:
            if (len(stack) == 5 and isinstance(stack[3], dict) and
                    stack[3]['op'] == 'dot'):
                # evaluate the simple dot
                self.compound_dot(stack[1], stack[2], stack[0])
            else:
                call_compound_kernel(self._get_rand_state_dev(), self.compute_capability, *stack)

        return stacks[-1][0]  # TODO: to be removed, used in partial

    def empty(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Allocate the space for a GPUTensor

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

            allocator (function, optional): Memory allocator.

        Returns:
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)

    def array(self, ary, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Converts a numpy array to a GPUTensor

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
            allocator (function, optional): Memory allocator.

        Returns:
            GPUTensor: newly created data structure reference
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
        Instantiate a new instance of the GPUTensor class setting each element
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
            allocator (function, optional): Memory allocator.

        Returns:
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)._assign(0)

    def ones(self, shape, dtype=None, name=None, persist_values=True,
             parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Instantiate a new instance of the GPUTensor class setting each element
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
            allocator (function, optional): Memory allocator.

        Returns:
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)._assign(1)

    def empty_like(self, other_ary, name=None):
        """
        Instantiate a new instance of this backend's Tensor class, with the
        shape taken from ary.

        Arguments:
            ary (tensor object): Tensor to inherit the dimensions of.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.

        Returns:
            Tensor: array object
        """
        # if other_ary is a numpy array it wont have attr persist_values or
        # allocator so use default values in that case.
        return GPUTensor(self, other_ary.shape, dtype=other_ary.dtype,
                         name=name,
                         persist_values=getattr(other_ary, 'persist_values', True),
                         allocator=getattr(other_ary, 'allocator', drv.mem_alloc),
                         rounding=self.round_mode)

    def zeros_like(self, other_ary, name=None):
        """
        Instantiate a new instance of this backend's Tensor class, with the
        shape taken from ary and populating each element with a value of 0.

        Arguments:
            ary (tensor object): Tensor to inherit the dimensions of.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.

        Returns:
            Tensor: array object
        """
        return GPUTensor(self, other_ary.shape, dtype=other_ary.dtype,
                         name=name, persist_values=other_ary.persist_values,
                         allocator=other_ary.allocator,
                         rounding=self.round_mode)._assign(0)

    def compound_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, bsum=None,
                     repeat=1, size=None):

        """
        Doing following operations (* is dot product)
        C = alpha * A * B   + beta * C
        C = alpha * A.T * B + beta * C
        C = alpha * A * B.T + beta * C.

        relu: if true applied before output (and prior to beta addition)

        size: one of 32x128, 128x32, 64x128, 128x64, 128x128.  Sometimes the
              fastest tiling isn't chosen for you.

        Arguments:
            A, B (GPUTensor): input operands
            C (GPUTensor): output
            alpha (float): scale A*B term
            beta (float): scale C term before sum
            relu (bool): whether to apply ReLu before output
            size(nxm): Sometimes the fastest tiling isn't chosen for you.
        """
        assert A.dtype.type == B.dtype.type == C.dtype.type

        if self.use_cudac_kernels or B.shape[1] == 1:
            for r in range(repeat):
                self.cublas_dot(A=A, B=B, C=C, alpha=alpha, beta=beta)

            if bsum is not None:
                bsum[:] = self.sum(C, 1)
            return C

        # one dimention must be contiguous
        assert min(A.strides) == 1
        assert min(B.strides) == 1
        assert min(C.strides) == 1

        lda = max(A.strides)
        ldb = max(B.strides)
        ldc = max(C.strides)

        if A.is_trans:
            opA = 't'
            if size not in ("32x64", "16x64"):
                lda *= 8 * A.dtype.itemsize  # saves a kernel register
        else:
            opA = 'n'

        if B.is_trans:
            opB = 't'
        else:
            opB = 'n'
            if size not in ("32x64", "16x64"):
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

        k_vec = 8 if sizeA in (16,32) or sizeB == 32 else 16

        vec_opt = None
        if op == "tn":
            if (m % 4 == 0 and n % 4 == 0 and
                A.strides[1] % 4 == 0 and B.strides[0] % 4 == 0):
                vec_opt = ("vec",)
        elif op == "nn":
            if (k % k_vec == 0 and n % 4 == 0 and
                A.strides[0] % k_vec == 0 and B.strides[0] % 4 == 0):
                vec_opt = ("vec",)
        elif op == "nt":
            if (k % k_vec == 0 and n % 4 == 0 and
                A.strides[0] % k_vec == 0 and B.strides[1] % k_vec == 0):
                vec_opt = ("vec",)

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

        kernel = kernel_specs.get_kernel("_".join((clss, op, size)), vec_opt)
        params = [
            (1, int(gridA), int(gridB)), (kernel.threads, 1, 1), self.stream,
            C.gpudata, A.gpudata, B.gpudata, alpha, beta, flags,
            int(lda), int(ldb), int(ldc), int(m), int(n), int(k),
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
            neon_logger.display("%7.3f msecs %4.0f gflops (%s_%s: %d,%d,%d) size:%s grid:(%d,%d)" %
                 (msecs, gflops, clss, op, m, n, k, size, gridA, gridB))
            if repeat > 1:
                return msecs, gflops
        if bsum is not None:
            bsum[:] = self.sum(C, 1)
        return C

    def batched_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, repeat=1, size=None):
        assert A.dtype.type == B.dtype.type == C.dtype.type

        if self.use_cudac_kernels:
            raise NotImplementedError("batched_dot is not implemented for Kepler.")

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
            vec_opt = ("vec",)
        else:
            vec_opt = None

        # nt and nn are more efficient with k%16==0
        if C.dtype.type is np.float16:
            clss = "hgemm"
        elif C.dtype.type is np.float32:
            clss = "sgemm"
        else:
            raise TypeError("Only floating point dot currently supported.")

        kernel = kernel_specs.get_kernel("_".join((clss, op, size)), vec_opt)
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
            gflops = (batch_loops * batch_grid * m * n * k * 2.0) / (msecs * 1000000.0)
            neon_logger.display("%7.3f msecs %4.0f gflops (%s_%s: %d,%d,%d) size:%s grid:(%d,%d,%d) loops:%d" %
                  (msecs, gflops, clss, op, m, n, k, size, batch_grid, gridA, gridB, batch_loops))
            if repeat > 1:
                return gflops

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

        # kernel restriction
        assert A.shape[1] % 32 == 0

        blocks = lambda x: max(int(math.ceil(x)), 1)

        pack_rows_kernel = binary.pack_rows()
        PACK_ROW_BLOCK_SIZE = 64
        Ab = self.empty((A.shape[0], int(A.shape[1]/32)), dtype=np.uint32)
        pack_rows_params = [
            (blocks(Ab.shape[0] * Ab.shape[1]/PACK_ROW_BLOCK_SIZE), 1),
            (PACK_ROW_BLOCK_SIZE, 1, 1),
            self.stream,
            A.gpudata, Ab.gpudata, Ab.shape[0] * Ab.shape[1]
        ]
        pack_rows_kernel.prepared_async_call(*pack_rows_params)

        pack_cols_kernel = binary.pack_cols()
        PACK_COL_BLOCK_SIZE = 32
        Bb = self.empty((int(B.shape[0]/32), B.shape[1]), dtype=np.uint32)
        pack_cols_params = [
            (blocks(B.shape[1]/PACK_COL_BLOCK_SIZE), blocks(B.shape[0]/PACK_COL_BLOCK_SIZE), 1),
            (PACK_COL_BLOCK_SIZE, PACK_COL_BLOCK_SIZE, 1),
            self.stream,
            B.gpudata, Bb.gpudata, B.shape[0], B.shape[1]
        ]
        pack_cols_kernel.prepared_async_call(*pack_cols_params)

        xnor_kernel = binary.XNOR_gemm()
        XNOR_BLOCK_SIZE = 16
        xnor_params = [
            (blocks(B.shape[1]/XNOR_BLOCK_SIZE), blocks(A.shape[0]/XNOR_BLOCK_SIZE), 1),
            (XNOR_BLOCK_SIZE, XNOR_BLOCK_SIZE, 1),
            self.stream,
            Ab.gpudata, Bb.gpudata, C.gpudata, A.shape[0], Ab.shape[1], B.shape[1]
        ]
        xnor_kernel.prepared_async_call(*xnor_params)

        if bsum is not None:
            bsum[:] = self.sum(C, 1)

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

    def shift(self, ary, shift_ary, value=True, out=None):
        """
        Shifts input array

        Arguments:
            ary: tensor
            shift_ary: tensor of shift amount
            out: reference to output
        """
        if not hasattr(ary, 'shape'):
            ary = self.array(np.array(ary))

        if not hasattr(shift_ary, 'shape'):
            shift_ary = self.array(np.array(shift_ary))

        if out is None:
            out = self.empty_like(ary)

        blocks = lambda x: max(int(math.ceil(x)), 1)

        shift_kernel = binary.shift()
        SHIFT_BLOCK_SIZE = 64
        sizeary = ary.shape[0] * ary.shape[1]
        shift_params = [
            (blocks(sizeary/SHIFT_BLOCK_SIZE), 1),
            (SHIFT_BLOCK_SIZE, 1, 1),
            self.stream,
            ary.gpudata, shift_ary.gpudata, out.gpudata, value, sizeary, shift_ary.shape[0], shift_ary.shape[1]
        ]
        shift_kernel.prepared_async_call(*shift_params)

        return out

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
            (shape[0], 1, 1), (32, 1, 1), self.stream, self._get_rand_state_dev(),
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
        """
        return ConvLayer(self, dtype, N, C, K, D, H, W, T, R, S,
                         pad_d, pad_h, pad_w, str_d, str_h, str_w,
                         dil_d, dil_h, dil_w)

    def fprop_conv(self, layer, I, F, O,
        X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
        relu=False, brelu=False, slope=0.0, repeat=1, layer_op=None):
        """
        fprop_conv:

        Required Arguments:
            layer: ConvLayer object created with conv_layer()
            I: input tensor  (actiavtions)
            F: filter tensor (weights)
            O: output tensor (actiavtions)

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
            relu: boolean flag to apply:
                O = max(O, 0) + slope*min(O, 0)
                can be combined with bias (where bias is added first)
            brelu: bprop_relu boolean flag to apply:
                O *= (X > 0) + slope*(X < 0)
                can be combined with bsum tensor to output bprop_bias

        repeat: used in benchmarking
        """
        assert layer.sizeI == I.size
        assert layer.sizeF == F.size
        assert layer.sizeO == O.size

        layer.fprop_kernels.bind_params(I, F, O, X, bias, bsum, alpha, beta, relu, brelu, slope)

        return self._execute_conv("fprop", layer, layer.fprop_kernels, repeat)

    def bprop_conv(self, layer, F, E, grad_I,
        X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
        relu=False, brelu=False, slope=0.0, repeat=1, layer_op=None):
        """
        bprop_conv:

        Required Arguments:
            layer: ConvLayer object created with conv_layer()
            E: error tensor (output gradient from previous layer)
            F: filter tensor (weights)
            grad_I: output tensor (gradient with respect to inputs)

        Compounding Options:
            X: tensor to use in bprop_relu or beta
                can be same as grad_I for beta accumulate (this is default when None)
                should be same shape as grad_I
            bias: (C,1) tensor to use for adding bias to output
                grad_I += bias
            bsum: (C,1) tensor to accumulate batch sum over (used in batchnorm or bprop_bias)
                bsum = sum(grad_I.reshape(C,-1), axis=1)
                the sum operation is fully deterministic
                if combined with brelu then brelu is applied first
            alpha, beta:
                grad_I = alpha*grad_I + beta*X
                grad_I = alpha*grad_I + beta*grad_I   (if X==grad_I)
            relu: boolean flag to apply:
                grad_I = max(grad_I, 0) + slope*min(grad_I, 0)
                can be combined with bias (where bias is added first)
            brelu: bprop_relu boolean flag to apply:
                grad_I *= (X > 0) + slope*(X < 0)
                can be combined with bsum tensor to output bprop_bias

        repeat: used in benchmarking
        """
        assert layer.sizeF == F.size
        assert layer.sizeO == E.size
        assert layer.sizeI == grad_I.size

        layer.bprop_kernels.bind_params(E, F, grad_I, X, bias, bsum, alpha, beta, relu, brelu, slope)

        return self._execute_conv("bprop", layer, layer.bprop_kernels, repeat)

    def update_conv(self, layer, I, E, grad_F, alpha=1.0, beta=0.0, repeat=1, grad_bias=None, layer_op=None):
        """
        update_conv:

        Required Inputs:
            layer: ConvLayer object created with conv_layer()
            I: input tensor (actiavtions)
            E: error tensor (output gradient from previous layer)
            grad_F: output tensor (gradient with respect to weights)

        Compounding Options:
        alpha, beta:
            O = alpha*O + beta*O

        repeat: used in benchmarking
        """
        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeF == grad_F.size

        if layer.NCK[0] < 4 and layer.TRS == (1, 1, 1):
            Ir = I.reshape((layer.NCK[1], -1))
            Er = E.reshape((layer.NCK[2], -1))
            Gr = grad_F.reshape((layer.NCK[1], -1))
            return self.compound_dot(A=Ir, B=Er.T, C=Gr, alpha=alpha, beta=beta, repeat=repeat)
        else:
            layer.updat_kernels.bind_params(I, E, grad_F, alpha, beta)
            return self._execute_conv("updat", layer, layer.updat_kernels, repeat)

    def _execute_conv(self, op, layer, kernels, repeat):
        # Warmup
        if repeat > 1:
            kernels.execute(max(repeat // 10, 1), unbind=False)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(stream=self.stream)

        kernels.execute(repeat)

        if self.bench or repeat > 1:
            end.record(stream=self.stream)
            end.synchronize()
            msecs  = end.time_since(start) / repeat
            gflops = layer.flops / (msecs * 1000000.0)
            #if layer.TRS[2] == 3:
            neon_logger.display("%7.3f msecs %5.0f gflops %6.0f (%s: %s)" %
                  (msecs, gflops, layer.flops / 1000000.0, op, layer))
            return msecs, gflops
        return 0, 0

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

        M: Depth of input
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
        lrn_opts = dict(T=1, R=1, S=1,
                        pad_c=pad_c,
                        pad_d=0, pad_h=0, pad_w=0,
                        str_c=1, str_d=1, str_h=1, str_w=1)

        return PoolLayer(lib=self, dtype=dtype, op=op, N=N, C=C, D=D, H=H, W=W, J=J, **lrn_opts)

    def fprop_lrn(self, layer, I, O, denom, alpha=1., beta=0., ascale=1., bpower=1., repeat=1):
        """
        Forward propagate lrn layer.

        Arguments:
            layer (PoolLayer): The pool layer object, specd for LRN
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
        WH = W * H
        DWH = D * W * H

        kernel_args = layer.fprop_kernel
        shared = layer.bprop_lut_size

        self._execute_lrn(layer, I, O, None, None, denom,
                          alpha, beta, ascale, bpower, kernel_args, shared, repeat)

    def bprop_lrn(self, layer, I, O, E, delta, denom,
                  alpha=1., beta=0., ascale=1., bpower=1., repeat=1):
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
        op = layer.op

        J, T, R, S = layer.JTRS
        C, D, H, W, N = layer.dimI
        K, M, P, Q, N = layer.dimO
        pad_c, pad_d, pad_h, pad_w = layer.padding
        str_c, str_d, str_h, str_w = layer.strides
        WH = W * H
        DWH = D * W * H

        kernel_args = layer.bprop_kernel
        shared = layer.bprop_lut_size

        self._execute_lrn(layer, I, O, E, delta, denom,
                          alpha, beta, ascale, bpower, kernel_args, shared, repeat)

    def _execute_lrn(self, layer, I, O, E, delta, denom,
                     alpha, beta, ascale, bpower, kernel_args, shared, repeat):

        assert I.dtype == O.dtype
        A_data = denom.gpudata if denom is not None else 0
        kernel = pooling.map_string2func(kernel_args[0], layer.dtype.str[1:], self.compute_capability)

        flags = 0
        params = [kernel_args[1], kernel_args[2], self.stream,
                  I.gpudata, O.gpudata, A_data, alpha, beta, ascale, bpower, flags]
        params.extend(kernel_args[3])

        if kernel_args[0][0] == "b":  # backprop kernel
            params = [kernel_args[1], kernel_args[2], self.stream,
                      I.gpudata, O.gpudata, E.gpudata, delta.gpudata, A_data,
                      alpha, beta, ascale, bpower, flags]
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
            neon_logger.display("%7.3f msecs (%s) grid:%s" % (msecs, layer, kernel_args[1]))

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
        kernel = pooling.map_string2func(kernel_args[0], layer.dtype.str[1:], self.compute_capability)
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
            neon_logger.display("%7.3f msecs (%s)" % (msecs, layer))


    def roipooling_fprop(self, I, rois, O, argmax, roi_count, fm_channel, fm_height, fm_width,
                            pooled_height, pooled_width, spatial_scale):
        """
        Function to perform fprop of ROIPooling.

        Arguments:
            I (Tensor): (C, H, W, N)
            rois (Tensor): (ROIs, 5)
            O (Tensor): (C, pooled_height, pooled_width, roi_count)
            argmax (Tensor): (C, pooled_height, pooled_width, roi_count)
        """
        thread = 1024
        assert roi_count == rois.shape[0]
        assert roi_count == O.shape[-1] == argmax.shape[-1]

        count = fm_channel * pooled_height * pooled_width * roi_count
        assert count == O.size == argmax.size
        assert argmax.dtype == np.int32

        def get_blocks(N, thread):
            return (N + thread - 1) // thread

        layer_dtype = I.dtype

        kernel = roipooling.map_string2func("fprop_roipooling", layer_dtype.str[1:])

        params = [(get_blocks(count, thread), 1, 1), (thread, 1, 1), self.stream,
                    count, roi_count, self.bsz, fm_channel, fm_height, fm_width,
                    pooled_height, pooled_width, I.gpudata,
                    rois.gpudata, O.gpudata, argmax.gpudata,
                    spatial_scale]

        kernel.prepared_async_call(*params)

    def roipooling_bprop(self, I, rois, O, argmax, roi_count, fm_channel, fm_height, fm_width,
                            pooled_height, pooled_width, spatial_scale):
        """
        Function to perform bprop of ROIPooling.

        Arguments:
            I (Tensor): input errors (C, pooled_height, pooled_width, roi_count)
            argmax (Tensor): max args from the fprp (C, pooled_height, pooled_width, roi_count)
            rois (Tensor): (ROIs, 5)
            O (Tensor): output deltas (C, H, W, N)
        """

        thread = 1024
        assert roi_count == rois.shape[0]
        # assert roi_count == I.shape[-1] == argmax.shape[-1]

        count = fm_channel * fm_height * fm_width * self.bsz
        assert count == O.size
        assert argmax.dtype == np.int32

        def get_blocks(N, thread):
            return (N + thread - 1) // thread

        layer_dtype = I.dtype

        kernel = roipooling.map_string2func("bprop_roipooling", layer_dtype.str[1:])

        params = [(get_blocks(count, thread), 1, 1), (thread, 1, 1), self.stream,
                    count, roi_count, self.bsz, fm_channel, fm_height, fm_width,
                    pooled_height, pooled_width, I.gpudata,
                    rois.gpudata, O.gpudata, argmax.gpudata,
                    spatial_scale]

        kernel.prepared_async_call(*params)

    def nms(self, detections, threshold, normalized=False):
        """
        Function to perform non-maximal supression.

        Arguments:
            detections (Tensor): detection boxes (box_count, 5), each row has
                                 (x1, y1, x2, y2, score). Assume the boxes have already
                                 been sorted based on score in descending order
            threshold (float): box overlap threshold, boxes with smaller overlaps will be kept
            normalized (bool): whether box coordinates are normalized to image dimensions.
                               This affects whether we use a +1 offset to compute box sizes.

        Outputs:
            keep_ind (list): list of indices
        """
        def div_ceil(N, thread):
            return int((N) / (thread) + ((N) % (thread) > 0))

        box_count = detections.shape[0]
        threadsPerBlock = 32
        # decide on how many blocks to use for each dimension, and use 2D blocks
        col_blocks = div_ceil(box_count, threadsPerBlock)

        assert detections.shape == (box_count, 5)

        mask_size = box_count*col_blocks
        output_mask = self.zeros((mask_size), dtype=np.uint32)

        params = [(col_blocks, col_blocks, 1), (threadsPerBlock, 1, 1), self.stream, box_count,
                   threshold, detections.gpudata, output_mask.gpudata, normalized]

        kernel = nms._get_nms_kernel()

        kernel.prepared_async_call(*params)

        mask_cpu = output_mask.get().ravel()
        scores = detections[:, -1].get()
        num_to_keep = 0
        keep = np.zeros((box_count), dtype=np.int32)
        remv = np.zeros((col_blocks), dtype=np.uint32)

        for i in range(box_count):
            nblock = int(i / threadsPerBlock)
            inblock = int(i % threadsPerBlock)

            if (remv[nblock] & (1 << inblock)) == 0 and scores[i] != 0:
                keep[num_to_keep] = i
                num_to_keep += 1
                for j in range(nblock, col_blocks):
                    remv[j] |= mask_cpu[i * col_blocks + j]

        return keep[:num_to_keep].tolist()

    def compound_fprop_bn(self, x, xsum, xvar, gmean, gvar, gamma, beta, y, eps, rho, compute_batch_sum,
                          accumbeta=0.0, relu=False, threads=None, repeat=1,
                          binary=False, inference=False, outputs=None, layer=None):
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
            beta (Tensor): location parameter
            y (Tensor): normalized output
            eps (float): constant for numerical stability
            rho (float): exponential window averaging constant
            accumbeta (float): value to scale output by before accumulating
            relu (bool): Compound ReLU activation in kernel
            threads (int): Number of GPU threads
            repeat (int): Repeats for benchmarking
            binary (bool): Binary shift based computations
        """
        assert xsum.dtype.type is np.float32

        if inference:
            xhat = (x - gmean) / self.sqrt(gvar + eps)  # Op-tree only
            y[:] = y * accumbeta + xhat * gamma + beta
            return

        if compute_batch_sum:
            xsum[:] = self.sum(x, axis=1)

        K = int(x.shape[0])
        N = int(x.shape[1])

        if threads is None:
            if N <= 8192:
                threads = 1 << max(5, int(round(log(N, 2))) - 3)
            else:
                occup = K / (128.0 * _get_sm_count())
                for t in (32, 64, 128, 256, 512):
                    if occup * t > 5.0:
                        threads = t
                        break
        if threads is None:
            threads = 1024

        params = [(K, 1, 1), (threads, 1, 1), x.backend.stream,
                  y.gpudata, xvar.gpudata, gmean.gpudata, gvar.gpudata,
                  x.gpudata, xsum.gpudata, gmean.gpudata, gvar.gpudata,
                  gamma.gpudata, beta.gpudata, eps, rho, accumbeta, N,
                  relu, binary]

        from neon.backends.float_ew import _get_bn_fprop_kernel

        kernel = _get_bn_fprop_kernel(x.dtype.str[1:], threads, self.compute_capability)

        self._execute_bn(kernel, params, repeat, x.nbytes * 2, N)

    def compound_bprop_bn(self, delta_out, grad_gamma, grad_beta, delta_in,
                          x, xsum, xvar, gamma, eps, threads=None,
                          repeat=1, binary=False, layer=None):
        """
        Function to perform batch normalization forward pass.

        Arguments:
            delta_out (Tensor): Delta buffer (where to write the output deltas)
            grad_gamma (Tensor): Gradient w.r.t. gamma
            grad_beta (Tensor): Gradient w.r.t. beta
            delta_in (Tensor): Delta buffer (where to get the input deltas)
            x (Tensor): feedforward input
            xsum (Tensor): Batch sum over PQN dimension
            xvar (Tensor): Batch variance
            gamma (Tensor): scale parameter
            eps (float): constant for numerical stability
            threads (int): Number of GPU threads
            repeat (int): Repeats for benchmarking
            binary (bool): Binary shift based computations
        """
        assert xsum.dtype.type is np.float32, "xsum should be fp32"

        K = int(x.shape[0])
        N = int(x.shape[1])

        if threads is None:
            if N <= 8192:
                threads = 1 << max(5, int(round(log(N, 2))) - 3)
            else:
                threads = 128 if K < 192 else 64

        params = [(K, 1, 1), (threads, 1, 1), x.backend.stream,
                  delta_out.gpudata, grad_gamma.gpudata, grad_beta.gpudata, delta_in.gpudata,
                  x.gpudata, xsum.gpudata, xvar.gpudata, gamma.gpudata, eps, N, binary]

        from neon.backends.float_ew import _get_bn_bprop_kernel

        kernel = _get_bn_bprop_kernel(x.dtype.str[1:], threads, self.compute_capability)

        self._execute_bn(kernel, params, repeat, x.nbytes * 4, N)

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
            neon_logger.display("%7.3f msecs %4.0f GBps %s(%d,%d,%d) %.1f" %
                  (msecs, bandwidth, kernel.name, blocks, N, threads, occup))


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
        from neon.backends.float_ew import (_get_lut_bprop_kernel,
                                            _get_sorting_kernel)
        embedding_dim = dW.shape[1]
        vocab_size = dW.shape[0]

        if pad_idx is None:
            pad_idx = int(-1)

        if self.deterministic:
            index_buffer = self.empty((error.shape[1],), dtype=np.int32)
            offset_buffer = self.empty((error.shape[1],), dtype=np.int32)
            word_counts = self.zeros((max(512, vocab_size) + 512,), dtype=np.int32)

            for kernel_id in range(5):
                threads = 512
                if kernel_id in [1, 3]:
                    blocks = vocab_size // (threads * 2)
                    if vocab_size % (threads * 2):
                        blocks = blocks + 1
                elif kernel_id == 2:
                    blocks = 1
                else:
                    blocks = error.shape[1] // threads
                    if error.shape[1] % threads:
                        blocks = blocks + 1

                params = [(blocks, 1, 1), (threads, 1, 1), inputs.backend.stream,
                          inputs.gpudata, index_buffer.gpudata, offset_buffer.gpudata, word_counts.gpudata,
                          max(512, vocab_size), error.shape[1]]
                kernel = _get_sorting_kernel(kernel_id, threads)
                kernel.prepared_async_call(*params)

            threads = 32
            blocks = error.shape[1]

            error_t[:] = error.T
            params = [(blocks, 1, 1), (threads, 1, 1), inputs.backend.stream,
                      inputs.gpudata, index_buffer.gpudata, dW.gpudata, error_t.gpudata,
                      nin, embedding_dim, vocab_size, pad_idx]

            kernel = _get_lut_bprop_kernel(error.dtype.str[1:], True)
            kernel.prepared_async_call(*params)
        else:
            threads = 32
            blocks = error.shape[1]

            error_t[:] = error.T
            params = [(blocks, 1, 1), (threads, 1, 1), inputs.backend.stream,
                      inputs.gpudata, dW.gpudata, error_t.gpudata, nin, embedding_dim, vocab_size,
                      pad_idx]

            kernel = _get_lut_bprop_kernel(error.dtype.str[1:])
            kernel.prepared_async_call(*params)

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
        if nout <= 1152 and self.bsz == 4 and (nout % 48) == 0:
            persistent_kernel = True
            num_blocks = nout // 48
        else:
            persistent_kernel = False
            num_blocks = (-(-h_s[0].shape[0] // 128)) * (-(-h_s[0].shape[1] // 32))
            num_blocks = (-(-num_blocks // 4))

        if (activation.classnm == 'Rectlinclip' and num_blocks <= self.sm_count and
                not self.use_cudac_kernels and activation.slope == 0):
            if h_s[0].base is not h_ff_s[0].base:
                if len(h_s[0].base.shape) == 3:
                    assert ((h_ff_s[0].base.shape[1] + 2) == h_s[0].base.shape[1])
                    h_buffer = h_s[0].base[:, 1:-1].reshape(nout, -1)
                    h_ff_buffer = h_ff_s[0].base.reshape(nout, -1)
                    h_buffer[:] = h_ff_buffer
                else:
                    h_s[0].base[:] = h_ff_s[0].base

            if num_used_steps is not None and num_used_steps < num_steps:
                num_steps = num_used_steps

            if persistent_kernel:
                self._persistent_rnn_fprop(W_recur, h_prev_s[0],
                                           h_s[0], bias, nout, nout,
                                           self.bsz, num_steps, activation,
                                           reverse)
            else:
                self._compound_unrolled_gemm(W_recur, h_prev_s[0], h_s[0],
                                             bias, nout, nout, self.bsz, num_steps,
                                             activation, reverse)
        else:
            if num_used_steps is not None and num_used_steps < num_steps:
                h_s = h_s[:num_used_steps]
                h_prev_s = h_prev_s[:num_used_steps]
                h_ff_s = h_ff_s[:num_used_steps]

            if reverse:
                steps = reversed(list(zip(h_s, h_prev_s, h_ff_s)))
            else:
                steps = list(zip(h_s, h_prev_s, h_ff_s))

            for (h, h_prev, h_ff) in steps:
                if h_ff is h:
                    self.compound_dot(W_recur, h_prev, h, beta=1.0)
                    h[:] = activation(h + bias)
                else:
                    self.compound_dot(W_recur, h_prev, h)
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
        if nout <= 1152 and self.bsz == 4 and (nout % 48) == 0:
            persistent_kernel = True
            num_blocks = nout // 48
        else:
            persistent_kernel = False
            num_blocks = (-(-delta_s[0].shape[0] // 128)) * (-(-delta_s[0].shape[1] // 32))
            num_blocks = (-(-num_blocks // 4))

        if (activation.classnm == 'Rectlinclip' and num_blocks <= self.sm_count and
                not self.use_cudac_kernels and activation.slope == 0):
            # Compute activation bprop for first timestep since there is
            # no compounded GEMM
            if reverse:
                delta_s[-1][:] = activation.bprop(h_s[-1]) * delta_s[-1]
            else:
                delta_s[0][:] = activation.bprop(h_s[0]) * delta_s[0]

            if num_used_steps is not None and num_used_steps < num_steps:
                num_steps = num_used_steps

            if reverse:
                B = delta_s[1]
                C = delta_s[0]
                H = h_s[0]
            else:
                B = delta_s[0]
                C = delta_s[1]
                H = h_s[1]

            if persistent_kernel:
                self._persistent_rnn_bprop(W_recur, B, C, H, nout, nout,
                                           self.bsz, num_steps - 1, activation,
                                           reverse)
            else:
                self._compound_unrolled_gemm_bprop(W_recur, B, C, H, nout, nout,
                                                   self.bsz, num_steps - 1,
                                                   activation, reverse)

            if reverse:
                self.compound_dot(W_recur, delta_s[0], delta_s[-1], beta=1.0)
            else:
                self.compound_dot(W_recur, delta_s[-1], delta_s[0], beta=1.0)
        else:
            if num_used_steps is not None and num_used_steps < num_steps:
                h_s = h_s[:num_used_steps]
                h_prev_s = h_prev_s[:num_used_steps]
                h_ff_s = h_ff_s[:num_used_steps]

            if reverse:
                steps = reversed(list(zip(h_s, delta_s, delta_prev_s)))
            else:
                steps = list(zip(h_s, delta_s, delta_prev_s))

            for (hs, in_deltas, prev_in_deltas) in steps:
                in_deltas[:] = activation.bprop(hs) * in_deltas
                self.compound_dot(W_recur, in_deltas, prev_in_deltas, beta=1.0)

    def _persistent_rnn_fprop(self, W, hprev, h, bias, nin, nout, unroll_stride,
                             num_steps, activation, reverse=False):
        assert W.dtype.type == h.dtype.type == bias.dtype.type
        assert activation.classnm == 'Rectlinclip'

        gpulock = _get_lock_data(4 * num_steps)
        drv.memset_d32_async(gpulock, 0, num_steps, self.stream)

        # one dimension must be contiguous
        assert min(h.strides) == 1
        assert min(hprev.strides) == 1
        assert min(W.strides) == 1

        reluclip = activation.xcut
        param_reverse = 1 if reverse else 0
        num_blocks = -(-nout // 48)

        kernel = kernel_specs.get_kernel("persistent_rnn_fprop")
        params = [
            (num_blocks, 1, 1), (kernel.threads, 1, 1), self.stream,
            h.gpudata, hprev.gpudata, bias.gpudata, W.gpudata, gpulock,
            h.strides[0] // 4, W.strides[0], self.bsz, num_steps, num_blocks,
            nout, param_reverse, reluclip]

        kernel.prepared_async_call(*params)

    def _persistent_rnn_bprop(self, W, dnext, d, h, nin, nout, unroll_stride,
                             num_steps, activation, reverse=False):
        assert W.dtype.type == h.dtype.type == d.dtype.type
        assert activation.classnm == 'Rectlinclip'

        gpulock = _get_lock_data(4 * num_steps)
        drv.memset_d32_async(gpulock, 0, num_steps, self.stream)

        # one dimension must be contiguous
        assert min(h.strides) == 1
        assert min(d.strides) == 1
        assert min(W.strides) == 1

        reluclip = activation.xcut
        param_reverse = 1 if reverse else 0
        num_blocks = -(-nout // 48)

        kernel = kernel_specs.get_kernel("persistent_rnn_bprop")
        params = [
            (num_blocks, 1, 1), (kernel.threads, 1, 1), self.stream,
            d.gpudata, dnext.gpudata, h.gpudata, W.gpudata, gpulock,
            d.strides[0] // 4, h.strides[0] // 4, W.strides[1], self.bsz,
            num_steps, num_blocks, nout, param_reverse, reluclip]

        kernel.prepared_async_call(*params)

    def _compound_unrolled_gemm(self, A, B, C, bias, nin, nout, unroll_stride, num_steps, activation,
                               reverse=False):
        assert A.dtype.type == B.dtype.type == C.dtype.type
        assert activation.classnm == 'Rectlinclip'

        gpulock = _get_lock_data(4)
        drv.memset_d32_async(gpulock, 0, 1, self.stream)

        # one dimension must be contiguous
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
        # NOTE: Only nn supported now
        assert op == "nn"

        m = nout
        n = unroll_stride
        k = nin

        # NOTE: Only 128x32 supported now
        sizeA = 128
        sizeB = 32

        gridA = m // sizeA + (m % sizeA != 0)
        gridB = n // sizeB + (n % sizeB != 0)

        if op == "nn":
            if (k % 8 == 0 and n % 4 == 0 and
                A.strides[0] % 8 == 0 and B.strides[0] % 4 == 0):
                op += "_vec"

        op = "sgemm_rnn_" + op + '_' + str(sizeA) + 'x' + str(sizeB)

        # Since the kernel uses inter-block synchronization, ensure that we don't
        # have more blocks than can run concurrently
        assert (gridA * gridB) < (4 * _get_sm_count())

        if reverse:
            flags = 4
        else:
            flags = 0

        kernel = kernel_specs.get_kernel(op)
        params = [
            (1, gridA, gridB), (kernel.threads, 1, 1), self.stream,
            C.gpudata, A.gpudata, B.gpudata, bias.gpudata, gpulock,
            1.0, 1.0, activation.xcut, flags,
            lda, ldb, ldc, m, n, k,
            0, 0, 0, 0, unroll_stride, unroll_stride, num_steps, gridA * gridB, gridA]

        kernel.prepared_async_call(*params)

    def _compound_unrolled_gemm_bprop(self, A, B, C, H, nin, nout, unroll_stride, num_steps, activation,
                               reverse=False):
        assert A.dtype.type == B.dtype.type == C.dtype.type
        assert activation.classnm == 'Rectlinclip'

        gpulock = _get_lock_data(4)
        drv.memset_d32_async(gpulock, 0, 1, self.stream)

        # one dimension must be contiguous
        assert min(A.strides) == 1
        assert min(B.strides) == 1
        assert min(C.strides) == 1
        assert min(H.strides) == 1

        lda = max(A.strides)
        ldb = max(B.strides)
        ldc = max(C.strides)
        ldh = max(H.strides)

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
        # NOTE: Only tn supported now
        assert op == "tn"

        m = nout
        n = unroll_stride
        k = nin

        # NOTE: Only 128x32 supported now
        sizeA = 128
        sizeB = 32

        gridA = m // sizeA + (m % sizeA != 0)
        gridB = n // sizeB + (n % sizeB != 0)

        if op == "tn":
            if (m % 4 == 0 and n % 4 == 0 and
                A.strides[1] % 4 == 0 and B.strides[0] % 4 == 0):
                op += "_vec"

        op = "sgemm_rnn_bprop_" + op + '_' + str(sizeA) + 'x' + str(sizeB)

        # Since the kernel uses inter-block synchronization, ensure that we don't
        # have more blocks than can run concurrently
        assert (gridA * gridB) < (4 * _get_sm_count())

        if reverse:
            flags = 4
        else:
            flags = 0

        kernel = kernel_specs.get_kernel(op)
        params = [
            (1, gridA, gridB), (kernel.threads, 1, 1), self.stream,
            C.gpudata, A.gpudata, B.gpudata, H.gpudata,
            gpulock, 1.0, 1.0, activation.xcut, flags,
            lda, ldb, ldc, ldh, m, n, k,
            0, 0, 0, 0, unroll_stride, unroll_stride, unroll_stride, num_steps,
            gridA * gridB, gridA]

        kernel.prepared_async_call(*params)

    def cublas_dot(self, A, B, C, alpha=1.0, beta=0.0):
        """
        Matrix multiplication using cublas library. Intended for use on Kepler
        GPUs where maxas kernels are not supported.

        C = alpha * (AB) + beta * C

        Arguments:
            A (Tensor): Input tensor
            B (Tensor): Input tensor
            C (Tensor): Output tensor
            alpha (float): Scalar for AB
            beta (float): Scalar for C
        """
        lda = max(A.strides)
        ldb = max(B.strides)
        ldc = max(C.strides)

        opA = 't' if A.is_trans else 'n'
        opB = 't' if B.is_trans else 'n'

        m = A.shape[0]
        n = B.shape[1]
        k = A.shape[1]

        # Swap A and B to map from C order to Fortran
        if A.dtype == np.float32 or (A.dtype == np.float16 and get_compute_capability() >= 5.2):
            if n != 1 or (opA == 't' and opB == 'n'):
                cublas.cublasSgemm(self.cublas_handle, opB, opA, n, m, k, alpha, B.gpudata,
                                   ldb, A.gpudata, lda, beta, C.gpudata, ldc)
            else:
                cublas.cublasSgemv(self.cublas_handle, 't', k, m, alpha, A.gpudata,
                                   k, B.gpudata, ldb, beta, C.gpudata, ldc)

        elif A.dtype == np.float16:
            #fp16 gemm not supported by cublas until 7.5, so do conversion
            A_temp = self._buf_malloc((A.shape[0], A.shape[1] * 2))
            B_temp = self._buf_malloc((B.shape[0], B.shape[1] * 2))
            C_temp = self._buf_malloc((C.shape[0], C.shape[1] * 2))

            A_fp32 = GPUTensor(self, A.shape, dtype=np.float32, gpudata=A_temp.gpudata,
                               strides=A.strides, is_trans=A.is_trans)
            B_fp32 = GPUTensor(self, B.shape, dtype=np.float32, gpudata=B_temp.gpudata,
                               strides=B.strides, is_trans=B.is_trans)
            C_fp32 = GPUTensor(self, C.shape, dtype=np.float32, gpudata=C_temp.gpudata,
                               strides=C.strides, is_trans=C.is_trans)

            A_fp32[:] = A
            B_fp32[:] = B
            C_fp32[:] = C

            if n != 1 or (opA == 't' and opB == 'n'):
                cublas.cublasSgemm(self.cublas_handle, opB, opA, n, m, k, alpha,
                                   B_fp32.gpudata, ldb, A_fp32.gpudata, lda, beta,
                                   C_fp32.gpudata, ldc)
            else:
                cublas.cublasSgemv(self.cublas_handle, 't', k, m, alpha, A_fp32.gpudata,
                                   k, B_fp32.gpudata, ldb, beta, C_fp32.gpudata, ldc)

            C[:] = C_fp32

            self._buf_free()
        else:
            raise TypeError("Unsupported type for cublas gemm")

    def copy_transpose(self, a, out, axes=None, repeat=1):
        """
        Function to perform a fast copy transpose/dimshuffle operation.
        Works just like numpy.transpose, but requires an output tensor argument.
        """
        assert a.dtype == out.dtype
        assert a.size == out.size
        assert a.gpudata != out.gpudata

        if axes is None:
            axes = tuple(range(len(a.shape) - 1,-1,-1))
        elif type(axes) is not tuple:
            axes = tuple(axes)

        assert all(out.shape[i] == a.shape[x] for i,x in enumerate(axes))

        from neon.backends.convolution import _get_copy_transpose_kernel

        kernel = _get_copy_transpose_kernel(a.dtype.str, a.shape, axes)

        args = kernel.args + a.strides + out.strides

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(kernel.grid, kernel.block,
                    self.stream, out.gpudata, a.gpudata, *args)

        if self.bench > 1 or repeat > 1:
            start, end = _get_events()
            start.record(self.stream)

        for r in range(repeat):
            kernel.prepared_async_call(kernel.grid, kernel.block,
                self.stream, out.gpudata, a.gpudata, *args)

        if self.bench > 1 or repeat > 1:
            end.record(self.stream)
            end.synchronize()
            msecs = end.time_since(start) / repeat
            bandwidth = a.nbytes * 2 / (msecs * 1024 * 1024)
            neon_logger.display("%7.3f msecs %4.0f GBps copy_transpose" % (msecs, bandwidth))

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
            self.less_equal(self.rand(), out, out)
        else:
            self.greater_equal(ary, 0, out)
        out[:] = 2 * out - 1
        return out

    def init_mark(self):
        """
        Generate a timing mark object.

        Returns:
            timing mark (pycude driver event)
        """
        return drv.Event()

    def record_mark(self, marker):
        """
        Mark the current time.

        Arguments:
            marker (time mark): timing mark generated by init_mark()
        """
        marker.record(self.stream)

    def synchronize_mark(self, marker):
        """
        Synchronize on the given marker.

        Arguments:
            marker (time mark): timing mark generated by init_mark()
        """
        marker.synchronize()

    def get_time(self, start, end):
        """
        Return time between start and end marks.

        Arguments:
            start (time maker): start time mark

            end (time marker): end time mark

        Returns:
            time elapsed between start and end time marks in milliseconds
        """
        return end.time_since(start)

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

    def bprop_mergebroadcast(self, ngLayer, layers, error_views, error, delta, out_shape, alpha, beta, alphas, betas):
        betas[-1] = beta
        for l, e, a, b in reversed(list(zip(layers, error_views, alphas, betas))):
            l.bprop(e, alpha=a * alpha, beta=b)

# Note the strides computed here do not include the dtype.itemsize
def _contiguous_strides(shape):
    if shape:
        strides = [1]
        for s in shape[:0:-1]:
            strides.append(strides[-1] * s)
        return tuple(strides[::-1])
    else:
        return ()

def _reshape_strides(orig_strides, orig_shape, new_shape):
    # Only contiguous dimensions can be reshaped
    matched_dims = 0
    for orig, new in zip(orig_shape, new_shape):
        if orig != new:
            break
        else:
            matched_dims = matched_dims + 1

    # Extend original shape to length of new shape
    orig_shape = tuple(list(orig_shape) + [1] * (len(new_shape) - len(orig_shape)))
    orig_strides = tuple(list(orig_strides) + [1] * (len(new_shape) - len(orig_strides)))

    reshape_size = np.prod(new_shape[matched_dims:])
    orig_size = np.prod(orig_strides[matched_dims]) * orig_shape[matched_dims]

    if orig_size != reshape_size:
        raise ValueError("Reshaping of non-contiguous dimensions unsupported.")

    new_strides = orig_strides[:matched_dims] + _contiguous_strides(new_shape[matched_dims:])
    return new_strides

@context_dependent_memoize
def _get_scratch_data(scratch_size):
    return drv.mem_alloc(scratch_size)

def _reset_scratch_data():
    try:
        delattr(_get_scratch_data.__wrapped__, '_pycuda_ctx_dep_memoize_dic')
    except AttributeError:
        pass
@context_dependent_memoize
def _get_lock_data(lock_size):
    return drv.mem_alloc(lock_size)


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
