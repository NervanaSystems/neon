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
Our GPU based backend interface and tensor data structure.  Our implementation
is derived from `cuda-convnet2 <https://code.google.com/p/cuda-convnet2/>`_
"""

from collections import defaultdict
import cudanet
import logging
import numpy
from time import time as now

from neon.backends.backend import Backend, Tensor
from neon.diagnostics.timing_decorators import FlopsDecorator
from neon.util.compat import range
from neon.util.error import TooSlowToImplementError

logger = logging.getLogger(__name__)


class GPUTensor(Tensor):
    """
    Our n-dimensional array data structure that can reside on host or on GPU
    device.  Our implementation is a wrapped `cudanet.CUDAMatrix` tensor, where
    cudanet is derived from cuda-convnet2.

    Arguments:
        obj (numpy.ndarray): The actual data values (will be converted
                             to a 2-d row matrix).  Python built-in types like
                             lists and tuples are also supported.
        dtype (None, optional): Underlying data type of the elements.
                                Ignored for this backend as all values are
                                stored in cudanet as float32's.
        persist_values (bool, optional): If set to True (the default), the
                                         values assigned to this Tensor will
                                         persist across multiple begin and end
                                         calls.  Setting to False may provide a
                                         performance increase if values do
                                         not need to be maintained across such
                                         calls

    Notes:
        This implementation currently has the following limitations:

        * only 2D shaped Tensors are supported (set in _min_dims)
        * All element values are stored as float32 (input may be converted if
          input of a differing type is passed)
        * Only contiguous rectangular slicing is supported.  Sliced assignment
          can only be done along a singular subsetted dimension (i.e. only row
          slice or column slice based assignment).

    """
    _tensor = None
    _min_dims = 2

    def __init__(self, obj, dtype=None, persist_values=True,
                 copy_to_device=True):
        if type(obj) == cudanet.CUDAMatrix:
            self._tensor = obj
            self.shape = self._tensor.shape
        else:
            if type(obj) == list:
                obj = numpy.array(obj)
            if isinstance(obj, numpy.ndarray):
                # CUDAMatrix only supports ndarrays with exactly 2 dimensions
                # (though the elements can be tuples/lists to create arbitrary
                # n dimensions)
                while obj.ndim < self._min_dims:
                    obj = obj.reshape(obj.shape + (1, ))
                if obj.ndim != self._min_dims:
                    raise ValueError("CUDAMatrix only supports %d-D"
                                     "matrices.  You specifed %d-D" %
                                     (self._min_dims, obj.ndim))
                logger.debug('Copying to GPU')
                if dtype not in (numpy.float32, numpy.int32, 'float32',
                                 'int32') or dtype is None:
                    logger.debug('dtype %s is unsupported in GPU '
                                 'backend, defaulting to float32', dtype)
                    obj = numpy.array(obj, dtype='float32')
                elif obj.dtype != dtype:
                    logger.debug('object dtype %s mismatch.  '
                                 'Converting to %s', obj.dtype, dtype)
                    obj = numpy.array(obj, dtype=dtype)
                self._tensor = cudanet.CUDAMatrix(obj)
                self.shape = self._tensor.shape
            else:
                self._tensor = obj
        self.dtype = dtype
        self.persist_values = persist_values

    @property
    def raw(self):
        return self._tensor

    def __str__(self):
        """
        Display a suitable representation of this Tensor.
        Note that this operation requires copying to host.

        Returns:
            str: the representation
        """
        return str(self._tensor.asarray())

    def __getstate__(self):
        """
        Defines what and how we go about serializing an instance of this class.

        Returns:
            numpy.ndarray: Representation of the underlying
                           `cudanet.CUDAMatrix` tensor
        """
        if type(self._tensor) == cudanet.CUDAMatrix:
            return self._tensor.asarray()
        else:
            return self._tensor

    def __setstate__(self, state):
        """
        Defines how we go about deserializing into an instance of this class.

        Arguments:
            state (numpy.ndarray): Serialized representation of the underlying
                                   `cudanet.CUDAMatrix` tensor to be unpacked.
        """
        self.__init__(state)
        if not hasattr(cudanet.CUDAMatrix, 'ones'):
            cudanet.cublas_init()

    def _slice_dim(self, _slice, dim=0):
        """
        Helper that actually performs a slice along the dimension passed.

        Arguments:
            _slice (int or slice): actual slice object specifying indices
            dim (int): dimension number. 0 is for rows, 1 for columns, etc.

        Returns:
            GPUTensor: view or new sliced copy

        Raises:
            TooSlowToImplementError: if invalid `_slice` provided (too
            complex to implement quickly).
        """
        res = self
        fn = res._tensor.row_slice_view
        if dim == 1:
            fn = res._tensor.col_slice_view
        if isinstance(_slice, int):
            _slice = slice(_slice, _slice + 1)
        if isinstance(_slice, slice):
            assert _slice.step is None or _slice.step == 1
            start, stop, stride = _slice.indices(self.shape[dim])
            res = GPUTensor(fn(start, stop))
        elif _slice is Ellipsis:
            pass
        else:
            # arbitrary long list, too expensive to support?
            raise TooSlowToImplementError("slice indexing too complex")
        return res

    def asnumpyarray(self):
        """
        Convert the GPUTensor to an in host memory `numpy.ndarray`.  A copy of
        the data may be made depending on where the GPUTensor normally resides.

        Returns:
            numpy.ndarray view or copy of the GPUTensor data.
        """
        self._tensor.copy_to_host()
        return self._tensor.numpy_array

    def __getitem__(self, key):
        """
        Extract a subset view of the items via slice style indexing
        along each dimension. e.g. A[5:10, :].  Each slice consists of
        start_idx:stop_idx:step_size triplets.  If step_size isn't specified it
        defaults to 1.  If start_idx isn't specified it defaults to 0.  If
        stop_idx isn't specified it defaults to the total number of elements
        along that dimension.  As such a slice value of ':' allows one to
        select all elements along that dimension.

        Arguments:
            key (int, slice, tuple): indices of each dimension's slice.

        Returns:
            GPUTensor: view of self corresponding to the subset items.

        Raises:
            IndexError: if invalid number of dimensions specified in key.

        See Also:
            take
        """
        res = self
        if isinstance(key, tuple):
            if len(key) > self._min_dims:
                raise IndexError("CUDAMatrix only supports %d-D matrices",
                                 self._min_dims)
            else:
                for idx in range(len(key) - 1, -1, -1):
                    res = res._slice_dim(key[idx], idx)
        else:
            res = res._slice_dim(key, 0)
        return res

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
            value (numeric array, GPUTensor): values to be assigned to the
                                              extracted element subset.  If an
                                              array it should be the same shape
                                              as what key indexes (or be
                                              broadcastable as such).

        Raises:
            IndexError: if invalid number of dimensions specified in key.
            ValueError: if invalid value type passed.
            TooSlowToImplementError: if arbitrarily indexed key passed.

        Notes:
            Currently, this implementation only supports assignment in which
            only a single dimension is subset.  That is, for a 4x4 matrix A,
            assignment to A[1:3, :] and A[:, 1:3] are ok, but A[1:3, 1:3] is
            not.  Attempts to perform such assignment will raise a
            TooSlowToImplementError.
        """
        if isinstance(value, GPUTensor):
            value = value._tensor
        elif not isinstance(value, (int, float, numpy.ndarray)):
            raise ValueError("can only assign GPUTensor's or numeric scalars")
        if isinstance(key, tuple):
            if len(key) > self._min_dims:
                raise IndexError("CUDAMatrix only supports %d-D matrices",
                                 self._min_dims)
            elif len(key) == self._min_dims:
                if isinstance(key[0], slice):
                    start, stop, stride = key[0].indices(self.shape[0])
                    if start == 0 and stop == self.shape[0]:
                        if isinstance(key[1], slice):
                            start, stop, stride = (key[1].indices(self.
                                                                  shape[1]))
                            self._tensor.set_col_slice(start, stop, value)
                        elif isinstance(key[1], int):
                            self._tensor.set_col_slice(key[1], key[1] + 1,
                                                       value)
                        else:
                            raise TooSlowToImplementError("arbitrary "
                                                          "indexing")
                    elif isinstance(key[1], slice):
                        start_1, stop_1, stride_1 = (key[1].indices(self.
                                                                    shape[1]))
                        if start_1 == 0 and stop_1 == self.shape[1]:
                            self._tensor.set_row_slice(start, stop, value)
                        else:
                            raise TooSlowToImplementError("arbitrary "
                                                          "indexing")
                    else:
                        raise TooSlowToImplementError("arbitrary "
                                                      "indexing")
                elif isinstance(key[0], int):
                    if isinstance(key[1], slice):
                        start_1, stop_1, stride_1 = (key[1].indices(self.
                                                                    shape[1]))
                        if start_1 == 0 and stop_1 == self.shape[1]:
                            self._tensor.set_row_slice(key[0], key[0] + 1,
                                                       value)
                        else:
                            raise TooSlowToImplementError("arbitrary "
                                                          "indexing")
                    else:
                        raise TooSlowToImplementError("arbitrary "
                                                      "indexing")
                else:
                    raise TooSlowToImplementError("arbitrary "
                                                  "indexing")
        else:
            # 1-D index, unless of form x[:] = value, we treat this as
            # x[key, :] = value
            if isinstance(key, slice):
                start, stop, stride = key.indices(self.shape[0])
                if start == 0 and stop == self.shape[0]:
                    # form x[:] = value
                    if isinstance(value, numpy.ndarray):
                        self._tensor.copy_from(value)
                    else:
                        self._tensor.assign(value)
                else:
                    self._tensor.set_row_slice(start, stop, value)
            else:
                self._tensor.set_row_slice(key, key + 1, value)

    def __delitem__(self, key):
        raise ValueError("cannot delete array elements")

    def set_host_mat(self, newarray):
        """
        Changes the host pointer for this tensor to point to a new numpy array
        and its associated data. newarray must be a numpy array
        """
        self._tensor.set_host_mat(newarray)

    def copy_to_device(self):
        self._tensor.copy_to_device()

    def copy_from(self, src):
        """
        Copy contents from src.

        Arguments:
            src (numpy.ndarray): the host-resident object to copy from
        """
        self._tensor.copy_from(src)

    def transpose(self):
        return TransposedGPUTensor(self._tensor, self._tensor.T)

    def reshape(self, shape):
        return GPUTensor(self._tensor.reshape(shape))

    def take(self, indices, axis=None):
        """
        Take returns a subset of a tensor specified by indices.
        Urs modified this to be consistent with numpy, where vectors
        get flipped to always be rows.
        """
        # we only support contiguous indices at the moment because this
        # is all cudanet supports efficiently.
        if isinstance(indices, int):
            indices = [indices, ]  # cudanet only supports 2D matrix
            if self._tensor.shape[0] == 1:
                axis = 1
                # fix the axis if we are dealing with a vector. This is a hack
                # and should be done differently.
        if (indices[-1] - indices[0] == len(indices) - 1):
            if axis == 0:
                return GPUTensor(self._tensor.get_row_slice(indices[0],
                                                            indices[-1] + 1))
            elif axis == 1:
                return GPUTensor(self._tensor.get_col_slice(indices[0],
                                                            indices[-1] + 1))
            elif axis is None:
                # we might be able to do this by first doing a reshape?
                raise TooSlowToImplementError("need to first reshape")
        else:
            raise TooSlowToImplementError("CUDAMatrix can't do arbitrary"
                                          " indexing efficiently")

    def fill(self, value):
        """
        Assign specified value to each element of this CPUTensor.

        Arguments:
            value (numeric): The value to be assigned to each element.

        Return:
            CPUTensor: updated view of the data.
        """
        self._tensor.assign(value)
        return self

    def sumsq(self, axis=None):
        """
        Sum of squares of elements of a CudanetTensor. If axis is None,
        all elements are summed and a numpy scalar returned. If axis is 1
        or 2, sum along that axis and return a CudanetTensor.
        """
        if axis is None:
            result = self._tensor.sumsq(axis=None)
            logger.debug('Copying to host')
            result.copy_to_host()
            return result.numpy_array[0][0]
        else:
            result = self._tensor.sumsq(axis=axis)
            logger.debug('major change in functionality of sum')
            return GPUTensor(result)

    def log(self):
        target = cudanet.empty(self.shape)
        cudanet.log(self._tensor, target)
        return GPUTensor(target)

    def exp(self):
        target = cudanet.empty(self.shape)
        cudanet.exp(self._tensor, target)
        return GPUTensor(target)


class TransposedGPUTensor(GPUTensor):

    """
    Transposed CUDAMatrix tensor
    """

    def __init__(self, obj, transposed):
        assert type(obj) == cudanet.CUDAMatrix
        self._tensor = transposed
        self.shape = (obj.shape[1], obj.shape[0])


class GPU(Backend):

    """
    Sets up a `cuda-convnet2 <https://code.google.com/p/cuda-convnet2/>`_
    based backend for matrix operations.

    Attributes:
    """
    default_dtype = 'float32'
    tensor_cls = GPUTensor

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.par = None
        if hasattr(self, 'device_id') is False or self.device_id is None:
            self.device_id = 0
        num_devices = cudanet.get_num_devices()
        if self.device_id >= num_devices:
            raise ValueError('Requested device (%d) is unavailable.' %
                             self.device_id)
        cudanet.set_device_id(self.device_id)
        cudanet.cublas_init()
        self.rng_init()

    def default_dtype_if_missing(self, in_dtype):
        if in_dtype is None:
            in_dtype = self.default_dtype
        return in_dtype

    def __del__(self):
        pass
        # cudanet.cublas_shutdown()
        # the above is what we ought to do, but generates Exceptions due to
        # a known cudanet issue as described here:
        # https://github.com/cudanet/cudanet/issues/19

    def empty(self, shape, dtype=None, persist_values=True):
        """
        Instantiate a new instance of the GPUTensor class without initializing
        each element's value.

        Arguments:
            shape (int, list): The size of each dimension of the Tensor.
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
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(cudanet.empty(shape), dtype, persist_values)

    def array(self, obj, dtype=None, persist_values=True):
        """
        Instantiate a new instance of the GPUTensor class based on the values
        and shape of obj passed.

        Arguments:
            obj (numpy.ndarray): The n-dimensional array of values to use in
                                 initializing the values of this Tensor.  Note
                                 that python built-in types like scalar
                                 integers and lists are supported.
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
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        ndarray = numpy.array(obj, dtype=dtype)
        if ndarray.ndim == 1:
            ndarray = ndarray.reshape((1, ndarray.shape[0]))
        return self.tensor_cls(ndarray, dtype, persist_values)

    def zeros(self, shape, dtype=None, persist_values=True):
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

        Returns:
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(cudanet.CUDAMatrix(numpy.zeros(shape,
                                                              dtype=dtype)),
                               dtype, persist_values)

    def ones(self, shape, dtype=None, persist_values=True):
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

        Returns:
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(cudanet.CUDAMatrix(numpy.ones(shape,
                                                             dtype=dtype)),
                               dtype, persist_values)

    def _unwrap(self, obj):
        """
        Helper that extracts and returns the raw data underlying obj (if it is
        a GPUTensor), otherwise returns the existing structure.

        Arguments:
            obj (numeric, GPUTensor): The object to extract raw data from

        Returns:
            numeric, cudanet.CUDAMatrix: raw data from object.
        """
        if isinstance(obj, self.tensor_cls):
            return obj._tensor
        else:
            return obj

    def copy(self, tsr):
        """
        Construct and return a deep copy of the GPUTensor passed.

        Arguments:
            tsr (GPUTensor): the object to copy

        Returns:
            GPUTensor: new array object with the same values as tsr.
        """
        assert type(tsr) == self.tensor_cls
        return self.tensor_cls(tsr._tensor.copy())

    def clip(self, a, a_min, a_max, out=None):
        if out is None:
            out = self.tensor_cls(cudanet.empty((a.shape[0], a.shape[1])),
                                  self.default_dtype_if_missing(None))
        cudanet.clip_range(a._tensor, a_min, a_max, out._tensor)
        return out

    def rng_init(self):
        seed = None
        if 'rng_seed' in self.__dict__:
            seed = self.rng_seed
        numpy.random.seed(seed)
        try:
            cudanet.cudanet_init_random(seed)
        except TypeError:
            if seed is not None:
                logger.warn("Must seed random number generator with an "
                            "integer.  You specified: %s", str(seed))
            cudanet.cudanet_init_random(0)

    def flop_timing_init(self, decorate_fc, decorate_conv, decorate_ew):
        """
        Initialize FLOP timing.  Wraps the specified MOP calls via a decorator
        to record elapsed time and number of operations.

        Arguments:
           decorate_fc (list): string giving the function names of fully
                               connected layer forward/backward/update calls
                               to time.
           decorate_conv (list): string giving the function names of
                                 convolutional layer forward/backward/update
                                 calls to time.
           decorate_ew (list): string giving the function names of element-wise
                               calls to time.

        Notes:
            Must be called prior to first flop_timing_start call
        """
        # output dictionaries where the timing diagnostics are stored
        self.time_dict = defaultdict(list)
        self.flop_dict = defaultdict(list)
        self.sync = cudanet.sync_stream
        self.flop_timer = FlopsDecorator(self)
        self.flop_timer.decorate(decorate_fc=decorate_fc,
                                 decorate_conv=decorate_conv,
                                 decorate_ew=decorate_ew)

    def flop_timinig_start(self):
        """
        Start a new FLOP timer.

        Returns:
            float: timestamp.
        """
        return now()

    def flop_timing_finish(self, start_time):
        """
        Complete current FLOP timing.

        Arguments:
            start_time (float): value returned from flop_timing_start

        Returns:
            float: elapsed time in seconds since prior flop_timing_start call.
        """
        cudanet.sync_stream()
        return 1000. * (now() - start_time)

    def uniform(self, low=0.0, high=1.0, size=1, dtype=None,
                persist_values=True):
        """
        Uniform random number sample generation.

        Arguments:
            low (numeric, optional): Minimal sample value that can be returned.
                                     Defaults to 0.0
            high (numeric, optional): Maximal sample value.  Open ended range
                                      so maximal value slightly less.
                                      Defaults to 1.0
            size (array_like or int, optional): Shape of generated samples
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
            GPUTensor: Of specified size filled with these random numbers.
        """
        seq = numpy.random.uniform(low, high, size)
        dtype = self.default_dtype_if_missing(None)
        return self.tensor_cls(numpy.array(seq, dtype), dtype, persist_values)

    def fill_uniform_thresh(self, tsr, keepthresh=0.5, dtype=None):
        """
        Uniform random number sample generation.

        Arguments:
            tsr (GPUTensor): Fill this with zeros or ones based on
                             sample values from uniform distribution.  Ones
                             are used where sample is > keepthresh, else zero.
            keepthresh (numeric, optional): Minimal sample value that can be
                                            returned to set element to one.
                                            Defaults to 0.5
        Returns:
            GPUTensor: Of specified size filled with these random numbers.
        """
        # This slow implementation is kept here in commented form should you
        # need to ensure consistency with CPU generated random numbers:
        # tsr._tensor.numpy_array[:] = numpy.array((numpy.random.uniform(
        #     size=tsr._tensor.shape) < keepthresh) / keepthresh,
        #     dtype=self.default_dtype_if_missing(None))
        # tsr.copy_to_device()

        # This implementation is faster but breaks consistency with CPU
        # backend based random numbers:
        tsr._tensor.randomize_uniform_thresh(keepthresh=keepthresh)

    def make_binary_mask(self, tsr, keepthresh=0.5, dtype=None):
        tsr._tensor.randomize_uniform_thresh(keepthresh=keepthresh)
        self.multiply(tsr, keepthresh, out=tsr)

    def gdm_compound(self, ps_item, us_item, vs_item, momentum_coef,
                     learning_rate, epoch):
        """
        Compound call that wraps the sequence of elementwise operations
        required to perform a Gradient Descent with Momentum update

        Arguments:
            ps_item (GPUTensor): parameter tensor (e.g. a weight matrix)
            us_item (GPUTensor): update tensor, contains gradient wrt. weights
            vs_item (GPUTensor): velocity tensor.
            momentum_coef (float): momentum coefficient.
            learning_rate (float): learning rate.
            epoch (int): epoch (used in conjunction with diagnostics).
        """
        self.multiply(vs_item, momentum_coef, out=vs_item)  # reduce old v
        self.multiply(us_item, learning_rate, out=us_item)  # reduce new up
        self.subtract(vs_item, us_item, out=vs_item)        # compute new v
        self.add(ps_item, vs_item, out=ps_item)             # apply new v

    def gdmwd_compound(self, ps_item, us_item, vs_item, momentum_coef,
                       learning_rate, wd, epoch):
        """
        Compound call that wraps the sequence of elementwise operations
        required to perform a Gradient Descent with Momentum and Weight Decay
        update.

        Arguments:
            ps_item (GPUTensor): parameter tensor (e.g. a weight matrix)
            us_item (GPUTensor): update tensor, contains gradient wrt. weights
            vs_item (GPUTensor): velocity tensor.
            momentum_coef (float): momentum coefficient.
            learning_rate (float): learning rate.
            wd (float): weight decay parameter.
            epoch (int): epoch (used in conjunction with diagnostics).
        """
        self.multiply(vs_item, momentum_coef, out=vs_item)
        self.multiply(us_item, learning_rate, out=us_item)
        self.subtract(vs_item, us_item, out=vs_item)

        self.multiply(ps_item, wd, out=us_item)
        self.multiply(us_item, learning_rate, out=us_item)
        self.subtract(vs_item, us_item, out=vs_item)

        self.add(ps_item, vs_item, out=ps_item)

    def normal(self, loc=0.0, scale=1.0, size=1, dtype=None,
               persist_values=True):
        """
        Gaussian/Normal random number sample generation

        Arguments:
            loc (numeric, optional): Where to center distribution.  Defaults
                                     to 0.0
            scale (numeric, optional): Standard deviaion.  Defaults to 1.0
            size (array_like or int, optional): Shape of generated samples
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
            GPUTensor: Of specified size filled with these random numbers.
        """
        seq = numpy.random.normal(loc, scale, size)
        dtype = self.default_dtype_if_missing(None)
        return self.tensor_cls(numpy.array(seq, dtype), dtype, persist_values)

    def add(self, left, right, out):
        """
        Perform element-wise addition on the operands left and right, storing
        the result in the GPUTensor out.  Each operand and out is assumed to
        have identical shape, or be broadcastable as such.

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(left, self.tensor_cls):
            left._tensor.add(self._unwrap(right), out._tensor)
        elif isinstance(right, self.tensor_cls):
            right._tensor.add(left, out._tensor)
        else:
            left = self.tensor_cls(left)
            left._tensor.add(right, out._tensor)
        return out

    def subtract(self, left, right, out):
        """
        Perform element-wise subtraction on the operands left and right,
        storing the result in the GPUTensor out.  Each operand and out is
        assumed to have identical shape, or be broadcastable as such.

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(left, self.tensor_cls):
            left._tensor.subtract(self._unwrap(right), out._tensor)
        elif isinstance(right, self.tensor_cls):
            right._tensor.subtract(left, out._tensor)
            out._tensor.mult(-1.0, out._tensor)
        else:
            left = self.tensor_cls(left)
            left._tensor.subtract(right, out._tensor)
        return out

    def multiply(self, left, right, out):
        """
        Perform element-wise multiplication on the operands left and right,
        storing the result in the GPUTensor out.  Each operand and out is
        assumed to have identical shape, or be broadcastable as such.

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(left, self.tensor_cls):
            left._tensor.mult(self._unwrap(right), out._tensor)
        elif isinstance(right, self.tensor_cls):
            right._tensor.mult(left, out._tensor)
        else:
            left = self.tensor_cls(left)
            left._tensor.mult(right, out._tensor)
        return out

    def divide(self, left, right, out):
        """
        Perform element-wise division on the operands left and right, storing
        the result in out.  Each operand and out is assumed to have identical
        shape, or be broadcastable as such.

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if not isinstance(left, self.tensor_cls):
            left = self.tensor_cls(left)
        left._tensor.divide(self._unwrap(right), out._tensor)
        return out

    def power(self, tsr, power, out):
        """
        Perform element-wise raise of tsr values to specified power,
        storing the result in GPUTensor out.  Both GPUTensor's should have
        identical shape.

        Arguments:
            tsr (GPUTensor): input to be transformed.
            power (GPUTensor, numeric): Exponentiated value to be applied to
                                        elements.  Examples include 2 (square),
                                        0.5 (sqaure root).
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        cudanet.pow(tsr._tensor, self._unwrap(power), out._tensor)
        return out

    def reciprocal(self, a, out):
        a._tensor.reciprocal(out._tensor)
        return out

    def dot(self, left, right, out, alpha=1, beta=0):
        """
        Perform sum product between the last axis of left and the second last
        axis of right, storing the result in out.  Note that this dot product
        is equivalent to the inner product if operands are vectors, and matrix
        multiplication if both operands are matrices.  We support BLAS Level 3
        general matrix multiplication (GEMM) functionality by including
        additional scalars alpha and beta.  The general form of the multiply
        is: out <- alpha * left . right + beta * out, but will be
        short-circuited to: out <- alpha * left . right if beta has value 0
        (the default).  All GPUTensor's should have commensurate shape or be
        broadcastable as such.

        Arguments:
            left (GPUTensor): left-hand side operand.
            right (GPUTensor): right-hand side operand.
            out (GPUTensor): where the result will be stored.  Note that this
                             object should differ from left and right.
            alpha (numeric, optional): scalar to multiply the resultant sum
                                       product by.  Defaults to 1.
            beta (numeric, optional): scalar to pre-multiply out values by
                                      prior to adding to sum product.  Defaults
                                      to 0, which implies no such addition of
                                      prior out values.

        Returns:
            GPUTensor: reference to out
        """
        cudanet.dot(left._tensor, right._tensor, out._tensor, beta, alpha)
        return out

    def equal(self, left, right, out):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(left, self.tensor_cls):
            left._tensor.equals(self._unwrap(right), out._tensor)
        elif isinstance(right, self.tensor_cls):
            right._tensor.equals(left, out._tensor)
        else:
            left = self.tensor_cls(left)
            left._tensor.equals(right, out._tensor)
        return out

    def not_equal(self, left, right, out):
        """
        Performs element-wise non-equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.equal(left, right, out)
        out._tensor.equals(0, out._tensor)
        return out

    def greater(self, left, right, out):
        """
        Performs element-wise greater than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if not isinstance(left, self.tensor_cls):
            left = self.tensor_cls(left)
        left._tensor.greater_than(self._unwrap(right), out._tensor)
        return out

    def greater_equal(self, left, right, out):
        """
        Performs element-wise greater than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        # we calculate >= as not <
        self.less(left, right, out)
        out._tensor.equals(0, out._tensor)
        return out

    def less(self, left, right, out):
        """
        Performs element-wise less than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if not isinstance(left, self.tensor_cls):
            left = self.tensor_cls(left)
        left._tensor.less_than(self._unwrap(right), out._tensor)
        return out

    def less_equal(self, left, right, out):
        """
        Performs element-wise less than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        # we calculate <= as not >
        self.greater(left, right, out)
        out._tensor.equals(0, out._tensor)
        return out

    def norm(self, tsr, order=None, axis=None, out=None):
        """
        Calculates and returns the vector p-norms of the GPUTensor along the
        specified axis.  The p-norm is defined on a vector A as
        :math:`||A||_p = \sum_i(|A_i|^p)^{1/p}`.

        Arguments:
            tsr (GPUTensor): the GPUTensor on which to find the norms
            order (int): The order or p upon which the norm is calculated.
                         Valid values include:
                         None, inf, -inf, 0, 1, -1, 2, -2, ...
            axis (int): The axis along which to compute vector norms.
            out (GPUTensor, optional): where to write the results to.  Must be
                                       of the expected result shape.  If not
                                       specified, a new buffer is created and
                                       returned.

        Returns:
            GPUTensor: p-norm of tsr along the specified axis.

        Raises:
            IndexError if invalid axis specified
            AttributeError if invalid order specified

        See Also:
            `numpy.linalg.norm`
        """
        if not isinstance(axis, int) or axis < 0 or axis >= len(tsr.shape):
            raise IndexError("invalid axis value: %s", axis)
        if not isinstance(order, (int, float)):
            raise AttributeError("invalid order value: %s", order)
        if out is None:
            out_shape = list(tsr.shape)
            out_shape[axis] = 1
            out = self.empty(out_shape)
        if order == float('Inf'):
            self.max(self.fabs(tsr), axis, out)
        elif order == float('-Inf'):
            self.min(self.fabs(tsr), axis, out)
        elif order == 0:
            tmp = self.zeros(tsr.shape)
            self.not_equal(tsr, tmp, tmp)
            self.sum(tmp, axis, out)
        else:
            tmp = self.empty(tsr.shape)
            self.power(self.fabs(tsr), order, tmp)
            self.sum(tmp, axis, out)
            self.power(out, (1.0 / order), out)
        return out

    def xcov(self, a, b, out):
        cudanet.xcov(a._tensor, b._tensor, out._tensor)

    def mean_norm(self, a, axis, out):
        a._tensor.mean_norm(axis, out._tensor)

    def exp(self, x, out):
        cudanet.exp(x._tensor, out._tensor)

    def log(self, x, out):
        cudanet.log(x._tensor, out._tensor)

    def logistic(self, x, out):
        cudanet.sigmoid(x._tensor, out._tensor)

    def tanh(self, x, out):
        cudanet.tanh(x._tensor, out._tensor)

    def rectlin(self, x, out):
        # x and out are the same buffer
        cudanet.maximum_scalar(x._tensor, 0., out._tensor)

    def rectlin_derivative(self, x, out):
        self.greater(x, 0, out=out)

    def rectleaky(self, x, slope, out):
        self.multiply(x, slope, out)
        cudanet.maximum(x._tensor, out._tensor, out._tensor)

    def rectleaky_derivative(self, x, slope, out):
        self.greater(x, 0, out=out)
        self.multiply(out, (1.0 - slope), out=out)
        self.add(out, slope, out=out)

    def sum(self, tsr, axes, out):
        """
        Calculates the summation of the elements along the specified axes.

        Arguments:
            tsr (Tensor): the Tensor on which to perform the sum
            axes (int, list, optional): the dimension(s) along which to sum.
                                        If set to None, we will sum over all
                                        dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out
        """
        if isinstance(axes, (tuple, list)):
            logger.warn("GPUTensor only supports single axis for sum.  "
                        "You specified: %s", str(axes))
        else:
            tsr._tensor.sum(axis=axes, target=out._tensor)
        return out

    def mean(self, tsr, axes, out):
        """
        Calculates the arithmetic mean of the elements along the specified
        axes.

        Arguments:
            tsr (Tensor): the Tensor on which to compute the average
            axes (int, list, optional): the dimension(s) along which to
                                        average.  If set to None, we will
                                        average over all dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out
        """
        if isinstance(axes, (tuple, list)):
            logger.warn("GPUTensor only supports single axis for mean.  "
                        "You specified: %s", str(axes))
        else:
            tsr._tensor.mean(axis=axes, target=out._tensor)
        return out

    def variance(self, tsr, axes, out, mean=None):
        """
        Calculates the variance of the elements along the specified
        axes.

        Arguments:
            tsr  (GPUTensor): the Tensor on which to compute the variance
            axes (int, list, optional): the dimension(s) along which to
                                        variance.  If set to None, we will
                                        variance over all dimensions.
            out (GPUTensor): where the result will be stored.
            mean (GPUTensor): the Tensor containing mean of tsr

        Returns:
            Tensor: reference to out
        """
        if mean is None:
            logger.error("GPUTensor requires mean to be specified.")
            raise ValueError("mean not specified")
        if isinstance(axes, (tuple, list)):
            logger.warn("GPUTensor only supports single axis for var.  "
                        "You specified: %s", str(axes))
        else:
            tsr._tensor.var(axis=axes, mean=mean._tensor, target=out._tensor)
        return out

    def min(self, tsr, axes, out):
        """
        Calculates the minimal element value along the specified axes.

        Arguments:
            tsr (GPUTensor): the GPUTensor on which to compute the minimum
            axes (int, list, optional): the dimension(s) along which to find
                                        the minimum.  If set to None, we will
                                        compute the overall minimal value
                                        across all dimensions.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(axes, (tuple, list)):
            logger.warn("GPUTensor only supports single axis for min.  "
                        "You specified: %s", str(axes))
        else:
            tsr._tensor.min(axis=axes, target=out._tensor)
        return out

    def max(self, tsr, axes, out):
        """
        Calculates the maximal element value along the specified axes.

        Arguments:
            tsr (GPUTensor): the GPUTensor on which to compute the maximum
            axes (int, list, optional): the dimension(s) along which to find
                                        the maximum.  If set to None, we will
                                        compute the overall maximal value
                                        across all dimensions.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(axes, (tuple, list)):
            logger.warn("GPUTensor only supports single axis for max.  "
                        "You specified: %s", str(axes))
        else:
            tsr._tensor.max(axis=axes, target=out._tensor)
        return out

    def argmin(self, tsr, axis, out):
        """
        Calculates the indices of the minimal element value along the specified
        axis.  If multiple elements contain the minimum, only the elements of
        the first are returned.

        Arguments:
            tsr (GPUTensor): The GPUTensor on which to find the minimum indices
            axis (int): The dimension along which to find the minimum.  If set
                        to None, find the overall minimum index of a flattened
                        representation of tsr.
            out (GPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape

        Returns:
            GPUTensor: reference to out
        """
        tsr._tensor.argmin(axis, target=out._tensor)
        return out

    def argmax(self, tsr, axis, out):
        """
        Calculates the indices of the maximal element value along the specified
        axis.  If multiple elements contain the maximum, only the elements of
        the first are returned.

        Arguments:
            tsr (GPUTensor): The GPUTensor on which to find the maximum indices
            axis (int): The dimension along which to find the maximum.  If set
                        to None, find the overall maximum index of a flattened
                        representation of tsr.
            out (GPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape

        Returns:
            GPUTensor: reference to out
        """
        tsr._tensor.argmax(axis, target=out._tensor)
        return out

    def fabs(self, x, out=None):
        """
        calculate the absolute value for floats.

        Arguments:
            x (GPUTensor): The GPUTensor on which to find the maximum indices
            out (GPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape

        Returns:
            GPUTensor: reference to out
        """
        if out is not None:
            res = cudanet.abs(x._tensor, out._tensor)
        else:
            # XXX: temporary fix.
            res = cudanet.abs(x._tensor, cudanet.empty(x.shape))
        return GPUTensor(res)

    def sqrt(self, x, out):
        res = cudanet.sqrt(x._tensor, out._tensor)
        return GPUTensor(res)

    def softmax(self, x, out):
        cudanet.softmax(x._tensor, out._tensor)

    def softmax_gradient(self, y, err, out):
        cudanet.softmax_grad(y._tensor, err._tensor, out._tensor)

    def nonzero(self, x):
        res = x._tensor.copy()
        res.equals(0)
        res.equals(0)
        return GPUTensor(res)

    def fprop_fc(self, out, inputs, weights, layer=None):
        """
        Forward propagate the inputs of a fully connected network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            weights (GPUTensor): The weight coefficient values for this layer.
            layer (Layer): The layer object.
        """
        cudanet.dot(weights._tensor, inputs._tensor, out._tensor)

    def bprop_fc(self, out, weights, deltas, layer=None):
        """
        Backward propagate the error through a fully connected network layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            weights (GPUTensor): The weight coefficient values for this layer.
            deltas (GPUTensor): The error values for this layer
            layer (Layer): The layer object.
        """
        cudanet.dot(weights.transpose()._tensor, deltas._tensor, out._tensor)

    def update_fc(self, out, inputs, deltas, layer=None):
        """
        Compute the updated gradient for a fully connected network layer.

        Arguments:
            out (GPUTensor): Where to store the updated gradient value.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (GPUTensor): The error values for this layer
            layer (Layer): The layer object.
        """
        cudanet.dot(deltas._tensor, inputs.transpose()._tensor, out._tensor)

    def fprop_conv(self, out, inputs, weights, ofmshape, ofmsize, ofmlocs,
                   ifmshape, links, nifm, padding, stride, ngroups, fpropbuf,
                   local=False):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            weights (GPUTensor): The weight coefficient values for this layer.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              number of height and width neurons).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).  For this
                              backend we expect these values to be square.
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            fpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  convolved outputs for a single receptive
                                  field.  Not used for this backend.
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
        """
        assert ifmshape[-2] == ifmshape[-1]
        cudanet.convolution(
            weights._tensor, inputs._tensor, out._tensor,
            ifmshape[-2], ofmshape[-2], ofmshape[-1], padding, stride, nifm,
            ngroups)

    def bprop_conv(self, out, weights, deltas, ofmshape, ofmsize, ofmlocs,
                   ifmshape, links, padding, stride, nifm, ngroups, bpropbuf,
                   local=False):
        """
        Backward propagate the error through a convolutional network layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            weights (GPUTensor): The weight coefficient values for this layer.
            deltas (GPUTensor): The error values for this layer
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            bpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  backpropagated error for a single receptive
                                  field
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
        """
        cudanet.deconvolve_errors(
            weights._tensor, deltas._tensor,
            out._tensor, ifmshape[-2], ifmshape[-1], ofmshape[-2],
            padding, stride, nifm, ngroups)

    def update_conv(self, out, inputs, weights, deltas, ofmshape, ofmsize,
                    ofmlocs, ifmshape, links, nifm, padding, stride, ngroups,
                    fwidth, updatebuf, local=False, layer=None):
        """
        Compute the updated gradient for a convolutional network layer.

        Arguments:
            out (GPUTensor): Where to store the updated gradient value.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            weights (GPUTensor): The weight coefficient values for this layer.
            deltas (GPUTensor): The error values for this layer
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            fwidth (int): Filter width.
            updatebuf (GPUTensor): Temporary storage buffer used to hold the
                                   updated gradient for a single receptive
                                   field
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
            layer (Layer): The layer object.
        """
        # Default sumwidth setting for most convolution layers except for
        # those with large output maps (in which case it's usually 4).
        # Following Khrizevsky's typical settings
        sumwidth = 3 if ofmshape[-2] < 32 else 4
        cudanet.deconvolve_wts(
            deltas._tensor, inputs._tensor, out._tensor,
            ifmshape[-2], ofmshape[-2], ofmshape[-1], fwidth,
            padding, stride, nifm, ngroups, sumwidth, local)

    def fprop_pool(self, out, inputs, op, ofmshape, ofmsize, ofmlocs, fshape,
                   ifmshape, links, nifm, padding, stride, fpropbuf):
        """
        Forward propagate the inputs of a Pooling network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            op (string): The type of pooling operation to apply.  We support
                         "max", "avg", "l2" currently.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              number of height and width neurons).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            fshape (tuple): Dimensions of each filter (typically height and
                            width).
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           pooling operation.
            stride (int): Number of neurons to shift the filter at each step.
            fpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  pooled outputs for a single receptive field.
        """
        op = op.lower()
        if op == "max":
            cudanet.max_pool(inputs._tensor, out._tensor, nifm, fshape[-1],
                             padding, stride, ofmshape[-1])
        elif op == "avg" or op == "mean":
            cudanet.avg_pool(
                imgs=inputs._tensor, target=out._tensor, channels=nifm,
                sizeX=fshape[-2], paddingStart=padding, moduleStride=stride,
                numModulesX=ofmshape[-2])
        elif op == "l2":
            cudanet.l2_pool(
                imgs=inputs._tensor, target=out._tensor, channels=nifm,
                sizeX=fshape[-2], paddingStart=padding, moduleStride=stride,
                numModulesX=ofmshape[-2])
        elif op == "unpool":
            cudanet.unpool_forward(
                smallMat=inputs._tensor, largeMat=out._tensor, channels=nifm,
                sizeX=fshape[-1], smallX=ifmshape[-1], largeX=ofmshape[-1])
        else:
            raise AttributeError("unexpected pooling op type: %s", op)

    def bprop_pool(self, out, fouts, inputs, deltas, op, ofmshape, ofmsize,
                   ofmlocs, fshape, fpsize, ifmshape, links, nifm, padding,
                   stride, bpropbuf):
        """
        Backward propagate the error through a pooling network layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            fouts (GPUTensor): Forward propagated outputs from the previous
                               layer.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (GPUTensor): The error values for this layer
            op (string): The type of pooling operation to apply.  We support
                         "max", "avg", "l2" currently.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                              each output feature map stored in out.
            fshape (tuple): Dimensions of each filter (typically height and
                            width).
            fpsize (int): The size of each filter.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           pooling operation.
            stride (int): Number of neurons to shift the filter at each step.
            bpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  backpropagated error for a single receptive
                                  field
        """
        op = op.lower()
        if op == "max":
            cudanet.max_pool_undo(inputs._tensor, deltas._tensor,
                                  fouts._tensor, out._tensor, fshape[-1],
                                  padding, stride, ofmshape[-1])
        elif op == "avg" or op == "mean":
            cudanet.avg_pool_undo(
                avgGrads=deltas._tensor, target=out._tensor, sizeX=fshape[-2],
                paddingStart=padding, moduleStride=stride,
                numModulesX=ofmshape[-2], imgSizeX=ifmshape[-2])
        elif op == "l2":
            cudanet.l2_pool_undo(
                imgs=inputs._tensor, l2Grads=deltas._tensor,
                l2Acts=fouts._tensor, target=out._tensor, sizeX=fshape[-2],
                paddingStart=padding, moduleStride=stride,
                numModulesX=ofmshape[-2])
        elif op == "unpool":
            cudanet.unpool_backward(
                largeMat=inputs._tensor, smallMat=out._tensor,
                channels=nifm, sizeX=fshape[-1], smallX=ifmshape[-1],
                largeX=ofmshape[-1])
        else:
            raise AttributeError("unexpected pooling op type: %s", op)

    def fprop_cmrnorm(self, out, inputs, ifmshape, nifm, ksize, alpha, beta):
        """
        Forward propagate the inputs of a CrossMap response normalization layer
        to produce output pre-activations (ready for transformation by an
        activation function).  The normalization is computed across feature
        maps at each pixel point.  The output will be same size as input.

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            nifm (int): Total number of input feature maps.
            ksize (int): Kernel size. This defines the channel indices to sum
                         over.
            alpha (int): scalar multiplier to multiply the normalization
                         denominator by.
            beta (int): scalar power to raise the normalization denominator by
            fpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  normalized outputs for a single receptive
                                  field.
        """
        cudanet.crossmap_response_norm(inputs._tensor, out._tensor, nifm,
                                       ksize, alpha, beta)

    def bprop_cmrnorm(self, out, fouts, inputs, deltas, ifmshape, nifm, ksize,
                      alpha, beta, bpropbuf):
        """
        Backward propagate the error through a CrossMap response normalization
        layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            fouts (GPUTensor): The forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (GPUTensor): The error values for this layer
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            nifm (int): Total number of input feature maps.
            ksize (int): Kernel size. This defines the channel indices to sum
                         over.
            alpha (int): scalar multiplier to multiply the normalization
                         denominator by.
            beta (int): scalar power to raise the normalization denominator by
            bpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  normalized outputs for a single receptive
                                  field.
        """
        cudanet.crossmap_response_norm_undo(inputs._tensor, deltas._tensor,
                                            fouts._tensor, out._tensor, nifm,
                                            ksize, alpha, beta)

    def fprop_lcnnorm(self, out, inputs, meandiffs, denoms, ifmshape, nifm,
                      ksize, alpha, beta):
        """
        Forward propagate the inputs of a local contrast normalization layer
        to produce output pre-activations (ready for transformation by an
        activation function).  The normalization is computed within feature
        maps at each pixel point.  The output will be same size as input.

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            meandiffs (GPUTensor): Storage buffer that keeps the difference
                                   between the avg pools surrounding each
                                   pixel and the pixel itself.  Should not be
                                   overwritten in between calls to fprop and
                                   bprop.
            denoms (GPUTensor): Storage buffer that keeps the denominators of
                                the normalization calculated during fprop.
                                Should not be overwritten in between calls to
                                fprop and bprop.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            nifm (int): Total number of input feature maps.
            ksize (int): Kernel size. This defines the channel indices to sum
                         over.
            alpha (int): scalar multiplier to multiply the normalization
                         denominator by.
            beta (int): scalar power to raise the normalization denominator by
        """
        cudanet.local_contrast_norm(imgs=inputs._tensor,
                                    meanDiffs=meandiffs._tensor,
                                    denoms=denoms._tensor, target=out._tensor,
                                    imgSizeX=ifmshape[-2], channels=nifm,
                                    sizeX=ksize, scale=alpha, power=beta)

    def bprop_lcnnorm(self, out, fouts, deltas, meandiffs, denoms, ifmshape,
                      nifm, ksize, alpha, beta):
        """
        Backward propagate the error through a local contrast normalization
        layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            fouts (GPUTensor): The forward propagated results.
            deltas (GPUTensor): The error values for this layer
            meandiffs (GPUTensor): Storage buffer that keeps the difference
                                   between the avg pools surrounding each
                                   pixel and the pixel itself.  Should not be
                                   overwritten in between calls to fprop and
                                   bprop.
            denoms (GPUTensor): Storage buffer that keeps the denominators of
                                the normalization calculated during fprop.
                                Should not be overwritten in between calls to
                                fprop and bprop.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            nifm (int): Total number of input feature maps.
            ksize (int): Kernel size. This defines the channel indices to sum
                         over.
            alpha (int): scalar multiplier to multiply the normalization
                         denominator by.
            beta (int): scalar power to raise the normalization denominator by

        """
        cudanet.local_contrast_norm_undo(meanDiffs=meandiffs._tensor,
                                         denoms=denoms._tensor,
                                         respGrads=deltas._tensor,
                                         respActs=fouts._tensor,
                                         target=out._tensor,
                                         channels=nifm,
                                         sizeX=ksize, scale=alpha, power=beta)

    def fprop_cmpool(self, out, inputs, weights, ifmshape, ifmsize):
        """
        Forward propagate the inputs of a CrossMap Pooling layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            weights (GPUTensor): The weight coefficient values for this layer.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            ifmsize (int): Total size of each input feature map.
        """
        # Let's do this the naive way for now
        cudanet.convolution(
            weights._tensor, inputs._tensor, out._tensor,
            ifmshape[-2], ifmshape[-2], ifmshape[-1], 0, 1,
            weights.shape[0], 1)

    def bprop_cmpool(self, out, weights, deltas, ifmshape, imfsize):
        """
        Backward propagate the error through a CrossMap pooling layer.

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            weights (GPUTensor): The weight coefficient values for this layer.
            deltas (GPUTensor): The error values for this layer
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            ifmsize (int): Total size of each input feature map.
        """
        self.fprop_cmpool(out, deltas, weights.transpose(), ifmshape, imfsize)

    def update_cmpool(self, out, inputs, deltas, ifmshape, ifmsize, updatebuf):
        """
        Compute the updated gradient for a CrossMap pooling layer.

        Arguments:
            out (GPUTensor): Where to store the updated gradient value.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (GPUTensor): The error values for this layer
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            ifmsize (int): Total size of each input feature map.
            updatebuf (GPUTensor): Temporary storage buffer used to hold the
                                   updated gradient for a single receptive
                                   field
        """
        nfilters = out.shape[0]/inputs.shape[0]
        cudanet.deconvolve_wts(
            deltas._tensor, inputs._tensor, out._tensor, ifmshape[-2],
            ifmshape[-2], ifmshape[-1], 1, 0, 1, nfilters, 1, ifmshape[-2])

    def exp_mavg(self, mavg, newval, rho):
        """
        Calculate the exponential moving average

        Arguments:
            mavg:  The running value of the moving average
            newval:  New sample to be added to the moving average
            rho:  Interpolation value
        """
        mavg._tensor.add_mult(newval._tensor, rho, 1.0 - rho)

    def ada_update(self, ps_item, us_item, gs_item, ds_item, ls_item, ss_item,
                   rho, epsilon):
        cudanet.adadelta_update(us_item._tensor, gs_item._tensor,
                                ds_item._tensor, ls_item._tensor, rho,
                                epsilon)
        self.add(ps_item, ls_item, out=ps_item)

    def rms_update(self, params, updates, run_squares, velocity, scratch_space,
                   gamma, epsilon, learning_rate, momentum_coef):

        # Update running squares
        self.multiply(run_squares, gamma, out=run_squares)
        self.multiply(updates, updates, out=scratch_space)
        self.multiply(scratch_space, 1.0 - gamma, out=scratch_space)
        self.add(run_squares, scratch_space, out=run_squares)

        # Now scale the gradient by lr / rms(grad) (with a epsilon term for
        # stability)
        self.sqrt(run_squares, out=scratch_space)
        self.add(scratch_space, epsilon, out=scratch_space)
        # reciprocal and multiply used instead of division because we don't
        # currently support scalar numerator
        self.reciprocal(scratch_space, out=scratch_space)
        self.multiply(learning_rate, scratch_space, out=scratch_space)
        self.multiply(scratch_space, updates, out=scratch_space)

        # Now update the params
        if momentum_coef == 0:
            self.subtract(params, scratch_space, out=params)
        else:
            self.multiply(velocity, momentum_coef, out=velocity)
            self.subtract(velocity, scratch_space, out=velocity)
            self.add(params, velocity, out=params)

    def sync_stream(self):
        cudanet.sync_stream()

    def set_weights(self, dev_weights, host_weights):
        """
        sets the GPUTensor dev_weights to the values in host_weights
        """
        dev_weights[:] = GPUTensor(numpy.array(host_weights, 'float32'))
