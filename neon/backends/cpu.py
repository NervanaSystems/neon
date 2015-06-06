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
Our CPU based backend interface and tensor data structure.  Our implementation
wraps :mod:`numpy` ndarray and related operations
"""

import logging
import numpy as np

from neon.backends.backend import Backend, Tensor
from neon.util.compat import range

logger = logging.getLogger(__name__)


class CPUTensor(Tensor):
    """
    Our basic n-dimensional array data structure that resides in host memory,
    and is meant to be manipulated on the CPU.  wrapped `numpy.ndarray` tensor.

    Arguments:
        obj (numpy.ndarray): the actual data values.  Python built-in
                             types like lists and tuples are also supported.
        dtype (numpy.ndtype, optional): underlying data type of the elements.
                                        If None will use float32.
        persist_values (bool, optional): If set to True (the default), the
                                         values assigned to this Tensor will
                                         persist across multiple begin and end
                                         calls.  Setting to False may provide a
                                         performance increase if values do
                                         not need to be maintained across such
                                         calls

    See also:
        CPU

    Notes:
        Unlike numpy, in this implementation we never collapse dimensions, and
        the minimal number of dimensions will be _min_dims (currently set to 2
        to match cudanet GPU implementation).  So a wrapped scalar will have
        dimension 1x1.
    """
    _tensor = None
    _min_dims = 2

    def __init__(self, obj, dtype=None, persist_values=True):
        if dtype is None:
            dtype = 'float32'
        if type(obj) != np.ndarray:
            self._tensor = np.array(obj, dtype)
        elif obj.dtype != dtype:
            self._tensor = obj.astype(dtype)
        else:
            self._tensor = obj
        while self._tensor.ndim < self._min_dims:
            self._tensor = self._tensor.reshape(self._tensor.shape + (1, ))
        self.shape = self._tensor.shape
        self.dtype = dtype
        self.persist_values = persist_values

    @property
    def raw(self):
        return self._tensor

    def __str__(self):
        """
        Display a suitable representation of this Tensor.

        Returns:
            str: the representation.
        """
        return str(self._tensor)

    def __repr__(self):
        return ("%s(%s)" % (self.__class__.__name__, str(self)))

    def _clean(self, val):
        """
        Replaces any CPUTensor indices with `numpy` arrays.

        Arguments:
            val (int, array_like, CPUTensor): the items to index by.

        Returns:
            int, array_like, CPUTensor: Transformed val
        """
        if isinstance(val, tuple):
            val = tuple(x._tensor.squeeze() if isinstance(x, self.__class__)
                        else x for x in val)
        if isinstance(val, self.__class__):
            val = val._tensor.squeeze()
        return val

    def asnumpyarray(self):
        """
        Convert the CPUTensor to an in host memory `numpy.ndarray`.  A copy of
        the data may be made depending on where the CPUTensor normally resides.

        Returns:
            numpy.ndarray view or copy of the CPUTensor data.
        """
        return self._tensor

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
            CPUTensor: view of self corresponding to the subset items.

        See Also:
            take
        """
        return self.__class__(self._tensor[self._clean(key)],
                              dtype=self._tensor.dtype)

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
        try:
            self._tensor[self._clean(key)] = self._clean(value)
        except ValueError:
            # can come about due to numpy's dimension collapsing. ex. trying to
            # assign a 5x1 value to a vector of length 5.  Not sure there's a
            # way to avoid the expensive reshape op here?
            clean_key = self._clean(key)
            req_shape = self._tensor[clean_key].shape
            self._tensor[clean_key] = np.reshape(self._clean(value), req_shape)

    def __delitem__(self, key):
        raise ValueError("cannot delete array elements")

    def copy_from(self, src):
        self._tensor[:] = src

    def transpose(self):
        return self.__class__(self._tensor.transpose(),
                              dtype=self._tensor.dtype)

    def reshape(self, shape):
        return self.__class__(self._tensor.reshape(shape),
                              dtype=self._tensor.dtype)

    def take(self, indices, axis=None):
        if type(indices) == self.__class__:
            indices = indices._tensor
        # if indices are nx1 or 1xn, much of our code assumes these dims are
        # collapsed, hence the squeeze call.
        if type(indices) == np.ndarray:
            indices = indices.squeeze()
        return self.__class__(self._tensor.take(indices, axis),
                              self._tensor.dtype)

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

    def repeat(self, repeats, axis):
        return self.__class__(self._tensor.repeat(repeats, axis))

    def log(self):
        return self.__class__(np.log(self._tensor))

    def exp(self):
        return self.__class__(np.exp(self._tensor))

    def sumsq(self, axis=None, dtype='float32', out=None):
        res = np.sum(self._tensor * self._tensor, axis, dtype, out)
        if axis is None:
            return res
        else:
            return self.__class__(res)


class CPU(Backend):

    """
    Sets up a :mod:`numpy` based backend for matrix ops.  By default, we use
    32-bit element data types for any arrays constructed.

    Attributes:
        default_dtype (dtype): default element data type.  We assume 32-bit
                               float
    See also:
        CPUTensor
    """
    default_dtype = 'float32'
    tensor_cls = CPUTensor

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.err_init()
        self.par = None
        self.rng_init()

    def default_dtype_if_missing(self, in_dtype):
        if in_dtype is None:
            in_dtype = self.default_dtype
        return in_dtype

    def empty(self, shape, dtype=None, persist_values=True):
        """
        Instantiate a new instance of the CPUTensor class without initializing
        individual element values.

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
            CPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.empty(shape, dtype), dtype, persist_values)

    def array(self, obj, dtype=None, persist_values=True):
        """
        Instantiate a new instance of the CPUTensor class setting each element
        value to what is specified in obj.

        Arguments:
            obj (numpy.ndarray): The data structure containing element values
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
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.array(obj, dtype), dtype, persist_values)

    def zeros(self, shape, dtype=None, persist_values=True):
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
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.zeros(shape, dtype), dtype, persist_values)

    def ones(self, shape, dtype=None, persist_values=True):
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
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.ones(shape, dtype), dtype, persist_values)

    def _unwrap(self, obj):
        """
        Helper that extracts and returns the raw data underlying obj (if it is
        a CPUTensor), otherwise returns the existing structure.

        Arguments:
            obj (numeric, CPUTensor): The object to extract raw data from

        Returns:
            numeric, numpy.ndarray: raw data from object.
        """
        if isinstance(obj, self.tensor_cls):
            return obj._tensor
        else:
            return obj

    def copy(self, tsr):
        """
        Construct and return a deep copy of the CPUTensor passed.

        Arguments:
            tsr (CPUTensor): the object to copy

        Returns:
            CPUTensor: new array object with the same values as tsr.
        """
        return self.tensor_cls(np.copy(tsr._tensor))

    def clip(self, a, a_min, a_max, out=None):
        if out is None:
            out = self.tensor_cls(np.empty_like(a._tensor))
        np.clip(a._tensor, a_min, a_max, out._tensor)
        return out

    def err_init(self):
        # support numpy.seterr settings:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html
        if ('seterr_handling' in self.__dict__ and self.seterr_handling is not
                None):
            logger.info("Updating numpy.seterr settings: %s",
                        str(self.seterr_handling))
            np.seterr(**self.seterr_handling)

    def rng_init(self):
        seed = None
        if 'rng_seed' in self.__dict__:
            seed = self.rng_seed
            logger.info("Seeding random number generator with: %s", str(seed))
        np.random.seed(seed)

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
            Tensor: Of specified size filled with these random numbers.
        """
        return self.tensor_cls(np.random.uniform(low, high, size), dtype,
                               persist_values)

    def fill_uniform_thresh(self, a, keepthresh=0.5, dtype=None):
        """
        Uniform random number sample generation.

        Arguments:
            a (dtype): CPUTensor to fill with zeros or ones based on whether
                       sample from uniform distribution is < keepthresh
            keepthresh (numeric, optional): Minimal sample value that can be
                                          returned. Defaults to 0.5
        Returns:
            Tensor: Of specified size filled with these random numbers.
        """
        a._tensor[:] = np.array(
            np.random.uniform(size=a._tensor.shape) < keepthresh,
            dtype=a._tensor.dtype)
        a._tensor[:] = a._tensor[:] / keepthresh

    def make_binary_mask(self, tsr, keepthresh=0.5, dtype=None):
        tsr._tensor[:] = np.array(
            np.random.uniform(size=tsr._tensor.shape) < keepthresh,
            dtype=tsr._tensor.dtype)

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
            Tensor: Of specified size filled with these random numbers.
        """
        return self.tensor_cls(np.random.normal(loc, scale, size), dtype,
                               persist_values)

    def add(self, left, right, out):
        """
        Perform element-wise addition on the operands left and right, storing
        the result in the CPUTensor out.  Each operand and out is assumed to
        have identical shape, or be broadcastable as such.

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.add(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def subtract(self, left, right, out):
        """
        Perform element-wise subtraction on the operands left and right,
        storing the result in the CPUTensor out.  Each operand and out is
        assumed to have identical shape, or be broadcastable as such.

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.subtract(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def multiply(self, left, right, out):
        """
        Perform element-wise multiplication on operands left and right,
        storing the result in the CPUTensor out.  Each operand and out is
        assumed to have identical shape, or be broadcastable as such.

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.multiply(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def divide(self, left, right, out):
        """
        Perform element-wise division on the operands left and right, storing
        the resultant values in the CPUTensor out.  Each operand and out is
        assumed to have identical shape, or be broadcastable as such.

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.divide(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def power(self, tsr, power, out):
        """
        Perform element-wise raise of tsr values to specified power,
        storing the result in CPUTensor out.  Both CPUTensor's should have
        identical shape.

        Arguments:
            tsr (CPUTensor): input to be transformed.
            power (CPUTensor, numeric): exponentiated value to be applied to
                                        elements.  Examples include 2 (square),
                                        0.5 (sqaure root).
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.power(tsr._tensor, self._unwrap(power), out._tensor)
        return out

    def reciprocal(self, a, out):
        np.divide(1.0, a._tensor, out._tensor)
        return out

    def dot(self, left, right, out, alpha=1, beta=0):
        """
        Perform sum product between the last axis of left and the second last
        axis of right, storing the result in out.  Note that this dot product
        is equivalent to the inner product if operands are vectors, and matrix
        multiplication if both operands are matrices.  We support BLAS Level 3
        general matrix multiplication (GEMM) functionality by including
        additional scalars alpha and beta.  The general form of the multiply
        is: out <- alpha * left * right + beta * out, but will be
        short-circuited to: out <- alpha * left * right if beta has value 0
        (the default).  All CPUTensor's should have commensurate shape or be
        broadcastable as such.

        Arguments:
            left (CPUTensor): left-hand side operand.
            right (CPUTensor): right-hand side operand.
            out (CPUTensor): where the result will be stored.  Note that this
                             object should differ from left and right.
            alpha (numeric, optional): scalar to multiply the resultant sum
                                       product by.  Defaults to 1.
            beta (numeric, optional): scalar to pre-multiply out values by
                                      prior to adding to sum product.  Defaults
                                      to 0, which implies no such addition of
                                      prior out values.

        Returns:
            CPUTensor: reference to out
        """
        if beta == 0:
            np.dot(left._tensor, right._tensor, out._tensor)
        else:
            np.multiply(out._tensor, beta, out._tensor)
            tmp = np.empty(out.shape)
            np.dot(left._tensor, right._tensor, tmp)
            np.multiply(tmp, alpha, tmp)
            np.add(out._tensor, tmp, out._tensor)
        return out

    def equal(self, left, right, out):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.equal(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def not_equal(self, left, right, out):
        """
        Performs element-wise non-equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.not_equal(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def greater(self, left, right, out):
        """
        Performs element-wise greater than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.greater(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def greater_equal(self, left, right, out):
        """
        Performs element-wise greater than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.greater_equal(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def less(self, left, right, out):
        """
        Performs element-wise less than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.less(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def less_equal(self, left, right, out):
        """
        Performs element-wise less than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor, numeric): left-hand side operand.
            right (CPUTensor, numeric): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.less_equal(self._unwrap(left), self._unwrap(right), out._tensor)
        return out

    def norm(self, tsr, order=None, axis=None, out=None):
        """
        Calculates and returns the vector p-norms of the CPUTensor along the
        specified axis.  The p-norm is defined on vector A as
        :math:`||A||_p = \sum_i(|A_i|^p)^{1/p}`.

        Arguments:
            tsr (CPUTensor): the CPUTensor on which to find the norms
            order (int): The order or p upon which the norm is calculated.
                         Valid values include:
                         None, inf, -inf, 0, 1, -1, 2, -2, ...
            axis (int): The axis along which to compute vector norms.
            out (CPUTensor, optional): where to write the results to.  Must be
                                       of the expected result shape.  If not
                                       specified, a new buffer is created and
                                       returned.

        Returns:
            CPUTensor: p-norm of tsr along the specified axis.

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
            np.max(np.abs(tsr._tensor), axis, out=out._tensor, keepdims=True)
        elif order == float('-Inf'):
            np.min(np.abs(tsr._tensor), axis, out=out._tensor, keepdims=True)
        elif order == 0:
            np.sum(tsr._tensor != 0, axis, out=out._tensor, keepdims=True)
        else:
            np.sum(np.abs(tsr._tensor) ** order, axis, out=out._tensor,
                   keepdims=True)
            np.power(out._tensor, (1.0 / order), out._tensor)
        return out

    def xcov(self, a, b, out):
        a0 = a._tensor - a._tensor.mean(1, keepdims=True)
        b0 = b._tensor - b._tensor.mean(1, keepdims=True)
        np.dot(a0, b0.T, out._tensor)
        return self.divide(out, a.shape[1], out=out)

    def mean_norm(self, a, axis, out):
        if (axis == -1 or not axis):
            out._tensor = a._tensor - a._tensor.mean()
        else:
            out._tensor = a._tensor - a._tensor.mean(axis, keepdims=True)

    def exp(self, x, out):
        np.exp(x._tensor, out=out._tensor)
        return out

    def log(self, x, out):
        np.log(x._tensor, out=out._tensor)
        return out

    def logistic(self, x, out):
        self.multiply(x, -1.0, out=out)
        self.exp(out, out=out)
        self.add(out, 1.0, out=out)
        self.reciprocal(out, out=out)
        return out

    def tanh(self, x, out):
        np.tanh(x._tensor, out=out._tensor)
        return out

    def rectlin(self, x, out):
        # x and out are the same buffer
        np.maximum(x._tensor, 0., out._tensor)
        return out

    def rectlin_derivative(self, x, out):
        self.greater(x, 0, out=out)
        return out

    def rectleaky(self, x, slope, out):
        self.multiply(x, slope, out=out)
        np.maximum(x._tensor, out._tensor, out._tensor)
        return out

    def rectleaky_derivative(self, x, slope, out):
        self.greater(x, 0, out=out)
        self.multiply(out, (1.0 - slope), out=out)
        self.add(out, slope, out=out)
        return out

    def sum(self, tsr, axes, out):
        """
        Calculates the summation of the elements along the specified axes.

        Arguments:
            tsr (CPUTensor): the Tensor on which to perform the sum
            axes (int, list, optional): the dimension(s) along which to sum.
                                        If set to None, we will sum over all
                                        dimensions.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.sum(tsr._tensor, axis=axes, out=out._tensor, keepdims=True)
        return out

    def mean(self, tsr, axes, out):
        """
        Calculates the arithmetic mean of the elements along the specified
        axes.

        Arguments:
            tsr (CPUTensor): the Tensor on which to compute the average
            axes (int, list, optional): the dimension(s) along which to
                                        average.  If set to None, we will
                                        average over all dimensions.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.mean(tsr._tensor, axis=axes, out=out._tensor, keepdims=True)
        return out

    def variance(self, tsr, axes, out, mean=None):
        """
        Calculates the sample variance of the elements along the specified
        axes.

        Arguments:
            tsr (CPUTensor): the Tensor on which to compute the variance
            axes (int, list, optional): the dimension(s) along which to
                                        variance.  If set to None, we will
                                        variance over all dimensions.
            out (CPUTensor): where the result will be stored.
            mean (CPUTensor, optional): The Tensor containing mean of tsr.
                                        Value currently ignored if specified.

        Returns:
            CPUTensor: reference to out
        """
        np.var(tsr._tensor, axis=axes, out=out._tensor, keepdims=True)
        return out

    def min(self, tsr, axes, out):
        """
        Calculates the minimal element value along the specified axes.

        Arguments:
            tsr (CPUTensor): the CPUTensor on which to compute the minimum
            axes (int, list, optional): the dimension(s) along which to find
                                        the minimum.  If set to None, we will
                                        compute the overall minimal value
                                        across all dimensions.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.amin(tsr._tensor, axis=axes, out=out._tensor, keepdims=True)
        return out

    def max(self, tsr, axes, out):
        """
        Calculates the maximal element value along the specified axes.

        Arguments:
            tsr (CPUTensor): the CPUTensor on which to compute the maximum
            axes (int, list, optional): the dimension(s) along which to find
                                        the maximum.  If set to None, we will
                                        compute the overall maximal value
                                        across all dimensions.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        np.amax(tsr._tensor, axis=axes, out=out._tensor, keepdims=True)
        return out

    def argmin(self, tsr, axis, out):
        """
        Calculates the indices of the minimal element value along the specified
        axis.  If multiple elements contain the minimum, only the elements of
        the first are returned.

        Arguments:
            tsr (CPUTensor): The CPUTensor on which to find the minimum indices
            axis (int): The dimension along which to find the minimum.  If set
                        to None, find the overall minimum index of a flattened
                        representation of tsr.
            out (CPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape

        Returns:
            CPUTensor: reference to out
        """
        try:
            tsr._tensor.argmin(axis, out._tensor)
        except (ValueError, TypeError):
            # numpy does not have the option to keepdims in the argmin result
            # so we may be dealing with mismatched shapes that we need to
            # restore in a costlier way.
            out._tensor[:] = np.reshape(tsr._tensor.argmin(axis), out.shape)
        return out

    def argmax(self, tsr, axis, out):
        """
        Calculates the indices of the maximal element value along the specified
        axis.  If multiple elements contain the maximum, only the elements of
        the first are returned.

        Arguments:
            tsr (CPUTensor): The CPUTensor on which to find the maximum indices
            axis (int): The dimension along which to find the maximum.  If set
                        to None, find the overall maximum index of a flattened
                        representation of tsr.
            out (CPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape

        Returns:
            CPUTensor: reference to out
        """
        try:
            tsr._tensor.argmax(axis, out._tensor)
        except (ValueError, TypeError):
            # numpy does not have the option to keepdims in the argmax result
            # so we may be dealing with mismatched shapes that we need to
            # restore in a costlier way.
            out._tensor[:] = np.reshape(tsr._tensor.argmax(axis), out.shape)
        return out

    def fabs(self, x, out=None):
        if out is not None:
            res = np.fabs(x._tensor, out._tensor)
        else:
            res = np.fabs(x._tensor)
        return self.tensor_cls(res)

    def sqrt(self, x, out):
        res = np.sqrt(x._tensor, out._tensor)
        return self.tensor_cls(res)

    def square(self, x, out):
        np.multiply(x._tensor, x._tensor, out._tensor)
        return out

    def cube(self, x, out):
        np.multiply(x._tensor, x._tensor, out._tensor)
        np.multiply(out._tensor, x._tensor, out._tensor)
        return out

    # Not part of the API - can be moved to a utility class.
    def hstack_maps(self, obj, nfm):
        """
        Stack the feature maps horizontally.
        """
        assert obj.shape[0] % nfm == 0
        return self.tensor_cls(np.hstack(np.vsplit(obj._tensor, nfm)))

    # Not part of the API - can be moved to a utility class.
    def vstack_maps(self, obj, nfm):
        """
        Stack the feature maps vertically.
        """
        assert obj.shape[1] % nfm == 0
        return self.tensor_cls(np.vstack(np.hsplit(obj._tensor, nfm)))

    def softmax(self, x, out):
        np.subtract(x._tensor, x._tensor.max(axis=0, keepdims=True),
                    out._tensor)
        np.exp(out._tensor, out._tensor)
        # This uses some temporary storage, but might be ok?
        np.divide(out._tensor, np.sum(out._tensor, axis=0, keepdims=True),
                  out._tensor)
        return out

    def softmax_gradient(self, y, err, out):
        a = np.einsum('ij,ji->i', err._tensor.T, y._tensor)
        np.subtract(err._tensor, a[np.newaxis], out._tensor)
        np.multiply(out._tensor, y._tensor, out._tensor)
        return out

    def fprop_fc(self, out, inputs, weights, layer=None):
        """
        Forward propagate the inputs of a fully connected network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (CPUTensor): Where to store the forward propagated results.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            weights (CPUTensor): The weight coefficient values for this layer.
            layer (Layer): The layer object.
        """
        self.dot(weights, inputs, out)

    def bprop_fc(self, out, weights, deltas, layer=None):
        """
        Backward propagate the error through a fully connected network layer.

        Arguments:
            out (CPUTensor): Where to store the backward propagated errors.
            weights (CPUTensor): The weight coefficient values for this layer.
            deltas (CPUTensor): The error values for this layer
            layer (Layer): The layer object.
        """
        self.dot(weights.transpose(), deltas, out)

    def update_fc(self, out, inputs, deltas, layer=None):
        """
        Compute the updated gradient for a fully connected network layer.

        Arguments:
            out (CPUTensor): Where to store the updated gradient value.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (CPUTensor): The error values for this layer
            layer (Layer): The layer object.
        """
        self.dot(deltas, inputs.transpose(), out)

    def fprop_conv(self, out, inputs, weights, ofmshape, ofmsize, ofmlocs,
                   ifmshape, links, nifm, padding, stride, ngroups, fpropbuf,
                   local=False):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (CPUTensor): Where to store the forward propagated results.
            inputs (CPUTensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            weights (CPUTensor): The weight coefficient values for this layer.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              number of height and width neurons).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (CPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            links (CPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            fpropbuf (CPUTensor): Temporary storage buffer used to hold the
                                  convolved outputs for a single receptive
                                  field.
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
        """
        fsize = links.shape[1]
        for dst in range(ofmsize):
            # Compute the weighted average of the receptive field
            # and store the result within the destination feature map.
            # Do this for all filters in one shot.
            rflinks = links[dst]
            if local is False:
                self.dot(weights.transpose(),
                         inputs.take(rflinks, axis=0), out=fpropbuf)
            else:
                self.dot(weights[(fsize*dst):(fsize*(dst+1))].transpose(),
                         inputs.take(rflinks, axis=0), out=fpropbuf)

            out[ofmlocs[dst]] = fpropbuf

    def bprop_conv(self, out, weights, deltas, ofmshape, ofmsize, ofmlocs,
                   ifmshape, links, padding, stride, nifm, ngroups, bpropbuf,
                   local=False):
        """
        Backward propagate the error through a convolutional network layer.

        Arguments:
            out (CPUTensor): Where to store the backward propagated errors.
            weights (CPUTensor): The weight coefficient values for this layer.
            deltas (CPUTensor): The error values for this layer
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (CPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (CPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            bpropbuf (CPUTensor): Temporary storage buffer used to hold the
                                  backpropagated error for a single receptive
                                  field
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
        """
        fsize = links.shape[1]
        out.fill(0.0)
        for dst in range(ofmsize):
            rflinks = links[dst]
            if local is False:
                self.dot(weights,
                         deltas.take(ofmlocs[dst], axis=0), bpropbuf)
            else:
                self.dot(weights[(fsize*dst):(fsize*(dst+1))],
                         deltas.take(ofmlocs[dst], axis=0), out=bpropbuf)
            self.add(bpropbuf, out.take(rflinks, axis=0), out=bpropbuf)
            out[rflinks] = bpropbuf

    def update_conv(self, out, inputs, weights, deltas, ofmshape, ofmsize,
                    ofmlocs, ifmshape, links, nifm, padding, stride, ngroups,
                    fwidth, updatebuf, local=False, layer=None):
        """
        Compute the updated gradient for a convolutional network layer.

        Arguments:
            out (CPUTensor): Where to store the updated gradient value.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            weights (CPUTensor): The weight coefficient values for this layer.
            deltas (CPUTensor): The error values for this layer
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (CPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (CPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            fwidth (int): Filter width.
            updatebuf (CPUTensor): Temporary storage buffer used to hold the
                                   updated gradient for a single receptive
                                   field
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
            layer (Layer): The layer object.
        """
        fsize = links.shape[1]
        out.fill(0.0)
        for dst in range(ofmsize):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps.
            rflinks = links[dst]
            eslice = deltas.take(ofmlocs[dst], axis=0)
            if eslice.shape[1] > 1:
                # vector eslices are treated as column vectors, so are already
                # in the correct form, otherwise we need to flip.
                eslice = eslice.transpose()
            if local is False:
                self.dot(inputs.take(rflinks, axis=0), eslice, out=updatebuf)
                self.add(out, updatebuf, out=out)
            else:
                self.dot(inputs.take(rflinks, axis=0), eslice,
                         out=out[(fsize*dst):(fsize*(dst+1))])

    def fprop_pool(self, out, inputs, op, ofmshape, ofmsize, ofmlocs, fshape,
                   ifmshape, links, nifm, padding, stride, fpropbuf):
        """
        Forward propagate the inputs of a Pooling network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (CPUTensor): Where to store the forward propagated results.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            op (string): The type of pooling operation to apply.  We support
                         "max", "avg", "l2" currently.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              number of height and width neurons).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (CPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            fshape (tuple): Dimensions of each filter (typically height and
                            width).
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            links (CPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           pooling operation.
            stride (int): Number of neurons to shift the filter at each step.
            fpropbuf (CPUTensor): Temporary storage buffer used to hold the
                                  pooled outputs for a single receptive field.
        """
        rinputs = self.hstack_maps(inputs, nifm)
        for dst in range(ofmsize):
            # For this output unit, get the corresponding receptive fields
            # within all input feature maps.
            rf = rinputs.take(links[dst], axis=0)
            if op.lower() == "max":
                # Save the index of the maximum value within the receptive
                # fields.
                ofmlocs[dst] = rf._tensor.argmax(axis=0)
                # Set the pre-activations to the maximum value.
                maxvals = rf[ofmlocs[dst], range(rf.shape[1])]
                fpropbuf[dst] = maxvals
            elif op.lower() == "avg" or op.lower() == "mean":
                fpropbuf[dst] = rf._tensor.mean(axis=0)
            elif op.lower() == "l2":
                fpropbuf[dst] = self.norm(rf, 2, axis=0)
            else:
                raise AttributeError("unexpected pooling op type: %s", op)
        out[:] = self.vstack_maps(fpropbuf, nifm)

    def bprop_pool(self, out, fouts, inputs, deltas, op, ofmshape, ofmsize,
                   ofmlocs, fshape, fpsize, ifmshape, links, nifm, padding,
                   stride, bpropbuf):
        """
        Backward propagate the error through a pooling network layer.

        Arguments:
            out (CPUTensor): Where to store the backward propagated errors.
            fouts (CPUTensor): Forward propagated outputs from the previous
                               layer.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (CPUTensor): The error values for this layer
            op (string): The type of pooling operation to apply.  We support
                         "max", "avg", "l2" currently.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (CPUTensor): Indices giving the location of each element in
                              each output feature map stored in out.
            fshape (tuple): Dimensions of each filter (typically height and
                            width).
            fpsize (int): The size of each filter.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (CPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           pooling operation.
            stride (int): Number of neurons to shift the filter at each step.
            bpropbuf (CPUTensor): Temporary storage buffer used to hold the
                                  backpropagated error for a single receptive
                                  field
        """
        op = op.lower()
        bpropbuf.fill(0.0)
        if op == "avg" or op == "mean":
            self.divide(deltas, fpsize, deltas)
            bprop_slice = self.empty([links.shape[1], bpropbuf.shape[1]])
        elif op == "max":
            col_inds = list(range(bpropbuf.shape[1]))
            bprop_slice = self.empty(bpropbuf.shape[1])
        elif op == "l2":
            rinputs = self.hstack_maps(inputs, nifm)
            rfouts = self.hstack_maps(fouts, nifm)
            bprop_slice = self.empty([links.shape[1], bpropbuf.shape[1]])
        rdeltas = self.hstack_maps(deltas, nifm)
        for dst in range(ofmsize):
            if op == "max":
                rflinks = links[dst]
                inds = rflinks.take(ofmlocs[dst], axis=0)
                # Because we are using advanced indexing into bpropbuf, a
                # copy is unavoidable, hence the additional temp buffer and
                # assignment back
                self.add(bpropbuf[inds, col_inds], rdeltas[dst], bprop_slice)
                bpropbuf[inds, col_inds] = bprop_slice[:]
            elif op == "avg" or op == "mean":
                self.add(bpropbuf[links[dst]], rdeltas[dst].transpose(),
                         bprop_slice)
                bpropbuf[links[dst]] = bprop_slice[:]
            elif op == "l2":
                inds = links[dst]
                rf = rinputs.take(inds, axis=0)
                denom = self.copy(rfouts[dst].transpose())
                # If the L2 norm is zero, the entire receptive field must be
                # zeros. In that case, we set the L2 norm to 1 before using
                # it to normalize the receptive field.
                denom[denom._tensor == 0] = 1
                self.divide(rf, denom, out=rf)
                self.multiply(rdeltas[dst].transpose(), rf, out=ofmlocs)
                self.add(bpropbuf[inds], ofmlocs, bprop_slice)
                bpropbuf[inds] = bprop_slice[:]
            else:
                raise AttributeError("unexpected pooling op type: %s", op)
        out[:] = self.vstack_maps(bpropbuf, nifm)

    def fprop_cmrnorm(self, out, inputs, ifmshape, nifm, ksize, alpha, beta):
        """
        Forward propagate the inputs of a CrossMap response normalization layer
        to produce output pre-activations (ready for transformation by an
        activation function).  The normalization is computed across feature
        maps at each pixel point.  The output will be same size as input.

        Arguments:
            out (CPUTensor): Where to store the forward propagated results.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            nifm (int): Total number of input feature maps.
            ksize (int): Kernel size. This defines the channel indices to sum
                         over.
            alpha (int): scalar multiplier to multiply the normalization
                         denominator by.
            beta (int): scalar power to raise the normalization denominator by
            fpropbuf (CPUTensor): Temporary storage buffer used to hold the
                                  normalized outputs for a single receptive
                                  field.
        """
        (H, W, N) = (ifmshape[-2], ifmshape[-1], inputs.shape[1])
        rinputs = inputs._tensor.reshape((nifm, H, W, N))
        rout = out._tensor.reshape((nifm, H, W, N))
        for i in range(nifm):
            x = rinputs[max(i-ksize/2, 0):min(i-ksize/2+ksize, nifm)]
            np.square(x).sum(axis=0, out=rout[i])
        self.multiply(out, alpha, out=out)
        self.add(out, 1.0, out=out)
        self.power(out, -beta, out=out)
        self.multiply(inputs, out, out=out)

    def bprop_cmrnorm(self, out, fouts, inputs, deltas, ifmshape, nifm, ksize,
                      alpha, beta, bpropbuf):
        """
        Backward propagate the error through a CrossMap response normalization
        layer.

        Arguments:
            out (CPUTensor): Where to store the backward propagated errors.
            fouts (CPUTensor): The forward propagated results.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (CPUTensor): The error values for this layer
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            nifm (int): Total number of input feature maps.
            ksize (int): Kernel size. This defines the channel indices to sum
                         over.
            alpha (int): scalar multiplier to multiply the normalization
                         denominator by.
            beta (int): scalar power to raise the normalization denominator by
            bpropbuf (CPUTensor): Temporary storage buffer used to hold the
                                  normalized outputs for a single receptive
                                  field.
        """
        (H, W, N) = (ifmshape[-2], ifmshape[-1], inputs.shape[1])
        rinputs = inputs.reshape((nifm, H, W, N))
        rout = out.reshape((nifm, H, W, N))
        rfouts = fouts.reshape((nifm, H, W, N))
        otemp = self.copy(rfouts)
        # We can do this because rinputs[rfouts == 0].sum() == 0
        otemp[otemp._tensor == 0] = 1.0
        self.divide(rinputs, otemp, out=otemp)
        itemp = self.copy(rinputs)
        # We can do this because rfouts[rinputs == 0].sum() == 0
        itemp[itemp._tensor == 0] = 1.0
        self.divide(rfouts, itemp, out=itemp)
        self.power(otemp, 1.0 / beta, out=otemp)
        self.multiply(otemp, rfouts, out=otemp)
        self.multiply(otemp, -2 * alpha * beta, out=otemp)
        rout.fill(0.0)
        for i in range(nifm):
            for j in range(max(i-ksize/2, 0), min(i-ksize/2+ksize, nifm)):
                self.multiply(otemp[i], rinputs[j], out=bpropbuf)
                if i == j:
                    self.add(bpropbuf, itemp[i], out=bpropbuf)
                self.add(rout[i], bpropbuf, out=rout[i])
        self.multiply(deltas, out, out=out)

    def fprop_lcnnorm(self, out, inputs, meandiffs, denoms, ifmshape, nifm,
                      ksize, alpha, beta):
        """
        Forward propagate the inputs of a local contrast normalization layer
        to produce output pre-activations (ready for transformation by an
        activation function).  The normalization is computed within feature
        maps at each pixel point.  The output will be same size as input.

        Arguments:
            out (CPUTensor): Where to store the forward propagated results.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            meandiffs (CPUTensor): Storage buffer that keeps the difference
                                   between the avg pools surrounding each
                                   pixel and the pixel itself.  Should not be
                                   overwritten in between calls to fprop and
                                   bprop.
            denoms (CPUTensor): Storage buffer that keeps the denominators of
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
        (H, W, N) = (ifmshape[-2], ifmshape[-1], inputs.shape[1])
        rinputs = inputs._tensor.reshape((nifm, H, W, N))
        rmeandiff = meandiffs._tensor.reshape((nifm, H, W, N))
        routputs = out._tensor.reshape((nifm, H, W, N))

        for y in xrange(H):
            starty = y - ksize/2
            yidx = range(max(starty, 0), min(starty + ksize, H))
            hh = len(yidx)
            for x in xrange(W):
                startx = x - ksize/2
                xidx = range(max(startx, 0), min(startx + ksize, W))
                ww = len(xidx)
                patch = rinputs.take(xidx, axis=1).take(
                    yidx, axis=2).reshape((nifm, hh, ww, N))
                rmeandiff[:, x, y, :] = rinputs[:, x, y, :] - patch.mean(
                    axis=(1, 2))

        for y in xrange(H):
            starty = y - ksize/2
            yidx = range(max(starty, 0), min(starty + ksize, H))
            hh = len(yidx)
            for x in xrange(W):
                startx = x - ksize/2
                xidx = range(max(startx, 0), min(startx + ksize, W))
                ww = len(xidx)
                patch = rmeandiff.take(xidx, axis=1).take(
                    yidx, axis=2).reshape((nifm, hh, ww, N))
                np.square(patch).sum(axis=(1, 2), out=routputs[:, x, y, :])

        self.multiply(out, alpha, out=denoms)
        self.add(denoms, 1, out=denoms)
        self.power(denoms, -beta, out=out)
        self.multiply(inputs, out, out=out)

    def bprop_lcnnorm(self, out, fouts, deltas, meandiffs, denoms, ifmshape,
                      nifm, ksize, alpha, beta):
        """
        Backward propagate the error through a local contrast normalization
        layer.

        Notes:
            This will overwrite fouts

        Arguments:
            out (CPUTensor): Where to store the backward propagated errors.
            fouts (CPUTensor): The forward propagated results.
            deltas (CPUTensor): The error values for this layer
            meandiffs (CPUTensor): Storage buffer that keeps the difference
                                   between the avg pools surrounding each
                                   pixel and the pixel itself.  Should not be
                                   overwritten in between calls to fprop and
                                   bprop.
            denoms (CPUTensor): Storage buffer that keeps the denominators of
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
        (H, W, N) = (ifmshape[-2], ifmshape[-1], fouts.shape[1])
        self.multiply(fouts, -2 * alpha * beta, out=fouts)
        self.multiply(fouts, deltas, out=fouts)
        self.divide(fouts, denoms, out=fouts)
        rfouts = fouts._tensor.reshape((nifm, H, W, N))
        rdeltas = out._tensor.reshape((nifm, H, W, N))

        offset = ksize/2 - ksize + 1
        for y in xrange(H):
            starty = y + offset
            yidx = range(max(starty, 0), min(starty + ksize, H))
            hh = len(yidx)
            for x in xrange(W):
                startx = x + offset
                xidx = range(max(startx, 0), min(startx + ksize, W))
                ww = len(xidx)
                patch = rfouts.take(xidx, axis=1).take(
                    yidx, axis=2).reshape((nifm, hh, ww, N))
                np.sum(patch, axis=(1, 2), out=rdeltas[:, x, y, :])

        self.multiply(out, meandiffs, out=out)
        self.power(denoms, -beta, out=fouts)
        self.multiply(deltas, fouts, out=fouts)
        self.add(out, fouts, out=out)

    def fprop_cmpool(self, out, inputs, weights, ifmshape, ifmsize):
        """
        Forward propagate the inputs of a CrossMap Pooling layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (CPUTensor): Where to store the forward propagated results.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            weights (CPUTensor): The weight coefficient values for this layer.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            ifmsize (int): Total size of each input feature map.
        """
        tmp = self.empty([ifmsize, out.shape[1]])
        for ofmind in range(weights.shape[1]):
            ofm = out[(ofmind * ifmsize):((ofmind + 1) * ifmsize)]
            ofm.fill(0.0)
            for ifmind in range(weights.shape[0]):
                ifm = inputs[(ifmind * ifmsize):((ifmind + 1) * ifmsize)]
                self.multiply(ifm, weights[ifmind, ofmind], tmp)
                self.add(ofm, tmp, ofm)

    def bprop_cmpool(self, out, weights, deltas, ifmshape, ifmsize):
        """
        Backward propagate the error through a CrossMap pooling layer.

        Arguments:
            out (CPUTensor): Where to store the forward propagated results.
            weights (CPUTensor): The weight coefficient values for this layer.
            deltas (CPUTensor): The error values for this layer
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            ifmsize (int): Total size of each input feature map.
        """
        self.fprop_cmpool(out, deltas, weights.transpose(), ifmshape, ifmsize)

    def update_cmpool(self, out, inputs, deltas, ifmshape, ifmsize, updatebuf):
        """
        Compute the updated gradient for a CrossMap pooling layer.

        Arguments:
            out (CPUTensor): Where to store the updated gradient value.
            inputs (CPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (CPUTensor): The error values for this layer
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            ifmsize (int): Total size of each input feature map.
            updatebuf (CPUTensor): Temporary storage buffer used to hold the
                                   updated gradient for a single receptive
                                   field
        """
        out.fill(0.0)
        for ofmind in range(out.shape[1]):
            ofmd = deltas[(ofmind * ifmsize):((ofmind + 1) * ifmsize)]
            for ifmind in range(out.shape[0]):
                ifm = inputs[(ifmind * ifmsize):((ifmind + 1) * ifmsize)]
                ofmd = ofmd.reshape((1, ofmd.shape[0] * ofmd.shape[1]))
                ifm = ifm.reshape((ifm.shape[0] * ifm.shape[1], 1))
                self.dot(ofmd, ifm, updatebuf)
                out[ifmind, ofmind] = updatebuf

    def exp_mavg(self, mavg, newval, rho):
        """
        Calculate the exponential moving average

        Arguments:
            mavg:  The running value of the moving average
            newval:  New sample to be added to the moving average
            rho:  Interpolation value
        """
        mavg._tensor[:] = rho * mavg._tensor + (1.0 - rho) * newval._tensor

    def ada_update(self, ps_item, us_item, gs_item, ds_item, ls_item, ss_item,
                   rho, epsilon):
        # Accumulate E[Grad^2]
        self.multiply(gs_item, rho, out=gs_item)
        self.multiply(us_item, us_item, out=ss_item)
        self.multiply(ss_item, 1.0 - rho, out=ss_item)
        self.add(gs_item, ss_item, out=gs_item)

        # Calculate Updates
        self.add(gs_item, epsilon, out=ss_item)
        self.add(ds_item, epsilon, out=ls_item)
        self.divide(ls_item, ss_item, out=ls_item)
        self.sqrt(ls_item, out=ls_item)
        self.multiply(ls_item, -1.0, out=ls_item)
        self.multiply(ls_item, us_item, out=ls_item)

        # Accumulate E[Delt^2]
        self.multiply(ds_item, rho, out=ds_item)
        self.multiply(ls_item, ls_item, out=ss_item)
        self.multiply(ss_item, 1.0 - rho, out=ss_item)
        self.add(ds_item, ss_item, out=ds_item)

        # Final update to the params
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
        self.divide(learning_rate, scratch_space, out=scratch_space)
        self.multiply(scratch_space, updates, out=scratch_space)

        # Now update the params
        if momentum_coef == 0:
            self.subtract(params, scratch_space, out=params)
        else:
            self.multiply(velocity, momentum_coef, out=velocity)
            self.subtract(velocity, scratch_space, out=velocity)
            self.add(params, velocity, out=params)

    def set_weights(self, dev_weights, host_weights):
        """
        copies the host_weights into dev_weights
        """
        dev_weights[:] = host_weights
