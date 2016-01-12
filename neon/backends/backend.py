# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
Defines Tensor and Backend class
"""

import numpy as np
import logging
from math import ceil

logger = logging.getLogger(__name__)


class OpCollection(object):
    """
    A collection of the set of operation strings
    """
    zero_operand_ops = {"rand", "onehot"}
    unary_ops = {"finite", "neg", "abs", "sgn", "sqrt", "sqr", "exp", "log",
                 "exp2", "log2", "sig", "sig2", "tanh", "tanh2", "transpose",
                 "safelog"}
    binary_ops = {"assign", "add", "sub", "mul", "div", "eq", "ne", "lt", "le",
                  "gt", "ge", "pow", "minimum", "maximum", "dot"}
    reduction_ops = {"sum", "max", "min", "argmax", "argmin"}
    float_ops = zero_operand_ops | unary_ops | binary_ops
    ew_ops = float_ops - {'dot', 'transpose'}


class Tensor(object):
    """
    The n-dimensional array data structure. GPUTensor and Tensor inherits
    Tensor. Depending on backend, may have additional keyword arguments.
    All non-keywords arguments shall be in exact same order as Tensor.

    Arguments:
        backend (Backend): backend of the tensor.
        shape (tuple, optional): shape of the tensor.
        dtype (numpy.ndtype, optional): underlying data type of the elements.
        name (str, optional): name indentifying the tensor (used in printing).
        persist_values (bool, optional): If set to True (the default), the
                                         values assigned to this Tensor will
                                         persist across multiple begin and
                                         end calls.  Setting to False may
                                         provide a performance increase if
                                         values do not need to be maintained
                                         across such calls

    See also:
        GPUTensor class, Tensor class

    Notes:
        Unlike numpy, in this implementation we never collapse dimensions, and
        the minimal number of dimensions will be _min_dims (currently set to
        2).  So a wrapped scalar will have dimension 1x1.
    """
    def __init__(self,
                 backend,
                 shape=None,
                 dtype=np.float32,
                 name=None,
                 persist_values=True):

        self.backend = backend
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.persist_values = persist_values
        self._min_dims = 2

    def __str__(self):
        """
        Returns a string representation of this Tensor.

        Returns:
            str: the representation.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def __repr__(self):
        """
        Returns a more unambiguous string representation of the Tensor.

        Returns:
            str: the string representation.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def __len__(self):
        """
        Return the size of the leading dimension of self.

        Returns:
            int: the size of the leading dimension.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def __setitem__(self, index, value):
        """
        Assign the specified value to a subset of elements found via slice
        style indexing along each dimension. e.g. A[5:10, :] = 4.5.
        Each slice consists of start_idx:stop_idx:step_size triplets.  If
        step_size isn't specified it defaults to 1.  If start_idx isn't
        specified it defaults to 0.  If stop_idx isn't specified it defaults
        to the total number of elements along that dimension.  As such a slice
        value of ':' allows one to select all elements along that dimension.

        Arguments:
            index (int, slice, tuple): indices of each dimension's slice.
            value (numeric array, Tensor): values to be assigned to the
                                          extracted element subset.  If an
                                          array it should be the same shape
                                          as what key indexes (or be
                                          broadcastable as such).

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        """
        Extract a subset view of the items via slice style indexing
        along each dimension. e.g. A[5:10, :]. Each slice consists of
        start_idx:stop_idx:step_size triplets.  If step_size isn't specified it
        defaults to 1.  If start_idx isn't specified it defaults to 0.  If
        stop_idx isn't specified it defaults to the total number of elements
        along that dimension.  As such a slice value of ':' allows one to
        select all elements along that dimension.

        Arguments:
            index (int, slice, tuple): indices of each dimension's slice.

        Returns:
            Tensor: view of self corresponding to the subset items.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def _assign(self, value):
        """
        Assign an input value to the Tensor. The NervanaCPU does clipping
        for int and uint types, when overflow happens

        Arguments:
            value (Tensor, OpTreNode, numeric): the value to be assigned.

        """
        raise NotImplementedError()

    def set(self, ary):
        """
        Copy host array to the tensor.

        Arguments:
            ary (numpy.ndarray): host array, needs to be contiguous

        Returns:
            Tensor: self
        """
        raise NotImplementedError()

    def get(self):
        """
        Copy tensor to host as numpy array.

        Returns:
            numpy.ndarray: A host numpy array
        """
        raise NotImplementedError()

    def asnumpyarray(self):
        """
        Convert the tensor to an in host memory `numpy.ndarray`.  A copy of the
        data may be made depending on where the Tensor normally resides.

        Returns:
            numpy.ndarray view or copy of the Tensor data.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def take(self, indices, axis, out=None):
        """
        Select a subset of elements from an array across an axis

        Arguments:
            indices (Tensor, numpy ndarray): indicies of elements to select
            axis (int): axis across which to select the values
            out (Tensor, numpy ndarray, optional): place the resultant values
                                                   into this array if
                                                   specified.

        Return:
            Tensor: Tensor with selected values

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def fill(self, value):
        """
        Assign specified value to each element of this Tensor.

        Arguments:
            value (numeric): The value to be assigned to each element.

        Return:
            Tensor: updated view of the data.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def copy(self, a):
        """
        Construct and return a deep copy of the Tensor passed.

        Arguments:
            a (Tensor): the object to copy

        Returns:
            Tensor: new array object with the same values as tsr.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def copy_from(self, a):
        """
        Copy contents from `a`.

        Arguments:
            a (numpy.ndarray): the host-resident object to copy from

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def reshape(self, *shape):
        """
        Adjusts the dimensions of the data to the specified shape.  The number
        of elements represented by the new shape must be the same as before.

        Arguments:
            shape (int, list): new length of each dimension

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @property
    def T(self):
        """
        Return a transposed view of the data.

        Returns:
            Tensor: transposed view of self.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def transpose(self, out=None):
        """
        Return a transposed view of the data.  Alias of .T property needed for
        MOP compatibility.

        Arguments:
            out (Tensor, numpy ndarray, optional): place the resultant values
                                                   into this array if
                                                   specified.

        Returns:
            Tensor: transposed view of self.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def hist(self, tag):
        """
        Compute a histogram of the current tensor values.

        Arguments:
            tag (string): Tag to identify the current state of the tensor,
                          useful for disambiguating multiple histograms of the
                          same tensor at different points in time.

        Returns:
            Tensor containing the histogram data.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def __add__(self, other):
        """
        Perform `add` operations.

        Arguments:
            other: the right-hand side operand

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("add", self, other)

    def __sub__(self, other):
        return OpTreeNode.build("sub", self, other)

    def __mul__(self, other):
        return OpTreeNode.build("mul", self, other)

    def __div__(self, other):
        return OpTreeNode.build("div", self, other)

    def __truediv__(self, other):
        return OpTreeNode.build("div", self, other)

    def __pow__(self, other):
        return OpTreeNode.build("pow", self, other)

    def __radd__(self, other):
        return OpTreeNode.build("add", other, self)

    def __rsub__(self, other):
        return OpTreeNode.build("sub", other, self)

    def __rmul__(self, other):
        return OpTreeNode.build("mul", other, self)

    def __rdiv__(self, other):
        return OpTreeNode.build("div", other, self)

    def __rtruediv__(self, other):
        return OpTreeNode.build("div", other, self)

    def __rpow__(self, other):
        return OpTreeNode.build("pow", other, self)

    def __eq__(self, other):
        return OpTreeNode.build("eq", self, other)

    def __ne__(self, other):
        return OpTreeNode.build("ne", self, other)

    def __lt__(self, other):
        return OpTreeNode.build("lt", self, other)

    def __le__(self, other):
        return OpTreeNode.build("le", self, other)

    def __gt__(self, other):
        return OpTreeNode.build("gt", self, other)

    def __ge__(self, other):
        return OpTreeNode.build("ge", self, other)

    def __abs__(self):
        return OpTreeNode.build("abs", self, None)

    def __neg__(self):
        return OpTreeNode.build("neg", self, None)


class Backend(object):
    """
    Backend interface used to manipulate Tensor data. This abstract base class
    defines what operations each concrete backend must support.
    NervanaGPU and NervanaCPU inherit Backend.

    Arguments:
        rng_seed (int, optional): random number generator seed value
        default_dtype (numpy.ndtype, optional): Elemental data type to use when
                                                creating new tensors if not
                                                otherwise specified.  Defaults
                                                to np.float32
        compat_mode (str, optional): Flag to match implementation of other
                                     libraries.  Currently only 'caffe' is
                                     supported, defaults to None.
    """
    def __init__(self, rng_seed=None, default_dtype=np.float32,
                 compat_mode=None):
        # dtype
        self.default_dtype = default_dtype

        # use RandomState instead of seed
        self.rng = np.random.RandomState(rng_seed)
        self.init_rng_state = self.rng.get_state()  # for resetting state

        # batch size
        self.bsz = None
        self._min_dims = 2

        if compat_mode is not None:
            if compat_mode == 'caffe':
                self.set_caffe_compat()
            else:
                raise ValueError('%s mode not supported currently' % compat_mode)
        else:
            self.compat_mode = None

    def output_dim(self, X, S, padding, strides, pooling=False):
        """
        compute along 1 dimension, with these sizes, what will be the output dimension

        Arguments:
            X (int): input data dimension
            S (int): filter dimension
            padding (int): padding on each side
            strides (int): striding
            pooling (bool): flag for setting pooling layer size
        """

        if self.check_caffe_compat() and pooling:
            size = int(ceil(float(X - S + 2 * padding)/strides)) + 1
            if padding > 0 and (size - 1)*strides >= X + padding:
                # decrement size if last pooling op is completely in padding
                size -= 1
        else:
            # normal neon output size determination
            size = (X - S + 2 * padding)/strides + 1
        return size

    def set_caffe_compat(self):
        """
        Set flag to make layers compatible with caffe in terms of conv and pool
        layer output size determination and dropout layer implementation
        """
        self.compat_mode = 'caffe'

    def check_caffe_compat(self):
        return self.compat_mode == 'caffe'

    def iobuf(self, dim0, x=None, dtype=None, name=None, persist_values=True,
              shared=None, parallelism=None):
        """
        Allocate input and output buffer for layer based on batch size. This
        is used because the layer does not know about the batch size.

        Arguments:
            dim0 (tuple or int): I/O buffer dimension for layer (without the
                                 axis specifying the batch size).
            x (data-type, optional): If present and not None, `x` will be
                                     returned directly. `x` will be not None if
                                     the buffer has already been allocated.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            name (str, optional): name indentifying the tensor (used in printing).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor will
                                             persist across multiple begin and
                                             end calls.  Setting to False may
                                             provide a performance increase if
                                             values do not need to be maintained
                                             across such calls
            shared (buffer, optional): If present will attempt to reuse the memory
                                       in shared to allocate the I/O buffer
            parallelism (str, optional): Indicates type of parallelism (Data,
                                         Model) employed by this buffer.
                                         Ignored on CPU and GPU backends,
                                         defaults to no parallelism.
        Returns:
            Tensor: array object
        """
        if x is not None:
            return x
        if isinstance(dim0, tuple):
            if (len(dim0) == 2):
                bufshape = (dim0[0], dim0[1] * self.bsz)
            else:
                bufshape = (np.prod(dim0), self.bsz)
        else:
            bufshape = (dim0, self.bsz)

        if shared is not None:
            if shared.shape == bufshape:
                return shared
            else:
                return shared.share(bufshape)
        else:
            return self.zeros(bufshape, dtype=dtype, name=name,
                              persist_values=persist_values)

    def rng_reset(self):
        """
        Reset the random state to the state where the Backend is first
        initialized.
        usually need to do: self.rng.set_state(self.init_rng_state)
        """
        raise NotImplementedError()

    def execute(self, node):
        """
        Execute the optree. There must be one and only one 'assign' op at the
        top of the optree when execute is called.

        Arguments:
            node (OpTreeNode): The op-tree to execute.
        """
        pass

    def begin(self, block, identifier):
        """
        Signal the start of a block of repeated computation (ex. at the start
        of a loop).  This operation can be used to help the compiler optimize
        instruction performance, but has no direct effect on calculations.
        It must be book-ended by a corresponding Backend.end() call.
        Note that multiple begin calls can appear adjacent in nested loops.

        Arguments:
            block (Block.attr): identifies the type of computation being worked
                                on based on Block attribute specified
            identifier (int): unique identifier for this particular iteration
                              of the block.  Will typically be something like
                              epoch number, mini-batch number, and so forth.

        See Also:
            :py:func:`~neon.backends.backend.Backend.end`,
        """
        pass

    def end(self, block, identifier):
        """
        Signal the corresponding end of a block of repeated computation
        (ex. at the end of a loop).  This operation can be used to help the
        compiler optimize performance, but has no direct effect on
        calculations.  It must be preceded by a corresponding Backend.begin()
        call.

        Arguments:
            block (Block.attr): identifies the type of computation being worked
                                on based on Block attribute specified
            identifier (int): unique identifier for this particular iteration
                              of the block.  Will typically be something like
                              epoch number, mini-batch number, and so forth.

        See Also:
            :py:func:`~neon.backends.backend.Backend.begin`,
        """
        pass

    def empty(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False):
        """
        Instantiate a new instance of this backend's Tensor class, without
        initializing element values.  This is slightly faster than
        :py:func:`~neon.backends.Backend.array`,
        :py:func:`~neon.backends.Backend.ones`,
        :py:func:`~neon.backends.Backend.zeros`, but the values will be
        random.

        Arguments:
            shape (int, list): length of each dimension of the Tensor.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            name (str, optional): name indentifying the tensor (used in printing).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls
            parallel (bool, optional): If True and using multi-GPU backend,
                                       replicate copies of this tensor across
                                       devices.  Defaults to False, and has no
                                       effect on CPU, or (single) GPU backends.
            distributed (bool, optional): If True and using multi-GPU backend,
                                          this tensor is fragmented and
                                          partitioned across devices.  Defaults
                                          to False, and has no effect on CPU,
                                          or (single) GPU backends.

        Returns:
            Tensor: array object

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.Backend.array`,
            :py:func:`~neon.backends.Backend.zeros`,
            :py:func:`~neon.backends.Backend.ones`
        """
        raise NotImplementedError()

    def array(self, ary, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        elements based on ary values.

        Arguments:
            ary (array_like): input array object to construct from.  Can be
                              built-in python scalar or list (of lists), or a
                              numpy.ndarray
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            name (str, optional): name indentifying the tensor (used in printing).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls
            parallel (bool, optional): If True and using multi-GPU backend,
                                       replicate copies of this tensor across
                                       devices.  Defaults to False, and has no
                                       effect on CPU, or (single) GPU backends.
            distributed (bool, optional): If True and using multi-GPU backend,
                                          this tensor is fragmented and
                                          partitioned across devices.  Defaults
                                          to False, and has no effect on CPU,
                                          or (single) GPU backends.

        Returns:
            Tensor: array object

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.Backend.empty`,
            :py:func:`~neon.backends.Backend.zeros`,
            :py:func:`~neon.backends.Backend.ones`
        """
        raise NotImplementedError()

    def zeros(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        Each element with a value of 0.

        Arguments:
            shape (int, list): length of each dimension of the Tensor.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            name (str, optional): name indentifying the tensor (used in printing).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls
            parallel (bool, optional): If True and using multi-GPU backend,
                                       replicate copies of this tensor across
                                       devices.  Defaults to False, and has no
                                       effect on CPU, or (single) GPU backends.
            distributed (bool, optional): If True and using multi-GPU backend,
                                          this tensor is fragmented and
                                          partitioned across devices.  Defaults
                                          to False, and has no effect on CPU,
                                          or (single) GPU backends.

        Returns:
            Tensor: array object

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.Backend.empty`,
            :py:func:`~neon.backends.Backend.ones`,
            :py:func:`~neon.backends.Backend.array`
        """
        raise NotImplementedError()

    def ones(self, shape, dtype=None, name=None, persist_values=True,
             parallel=False, distributed=False):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        Each element with a value of 1.

        Arguments:
            shape (int, list): length of each dimension of the Tensor.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            name (str, optional): name indentifying the tensor (used in printing).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls
            parallel (bool, optional): If True and using multi-GPU backend,
                                       replicate copies of this tensor across
                                       devices.  Defaults to False, and has no
                                       effect on CPU, or (single) GPU backends.
            distributed (bool, optional): If True and using multi-GPU backend,
                                          this tensor is fragmented and
                                          partitioned across devices.  Defaults
                                          to False, and has no effect on CPU,
                                          or (single) GPU backends.

        Returns:
            Tensor: array object

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.empty`,
            :py:func:`~neon.backends.backend.Backend.zeros`,
            :py:func:`~neon.backends.backend.Backend.array`
        """
        raise NotImplementedError()

    def empty_like(self, other_ary, name=None, persist_values=True):
        """
        Instantiate a new instance of this backend's Tensor class, with the
        shape taken from other_ary.

        Arguments:
            other_ary (tensor object): Tensor to inherit the dimensions of.
            name (str, optional): name indentifying the tensor (used in printing).
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls.

        Returns:
            Tensor: array object

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.Backend.empty`,
            :py:func:`~neon.backends.Backend.ones`,
            :py:func:`~neon.backends.Backend.array`
        """
        raise NotImplementedError()

    def zeros_like(self, other_ary, name=None, persist_values=True):
        """
        Instantiate a new instance of this backend's Tensor class, with the
        shape taken from other_ary and populating each element with a value of 0.

        Arguments:
            other_ary (tensor object): Tensor to inherit the dimensions of.
            name (str, optional): name indentifying the tensor (used in printing).
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls.
        Returns:
            Tensor: array object

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.Backend.empty`,
            :py:func:`~neon.backends.Backend.ones`,
            :py:func:`~neon.backends.Backend.array`
        """
        raise NotImplementedError()

    def dot(self, a, b, out=None):
        """
        Dot product of two Tensors.

        Arguments:
            a (Tensor): left-hand side operand.
            b (Tensor): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
                                    Note that this object should differ from
                                    left and right.

        Returns:
            OpTreeNode: the resulting op-tree from this operation.
        """
        return OpTreeNode.build("dot", a, b, out=out)

    def compound_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False):
        """
        Perform one of the following operations (* is dot product)
        C = alpha * A * B   + beta * C
        C = alpha * A.T * B + beta * C
        C = alpha * A * B.T + beta * C

        relu: if true, applied before output (and prior to beta addition)

        The operation will be short-circuited to: out <- alpha * left * right
        if beta has value 0 (the default).

        Arguments:
            A (Tensor): left-hand side operand.
            B (Tensor): right-hand side operand.
            C (Tensor): output operand
            alpha (float. optional): scale A*B term
            beta (float, optional): scale C term before sum
            relu (bool, optional): If True apply ReLu non-linearity before
                                   output.  Defaults to False.
        """
        raise NotImplementedError()

    def batched_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False):
        """
        Perform one of the following operations:
        1. For fprop: A(K, C), B(X,C,N), C(X,K,N) --> call batched_dot(A, B, C)
        2. For bprop: A(K, C), B(X,K,N), C(X,C,N) --> call batched_dot(A.T, B, C)
        3. For update: A(X,K,N), B(X,C,N), C(K,C) --> call batched_dot(A, B.T, C)

        Arguments:
            A (Tensor): left-hand input operand
            B (Tensor): right-hand input operand
            C (Tensor): output operand
            alpha (float. optional): scale A*B term
            beta (float, optional): scale C term before sum
            relu (bool, optional): If True apply ReLu non-linearity before
                                   output.  Defaults to False.
        """
        raise NotImplementedError()

    def make_binary_mask(self, out, keepthresh=0.5):
        """
        Create a binary mask for dropout layers.

        Arguments:
            out (Tensor): Output tensor
            keepthresh (float, optional): fraction of ones. Defaults to 0.5
        """
        raise NotImplementedError()

    def add(self, a, b, out=None):
        """
        Perform element-wise addition on the operands, storing the resultant
        values in the out Tensor. Each operand and out must have identical
        shape or be broadcastable as such.

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("add", a, b, out=out)

    def subtract(self, a, b, out=None):
        """
        Perform element-wise subtraction on the operands, storing the resultant
        values in the out Tensor. Each operand and out must have identical
        shape or be broadcastable as such.

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("sub", a, b, out=out)

    def multiply(self, a, b, out=None):
        """
        Perform element-wise multiplication on the operands, storing the
        resultant values in the out Tensor. Each operand and out must have
        identical shape or be broadcastable as such.

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("mul", a, b, out=out)

    def divide(self, a, b, out=None):
        """
        Perform element-wise division on the operands, storing the
        resultant values in the out Tensor. Each operand and out must have
        identical shape or be broadcastable as such.

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("div", a, b, out=out)

    def true_divide(self, a, b, out=None):
        """
        Here it is an alias of divide.
        Instead of the Python traditional 'floor division', this returns a
        true division.

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("div", a, b, out=out)

    def power(self, a, b, out=None):
        """
        Perform element-wise raise of tsr values to specified power,
        storing the result in Tensor out. Both Tensor's should have identical
        shape.

        Arguments:
            a (Tensor): input to be transformed.
            b (Tensor, numeric): exponentiated value to be applied to
                                     element.  Examples include 2 (square),
                                     0.5 (sqaure root).
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("pow", a, b, out=out)

    def reciprocal(self, a, out=None):
        """
        Perform element-wise reciprocal of Tensor `a`, storing the result in
        Tensor out. Both Tensor's should have identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            power (Tensor, numeric): exponentiated value to be applied to
                                     element.  Examples include 2 (square),
                                     0.5 (sqaure root).
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("div", 1., a, out=out)

    def negative(self, a, out=None):
        """
        Perform element-wise negation of Tensor `a`, storing the result in
        Tensor out. Both Tensor's should have identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("neg", a, None, out=out)

    def sgn(self, a, out=None):
        """
        Perform element-wise indication of the sign of Tensor `a`, storing the
        result in Tensor out. Both Tensor's should have identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("sgn", a, None, out=out)

    def absolute(self, a, out=None):
        """
        Perform element-wise absolute value of Tensor `a`, storing the result in
        Tensor out. Both Tensor's should have identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("abs", a, None, out=out)

    def fabs(self, a, out=None):
        """
        Perform element-wise absolute value of Tensor `a`, storing the result
        in Tensor out. Both Tensor's should have identical shape. Implemented as
        an alias of absolute.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("abs", a, None, out=out)

    def sqrt(self, a, out=None):
        """
        Perform element-wise square-root of Tensor `a`, storing the result in
        Tensor out. Both Tensor's should have identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("sqrt", a, None, out=out)

    def square(self, a, out=None):
        """
        Perform element-wise square of Tensor `a`, storing the result in Tensor
        out. Both Tensor's should have identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("sqr", a, None, out=out)

    def exp(self, a, out=None):
        """
        Perform element-wise exponential transformation on Tensor `a`, storing
        the result in Tensor out. Both Tensor's should have identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("exp", a, None, out=out)

    def exp2(self, a, out=None):
        """
        Perform element-wise 2-based exponential transformation on Tensor `a`,
        storing the result in Tensor out. Both Tensor's should have identical
        shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("exp2", a, None, out=out)

    def safelog(self, a, out=None):
        """
        Perform element-wise natural logarithm transformation on Tensor `a`,
        storing the result in Tensor out. Both Tensor's should have identical
        shape.  This log function has built in safety for underflow.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("safelog", a, None, out=out)

    def log(self, a, out=None):
        """
        Perform element-wise natural logarithm transformation on Tensor `a`,
        storing the result in Tensor out. Both Tensor's should have identical
        shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("log", a, None, out=out)

    def log2(self, a, out=None):
        """
        Perform element-wise 2-based logarithm transformation on Tensor `a`,
        storing the result in Tensor out. Both Tensor's should have identical
        shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("log2", a, None, out=out)

    def sig(self, a, out=None):
        """
        Perform element-wise sigmoid transformation on Tensor `a`,
        storing the result in Tensor out. Both Tensor's should have identical
        shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("sig", a, None, out=out)

    def sig2(self, a, out=None):
        """
        Perform element-wise 2-based sigmoid logarithm transformation on
        Tensor `a`, storing the result in Tensor out. Both Tensor's should
        have identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("sig2", a, None, out=out)

    def tanh(self, a, out=None):
        """
        Perform element-wise hyperbolic tangent transformation on Tensor `a`,
        storing the result in Tensor out. Both Tensor's should have identical
        shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("tanh", a, None, out=out)

    def tanh2(self, a, out=None):
        """
        Perform element-wise 2-based hyperbolic tangent transformation on Tensor
        `a`, storing the result in Tensor out. Both Tensor's should have
        identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("tanh2", a, None, out=out)

    def finite(self, a, out=None):
        """
        Perform element-wise test of finiteness (not infinity or not Not a
        Number) on Tensor `a`, storing the result in Tensor out. Both Tensor's
        should have identical shape.

        Arguments:
            a (Tensor): input to be transformed.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("finite", a, None, out=out)

    def equal(self, a, b, out=None):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out. Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("eq", a, b, out=out)

    def not_equal(self, a, b, out=None):
        """
        Performs element-wise non-equality testing on each element of left and
        right, storing the result in out. Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("ne", a, b, out=out)

    def less(self, a, b, out=None):
        """
        Performs element-wise less than testing on each element of left and
        right, storing the result in out. Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("lt", a, b, out=out)

    def less_equal(self, a, b, out=None):
        """
        Performs element-wise less than or equal testing on each element of
        left and right, storing the result in out. Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("le", a, b, out=out)

    def greater(self, a, b, out=None):
        """
        Performs element-wise greater than testing on each element of left and
        right, storing the result in out. Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only theshape op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("gt", a, b, out=out)

    def greater_equal(self, a, b, out=None):
        """
        Performs element-wise greater than or equal testing on each element of
        left and right, storing the result in out. Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("ge", a, b, out=out)

    def maximum(self, a, b, out=None):
        """
        Performs element-wise maximum value assignment based on corresponding
        elements of left and right, storing the result in out. Each operand is
        assumed to be the same shape (or broadcastable as such).

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("maximum", a, b, out=out)

    def minimum(self, a, b, out=None):
        """
        Performs element-wise minimum value assignment based on corresponding
        elements of left and right, storing the result in out. Each operand is
        assumed to be the same shape (or broadcastable as such).

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("minimum", a, b, out=out)

    def clip(self, a, a_min, a_max, out=None):
        """
        Performs element-wise clipping of Tensor `a`, storing the result in out.
        The clipped value will be between [a_min, a_max].

        Arguments:
            a (Tensor, numeric): left-hand side operand.
            b (Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return self.minimum(self.maximum(a, a_min), a_max, out=out)

    def sum(self, a, axis=None, out=None, keepdims=True):
        """
        Calculates the summation of the elements along the specified axis.

        Arguments:
            a (Tensor): the Tensor on which to perform the sum
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will sum over all
                                  dimensions.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
            keepdims (bool, optional): Keep the axes being computed over in the
                                       output (with size 1), instead of
                                       collapsing.  Defaults to True.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        if axis is None:
            return OpTreeNode.build("sum", OpTreeNode.build("sum", a, None, axis=0),
                                    None, axis=1, out=out)
        return OpTreeNode.build("sum", a, None, axis=axis, out=out)

    def max(self, a, axis=None, out=None, keepdims=True):
        """
        Calculates the maximal element value along the specified axes.

        Arguments:
            a (Tensor): the Tensor on which to perform the operation
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will take max over all
                                  dimensions.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
            keepdims (bool, optional): Keep the axes being computed over in the
                                       output (with size 1), instead of
                                       collapsing.  Defaults to True.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        if axis is None:
            return OpTreeNode.build("max", OpTreeNode.build("max", a, None, axis=0),
                                    None, axis=1, out=out)
        return OpTreeNode.build("max", a, None, axis=axis, out=out)

    def min(self, a, axis=None, out=None, keepdims=True):
        """
        Calculates the minimal element value along the specified axes.

        Arguments:
            a (Tensor): the Tensor on which to perform the operation
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will take min over all
                                  dimensions.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
            keepdims (bool, optional): Keep the axes being computed over in the
                                       output (with size 1), instead of
                                       collapsing.  Defaults to True.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        if axis is None:
            return OpTreeNode.build("min", OpTreeNode.build("min", a, None, axis=0),
                                    None, axis=1, out=out)
        return OpTreeNode.build("min", a, None, axis=axis, out=out)

    def argmax(self, a, axis=1, out=None, keepdims=True):
        """
        Calculates the indices of the maximal element value along the specified
        axis.  If multiple elements contain the maximum, only the indices of
        the first are returned.

        Arguments:
            a (Tensor): the Tensor on which to perform the operation
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will take argmax over all
                                  dimensions.  Defaults to 1
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
            keepdims (bool, optional): Keep the axes being computed over in the
                                       output (with size 1), instead of
                                       collapsing.  Defaults to True.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("argmax", a, None, axis=axis, out=out)

    def argmin(self, a, axis=1, out=None, keepdims=True):
        """
        Calculates the indices of the minimal element value along the specified
        axis.  If multiple elements contain the minimum, only the indices of
        the first are returned.

        Arguments:
            a (Tensor): the Tensor on which to perform the operation
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will take argmin over all
                                  dimensions.  Defaults to 1
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
            keepdims (bool, optional): Keep the axes being computed over in the
                                       output (with size 1), instead of
                                       collapsing.  Defaults to True.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return OpTreeNode.build("argmin", a, None, axis=axis, out=out)

    def mean(self, a, axis=None, partial=None, out=None, keepdims=True):
        """
        Calculates the arithmetic mean of the elements along the specified
        axes.

        Arguments:
            a (Tensor): the Tensor on which to perform the operation
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will take mean over all
                                  dimensions.  Defaults to None
            partial (bool, optional): Not currently used.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
            keepdims (bool, optional): Keep the axes being computed over in the
                                       output (with size 1), instead of
                                       collapsing.  Defaults to True.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        shape = a.shape
        if axis is None:
            return self.multiply(self.sum(a), 1.0 / (shape[0] * shape[1]), out=out)
        return self.multiply(self.sum(a, axis=axis), 1.0 / shape[axis], out=out)

    def var(self, a, axis=None, partial=None, out=None, keepdims=True):
        """
        Calculates the variance of the elements along the specified
        axes.

        Arguments:
            a (Tensor): the Tensor on which to perform the operation
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will take var over all
                                  dimensions.  Defaults to None
            partial (bool, optional): Not currently used.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
            keepdims (bool, optional): Keep the axes being computed over in the
                                       output (with size 1), instead of
                                       collapsing.  Defaults to True.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        if axis is None:
            return self.mean(self.square(a - self.mean(a)), out=out)
        return self.mean(self.square(a - self.mean(a, axis=axis)), axis=axis, out=out)

    def std(self, a, axis=None, partial=None, out=None, keepdims=True):
        """
        Calculates the standard deviation of the elements along the specified
        axes.

        Arguments:
            a (Tensor): the Tensor on which to perform the operation
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will take std over all
                                  dimensions.
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
            partial (bool, optional): Not currently used.
            keepdims (bool, optional): Keep the axes being computed over in the
                                       output (with size 1), instead of
                                       collapsing.  Defaults to True.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        return self.sqrt(self.var(a, axis=axis, partial=partial, out=out))

    def take(self, a, indices, axis, out=None):
        """
        Extract elements based on the indices along a given axis.

        Arguments:
            a (Tensor): the Tensor on which to perform the operation
            indices (Tensor, numpy ndarray): indicies of elements to select
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will extract over all
                                  dimensions (flattened first)
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
        """
        return a.take(indices, axis, out)

    def onehot(self, indices, axis, out=None):
        """
        Generate optree for converting `indices` to a onehot representation

        Arguments:
            indices (Tensor): Elements must be of numpy integer type for gpu
                              onehot to work.
            axis (int): the axis along the feature length dimension
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        if axis not in (0, 1):
            raise ValueError("bad axis for onehot")
        return OpTreeNode.build("onehot", None, None, idx=indices, axis=axis, out=out)

    def update_fc_bias(self, err, out):
        """
        Compute the updated bias gradient for a fully connected network layer.

        Arguments:
            err (Tensor): backpropagated error
            out (Tensor): Where to store the updated gradient value.
        """
        self.ng.sum(err, axis=1, out=out)

    def add_fc_bias(self, inputs, bias):
        """
        Add the bias for a fully connected network layer.

        Arguments:
            inputs (Tensor): the input to update.
            bias (Tensor): the amount to increment
        """
        self.ng.add(inputs, bias, out=inputs)

    def conv_layer(self, dtype,
                   N, C, K,
                   D=1, H=1, W=1,
                   T=1, R=1, S=1,
                   pad_d=0, pad_h=0, pad_w=0,
                   str_d=1, str_h=1, str_w=1,
                   relu=False, bsum=False, deterministic_update=False):
        """
        Create a new ConvLayer parameter object.
        This is then passed as an argument to all the convolution operations.

        Arguments:
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.

            N (int): Number of images in mini-batch
            C (int): Number of input feature maps
            K (int): Number of output feature maps

            D (int, optional): Depth of input image.  Defaults to 1
            H (int, optional): Height of input image.  Defaults to 1
            W (int, optional): Width of input image.  Defaults to 1

            T (int, optional): Depth of filter kernel.  Defaults to 1
            R (int, optional): Height of filter kernel.  Defaults to 1
            S (int, optional): Width of filter kernel.  Defaults to 1

            pad_d (int, optional): amount of zero-padding around the depth edge
                                   Defaults to 0.
            pad_h (int, optional): amount of zero-padding around the height edge
                                   Defaults to 0.
            pad_w (int, optional): amount of zero-padding around the width edge
                                   Defaults to 0.

            str_d (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1
            str_h (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1
            str_w (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1

            relu (bool, optional): apply a relu transform to the output for
                                   fprop or bprop.  Defaults to False

            bsum (bool, optional): calculate the sum along the batchnorm axis
                                   for fprop or bprop.  Outputs an fp32 tensor
                                   of size Kx1.  Defaults to False.

            deterministic_update (bool, optional): eleminate atomic adds in the
                                                   update operation.  Increases
                                                   reproducibility but runs
                                                   slower.  Defaults to False.
        """
        raise NotImplementedError()

    def fprop_conv(self, layer, I, F, O, alpha=1.0, relu=False, repeat=1):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output

        Arguments:
            layer: the conv layer as a parameter object
            I (Tensor): inputs
            F (Tensor): the weights (filters)
            O (Tensor): outputs
            alpha (float, optional): linear scaling.  Defaults to 1.0
            relu (bool, optional): apply ReLu before output.  Default not to.
            repeat (int, optional): Repeat this operation the specified number
                                    of times.  Defaults to 1.
        """
        raise NotImplementedError()

    def bprop_conv(self, layer, F, E, grad_I, alpha=1.0, repeat=1):
        """
        Backward propagate the error through a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            F (Tensor): the weights (filters)
            E (Tensor): errors
            grad_I (Tensor): gradient to inputs (output delta)
            alpha (float, optional): linear scaling.  Defaults to 1.0
            repeat (int, optional): Repeat this operation the specified number
                                    of times.  Defaults to 1.
        """
        raise NotImplementedError()

    def update_conv(self, layer, I, E, grad_F, alpha=1.0, repeat=1):
        """
        Compute the updated gradient for a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            I (Tensor): the inputs
            E (Tensor): the errors
            grad_F (Tensor): filter gradients (weights) to update.
            alpha (float, optional): linear scaling.  Defaults to 1.0
            repeat (int, optional): Repeat this operation the specified number
                                    of times.  Defaults to 1.
        """
        raise NotImplementedError()

    def deconv_layer(self, dtype,
                     N, C, K,
                     P, Q,
                     R=1, S=1,
                     pad_d=0, pad_h=0, pad_w=0,
                     str_d=1, str_h=1, str_w=1):
        """
        Create a new Deconvolution parameter object.
        This then is passed as an argument to all deconvolution kernels.

        Arguments:
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.

            N (int): Number of images in mini-batch
            C (int): Number of input feature maps
            K (int): Number of output feature maps

            P (int): Height of output
            Q (int): Width of output

            R (int, optional): Height of filter kernel.  Defaults to 1
            S (int, optional): Width of filter kernel.  Defaults to 1

            pad_d (int, optional): amount of zero-padding around the depth edge
                                   Defaults to 0.
            pad_h (int, optional): amount of zero-padding around the height edge
                                   Defaults to 0.
            pad_w (int, optional): amount of zero-padding around the width edge
                                   Defaults to 0.

            str_d (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1
            str_h (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1
            str_w (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1

        Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.
        """
        raise NotImplementedError()

    def pool_layer(self, dtype,
                   op, N, C,
                   D=1, H=1, W=1,
                   J=1, T=1, R=1, S=1,
                   pad_j=0, pad_d=0, pad_h=0, pad_w=0,
                   str_j=None, str_d=None, str_h=None, str_w=None):
        """
        Create a new PoolLayer parameter object.
        This then is passed as an argument to all pooling kernels.

        Arguments:
            op (str): "max", "avg", "l2" pooling (currently bprop only supports
                      max, but not avg and l2)
            N (int): Number of images in mini-batch

            C (int): Number of input feature maps
            D (int, optional): Depth of input image.  Defaults to 1
            H (int, optional): Height of input image.  Defaults to 1
            W (int, optional): Width of input image.  Defaults to 1

            J (int, optional): Size of feature map pooling window
                               (maxout n_pieces).  Defaults to 1
            T (int, optional): Depth of pooling window.  Defaults to 1
            R (int, optional): Height of pooling window.  Defaults to 1
            S (int, optional): Width of pooling window.  Defaults to 1

            pad_j (int, optional): amount of zero-padding around the fm pooling
                                   window edge.  Defaults to 0.
            pad_d (int, optional): amount of zero-padding around the depth edge
                                   Defaults to 0.
            pad_h (int, optional): amount of zero-padding around the height edge
                                   Defaults to 0.
            pad_w (int, optional): amount of zero-padding around the width edge
                                   Defaults to 0.

            str_d (int, optional): factor to step the filters by in the fm
                                   pooling window direction.  Defaults to 1
            str_d (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1
            str_h (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1
            str_w (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1

        Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.
        """
        raise NotImplementedError()

    def fprop_pool(self, layer, I, O):
        """
        Forward propagate pooling layer.

        Arguments:
            layer (PoolLayer): The pool layer object, different backends have
                               different pool layers.
            I (Tensor): Input tensor.
            O (Tensor): output tensor.
        """
        raise NotImplementedError()

    def bprop_pool(self, layer, I, E, grad_I):
        """
        Backward propagate pooling layer.

        Arguments:
            layer (PoolLayer): The pool layer object. Different backends have
                               different pool layers.
            I (Tensor): Input tensor.
            E (Tensor): Error tensor.
            grad_I (Tensor): Gradient tensor (delta)
        """
        raise NotImplementedError()

    def compound_bprop_lut(self, nin, inputs, error, error_t, dW, pad_idx, alpha=1.0, beta=0):
        """
        Backward propagate lookup table layer.

        Arguments:
            nin (integer): Number of input word_ids.
            inputs (Tensor): Input tensor.
            error (Tensor): Error tensor.
            error_t (Tensor): Transposed error tensor.
            dW (Tensor): Gradient tensor (delta).
            pad_idx (integer):
            alpha (float):
            beta (float):
        """
        raise NotImplementedError()


# For constructing an op tree used in lazy evaluation
class OpTreeNode(tuple):
    """
    An OpTreeNode is a tuple of length 3. The first element is a dict
    specifying the operation, and the second and third elements specify the
    operands. From an op-tree's tree perspective, think about the 3
    elements as 3 nodes. The second and third element are the left and right
    child of the first element.
    """
    def __new__(cls, *args):
        return tuple.__new__(cls, args)

    def __str__(self):
        s = '(' + str(self[0])
        s += ', '
        if isinstance(self[1], Tensor):
            if self[1].name and self[1].name is not None:
                s += self[1].name
            else:
                s += 'tensor-' + hex(id(self[1]))
        else:
            s += str(self[1])
        s += ', '
        if isinstance(self[2], Tensor):
            if self[2].name and self[2].name is not None:
                s += self[2].name
            else:
                s += 'tensor-' + hex(id(self[2]))
        else:
            s += str(self[2])
        s += ')'
        return s

    def __repr__(self):
        return self.__str__()

    def key(self):
        """
        Returns a key for identifying the optree. The key is depended on the ops
        and the id of the tensors. Since __eq__ is overloaded, need to manage
        the hashing of the OpTreeNode manually.

        Returns:
            tuple: optree key
        """
        stack = self.traverse(list())
        for i in range(len(stack)):
            if type(stack[i]) is dict:
                if 'axis' in stack[i]:
                    stack[i] = (stack[i]['op'], stack[i]['axis'])
                else:
                    stack[i] = (stack[i]['op'])

        return tuple(stack)

    def intrinsic_key_maps(self):
        """
        Returns the intrinsic key, tensor_index_map and index_tensor_map
        for the purpose of identifying a optree. The key is depended on the ops
        tensors dimensions and the relaion among the tensors.

        x0 * x1 + x0 * x2 will have the same intrinsic key as y0 * y1 + y0 * y2,
        if xi and yi have the same shape.

        In tensor_index_map and index_tensor_map, tensors has a one-to-one
        mapping with indices. The index of the tensor is depended on the first
        occurance of the tensor in the post-order traversal of the optree.

        Returns:
            (intrinsic_key, tensor_index_map, index_tensor_map)

        """
        stack = self.traverse(list())
        tensor_index = 0
        tensor_index_map = {}
        index_tensor_map = {}
        for i in range(len(stack)):
            if type(stack[i]) is dict:
                if 'axis' in stack[i]:
                    stack[i] = (stack[i]['op'], stack[i]['axis'])
                else:
                    stack[i] = (stack[i]['op'])
            elif isinstance(stack[i], Tensor):
                # use interger to replace tensor
                if stack[i] in tensor_index_map:
                    stack[i] = (tensor_index_map[stack[i]], stack[i].shape)
                else:
                    # put tensor in dict
                    tensor_index_map[stack[i]] = tensor_index
                    index_tensor_map[tensor_index] = stack[i]
                    stack[i] = (tensor_index, stack[i].shape)
                    tensor_index += 1

        return (tuple(stack), tensor_index_map, index_tensor_map)

    @staticmethod
    def build(op, a, b, out=None, **kwargs):
        """
        Build OpTreeNode.

        Arguments:
            a (OpTreeNode, Tensor, numeric): left-hand side operand.
            b (OpTreeNode, Tensor, numeric): right-hand side operand.
            out (Tensor, optional): where the result will be stored. If out is
                                    not None, the op-tree will be executed.
            kwargs: optional argument such as axis of the reducion.
        """
        # check type
        for arg in (a, b):
            if not isinstance(arg, (int, float, Tensor, OpTreeNode, type(None))):
                return NotImplemented
        # get shape
        out_shape = [1, 1]
        if isinstance(a, (OpTreeNode, Tensor)):
            a_shape = a.shape
        elif isinstance(a, (float, int)):
            a_shape = [1, 1]
        else:
            a_shape = [0, 0]
        if isinstance(b, (OpTreeNode, Tensor)):
            b_shape = b.shape
        elif isinstance(b, (float, int)):
            b_shape = [1, 1]
        else:
            b_shape = [0, 0]

        # TODO: fix shape in smarter way
        if len(a_shape) == 1:
            a_shape = a_shape + (1,)
        if len(b_shape) == 1:
            b_shape = b_shape + (1,)

        if op in OpCollection.ew_ops:
            for i in range(2):
                out_shape[i] = max(a_shape[i], b_shape[i])
        elif op in OpCollection.reduction_ops:
            if "axis" in kwargs:
                out_shape = list(a_shape)
                out_shape[kwargs["axis"]] = 1
            else:
                pass  # [1, 1]
        elif op == "assign":
            out_shape = a_shape
        elif op == "dot":
            assert (len(a_shape) == len(b_shape) and len(b_shape) == 2 and
                    a_shape[1] == b_shape[0])
            out_shape = (a_shape[0], b_shape[1])
        elif op == "transpose":
            assert b is None
            out_shape = tuple(reversed(a_shape))
        else:
            raise TypeError("%s is not a valid operation" % op)
        out_shape = tuple(out_shape)

        # build op dict
        op_dict = {"op": op, "shape": out_shape}
        op_dict.update(kwargs)

        node = OpTreeNode(op_dict, a, b)

        # execute explicit assignment
        if op == "assign":
            return node.execute()

        # passing in an out value counts as assignment
        if out is not None:
            return OpTreeNode({"op": "assign"}, out, node).execute()

        # delay execution until assignment
        return node

    def execute(self):
        """
        Execute the optree. When calling `execute()`, there must be one and only
        one `assign` operation at the very top of the op-tree. The corresponding
        backend's execute function will be called.
        """

        assert(self[0]["op"] == "assign")

        backend = self[1].backend

        if isinstance(backend, Backend):
            return backend.execute(self)
        else:
            raise NotImplementedError()

    def traverse(self, stack):
        """
        Post order walk op tree and produce postfix stack

        Arguments:
            stack (list): user shall give empty list like `list()`, then it's
                          used recursively to construct the post-order stack.
        """
        # Left
        if isinstance(self[1], OpTreeNode):
            self[1].traverse(stack)
        elif self[1] is not None:
            stack.append(self[1])

        # Right
        if isinstance(self[2], OpTreeNode):
            self[2].traverse(stack)
        elif self[2] is not None:
            stack.append(self[2])

        stack.append(self[0])

        return stack

    @property
    def T(self):
        return OpTreeNode.build("transpose", self, None)

    def transpose(self, out=None):
        """
        Return a transposed view of the data.
        """
        if out:
            return OpTreeNode.build("assign", out, self.T)
        return self.T

    @staticmethod
    def optree_to_list(optree):
        """
        convert optree to list of lists recursively
        """
        if isinstance(optree, OpTreeNode):
            return list(map(OpTreeNode.optree_to_list, optree))
        else:
            return optree

    @staticmethod
    def list_to_optree(l):
        """
        convert list to optree recursively
        """
        if isinstance(l, list):
            return OpTreeNode(*map(OpTreeNode.list_to_optree, l))
        else:
            return l

    @property
    def shape(self):
        """
        return the shape of the OpTreeNode
        """

        if isinstance(self, OpTreeNode):
            return self[0]['shape']

        if isinstance(self, Tensor):
            return self.shape

        # scalar
        return (1, 1)

    @staticmethod
    def _pretty_print(node):
        operators = {'add': '+',
                     'sub': '-',
                     'mul': '*',
                     'div': '/',
                     'pow': '**'}
        s = ''
        if isinstance(node, Tensor):
            if node.name:
                s = node.name
            else:
                s = 'tensor-' + hex(id(node))
        elif isinstance(node, OpTreeNode):
            if node[2]:
                s += OpTreeNode._pretty_print(node[1]) + ' '
                if node[0]['op'] in operators:
                    s += operators[node[0]['op']]
                else:
                    s += node[0]['op']
                s += ' ' + OpTreeNode._pretty_print(node[2])
            else:
                s = node[0]['op'] + ' ' + OpTreeNode._pretty_print(node[1])
            s = '(' + s + ')'
        else:
            s = str(node)  # TODO
            s = '(' + s + ')'
        return s

    def pp(self):
        """
        Pretty print of the optree

        Arguments:
            node (OpTreeNode): the top node of the op-tree to print

        Returns:
            str: string representation of the op-tree
        """
        return OpTreeNode._pretty_print(self)

    def asnumpyarray(self):
        """
        Returns the evaluated value of the optree as a host numpy.ndarray.
        Allocates new memory, usually used for debug.

        Returns:
            numpy.ndarray: evaluated value
        """
        return self.astensor().get()

    def astensor(self):
        """
        Returns the evaluated value of the optree as a Tensor.
        Allocates new memory, usually used for debug.

        Returns:
            Tensor: evaluated value
        """
        stack = self.traverse(list())

        be = None
        for s in stack:
            if isinstance(s, Tensor):
                be = s.backend
                break
        if be is None:
            raise ValueError("No tensor object in op_tree")

        buf = be.empty(self.shape)
        buf[:] = self
        return buf

    def __add__(self, other):
        return self.build("add", self, other)

    def __sub__(self, other):
        return self.build("sub", self, other)

    def __mul__(self, other):
        return self.build("mul", self, other)

    def __div__(self, other):
        return self.build("div", self, other)

    def __truediv__(self, other):
        return self.build("div", self, other)

    def __pow__(self, other):
        return self.build("pow", self, other)

    def __radd__(self, other):
        return self.build("add", other, self)

    def __rsub__(self, other):
        return self.build("sub", other, self)

    def __rmul__(self, other):
        return self.build("mul", other, self)

    def __rdiv__(self, other):
        return self.build("div", other, self)

    def __rtruediv__(self, other):
        return self.build("div", other, self)

    def __rpow__(self, other):
        return self.build("pow", other, self)

    def __eq__(self, other):
        return self.build("eq", self, other)

    def __ne__(self, other):
        return self.build("ne", self, other)

    def __lt__(self, other):
        return self.build("lt", self, other)

    def __le__(self, other):
        return self.build("le", self, other)

    def __gt__(self, other):
        return self.build("gt", self, other)

    def __ge__(self, other):
        return self.build("ge", self, other)

    def __abs__(self):
        return self.build("abs", self, None)

    def __neg__(self):
        return self.build("neg", self, None)
