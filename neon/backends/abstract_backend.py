# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
Defines interface that any backend must implement
"""
import abc
from future.utils import with_metaclass


class Backend_ABC_Meta(abc.ABCMeta):
    """
    metaclass for the backend objects
    takes care of registering all the backend subclasses
    """
    def __init__(self, name, bases, dict_):
        if not hasattr(self, 'backends'):
            self.backends = {}
        else:
            name = getattr(self, 'backend_name', name)
            if name not in ['Backend']:
                self.backends[name] = self
        super(Backend_ABC_Meta, self).__init__(name, bases, dict_)


class AbstractBackend(with_metaclass(Backend_ABC_Meta, object)):
    def __del__(self):
        self.cleanup_backend()

    @abc.abstractmethod
    def cleanup_backend(self):
        """Release any resources that have been acquired by this backend."""
        raise NotImplementedError()

    @abc.abstractmethod
    def gen_rng(self, seed=None):
        """
        Setup the random number generator(s) and store the state
        in self.init_rng_state.

        Arguments:
            seed (int or None): RNG seed, if the seed is None,
                                then a seed will be randomly chosen

        Returns:
            np.random.RandomState: numpy RNG
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rng_get_state(self, state):
        """
        Get the random number generator state to a specific state.

        Returns a tuple since some backends have multiple RNG states
        (e.g. on-host and on-device)

        Returns:
            tuple: array of numpy ndarray which defines the current
                   state of the RNGs
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rng_reset(self):
        """
        Reset the random state to the state where the Backend is first
        initialized.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rng_set_state(self, state):
        """
        Set the random number generator state to a specific state.

        Arguments:
            state (np.array): array which is used to define the RNG
                              state
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
            :py:func:`~neon.backends.backend.Backend.array`,
            :py:func:`~neon.backends.backend.Backend.zeros`,
            :py:func:`~neon.backends.backend.Backend.ones`
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
            :py:func:`~neon.backends.backend.Backend.empty`,
            :py:func:`~neon.backends.backend.Backend.zeros`,
            :py:func:`~neon.backends.backend.Backend.ones`
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def zeros(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        each element with a value of 0.

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
            :py:func:`~neon.backends.backend.Backend.ones`,
            :py:func:`~neon.backends.backend.Backend.array`
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def ones(self, shape, dtype=None, name=None, persist_values=True,
             parallel=False, distributed=False):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        each element with a value of 1.

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

    @abc.abstractmethod
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
            :py:func:`~neon.backends.backend.Backend.empty`,
            :py:func:`~neon.backends.backend.Backend.ones`,
            :py:func:`~neon.backends.backend.Backend.array`
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
            :py:func:`~neon.backends.backend.Backend.empty`,
            :py:func:`~neon.backends.backend.Backend.ones`,
            :py:func:`~neon.backends.backend.Backend.array`
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
    def compound_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False):
        """
        Perform one of the following operations (* is dot product)
        C = alpha * A * B   + beta * C
        C = alpha * A.T * B + beta * C
        C = alpha * A * B.T + beta * C.

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

    @abc.abstractmethod
    def batched_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False):
        """
        Perform one of the following operations:
        1 For fprop: A(K, C), B(X,C,N), C(X,K,N) --> call batched_dot(A, B, C)
        2 For bprop: A(K, C), B(X,K,N), C(X,C,N) --> call batched_dot(A.T, B, C)
        3 For update: A(X,K,N), B(X,C,N), C(K,C) --> call batched_dot(A, B.T, C)

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

    @abc.abstractmethod
    def make_binary_mask(self, out, keepthresh=0.5):
        """
        Create a binary mask for dropout layers.

        Arguments:
            out (Tensor): Output tensor
            keepthresh (float, optional): fraction of ones. Defaults to 0.5
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
    def onehot(self, indices, axis, out=None):
        """
        Generate optree for converting `indices` to a onehot representation.

        Arguments:
            indices (Tensor): Elements must be of numpy integer type for gpu
                              onehot to work.
            axis (int): the axis along the feature length dimension
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            OpTreeNode: the resulting op-tree
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update_fc_bias(self, err, out):
        """
        Compute the updated bias gradient for a fully connected network layer.

        Arguments:
            err (Tensor): backpropagated error
            out (Tensor): Where to store the updated gradient value.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def add_fc_bias(self, inputs, bias):
        """
        Add the bias for a fully connected network layer.

        Arguments:
            inputs (Tensor): the input to update.
            bias (Tensor): the amount to increment
        """
        self.ng.add(inputs, bias, out=inputs)

    @abc.abstractmethod
    def conv_layer(self, dtype,
                   N, C, K,
                   D=1, H=1, W=1,
                   T=1, R=1, S=1,
                   pad_d=0, pad_h=0, pad_w=0,
                   str_d=1, str_h=1, str_w=1,
                   relu=False, bsum=False):
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
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def fprop_conv(self, layer, I, F, O, alpha=1.0, relu=False, repeat=1):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output.

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

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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

            str_j (int, optional): factor to step the filters by in the fm
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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
        raise NotImplementedError()
