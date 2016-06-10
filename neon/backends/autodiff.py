# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
Automatic differentiation of optrees
Supports:
    - elementwise operations
      - unary
      - binary
    - dot
    - reductions (<2d, same as nervanagpu)
To support:
    - batched_dot
    - zero-operand operations
    - slicing (need to modify tensor view)
TODO:
    - make use of empty_like
    - intrinsic key for caching
"""
from __future__ import division
from builtins import object, zip
from neon.backends.backend import OpTreeNode, Tensor
import numpy as np
from functools import wraps

_scalar_types = {int, float, np.float16, np.float32, np.uint8, np.int8,
                 np.uint16, np.int16, np.uint32, np.int32}


class GradUtil(object):

    """
    Utility class for calculating gradients.
    """

    @staticmethod
    def get_grad_back(grad_node):
        """
        Get left and right gradient increments from back-propagation.

        Arguments:
            grad_node (GradNode): The GradNode to perform gradient
                                  back-propagation on.
        """
        if not grad_node:
            return None
        # dissemble node
        x = grad_node.left.op_tree if grad_node.left else None
        y = grad_node.right.op_tree if grad_node.right else None
        z = grad_node.op_tree
        dz = grad_node.grad_op_tree
        op_dict = z[0]
        be = grad_node.ad.be
        # get element-wise gradient increments
        op = grad_node.op_tree[0]['op']
        grad_increments = grad_map[op](x, y, z, dz, op_dict, be)
        # unbroacat to input dimensions
        left_increment = GradUtil._unbroadcast(grad_increments[0], x, be)
        right_increment = GradUtil._unbroadcast(grad_increments[1], y, be)

        return (left_increment, right_increment)

    @staticmethod
    def _unbroadcast(grad_op_tree, x, be):
        """
        Reverse broadcast from shape(grad_op_tree) to shape(x)

        Arguments:
            grad_op_tree (OpTreeNode or Tensor): The OpTreeNode to broadcast.
            x (OpTreeNode or Tensor): Provides the dimension to be broadcasted to.
            be: (Backend): The backend to be used.

        Returns:
            OpTreeNode or Tensor: The broadcasted result.
        """

        if (not grad_op_tree) or (not x):
            return grad_op_tree
        if type(x) in _scalar_types:
            return 0.
        in_shape = x.shape
        out_shape = grad_op_tree.shape

        if in_shape == out_shape:
            return grad_op_tree
        elif len(in_shape) == 2 and len(out_shape) == 2:
            # broadcasts
            if in_shape == (1, 1):
                # [1 * 1] -> [m * n]
                return be.sum(grad_op_tree)
            elif in_shape[0] == out_shape[0] and in_shape[1] == 1:
                # [m * 1] -> [m * n]
                return be.sum(grad_op_tree, axis=1)
            elif in_shape[0] == 1 and in_shape[1] == out_shape[1]:
                # [1 * n] -> [m * n]
                return be.sum(grad_op_tree, axis=0)
            # reductions
            elif ((out_shape[0] == in_shape[0] and out_shape[1] == 1) or
                  (out_shape[0] == 1 and out_shape[1] == in_shape[1])):
                # TODO cleaner way to broadcast
                return 0 * x + grad_op_tree
            else:
                return NotImplemented
        else:
            return NotImplemented

    @staticmethod
    def is_invalid(grad_op_tree, be):
        """
        Test if the result of grad_op_tree contains Nan, inf, -inf, or
        abnormally large or small numbers. Only for debug purpose.

        Arguments:
            grad_op_tree (OpTreeNode or Tensor): The tensor or op-tree to test.
            be (Backend): The backend to be used.

        Returns:
            bool: Whether the result contains Nan, inf, -inf, or abnormally
                  large or small numbers
        """
        grad_op_tree_val = be.empty(grad_op_tree.shape)
        grad_op_tree_val[:] = grad_op_tree
        grad_op_tree_val_np = grad_op_tree_val.get().reshape(-1,)
        for val in grad_op_tree_val_np:
            if not (-50000. < val < 50000.):
                return True
        else:
            return False

    """
    (Applies to the following grad functions)
    Return gradients for these operations.

    Arguments:
        x (Tensor, int, float, OpTreeNode): Left operand.
        y (Tensor, int, float, OpTreeNode): Right operand.
        z (Tensor, int, float, OpTreeNode): `z = x op y`
        dz (Tensor, int, float, OpTreeNode): Gradient w.r.t.`z`
        op_dict (dict): Dictionary specifying the operation.
        be (Backend): The backend of the tensors.

    Returns:
        Tuple: (left_increment, right_increment)
    """

    # derivatives
    @staticmethod
    def _zero_grad_unary(x, y, z, dz, op_dict, be):
        return (dz * 0., None)

    @staticmethod
    def _zero_grad_binary(x, y, z, dz, op_dict, be):
        return (dz * 0., dz * 0.)

    @staticmethod
    def _add_grad(x, y, z, dz, op_dict, be):
        return (dz, dz)

    @staticmethod
    def _mul_grad(x, y, z, dz, op_dict, be):
        return (dz * y, dz * x)

    @staticmethod
    def _sub_grad(x, y, z, dz, op_dict, be):
        return (dz, -dz)

    @staticmethod
    def _neg_grad(x, y, z, dz, op_dict, be):
        return (-dz, None)

    @staticmethod
    def _pow_grad(x, y, z, dz, op_dict, be):
        return (dz * y * x ** (y - 1.), dz * z * be.log(x))

    @staticmethod
    def _div_grad(x, y, z, dz, op_dict, be):
        return (dz / y, -dz * x / be.square(y))

    @staticmethod
    def _dot_grad(x, y, z, dz, op_dict, be):
        return (be.dot(dz, y.T), be.dot(x.T, dz))

    @staticmethod
    def _abs_grad(x, y, z, dz, op_dict, be):
        return (dz * be.sgn(x), None)

    @staticmethod
    def _sqrt_grad(x, y, z, dz, op_dict, be):
        return (dz * 0.5 / z, None)

    @staticmethod
    def _sqr_grad(x, y, z, dz, op_dict, be):
        return (dz * 2. * x, None)

    @staticmethod
    def _exp_grad(x, y, z, dz, op_dict, be):
        return (dz * z, None)

    @staticmethod
    def _exp2_grad(x, y, z, dz, op_dict, be):
        return (dz * z * be.log(2.), None)

    @staticmethod
    def _log_grad(x, y, z, dz, op_dict, be):
        return (dz / x, None)

    @staticmethod
    def _log2_grad(x, y, z, dz, op_dict, be):
        return (dz / x / be.log(2.), None)

    @staticmethod
    def _sig_grad(x, y, z, dz, op_dict, be):
        return (dz * z * (1. - z), None)

    @staticmethod
    def _sig2_grad(x, y, z, dz, op_dict, be):
        return (dz * z * (1. - z) * be.log(2.), None)

    @staticmethod
    def _tanh_grad(x, y, z, dz, op_dict, be):
        return (dz * (1. - be.square(z)), None)

    @staticmethod
    def _tanh2_grad(x, y, z, dz, op_dict, be):
        return (dz * (1. - be.square(z)) * be.log(2.), None)

    @staticmethod
    def _max_grad(x, y, z, dz, op_dict, be):
        return (dz * (x == z), None)

    @staticmethod
    def _min_grad(x, y, z, dz, op_dict, be):
        return (dz * (x == z), None)

    @staticmethod
    def _maximum_grad(x, y, z, dz, op_dict, be):
        return (dz * be.greater_equal(x, y), dz * be.greater_equal(y, x))

    @staticmethod
    def _minimum_grad(x, y, z, dz, op_dict, be):
        return (dz * be.less_equal(x, y), dz * be.less_equal(y, x))

    @staticmethod
    def _sum_grad(x, y, z, dz, op_dict, be):
        assert('axis' in op_dict and (op_dict['axis'] in (0, 1)))
        return (dz, None)  # will be unbroadcasted

    @staticmethod
    def _transpose_grad(x, y, z, dz, op_dict, be):
        return (dz.T, None)


grad_map = {
    # zero gradients
    'eq': GradUtil._zero_grad_binary,
    'lt': GradUtil._zero_grad_binary,
    'le': GradUtil._zero_grad_binary,
    'gt': GradUtil._zero_grad_binary,
    'ge': GradUtil._zero_grad_binary,
    'sgn': GradUtil._zero_grad_unary,
    'finite': GradUtil._zero_grad_unary,
    'argmax': GradUtil._zero_grad_unary,
    'argmin': GradUtil._zero_grad_unary,
    # binary operations
    'add': GradUtil._add_grad,
    'mul': GradUtil._mul_grad,
    'sub': GradUtil._sub_grad,
    'pow': GradUtil._pow_grad,
    'div': GradUtil._div_grad,
    'dot': GradUtil._dot_grad,
    # unary operations
    'neg': GradUtil._neg_grad,
    'abs': GradUtil._abs_grad,
    'sqrt': GradUtil._sqrt_grad,
    'sqr': GradUtil._sqr_grad,
    'exp': GradUtil._exp_grad,
    'exp2': GradUtil._exp2_grad,
    'log': GradUtil._log_grad,
    'log2': GradUtil._log2_grad,
    'sig': GradUtil._sig_grad,
    'sig2': GradUtil._sig2_grad,
    'tanh': GradUtil._tanh_grad,
    'tanh2': GradUtil._tanh2_grad,
    'max': GradUtil._max_grad,
    'min': GradUtil._min_grad,
    'maximum': GradUtil._maximum_grad,
    'minimum': GradUtil._minimum_grad,
    # reduction operations
    'sum': GradUtil._sum_grad,
    # transpose
    'transpose': GradUtil._transpose_grad
}


def memoize_autodiff(func):
    """
    Memoize to avoid rebuilding of the gradient tree.

    Arguments:
        func (Function): Function to memoize.
    """
    cache = {}

    @wraps(func)
    def memoizer(op_tree, be, next_error=None):
        """
        If params in the caches, return results directly. Othewise, add to cache
        and return the results.

        Arguments:
            op_tree (OpTreeNode): the op-tree to supply to the func.
            be (Backend): computation backend to supply to the func.
            next_error (Tensor or OpTreeNode, optional): next layer's error to
                                                         supply to the func.
        """
        key = (op_tree.key(), be, next_error)
        if key not in cache:
            cache[key] = func(op_tree, be, next_error)
            # print 'created grad_tree cache'
        return cache[key]
    return memoizer


@memoize_autodiff
class Autodiff(object):

    """
    Automatic differentiation given an op-tree.

    Arguments:
        op_tree (OpTreeNode): the op-tree to take gradient of
        be (Backend): computation backend used
        next_error (Tensor or OpTreeNode, optional): next layer's error, usually
                                                     self.delta in a layer. If
                                                     set to None, then automatically
                                                     the default value is tensor
                                                     ones() in output shape
    """

    __slots__ = ['op_tree', 'be', 'dtype', 'next_error', 'map_tensor_grad_node',
                 'map_tensor_grad_op_tree', 'grad_node']

    def __init__(self, op_tree, be, next_error=None):
        # check type
        assert (type(op_tree) in _scalar_types or type(op_tree) == OpTreeNode or
                isinstance(op_tree, Tensor)), "op_tree type not supported"
        assert be is not None

        # attributes
        self.op_tree = op_tree
        self.be = be
        self.dtype = be.default_dtype
        if next_error is not None:
            assert next_error.shape == op_tree.shape, (
                "next_error.shape %s must be consistant with op_tree.shape %s"
                % (next_error.shape, op_tree.shape))
            self.next_error = next_error
        else:
            self.next_error = self.be.ones(op_tree.shape)

        self.map_tensor_grad_node = {}  # for building grad tree
        self.map_tensor_grad_op_tree = {}  # quick access to grad_op_tree

        # build_grad
        self.grad_node = GradNode(op_tree, self)
        if self.next_error:
            self.grad_node.grad_op_tree = self.next_error
        else:
            self.grad_node.grad_op_tree = self.be.ones(self.op_tree.shape)
        self.grad_node.build_grad()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """
        Perform cleanup on object deletion.
        """
        if self.grad_node is not None:
            self.grad_node.cleanup()
        self.grad_node = None
        self.dtype = None
        self.next_error = None
        self.op_tree = None
        self.be = None

    def back_prop_grad(self, tensors, gradients):
        """
        Back-propagate the gradient of the `tensors` to `gradients`.

        Arguments:
            Tensors (list): List of Tensors to compute gradients.
            Gradient (list): List of Tensors, as output buffers of the
                             Gradients.
        """
        # avoid tensor reused as a grad_buffer
        for grad_buffer in gradients:
            assert(grad_buffer._original_base not in self.map_tensor_grad_op_tree)

        skipped_tensor = None
        for tensor, grad_buffer in zip(tensors, gradients):
            if grad_buffer is self.next_error:
                # next_error reused as a grad_buffer
                skipped_tensor = tensor
            else:
                grad_buffer[:] = self.map_tensor_grad_op_tree.get(
                    tensor._original_base, grad_buffer * 0.)

        if skipped_tensor:
            self.next_error[:] = self.map_tensor_grad_op_tree.get(
                skipped_tensor._original_base, self.next_error * 0.)

    def get_grad_op_tree(self, tensors):
        """
        Get gradient op_trees w.r.t the list of `tensors`. If a tensor is not
        used, its gradient will be set to zero.

        Arguments:
            Tensors (list): List of Tensors to compute gradients.

        Returns
            list: A list of op_trees, each of them is the gradent of the input
                  tensor.
        """
        grad_op_trees = []
        for tensor in tensors:
            grad_op_trees.append(
                self.map_tensor_grad_op_tree.get(tensor._original_base, tensor * 0.))
        return grad_op_trees

    def get_grad_tensor(self, tensors):
        """
        Get gradient values in type Tensor w.r.t the list of `tensors`. If a
        tensor is not used, its gradient will be set to zero.

        Arguments:
            Tensors (list): List of Tensors to compute gradients on.

        Returns
            list: A list of Tensors, each of them is the gradent of the input
                  tensor.
        """
        grad_op_trees = self.get_grad_op_tree(tensors)
        grad_vals = []
        for grad_op_tree in grad_op_trees:
            grad_val = self.be.empty(grad_op_tree.shape)
            grad_val[:] = grad_op_tree
            grad_vals.append(grad_val)
        return grad_vals

    def get_grad_asnumpyarray(self, tensors):
        """
        Get gradient values as numpy array w.r.t the list of `tensors`. If a
        tensor is not used, its gradient will be set to zero.

        Arguments:
            Tensors (list): List of Tensors to compute gradients.

        Returns
            list: A list of numpy.ndarray, each of them is the gradient of the
                  input tensor.
        """
        grad_vals = self.get_grad_tensor(tensors)
        for i in range(len(grad_vals)):
            grad_vals[i] = grad_vals[i].get().astype(self.dtype)
        return grad_vals


class GradNode(object):

    """
    A node in grad_tree. A GradNode contains the op_optree and the grad_op_tree
    at this location of the grad_tree, and it also has pointers to the left and
    right child in the grad_tree.
    """

    __slots__ = ['op_tree', 'grad_op_tree', 'ad', 'left', 'right']

    def __init__(self, op_tree, ad):
        """
        Arguments:
            op_tree (OpTreeNode or Tensor): the op_tree at this grad_node
            ad (Autodiff): the autodiff object with global op_tree, next_error and dicts
        """
        # check op_tree
        assert op_tree is not None

        # attributes
        self.op_tree = op_tree  # forward op_tree
        self.grad_op_tree = None  # backward gradient op_tree
        self.ad = ad  # info about autodiff object
        self.left = None
        self.right = None

        # build GradNode recursively
        if isinstance(op_tree, Tensor):
            # save to ad.map_tensor_grad_node
            if op_tree._original_base not in ad.map_tensor_grad_node:
                ad.map_tensor_grad_node[op_tree._original_base] = self
        elif type(op_tree) == OpTreeNode:
            # init recursively
            if op_tree[1] is not None:
                if (isinstance(op_tree[1], Tensor) and
                        op_tree[1]._original_base in ad.map_tensor_grad_node):
                    # seen tensor before
                    self.left = ad.map_tensor_grad_node[op_tree[1]._original_base]
                else:
                    # build recursively
                    self.left = GradNode(op_tree[1], ad)
            if op_tree[2] is not None:
                if (isinstance(op_tree[2], Tensor) and
                        op_tree[2]._original_base in ad.map_tensor_grad_node):
                    # seen tensor before
                    self.right = ad.map_tensor_grad_node[op_tree[2]._original_base]
                else:
                    # build recursively
                    self.right = GradNode(op_tree[2], ad)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """
        Perform cleanup on object deletion.
        """
        self.op_tree = None
        self.grad_op_tree = None
        self.ad = None

        if self.left is not None:
            self.left.cleanup()
        self.left = None

        if self.right is not None:
            self.right.cleanup()
        self.right = None

    def build_grad(self):
        """
        Actually back-propagate the gradient.
        """
        # self.grad_op_tree shall be set by ad or parent grad_node
        assert self.grad_op_tree is not None

        if type(self.op_tree) == OpTreeNode:
            # get increment
            (left_increment, right_increment) = GradUtil.get_grad_back(self)

            # left increment
            if self.left.grad_op_tree is None:
                self.left.grad_op_tree = left_increment
            else:
                self.left.grad_op_tree = self.left.grad_op_tree + \
                    left_increment

            # left recursive
            self.left.build_grad()

            # check if right increment
            if right_increment is None:
                return

            # right increment
            if self.right.grad_op_tree is None:
                self.right.grad_op_tree = right_increment
            else:
                self.right.grad_op_tree = self.right.grad_op_tree + \
                    right_increment

            # right recursive
            self.right.build_grad()

        elif isinstance(self.op_tree, Tensor):
            self.ad.map_tensor_grad_op_tree[self.op_tree._original_base] = self.grad_op_tree
