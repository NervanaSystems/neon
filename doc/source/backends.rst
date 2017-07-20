.. ---------------------------------------------------------------------------
.. Copyright 2015 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

neon backend
============

neon features highly optimized CPU (MKL) and GPU computational backends for
fast matrix operations. Understanding how to work with backend is
critical to creating custom layers or cost functions. In fact, the
backend API is exposed, allowing direct access for any application.

In this guide, we will first demonstrate how to directly call the
backend, and then introduce Op-Trees and neon's automatic
differentiation feature. These operations are used extensively for
creating custom layers, costs, and metrics.

The neon backend is easily swappable, meaning that the same code will
run for both the GPU and CPU backends.

Directly calling the backend
----------------------------

To generate an MKL backend, we call

.. code-block:: python

    from neon.backends import gen_backend

    be = gen_backend(backend='mkl')

The method :py:meth:`gen_backend()<neon.backends.gen_backend>` takes several optional arguments (see the :doc:`API<api>` for a full list).

The |Tensor| class is used to represent multidimensional arrays where
each element is of a consistent datatype. We provide methods to
instantiate and copy instances of this data structure, as well as
initialize elements, reshape dimensions, and access metadata.

Let's initialize a |Tensor| of zeros with shape ``(100,100)``.

.. code-block:: python

    from neon.backends.backend import Tensor

    myTensor = be.zeros((100, 100))

There are also other ways of initializing a Tensor:

.. code-block:: python

    import numpy as np

    # initialize a numpy array of zeros
    array_of_zeros = np.zeros((100, 100))

    # 1. array of zeros with the shape like the input array
    myTensor = be.zeros_like(array_of_zeros)

    # 2. initialize a tensor with same values as the input array
    myTensor = be.array(array_of_zeros)

    # 3. initialize an empty Tensor, then set elements to zero
    myTensor = be.empty((100, 100))
    myTensor[:] = 0

    # 4. initialize an empty Tensor, then fill the elements to zero
    myTensor = be.empty((100, 100))
    myTensor.fill(0.0)

    # 5. deep copy another Tensor
    yourTensor = be.zeros((100, 100))
    myTensor = yourTensor.copy(yourTensor)

To view the elements of a Tensor, you need to first copy into host
memory as a numpy array via

.. code-block:: python

    myNumpyArray = myTensor.get()

The |Tensor| datatype supports all the operations you would expect
from a multi-dimensional array:

* Fancy slicing (``myTensor[:,1]``, ``myTensor[:, 3:10]``)
* Basic element-wise arithmetic (``myTensor*yourTensor``, ``myTensor+yourTensor``, etc.)
* Transcendental functions (``be.exp(myTensor)``, ``be.sqrt(myTensor)``)
* Logical operations (``myTensor == yourTensor``, ``myTensor > yourTensor``)
* Dot product (``be.dot(myTensor,yourTensor)``)
* Summary statistics (``be.max(myTensor)``, ``be.mean(myTensor)``)

For a full list, see the :doc:`API<api>` documentation.

Using these tools, we can construct, for example, the logistic function

.. code-block:: python

    f = 1/(1+be.exp(-1*myTensor))

The backend creates ``f`` as a graph representation of numerical
operations (Op-Tree). The neon backend performs sequences of operations
using a lazy evaluation scheme where operations are pushed onto an
OpTree and only evaluated when an explicit assignment is made using
Op-Tree syntax (``optree[:]``):

.. code-block:: python

    fval = be.empty((100, 100)) # allocate space for output
    fval[:] = f # execute the op-tree

Op-Trees
--------

Op-tree (as in |OpTreeNode| class) is a graph representation of
numerical operations. We are going to start by looking at a minimal
example:

.. code-block:: python

    from neon.backends import gen_backend

    be = gen_backend('cpu')
    x0 = be.ones((2, 2), name='x0')
    x1 = be.ones((2, 2), name='x1')
    f = x0 + x1               # op-tree creation
    f_val = be.empty((2, 2))  # output buffer allocation
    f_val[:] = f              # execution

    print(f)                  # print the op-tree tuple
    print(f_val.get())        # get the device tensor

This prints:

.. code-block:: python

    ({'shape': (2, 2), 'op': 'add'}, x0, x1) [[ 2.  2.]  [ 2.  2.]]

The tuple ``({'shape': (2, 2), 'op': 'add'}, x0, x1)`` is the op-tree.
There are two different types of nodes in an op-tree. The dict
``{‘shape’: (2, 2), ‘op’: ‘add’}`` is the “op”, containing the
operations, properties of the operation (such as axis) and the shape of
the output. The other two nodes are the “numeric node”, containing
Tensor or constant (float, int).

Relation between OpTreeNode and op-tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  |OpTreeNode| is the class inherited from tuple. An |OpTreeNode|
   is a tuple of length 3. The first element is a dict specifying the
   operation, and the second and third elements specify the operands.
-  From an op-tree’s tree perspective, think about the 3 elements as 3
   nodes. The second and third element are the left and right child of
   the first element (the dict).

Op-Tree Creation
~~~~~~~~~~~~~~~~

An Op-Tree can be created in several ways:

1. **Operator overload**. Most of the common numerical operators between
   Tensor and OpTreeNode are overloaded. Operations between a Tensor and
   an OpTreeNode will produce an OpTreeNode. For example, if we have
   Tensors ``x1`` and ``x2`` and OpTreeNodes ``t1`` and ``t2``, we can
   create an OpTreeNode ``f`` by:

   .. code-block:: python

       # f are OpTreeNode in the following example
       f = x1 + x2 # Tensor + Tensor
       f = x1 + t1 # Tensor + OpTreeNode
       f = t1 + t2 # OpTreeNode + OpTreeNode

2. **Backend functions**. An Op-tree can be built by calling backend
   functions using syntax similar to numpy. For example:

   .. code-block:: python

       f = ng.mean(x) # f is an optree

3. **OpTreeNode.build()**. This method is called internally in the
   first two cases. The build function does the type checking and
   appends the shape to the op\_dict. When the first op is ‘assign’, the
   op-tree will be executed automatically.

   .. code-block:: python

       OpTreeNode.build("add", a, b) # binary ops
       OpTreeNode.build("sqrt", a, None, out=out_buffer) # unary ops

4. **OpTreeNode’s init function**. Lastly, we can build an op-tree from
   the constructor of the OpTreeNode class. This is usually called
   internally, giving us complete control of the contents of the OpTree.
   However, this approach does not do type checking or shape
   calculation, so tread carefully.

   .. code-block:: python

       OpTreeNode(op_dict, a, b)

Op-Tree Execution
~~~~~~~~~~~~~~~~~

Usually, the execution of an op-tree is triggered by assignment:

.. code-block:: python

    f_val[:] = f

Here is what happens under the hood:

1. An new op-tree with assignment is built based on f:

   .. code-block:: python

       OpTreeNode.build("assign", f_val, f)

2. Then this new op-tree is executed:

   .. code-block:: python

       OpTreeNode.build("assign", f_val, f).execute()

3. The corresponding backend’s execute function will be called and the
   value of ``f`` will be written to Tensor ``f_val``.

Property of the op-tree
~~~~~~~~~~~~~~~~~~~~~~~

The |OpTreeNode| class is inherited from tuple, making |OpTreeNode|
efficient and immutable. If we want to modify the op-tree (for example
swapping all instance of |Tensor| x1 to x2), consider modifying the
post-order stack (which is a list) of the optree directly.

An op-tree is a binary tree. It has the following properties:

-  Except for the root node, every node has exactly one parent.
-  All leaf nodes are “numeric nodes” and all internal nodes are “op
   nodes”.
-  An “op node” can have zero, one or two children, depending on whether
   it is a zero-operand, unary or binary operation.

Automatic differentiation
-------------------------

Automatic differentiation can be achieved given an op-tree (see
Op-Tree). In the following examples, we will explain how to get
differentiation from a compound operation or from a layer that does
batch normalization.

Example: use autodiff based on an op-tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct an op-tree from a compound operation.

.. code-block:: python

    from neon.backends import gen_backend, Autodiff
    import numpy as np
    be = gen_backend('nervanagpu')
    x0 = be.array(np.ones((3, 3)) * 1., name='x0')
    x1 = be.array(np.ones((3, 3)) * 2., name='x1')
    print '# example 0'
    f = x0 * x0 + x0 * x1

Construct an |Autodiff| object using the op-tree

.. code-block:: python

    ad = Autodiff(op_tree=f, be=be, next_error=None)
    print ad.get_grad_asnumpyarray([x0, x1]) # result is [2 * x0 + x1, x0]

If an op-tree is obtained from a forward propagation process:

.. code-block:: python

    # fprop optree
    # - different from theano, we need explicit tensors to build the optree
    f = be.tanh((x0 * x1) + (x0 / x1))
    # Create Autodiff object
    # - the object will be memoized and reused, so it's safe to call Autodiff on
    #   the same optree multiple times
    # - when next_error is None, it will be set to be.ones() of the output shape,
    #   in neon, next_error is set to the next layer's back prop error
    ad = Autodiff(op_tree=f, be=be, next_error=None)

The gradient with respect to certain variables can be called from an
|Autodiff| object

.. code-block:: python

    # print gradient optree
    # - in the printed result, the unnamed tensor is ones() of the output shape
    [x0_grad_op_tree, x1_grad_op_tree] = ad.get_grad_op_tree([x0, x1])
    print(OpTreeNode.pp(x0_grad_op_tree))
    print(OpTreeNode.pp(x1_grad_op_tree))

|Autodiff| provides a few other functions:

-  ``back_prop_grad``: back prop gradients to the specified buffers,
   most efficient (shall be used in most cases with pre-allocated
   memory)
-  ``get_grad_op_tree``: get the gradient optrees
-  ``get_grad_tensor``: get the gradient tensors, it will allocate
   device memory
-  ``get_grad_asnumpyarray``: get gradients as numpy array, it will
   allocate host memory

Here is an example of |Autodiff| applied to a dynamically generated
optree:

.. code-block:: python

    x0 = be.array(np.random.randint(10, size=(3, 3)), name='x0')
    def my_loop(x):
        y = be.zeros(OpTreeNode.shape(x), name='tensor-zero')
        for _ in range(x.get()[0, 0]):
            y = y + x
        return y
    def my_condition(x):
        if x.get()[0, 0] % 2 == 0:
            return be.sig(x)
        else:
            return be.tanh(x)
    f = my_loop(x0) + my_condition(x0)
    ad = Autodiff(f, be)
    x0_grad_tree = ad.get_grad_op_tree([x0])[0]
    print(x0.get())
    print(OpTreeNode.pp(f))
    print(OpTreeNode.pp(x0_grad_tree))

Custom Kernels for GPU
----------------------

We use `pycuda <https://mathema.tician.de/software/pycuda/>`_ to wrap custom kernels for the GPU. For example, see `float_ew <https://github.com/NervanaSystems/neon/blob/master/neon/backends/float_ew.py#L1066>`_.


.. |Tensor| replace:: :py:class:`.Tensor`
.. |OpTreeNode| replace:: :py:class:`.OpTreeNode`
.. |Autodiff| replace:: :py:class:`.Autodiff`
