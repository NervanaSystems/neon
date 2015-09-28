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

Op-Tree
========

Op-tree (as in OpTreeNode class) is a graph representation of numerical
operations. We are going to start by looking at a minimal example:

.. code:: python

    from neon.backends import gen_backend

    be = gen_backend('cpu')
    x0 = be.ones((2, 2), name='x0')
    x1 = be.ones((2, 2), name='x1')
    f = x0 + x1               # op-tree creation
    f_val = be.empty((2, 2))  # output buffer allocation
    f_val[:] = f              # execution

    print(f)                  # print the op-tree tuple
    print(f_val.get())        # get the device tensor

this gives:

.. parsed-literal::

    ({'shape': (2, 2), 'op': 'add'}, x0, x1)
    [[ 2.  2.]
     [ 2.  2.]]

The tuple ``({'shape': (2, 2), 'op': 'add'}, x0, x1)`` is the op-tree.
There are two different types of nodes in an op-tree. The dict {'shape':
(2, 2), 'op': 'add'} is the "op", containing the operations, properties
of the operation (such as axis) and the shape of the output. The other
two nodes are the "numeric node", containing Tensor or constant (float,
int).

Relation between OpTreeNode and op-tree
---------------------------------------

-  OpTreeNode is the class inherited from tuple. An OpTreeNode is a
   tuple of length 3. The first element is a dict specifying the
   operation, and the second and third elements specify the operands.
-  From an op-tree's tree perspective, think about the 3 elements as
   3 nodes. The second and third element are the left and right child of
   the first element (the dict).


Op-Tree Creation
----------------

1. **Operator overload.** Most of the common numerical operators betweenTensor 
and OpTreeNode are overloaded. Operations between a Tensor and an OpTreeNode 
will produce an OpTreeNode. For example, we can do:

.. code:: python

    # f are OpTreeNode in the following example
    f = x1 + x2 # Tensor + Tensor
    f = x1 + t1 # Tensor + OpTreeNode
    f = t1 + t2 # OpTreeNode + OpTreeNode

2. **Backend functions.** Op-tree can be built from calling backend functions 
using syntax similar to numpy, for example:

.. code:: python

    f = ng.mean(x) # f is an optree

3. **`OpTreeNode.build()`.** The `OpTreeNode.build()` is called internally by 
the first two cases. The build function does the type checking and appends the shape 
to the op_dict. When the first op is 'assign', the op-tree will be executed automatically.

.. code:: python

    OpTreeNode.build("add", a, b) # binary ops
    OpTreeNode.build("sqrt", a, None, out=out_buffer) # unary ops

4. **OpTreeNode's init function.** Lastly, we could build an op-tree from the 
initializer or the OpTreeNode class. It is also usually called internally. This 
gives us complete control of the contents of the OpTree, but it does not give 
the type checking and shape calculation. So be careful.

.. code:: python

    OpTreeNode(op_dict, a, b)


Op-Tree Execution
-----------------

Usually, the execution of an op-tree is triggered by assignment:

.. code:: python

    f_val[:] = f

Here is what happens under the hood:

- An new op-tree with assignment is built based on `f`:

.. code:: python

    OpTreeNode.build("assign", f_val, f)

- Then this new op-tree is executed:

.. code:: python

    OpTreeNode.build("assign", f_val, f).execute()

- The corresponding backend's execute function will be called and the value of `f`
  will be written to Tensor `f_val`.

Property of the op-tree
-----------------------

The OpTreeNode class is inherited from tuple, making OpTreeNode efficient and 
immutable. If we want to modify the op-tree (for example swapping all instance 
of Tensor `x1` to `x2`), consider modifying the post-order stack (which is a list) 
of the optree directly.


Tree structure & number of operands
-----------------------------------

An op-tree is a binary tree. It has the following properties:

- Except for the root node, every node has exactly one parent.
- All leaf nodes are "numeric nodes" and all internal nodes are "op nodes".
- An "op node" can have zero, one or two children, depending on whether it is a
  zero-operand, unary or binary operation.

