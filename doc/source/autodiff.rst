Automatic differentiation
=========================

Automatic differentiation can be achieved given an op-tree (see :doc:`optree`). In the 
following examples, we will explain how to get differentiation from a compound operation 
or from a layer that does batch normalization.

Example: use autodiff based on an op-tree
---------------------------------------------

Construct an op-tree from a compound operation.

.. code:: python

    from neon.backends import gen_backend, Autodiff
    from neon.backends.backend import OpTreeNode # just for pretty print here
    import numpy as np
    be = gen_backend('nervanagpu')
    x0 = be.array(np.ones((3, 3)) * 1., name='x0')
    x1 = be.array(np.ones((3, 3)) * 2., name='x1')
    print '# example 0'
    f = x0 * x0 + x0 * x1

Construct an autodiff object using the op-tree

.. code:: python

    ad = Autodiff(op_tree=f, be=be, next_error=None)
    print ad.get_grad_asnumpyarray([x0, x1]) # result is [2 * x0 + x1, x0]
    
If an op-tree is obtained from a forward propagation process

.. code:: python

    # fprop optree
    # - different from theano, we need explicit tensors to build the optree
    f = be.tanh((x0 * x1) + (x0 / x1))
    # create Autodiff object
    # - the object will be memoized and reused, so it's safe to call Autodiff on
    #   the same optree multiple times
    # - when next_error is None, it will be se to be.ones() of the output shape,
    #   in neon, next_error is set to the next layer's back prop error
    ad = Autodiff(op_tree=f, be=be, next_error=None)
    # print gradient optree
    # - in the printed result, the unamed tensor is ones() of the output shape
    
The gradient with respect to certain variables can be called from an autodiff object

.. code:: python

    [x0_grad_op_tree, x1_grad_op_tree] = ad.get_grad_op_tree([x0, x1])
    print(OpTreeNode.pp(x0_grad_op_tree))
    print(OpTreeNode.pp(x1_grad_op_tree))

Autodiff provides a few other functions:

    * ad.back_prop_grad: back prop gradients to the specified buffers, most efficient (shall be used in most cases with pre-allocated memory)
    * ad.get_grad_op_tree: get the gradient optrees
    * ad.get_grad_tensor: get the gradient tensors, it will allocate device memory
    * ad.get_grad_asnumpyarray: get gradients asnumpy array, it will allocate host memory


An example of autodiff applied to a dynamically generated optrees

.. code:: python

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


Example: use autodiff in a Batch-normalization layer
-----------------------------------------------------

In the layer class, a function is used to create the op-tree object based on the forward propagation. 

.. code:: python

    def get_forward_optree(self):
        """
        Initialize the fprop optree for batchnorm.
        """
        # get fprop op-tree
        xvar = self.be.var(self.x, axis=1)
        xmean = self.be.mean(self.x, axis=1)
        xhat = (self.x - xmean) / self.be.sqrt(xvar + self.eps)
        return xhat * self.gamma + self.beta

In the fprop function, the fprop in the form of an op-tree is saved.

.. code:: python

    self.fprop_op_tree = self.get_forward_optree()


In the bprop function, the errors are derived from auto-differentiating on the fprop op-tree.

.. code:: python

    def bprop(self, error):
        """
        Use Autodiff.back_prop_grad to back propagate gradients for the
        corresponding tensors.
        """
        if not self.deltas:
            self.deltas = error.reshape(self.bn_shape)

        # autodiff will automatically cache and reuse the object
        # if we know the `error` buffer at init, we can also create the autodiff
        # object at layer's init
        ad = Autodiff(self.fprop_op_tree, self.be, next_error=self.deltas)

        # back propagate
        ad.back_prop_grad([self.x, self.gamma, self.beta],
                          [self.deltas, self.grad_gamma, self.grad_beta])

        return error


















