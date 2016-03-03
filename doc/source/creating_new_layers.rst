Creating new layers
===================

A simple layer
--------------

To implement a simple custom layer in neon, write a Python class that
subclasses Layer (:py:class:`neon.layers.Layer<.Layer>`). |Layer| is a subclass of
:py:class:`neon.NervanaObject`, which contains a static instance of the computational
backend. The backend is exposed in the class as ``self.be``.

At minimum, the layer must implement |configure| to
properly set the input/output shapes as well as the |fprop| and
|bprop| methods for forward and backward propagation, respectively.
For computations that require pre-allocating buffer space, also
implement :py:meth:`.Layer.allocate()`.

Here is a custom layer that multiples the input by two.

.. code-block:: python


    class MultiplyByTwo(Layer):
        " A layer that multiples the input by two. "

        # constructor and initialize buffers
        def __init__(self, name=None):
            super(MultiplyByTwo, self).__init__(name)

        # configure the layer input and output shapes
        def configure(self, in_obj):
            super(MultiplyByTwo, self).configure(in_obj)
            self.out_shape = self.in_shape
            return self

        # compute the fprop
        def fprop(self, inputs, inference=False):
            self.outputs = inputs
            self.outputs[:] = inputs*2
            return self.outputs

        # backprop the gradients
        def bprop(self, error):
            error[:] = 2*self.error
            return error


Let's break this down. Because this layer does not change the shape of
the data (as opposed to convolutional or pooling layers, for example), |configure| simply sets the shape of the output to the
shape of the input layer ``in_obj``. During model initialization, neon
forward-propagates the shapes through the layers, calling |configure|.

|Layer| has a tensor ``self.outputs`` that is pre-allocated, which we
use as a buffer to store the outputs. For |bprop| and |fprop|, we execute the backend multiply operations and return the
Tensor results.

Auto-differentiation
--------------------

Neon provides an auto-differentiation feature, which is particularly
useful for defining |bprop|. Computations with our backend
are first stored as a graph of numerical calculations (see
:doc:`backend<backends>`). For example, we define the logistic function

.. code-block:: python

    myTensor = be.zeros((10,10), name = 'myTensor')
    f = 1/(1+be.exp(-1*myTensor))

Then, ``f`` is an op-tree (:py:class:`neon.backends.backend.OpTreeNode`). We execute the op-tree by calling the proper syntax.

.. code-block:: python

    fval = be.empty((10,10)) # allocate space for output
    fval[:] = f # execute the op-tree

We compute the gradients from an op-tree by calling |Autodiff| and
passing the op-tree ``f`` and the backend ``be``:

.. code-block:: python

    from neon.backends import Autodiff

    myAutodiff = Autodiff(op_tree = f, be = be)

Then we retrieve the gradients by calling :py:meth:`.get_grad_tensor`
and passing the tensor ``myTensor``.

.. code-block:: python

    grads = myAutodiff.get_grad_tensor(myTensor)

There are two other methods for computing the gradient. The method
``get_grad_asnumpyarray`` returns a numpy array instead of a tensor.
Relevant for constructing layers is the ``back_prop_grad`` function,
which stores the result in the provided tensor.

.. code-block:: python

    grads = be.empty((10,10))
    myAutodiff.back_prop_grad(myTensor,grads)

Example layer with autodiff
---------------------------

We can put this into action with a layer that applies the logistic
function and uses |Autodiff| for computing the gradients.

.. code-block:: python

    class Logistic(Layer):
        " A layer that applies the logistic function "

        # constructor and initialize buffers
        def __init__(self, name=None):
            super(Logistic, self).__init__(name)
            self.fprop_op_tree = None

        # configure the layer input and output shapes
        def configure(self, in_obj):
            super(Logistic, self).configure(in_obj)
            self.out_shape = self.in_shape
            return self

        # compute the fprop
        def fprop(self, inputs, inference):
            # define and then execute the op-tree
            self.fprop_op_tree = 1/(self.be.exp(-1*inputs) + 1.0)
            self.outputs[:] = self.fprop_op_tree
            return self.outputs

        # backprop the gradients
        def bprop(self, error):
            # generate the autodiff
            ad = Autodiff(self.fprop_op_tree, self.be)

            # compute the gradient with autodiff
            error[:] = ad.get_grad_Tensor(self.fprop_op_tree, self.be, self.outputs)
            return error

Layers with parameters
----------------------

For simple layers that do not carry any weights, inheriting from
|Layer| suffices. However, if the layer has weight parameters (e.g.
linear, convolutional, etc.), neon has a class |ParameterLayer| with
common functionality for storing and tracking weights.

This class has the variables ``W`` (Tensor) for storing the weights and
implements :py:meth:`~.ParameterLayer.allocate` to allocate the buffers for ``W`` and
initialize ``W`` with the provided initializer. New layers with weights
should subclass from |ParameterLayer| and implement
:py:meth:`~.ParameterLayer.configure`, |fprop|, and |bprop|.

.. |Layer| replace:: :py:class:`Layer<neon.layers.layer.Layer>`
.. |ParameterLayer| replace:: :py:class:`ParameterLayer<neon.layers.layer.ParameterLayer>`
.. |bprop| replace:: :py:meth:`~.Layer.bprop`
.. |fprop| replace:: :py:meth:`~.Layer.fprop`
.. |configure| replace:: :py:meth:`~.Layer.configure()`
.. |Autodiff| replace:: :py:class:`.Autodiff`
