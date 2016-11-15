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

.. currentmodule:: neon

.. |Tensor| replace:: :py:meth:`~neon.backends.backend.Tensor`
.. |Backend| replace:: :py:meth:`~neon.backends.backend.Backend`

ML Operational Layer (MOP) API
==============================

We expose the following API which we refer to as our Machine learning
Operational Layer (MOP layer). This layer abstracts the backend, so the
same operations can be performed on a CPU, GPU, or future hardware
backends.

The API consists of two interface classes:

.. csv-table::
    :header: "Interface", "Description"
    :widths: 20, 30
    :delim: |

    :py:class:`neon.backends.Tensor<.Tensor>`| :math:`n`-dimensional array data structure
    :py:class:`neon.backends.Backend<.Backend>` | Backend interface used to manipulate Tensor data

Basic Data Structure
--------------------

The :py:class:`.Tensor` class is used to represent an arbitrary dimensional array
in which each element is stored using a consistent underlying type. For
the CPU and GPU backends, Tensors are stored as :py:class:`.CPUTensor` and
:py:class:`.GPUTensor` subclasses, respectively.

We have the ability to instantiate and copy instances of this data
structure, as well as initialize its elements, reshape its dimensions,
and access metadata.

Tensor Creation
---------------

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.empty<.Backend.empty>` | Instantiate an empty Tensor
    :py:meth:`neon.backends.Backend.array<.Backend.array>` | Instantiate a new Tensor, populating elements based on a provided array.
    :py:meth:`neon.backends.Backend.zeros<.Backend.zeros>` | Instantiate a new Tensor, populating each element with the value of 0.
    :py:meth:`neon.backends.Backend.ones<.Backend.ones>` | Instantiate a new Tensor, populating each element with the value of 1.
    :py:meth:`neon.backends.Tensor.copy<.Tensor.copy>` | Construct and return a deep copy of the Tensor passed.

Tensor Manipulation
-------------------

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Tensor.get<.Tensor.get>` |	Convert the tensor to a host memory numpy.ndarray.
    :py:meth:`neon.backends.Tensor.take<.Tensor.take>` | Select a subset of elements from an array across an axis
    :py:meth:`neon.backends.Tensor.__getitem__<.Tensor.__getitem__>` |	Extract a subset view of the items via slice style indexing along each dimension.
    :py:meth:`neon.backends.Tensor.__setitem__<.Tensor.__setitem__>` | Assign the specified value to a subset of elements found via slice style indexing along each dimension.
    :py:meth:`neon.backends.Tensor.fill<.Tensor.fill>` |	Assign specified value to each element of this Tensor.
    :py:meth:`neon.backends.Tensor.transpose<.Tensor.transpose>` | Return a transposed view of the data.
    :py:meth:`neon.backends.Tensor.reshape<.Tensor.reshape>` | Adjust the dimensions of the data to the specified shape.
    :py:meth:`neon.backends.Backend.take<neon.backends.Backend.take>` | Select a subset of elements (based on provided indices) from a supplied dimension


Arithmetic Operations
---------------------

Unary and binary arithmetic operations can be performed on ``Tensor``
objects. In all cases, the user must pre-allocate correctly sized output
to store the result. All the below operations are performed
element-wise.

Element-wise Binary Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These element-wise binary operations perform the following operations.
An optional ``out=`` argument can be passed to store the result as a
Tensor. If ``out=None`` (default), an op-tree is returned.

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.add(a, b)<.Backend.add>` | :math:`a+b`
    :py:meth:`neon.backends.Backend.subtract(a,b)<.Backend.subtract>` | :math:`a-b`
    :py:meth:`neon.backends.Backend.multiply(a, b)<.Backend.multiply>` | :math:`a\times b`
    :py:meth:`neon.backends.Backend.divide(a, b)<.Backend.divide>` | :math:`\frac{a}{b}`
    :py:meth:`neon.backends.Backend.dot(a, b)<.Backend.dot>` | :math:`a \cdot b`
    :py:meth:`neon.backends.Backend.power(a, b)<.Backend.power>` | :math:`a^b`
    :py:meth:`neon.backends.Backend.maximum(a, b)<.Backend.maximum>`| :math:`\max(a, b)`
    :py:meth:`neon.backends.Backend.minimum(a, b)<.Backend.minimum>` | :math:`\min(a,b)`
    :py:meth:`neon.backends.Backend.clip(a, a_min, a_max)<.Backend.clip>` | Clip each element of :math:`a` between the corresponding elements in :math:`a_\text{min}` and :math:`a_\text{max}`

Element-wise Unary Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These element-wise operations operate on one input Tensor or Op-tree.
Similar to the binary operations, an optional ``out=`` argument can be
passed to store the result as a Tensor. If ``out=None`` (default), an
op-tree is returned.

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.log(a)<.Backend.log>` | :math:`\log(a)`
    :py:meth:`neon.backends.Backend.log2(a)<.Backend.log2>` | :math:`\log_2(a)`
    :py:meth:`neon.backends.Backend.exp(a)<.Backend.exp>` | :math:`e^a`
    :py:meth:`neon.backends.Backend.exp2(a)<.Backend.exp2>` | :math:`2^a`
    :py:meth:`neon.backends.Backend.abs(a)<.Backend.abs>` | :math:`abs(a)`
    :py:meth:`neon.backends.Backend.sgn(a)<.Backend.sgn>` | if :math:`x<0`, :math:`-1`; if :math:`x=0`, :math:`0`; if :math:`x>0`, :math:`1`
    :py:meth:`neon.backends.Backend.sqrt(a)<.Backend.sqrt>` | :math:`\sqrt{a}`
    :py:meth:`neon.backends.Backend.square(a)<.Backend.square>` | :math:`a^2`
    :py:meth:`neon.backends.Backend.reciprocal(a)<.Backend.reciprocal>` | :math:`1/a`
    :py:meth:`neon.backends.Backend.negative(a)<.Backend.negative>` | :math:`-a`
    :py:meth:`neon.backends.Backend.sig(a)<.Backend.sig>` | :math:`1/(1+\exp(-a))`
    :py:meth:`neon.backends.Backend.tanh(a)<.Backend.tanh>` | :math:`\tanh(a)`
    :py:meth:`neon.backends.Backend.finite(a)<.Backend.finite>` | Element-wise test for finiteness (e.g. :math:`a \neq \infty`)

Element-wise Logical Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods perform element-wise logical testing on each corresponding
element of the input :math:`a` and :math:`b`. As before, an optional ``out=`` argument can be passed to store the
output.

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.equal(a, b)<.Backend.equal>` |	:math:`a=b`
    :py:meth:`neon.backends.Backend.not_equal(a, b)<.Backend.not_equal>` |	:math:`a\neq b`
    :py:meth:`neon.backends.Backend.greater(a, b)<.Backend.greater>` |	:math:`a>b`
    :py:meth:`neon.backends.Backend.greater_equal(a, b)<.Backend.greater_equal>` |	:math:`a \geq b`
    :py:meth:`neon.backends.Backend.less(a, b)<.Backend.less>` |	:math:`a<b`
    :py:meth:`neon.backends.Backend.less_equal(a, b)<.Backend.less_equal>` |	:math:`a\leq b`

Summary Operations
~~~~~~~~~~~~~~~~~~

These operations perform a summary calculation over a single provided
tensor ``a`` along a specified ``axis`` dimension. If ``axis=None``
(default), the calculation is performed over all the dimensions.

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.sum<.Backend.sum>` | Sum elements over the ``axis`` dimension
    :py:meth:`neon.backends.Backend.mean<.Backend.mean>` | Compute the arithmetic mean over the ``axis`` dimension
    :py:meth:`neon.backends.Backend.var<.Backend.var>` | Compute the variance of the elements over the ``axis`` dimension
    :py:meth:`neon.backends.Backend.std<.Backend.std>` | Compute the standard deviation of the elements along the ``axis`` dimension
    :py:meth:`neon.backends.Backend.min<.Backend.min>` |	Calculate the minimal element value over the ``axis`` dimension
    :py:meth:`neon.backends.Backend.max<.Backend.max>` |	Calculate the maximal element value over the ``axis`` dimension
    :py:meth:`neon.backends.Backend.argmin<.Backend.argmin>` | Calculate the indices of the minimal element value along the ``axis`` dimension
    :py:meth:`neon.backends.Backend.argmax<.Backend.argmax>` | Calculate the indices of the maximal element value along the ``axis`` dimension

Random Number Generator
~~~~~~~~~~~~~~~~~~~~~~~

Both the ``NervanaGPU`` and ``NervanaCPU`` backends use the numpy random
number generator (``np.random.RandomState``).

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.gen_rng<.Backend.gen_rng>` |	Setup the numpy rng and store the initial state. Called by the backend constructor
    :py:meth:`neon.backends.Backend.rng_get_state<.Backend.rng_get_state>` | Return the state of the rng
    :py:meth:`neon.backends.Backend.rng_set_state<.Backend.rng_set_state>` | Set the state of the rng
    :py:meth:`neon.backends.Backend.rng_reset<.Backend.rng_reset>` | Reset the state to the initial state
    :py:meth:`neon.backends.NervanaGPU.rand<.nervanagpu.NervanaGPU.rand>` | Generate random numbers uniformly distributed between 0 and 1.

To generate a Tensor with shape ``(100,100)``, where each element is
uniformly distributed between 0 and 1, we can call

.. code-block:: python

    myTensor = be.empty((100,100))
    myTensor[:] = be.rand()

Loop indicators
~~~~~~~~~~~~~~~

These two methods signal the start (and end) of a block of repeated
computations, such as a loop. This operation can be used to help the
compiler optimize instruction performance, but has no direct effect on
the underlying calculations. Each ``Backend.start()`` call must be
book-ended by a corresponding ``Backend.end()`` call. Note that multiple
``begin`` calls can appear adjacent in nested loops.

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.begin<.Backend.begin>` | Signal the start of a block
    :py:meth:`neon.backends.Backend.end<.Backend.end>` |	Signal the end of a block

Higher-level Support
--------------------

We have taken common operations used by neural network layers and
performed many optimizations. Many of these operations include custom
objects (such as :py:class:`ConvLayer<.layer_cpu.ConvLayer>` or :py:class:`PoolLayer<.layer_cpu.PoolLayer>`) which are used to track
the layer parameters and cache some calculations.

Operations
~~~~~~~~~~

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.onehot<.Backend.onehot>` | Generate a one-hot representation from a set of indices (see :doc:`Loading data<loading_data>`)
    :py:meth:`neon.backends.Backend.fill_normal<.NervanaGPU.fill_normal>` | Fill a tensor with gaussian random variables
    :py:meth:`neon.backends.Backend.compound_dot<.Backend.compound_dot>` | Depending on the size of the input :math:`A`, :math:`B`, :math:`C`, perform :math:`\alpha A B + \beta C`
    :py:meth:`neon.backends.NervanaGPU.make_binary_mask<.NervanaGPU.make_binary_mask>` | Create a randomized binary mask for dropout layers
    :py:meth:`neon.backends.NervanaCPU.make_binary_mask<.NervanaCPU.make_binary_mask>` | Create a randomized binary mask for dropout layers

Convolutional Layers
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.conv_layer<.Backend.conv_layer>` | Create a ``ConvLayer`` object that holds the filter parameters. This is passed to the below functions.
    :py:meth:`neon.backends.Backend.fprop_conv<.Backend.fprop_conv>` | Forward propagate the inputs of a convolutional network layer
    :py:meth:`neon.backends.Backend.bprop_conv<.Backend.bprop_conv>` | Backward propagate the error through a convolutional network layer.
    :py:meth:`neon.backends.Backend.update_conv<.Backend.update_conv>` |	Compute the updated gradient for a convolutional network layer.
    :py:meth:`neon.backends.Backend.deconv_layer<.Backend.deconv_layer>` | Create a ``DeconvLayer`` object that holds the filter parameters. This is passed to the above functions.

Pooling Layers
~~~~~~~~~~~~~~

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.pool_layer<.Backend.pool_layer>` |	Create a new ``PoolLayer`` parameter object.
    :py:meth:`neon.backends.Backend.fprop_pool<.Backend.fprop_pool>` | Forward propagate pooling layer.
    :py:meth:`neon.backends.Backend.bprop_pool<.Backend.bprop_pool>` | Backward propagate pooling layer.

The below methods implement
`ROI-pooling <http://arxiv.org/pdf/1504.08083.pdf>`__, where the pooling
window sizes are themselves hyper-parameters.

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.roipooling_fprop<.NervanaGPU.roipooling_fprop>` | Forward propagate through ROI-pooling layer
    :py:meth:`neon.backends.Backend.roipooling_bprop<.NervanaGPU.roipooling_bprop>` | Backward propagate through ROI-pooling layer

Local Response Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.lrn_layer<.NervanaGPU.lrn_layer>` | Create the `LRNLayer` parameter object
    :py:meth:`neon.backends.Backend.fprop_lrn<.NervanaGPU.fprop_lrn>` | Forward propagate through LRN layer
    :py:meth:`neon.backends.Backend.bprop_lrn<.NervanaGPU.bprop_lrn>` | Backward propagate through LRN layer

Batch Normalization
~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.compound_fprop_bn<.NervanaGPU.compound_fprop_bn>` | Forward propagate through BatchNorm layer
    :py:meth:`neon.backends.Backend.compound_bprop_bn<.NervanaGPU.compound_bprop_bn>` | Backward propagate through BatchNorm layer


Linear layer with bias
~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Method", "Description"
    :widths: 20, 40
    :delim: |

    :py:meth:`neon.backends.Backend.update_fc_bias<.Backend.update_fc_bias>` | Compute the updated bias gradient for a fully connected layer
    :py:meth:`neon.backends.Backend.add_fc_bias<.Backend.add_fc_bias>` | Add bias to a fully connected layer
