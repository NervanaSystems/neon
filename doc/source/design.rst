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

Design Decisions
================

Computation backend
-------------------
* All objects inherit from ``NervanaObject`` which has a static ``be`` variable
  which is the computation backend being used (gpu, mkl, or cpu).

  * ``be`` stores other important attributes like batch size and data type.
  * A backend must first be generated before running a model using ``gen_backend``.
  * If swapping backends, then buffers must be reinitialized by reinstantiating the
    model layers and calling fprop with the new generated backend.

Data Layout
-----------

Neon's layers internally store data as two-dimensional tensors. For convolution and pooling layers, the data
is formatted in :math:`(C, H, W, N)` layout (:math:`C` = channels, :math:`H` = height, :math:`W` = width, :math:`N` = batch size), and represented as a tensor of shape :math:`(F, N)`, where :math:`F = C * H *W`.

For recurrent layers, the time dimension :math:`T` is added to the :math:`N` dimension, so the data format is :math:`(F, T*N)`. The second dimension is ordered by incrementing the batch index first: :math:`t_1n_1, t_1n_2, ... t_1n_N, t_2n_1, t_2n_2, ...`

Layers
------
* Most layers are in layer.py, recurrent layers are in recurrent.py, and merge layers
  for concatenating or summing input are in merge.py.

Composite layers
'''''''''''''''''
* Some layers (for convenience) are composite layers made as lists of other layers.

  * Conv is a list of Convolution, Bias, and Activation layers
  * Affine is a list of Linear, Bias, and Activation layers
  * This allows flexibility in adding optional bias and activation layers without
    having to specify these as separate layers.

Layer buffer allocations
''''''''''''''''''''''''''''
* Data buffers

  * A layer infers input shape from previous layers and initializes buffers accordingly.
  * Pre-allocating activation buffers allows buffer reuse and reduces memory usage.
  * Buffers will be reinitialized during the next fprop if the layer is reinstantiated.

* Parameter layers (``Linear``, ``Bias``, ``Convolution``, and ``BatchNorm``) maintain
  their own parameters ``W``, gradients ``dW``, and states ``states`` (for the optimizer).
* In general, layer buffer allocation is kicked off by the containing model, being
  called prior to the first ``fit`` or ``eval`` call.

Initialization
'''''''''''''''

* Weight initialization routines are in ``initializers.py`` and all have a
  ``fill`` method that describe how they will fill a given param buffer.
* The weight initialization object is passed to the layer constructor and
  the layer will fill the parameters during ``init_params``.

Models
------

Model container
''''''''''''''''
* The model provides a container of all the network layers and provides function calls
  to run and train the network.  It is also responsible for initializing and
  allocating layer parameter buffers.
* We can create a list of layers and give that to the model.
* When forward or backward propagation functions are called, the model will iterate
  through all the layers to forward pass the inputs and backward pass the errors.

Learning
''''''''
* When training the model, the following necessary components will be provided:

  * a training set object that can iterate over training data
  * an optimizer that applies to all the layer updates or a multi-optimizer that
    maps different optimizers to different layers by layer name
  * a cost function to compute the error
  * callback object that configures whether to use a validation set and how frequent
    in the training to validate, whether to get progress bar display, etc. For more
    information, see :doc:`neon fundamentals -- callbacks <callbacks>`.

* During update, the model sends a list containing all layers with learnable parameters
  to the optimizer.

  * The optimizer will then grab a tuple of ``(W, dW, state)`` from each layer and apply
    the updates.

Choice of sizes
---------------
* We will get better utilization if we pick more friendly sizes for batch size,
  sequence length, or feature size.
* Our GPU kernels are optimized for sizes being multiples of 4.
* In many of our examples, we use parameters from reference implementations. However,
  it is recommended to use multiples of 4. In many cases, zero-padding is needed to
  implement the same model.
