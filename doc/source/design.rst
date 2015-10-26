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
  which is the computation backend being used (gpu or cpu).

  * ``be`` stores other important attributes like batch size and data type.
  * A backend must first be generated before running a model using ``gen_backend``.
  * If swapping backends, then buffers must be reinitialized by reinstantiating the
    model layers and calling fprop with the new generated backend.

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
  * Buffers will be reinitalized during the next fprop if the layer is reinstantiated.

* Parameter layers (``Linear``, ``Bias``, ``Convolution``, and ``BatchNorm``) maintain
  their own parameters ``W``, gradients ``dW``, and states ``states`` (for the optimizer).

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
  to run and train the network.
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
    information, please checkout neon fundamentals -- callbacks

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

