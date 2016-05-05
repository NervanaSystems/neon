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

Models
======

The :py:class:`Model<neon.models.model>` class represents the network architecture,
either provided as a layer container, a list of layers, or loaded from a
previously saved model file.

Lifecycle of a model
--------------------

Instantiation
~~~~~~~~~~~~~

We create a model by passing either a list of layers or a container. If
a list of layers is provided, the model will wrap the layers in a
:py:class:`Sequential<neon.layers.container.Sequential>` container. Combinations of lists and
containers is not supported. When a model is instantiated, the layer
shapes are not determined until a training set is provided.

Training
~~~~~~~~

To train, call the ``model.fit()`` function and provide

.. csv-table::
    :header: Argument, Description
    :widths: 20, 50
    :delim: |

    dataset | An iterable of minibatches of the dataset (e.g. :py:class:`ArrayIterator<neon.data.dataiterator.ArrayIterator>`).
    cost | Cost function to apply to the output of the last layer (:py:class:`neon.transforms.Costs<neon.transforms.cost.Cost>`)
    optimizer | The learning rule for updating the model parameters (:py:mod:`neon.optimizers`)
    num_epochs | Number of iterations over the dataset
    callbacks | Functions to run at the start/end of each epoch/minibatch (:py:mod:`neon.callbacks`)



When ``model.fit()`` is called and training data provided, the model is
first initialized with ``model.initialize()``. During initialization,
the dataset is propagated through the layers to call each layer's
``configure()`` method to set the input and output shapes. Then, the
appropriate buffers are allocated with each layer's ``allocate()``
method. Note that a model object can only be initialized once.

During training, the model iterates through mini-batches of the dataset,
calling the forward and backward propagation functions to compute the
gradients according to the provided ``cost`` and update the weights
based on the ``optimizer``. The length of training is controlled by the
``num_epochs`` argument. Callbacks can also be configured to end training
when certain exit conditions are met.

.. code-block:: python

    # Pseudo-code of training procedure in neon.models.model
    for (x_train, y_train) in dataset:

        # fprop through the layers
        x_train = self.fprop(x_train)

        # get deltas in the cost
        delta = self.cost.get_errors(x_train, y_train)

        # backprop the deltas through the layers
        self.bprop(delta)

        # update the weights
        self.optimizer.optimize(self.layer, epoch=epoch)

Evaluation
~~~~~~~~~~

When training is completed, the model can be evaluated against a
provided Metric and dataset with the ``model.eval(dataset, metric)``
method. This method iterates over the provided dataset, and calls
``fprop`` to obtain the model output. For efficient inference, the model
calls ``fprop`` with ``inference=True`` argument to avoid unneeded
memory and computation.

To directly obtain the model outputs for a specific dataset, the
``model.get_outputs(dataset)`` method can also be called, which returns
a numpy array with the final layer output for each example in the
dataset.

Inspecting the model
--------------------

The easiest way of inspecting a model's weights is by accessing the
layer parameters directly. For example, to get the Tensor for the first
layer in a model, call:

.. code-block:: python

    mlp.layers.layers[0].W

To get the entire model configuration and weights, call

.. code-block:: python

    pdict = model.get_description(get_weights=True)

We can now inspect each layer by obtaining a list of dicts, one for each
layer:

.. code-block:: python

    ldict = pdict['model']['config']['layers']

Each layer dict has three keys:

* ``'config'``: arguments passed to the constructor (e.g., name, weight initializer)
* ``'type'``: layer class (e.g. ``neon.layers.layer.Linear``)
* ``'params'``: dict of layer parameters (e.g. ``'W'`` for the weight matrix)

For example, we can obtain a numpy array with the weight matrix of the
first layer by calling

.. code-block:: python

    W = ldict[0]['params']['W']

    # or more directly,
    W = pdict['model']['config']['layers'][0]['params']['W']

Note that this copies all the data from the GPU device to host to
produce the numpy array values.

Loading and saving models
-------------------------

The entire model (layers, per layer weights, epochs run, optimizer
states, etc.) can be saved and loaded from disk with neon's
serialization feature.

There are two ways to save a model. One can call, after fitting is
complete:

.. code-block:: python

    model.save_params("mnist_model.prm")

This will save the model objects into "save\_path.prm". Alternatively,
the command line argument ``--serialize n`` will save the model every
``n`` epochs:

.. code-block:: python

    python mnist_mlp.py --save_path mnist_model.prm --serialize 1 -e 3 \

Then, the model will be saved every epoch of training.

To load the model, pass the file to the ``model`` constructor:

.. code-block:: python

    new_model = Model("mnist_model.prm")
