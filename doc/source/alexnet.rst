AlexNet example
===============

This tutorial introduces image classification with convolutional neural
networks. We will train AlexNet, a seven-layer convolutional network, on
a truncated dataset of ImageNet, a large corpus of labeled natural
images.

In particular, you will learn how to:

1. Work with a training set that is too large to fit in memory. We will configure neon to load the images in batches for training.
2. Define different learning schedules for different layers
3. Debug shape mismatch issues for convolutional layers

Among deep learning libraries, Neon has the fastest computation times on
convolutional networks similar to this example.

ImageNet dataset
----------------

ImageNet consists of around 1 million color images drawn from 1000
object categories. Each image is of varying size. The challenge is to
build a network that classifies each image into one of the object
categories.

The entire dataset is 300 gigabytes, which is too large to load into
memory.

We first use ``batch_writer.py`` to chunk the data into macrobatch
files. The writer also resizes each image to a 256x256x3 matrix (height
x width x color channels). See :doc:`Loading data <loading_data>` for
more details.

.. code-block:: bash

    python neon/data/batch_writer.py  --data_dir /usr/local/data/tmp \                                      --image_dir /usr/local/data/I1K/imagenet_orig \ --set_type i1k

During training, these macrobatches are loaded into memory individually
and then split into minibatches.

Model Specification
-------------------

.. code-block:: python

    from neon.initializers import Constant, Gaussian
    from neon.layers import Conv, Dropout, Pooling, Affine
    from neon.transforms import Rectlin, Softmax

    init_gauss1 = Gaussian(scale=0.01)
    init_gauss3 = Gaussian(scale = 0.03)

For ease of reading, we break apart the layer specifications.

.. code-block:: python

    layers = []

We start with two pairs of convolution and pooling. The convolution
layer takes a tuple specifying the (width, height, # of filters) of the
layer, as well as the strides and padding.

.. code-block:: python

    layers.append([
                  Conv((11, 11, 64), init=init_gauss1, bias=Constant(0),
                       activation=Rectlin(), padding=3, strides=4),
                  Pooling(3, strides=2),
                  Conv((5, 5, 192), init=init_gauss1, bias=Constant(1),
                       activation=Rectlin(), padding=2),
                  Pooling(3, strides=2)])

Then, we add three convolutional layers, and one final pooling layer

.. code-block:: python

    layers.append([
                  Conv((3, 3, 384), init=init_gauss1, bias=Constant(0),
                       activation=Rectlin(), padding=1),
                  Conv((3, 3, 256), init=init_gauss1, bias=Constant(1),
                       activation=Rectlin(), padding=1),
                  Conv((3, 3, 256), init=init_gauss1, bias=Constant(1),
                       activation=Rectlin(), padding=1),
                  Pooling(3, strides=2)])

Now we concatenate two fully connected (affine) layers, with dropout
layers interspersed for regularization.

.. code-block:: python

    layers.append([
                  Affine(nout=4096, init=init_gauss1, bias=Constant(1),
                         activation=Rectlin()),
                  Dropout(keep=0.5),
                  Affine(nout=4096, init=init_gauss1, bias=Constant(1),
                         activation=Rectlin()),
                  Dropout(keep=0.5)])

Finally, we add a output layer with 1000 units for each object category,
configured with the ``Softmax`` activation. This layer is equivalent to
logistic regression.

.. code-block:: python

    layers.append([
                   Affine(nout=1000, init=init_gauss1, bias=Constant(-7), activation=Softmax())])


Learning schedules
------------------

The training process for this model is more sophisticated than the
previous example. We define a schedule for dropping the learning rate
over the course of training. We also use different learning rates and
schedules for the bias weights.

First, set up the optimizer for the bias weights. We want to drop the
learning rate by 1/10 at the end of epoch 44.

.. code-block:: python

    from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule

    # Set up optimizer for the bias weights
    # At the end of epoch 44, drop the learning rate by 1/10
    bias_schedule = Schedule([44], 0.1)
    opt_bias = GradientDescentMomentum(0.02, 0.9, schedule=bias_schedule)

For all the other layers, we use a slower initial learning rate, and
drop the learning rates by 0.15 at epochs 22, 44, and 65.

.. code-block:: python

    weight_sched = Schedule([22, 44, 65], 0.15)
    opt_default = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched)

We define a :py:class:`MultiOptimizer<neon.optimizers.optimizer.MultiOptimizer>` and pass a mapping of layers to optimizers in the form of a dictionary. The keys can either be: ``default``, a layer class name (e.g. ``Bias``), or the Layer's name attribute. The latter takes precedence for finer layer-to-layer control (Don't name your layers ``default``).

.. code-block:: python

    # Define the mapping of layers to optimizers
    opt = MultiOptimizer({'default': opt_default, 'Bias': opt_bias})

Now, the bias layers will use ``opt_bias``, and all other layers will
use the ``opt_default`` optimizer.

Train and evaluate model
------------------------

Now we are ready to train the model for one epoch to obtain some
interesting results.

.. code-block:: python

    from neon.transforms import CrossEntropyMulti, TopKMisclassification
    from neon.callbacks.callbacks import Callbacks
    from neon.models import Model

    # define cost, metric, and callbacks

    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    valmetric = TopKMisclassification(k=5)
    callbacks = Callbacks(model, eval_set=test, metric=valmetric)

    # train model
    model.fit(train, optimizer=opt, num_epochs=1, cost=cost, callbacks=callbacks)
