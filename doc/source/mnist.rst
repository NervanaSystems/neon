MNIST Example
=============

MNIST is a computer vision dataset consisting of 70,000 images of handwritten digits.
Each image has 28x28 pixels for a total of 784 features, and is associated with
a digit between 0-9.

In this tutorial, we will construct a multi-layer perceptron (also
called softmax regression) to recognize each image. Note that this
tutorial assumes some basic familiarity with python and machine
learning.

This tutorial is similar to the model specified in ``examples/mnist_mlp.py``.

Preamble
--------

The first step is to set up the argument parser, which enables
customizing options with flags (see the previous chapter).

.. code-block:: python

    #!/usr/bin/env python
    from neon.util.argparser import NeonArgparser

    parser = NeonArgparser(__doc__)
    args = parser.parse_args()

By default, :py:meth:`~.parse_args()` will create a computational
:doc:`backend <backends>` on a GPU, if present, or a CPU.

MNIST dataset
-------------

The MNIST dataset can be found on `Yann LeCunn's
website <http://yann.lecun.com/exdb/mnist/>`__. We have included an easy
function that downloads the MNIST dataset into your ``nervana/data/``
directory and loads it into memory.

.. code-block:: python

    from neon.data import MNIST

    mnist = MNIST()

    (X_train, y_train), (X_test, y_test), nclass = mnist.load_data()

This function automatically splits the images ``X`` and labels ``y``
into training (60,000 examples) and testing (10,000 examples) data. The
training images ``X_train`` is a numpy array with shape
``(num_examples, num_features) = (60000, 784)``.

During training, neon iterates over the training examples to compute the
gradients. We use the following commands to set up the :py:class:`.ArrayIterator`
object that we send to the optimizer.

.. code-block:: python

    from neon.data import ArrayIterator

    # setup training set iterator
    train_set = ArrayIterator(X_train, y_train, nclass=nclass)
    # setup test set iterator
    test_set = ArrayIterator(X_test, y_test, nclass=nclass)

For small datasets like MNIST, this step may seem trivial. However, for
large datasets that cannot fit into memory (e.g.
`ImageNet <http://image-net.org/>`__ or
`Sports-1M <http://cs.stanford.edu/people/karpathy/deepvideo/>`__), the
data has to be efficiently loaded and fed to the optimizer in batches.
This requires more advanced iterators described in :doc:`Loading Data <loading_data>`.

Since it is a common function, the data iterator generation for stock
datasets can be done directly through helper methods contained in the
DataSet class.  For example, the MNIST training and validation set
iterators can be obtained with the following code:

.. code-block:: python

    from neon.data import MNIST
    mnist = MNIST()
    train_set = mnist.train_iter
    test_set = mnist.valid_iter

Model specification
-------------------

Training a deep learning model in neon requires specifying the dataset,
a list of layers, a cost function, and the learning rule. Here we guide
you through each item in turn.

Initializing weights
~~~~~~~~~~~~~~~~~~~~

Neon supports many ways of initializing weight matrices. In this
tutorial, we initialize the weights using a Gaussian distribution with
zero mean and 0.01 standard deviation.

.. code-block:: python

    from neon.initializers import Gaussian

    init_norm = Gaussian(loc=0.0, scale=0.01)

Model architecture
~~~~~~~~~~~~~~~~~~

The model is specified as a list of layers. For classifying MNIST
images, we use a multi-layer perceptron with fully connected layers.

-  Affine (i.e. fully-connected) layer with 100 hidden units and a
   `rectified linear <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`__
   activation function, defined as :py:class:`Rectlin()<neon.transforms.activation.Rectlin>`.
-  An output layer with 10 units to match the number of labels in the
   MNIST dataset. We use the :py:class:`Softmax()<neon.transforms.activation.Softmax>` activation function to ensure
   the outputs sum to one and are within the range :math:`[0, 1]`.

   .. code-block:: python

    from neon.layers import Affine
    from neon.transforms import Rectlin, Softmax

    layers = []
    layers.append(Affine(nout=100, init=init_norm, activation=Rectlin()))
    layers.append(Affine(nout=10, init=init_norm,
                         activation=Softmax()))

We initialize the weights in each layer with the ``init_norm`` defined
previously. Neon supports many other layer types (convolutional,
pooling, recurrent, etc.) that will be described in subsequent examples.
We then construct the model via

.. code-block:: python

    # initialize model object
    from neon.models import Model

    mlp = Model(layers=layers)

Costs
~~~~~

The cost function is wrapped within a :py:class:`.GeneralizedCost` layer, which
handles the comparison of the outputs with the provided labels in the
dataset. One common cost function which we use here is the `cross
entropy loss <https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression>`__.

.. code-block:: python

    from neon.layers import GeneralizedCost
    from neon.transforms import CrossEntropyMulti

    cost = GeneralizedCost(costfunc=CrossEntropyMulti())

To read more about costs, read :doc:`Costs and
metrics <costs>`.

Learning rules
~~~~~~~~~~~~~~

For learning, we use `stochastic gradient
descent <http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/>`__
with a learning rate of 0.1 and momentum coefficient of 0.9.

.. code-block:: python

    from neon.optimizers import GradientDescentMomentum

    optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

Additional optimizers and optional arguments are discussed in
:doc:`Optimizers <optimizers>`.

Callbacks
~~~~~~~~~

Neon provides an API for calling operations during the model fit (see
:doc:`Callbacks <callbacks>`). Here we set up the default callback,
which is displaying a progress bar for each epoch.

.. code-block:: python

    from neon.callbacks.callbacks import Callbacks

    callbacks = Callbacks(mlp, eval_set=test_set, **args.callback_args)

Putting it all together
~~~~~~~~~~~~~~~~~~~~~~~

We are ready to put all the ingredients together and run our model!

.. code-block:: python

    mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost,
            callbacks=callbacks)

At the beginning of the fitting procedure, neon propagates ``train_set``
through the model to set the input and output shapes of each layer. Each
layer has a ``configure()`` method that determines the appropriate layer
shapes, and an ``allocate()`` method to set up the needed buffers for
holding the forward propagation information.

During the training, neon sends batches of the training data through the
model, calling each layers' ``fprop()`` and ``bprop()`` methods to
compute the gradients and update the weights.

Using the trained model
-----------------------

Now that the model is successfully trained, we can use the trained model
to classify a novel image, measure performance, and visualize the
weights and training results.

Inference
~~~~~~~~~

Given a set of images such as those contained in the iterable
``test_set``, we can fetch the output of the final model layer via

.. code-block:: python

    results = mlp.get_outputs(test_set)

The variable ``results`` is a numpy array with shape
``(num_test_examples, num_outputs) = (10000,10)`` with the model
probabilities for each label.

Performance
~~~~~~~~~~~

Neon supports convenience functions for evaluating performance using
custom metrics. Here we measure the misclassification rate on the held
out test set.

.. code-block:: python

    from neon.transforms import Misclassification

    # evaluate the model on test_set using the misclassification metric
    error = mlp.eval(test_set, metric=Misclassification())*100
    print('Misclassification error = %.1f%%' % error)

Next steps
~~~~~~~~~~

This simple example guides you through the basic operations needed to
create and fit a neural network. However, neon contains a rich feature
set of customizable layers, metrics, and options. To learn more, we
recommend reading through the :doc:`CIFAR10 tutorial <cifar10>`,
which introduces convolutional neural networks.
