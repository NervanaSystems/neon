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
..  ---------------------------------------------------------------------------

Tutorials 
=========

How to run a model
------------------
With the virtual environment activated, there are two ways to run
models through neon. The first is to use the neon executable and pass
it a model specified by a YAML file. The second way is to specify
the model directly through a python script. There are a number of
examples available in the ``examples`` directory. The one we focus on
here is an MLP trained on the MNIST dataset.

YAML based example
''''''''''''''''''

From the neon repository directory, type

.. code-block:: bash

    neon examples/mnist_mlp.yaml

On the first run, neon will download the MNIST dataset. It will create
a ``~/nervana`` directory where the raw datasets are kept, and store
processed batches in the neon repository directory. Once that is done,
it will train a simple MLP model on the dataset and report
cross-entropy error after each epoch.

Python script example
'''''''''''''''''''''

The same model is available in a python script format that can be called
directly without using a YAML specification to create the model. To
run the script, type

.. code-block:: bash

    examples/mnist_mlp.py

This will run an identical MLP model and print the final
misclassification error after running for 10 epochs.


Simple MLP walk-through
-----------------------

This example follows the model from a slightly simplified version of
:py:obj:`examples/mnist_mlp.py`.

The first step is to set up a logger and argument parser. The logging
module gives us control over printing messages to stdout or to file,
and controls verbosity of the output.
:py:class:`NeonArgparser<neon.util.argparser.NeonArgparser>` is used to
parse command line arguments, such as number of training epochs, how
often to run cross-validation, where to save the model, etc. It also
controls backend settings, such as running on GPU or CPU, which
datatype to use, and how rounding is performed. For a full list of
arguments, run ``neon --help`` and see :py:obj:`examples/mnist_mlp.py`
for an example of how to work with the parser.

.. code-block:: python

    import logging
    logger = logging.getLogger()

    # parse the command line arguments
    from neon.util.argparser import NeonArgparser
    parser = NeonArgparser()
    args = parser.parse_args()

Backend Setup
'''''''''''''
The backend is controlled via the ``-b`` command line argument, which takes a
single parameter.  This parameter value can be ``gpu`` to select our
:py:class:`NervanaGPU<neon.backends.nervanagpu.NervanaGPU>` based backend
or ``cpu`` to select :py:class:`NervanaCPU<neon.backends.nervanacpu.NervanaCPU>`
as the backend. By default, the GPU backend is used on machines with a Maxwell
capable GPU. On machines where no compatible GPU is found, neon will
automatically fail back to using the CPU. The following block of code sets up
the backend.

.. code-block:: python

    from neon.backends import gen_backend
    be = gen_backend(backend=args.backend,
                     batch_size=128,
                     rng_seed=args.rng_seed,
                     device_id=args.device_id,
                     default_dtype=args.datatype,
                     stochastic_round=False)


The :py:func:`gen_backend` function will handle generating and
switching backends. When called repeatedly, it will clean up an
existing backend and generate a new one. If a GPU backend was
generated previously, then :py:func:`gen_backend` will destroy the
existing context and delete the backend object. See :doc:`backends`
for detail on all the options that can be set when generating a backend.

The minibatch size for training defaults to 128 input items and
stochastic rounding (mainly useful for estimating models in 16 bit
precision) is disabled. The ``rng_seed`` argument can be used to specify a
fixed random seed, ``device_id`` controls which GPU to run on if multiple
GPUs are available, and the ``default_dtype`` can be used to specify a 32
or 16 bit data type.


Loading a Dataset
'''''''''''''''''

To load the MNIST dataset, the :py:func:`load_mnist` function is included
with the ``neon/data/datasets.py`` utility. The data is set up on the
GPU as a :py:class:`DataIterator<neon.data.dataiterator.DataIterator>`, which
provides an interface to iterate over mini-batches after pre-loading them into
device memory.

.. code-block:: python

    from neon.data import DataIterator, load_mnist
    # split into train and tests sets
    (X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
    # setup training set iterator
    train_set = DataIterator(X_train, y_train, nclass=nclass)
    # setup test set iterator
    test_set = DataIterator(X_test, y_test, nclass=nclass)


See :doc:`datasets`  to learn how to load the other datasets or add your own.


Weight Initialization
'''''''''''''''''''''

Neon supports initializing weight matrices with constant, uniform, Gaussian,
and automatically scaled uniform (Glorot initialization) distributed values.
This example uses :py:class:`Gaussian<neon.initializers.initializer.Gaussian>`
initialization with zero mean and 0.01 standard deviation.

.. code-block:: python

    from neon.initializers import Gaussian
    init_norm = Gaussian(loc=0.0, scale=0.01)

The weights will be initialized below when the layers are created.

Learning Rules
''''''''''''''

The examples uses :py:class:`Gradient Descent with Momentum<neon.optimizers.optimizer.GradientDescentMomentum>`
as the learning rule:

.. code-block:: python

    from neon.optimizers import GradientDescentMomentum
    optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9,
                                        stochastic_round=args.rounding)

If stochastic rounding is used, it is applied exclusively to weight updates, so
it is passed as a parameter to the optimizer.

Layers
''''''

The model is specified as a list of layer instances, which are defined
by a layer type and an activation function. This example uses affine
(i.e. fully-connected) layers with a rectified linear activation on
the hidden layer and a logistic activation on the output layer. We set
our final layer to have 10 units in order to match the number of
labels in the MNIST dataset.  The ``shortcut`` parameter in the logistic
activation allows one to forego computing and returning the actual derivative
during backpropagation, but can only be used with an appropriately paired cost
function like cross entropy.

.. code-block:: python

    from neon.layers import Affine
    from neon.transforms import Rectlin, Logistic

    layers = []
    layers.append(Affine(nout=100, init=init_norm, activation=Rectlin()))
    layers.append(Affine(nout=10, init=init_norm,
                         activation=Logistic(shortcut=True)))


Other layer types that are not used in this example include
convolution and pooling layers. They are described in :doc:`layers`. Weight
layers take an initializer for the weights, which we have defined in
``init_norm`` above.


Costs
'''''

The cost function is wrapped into a ``GeneralizedCost`` layer, which handles
the comparison of the cost function outputs with the labels provided with the
data set. The cost function passed into the cost layer is the cross-entropy
transform in this example.

.. code-block:: python

    from neon.layers import GeneralizedCost
    from neon.transforms import CrossEntropyBinary
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())


Model
'''''

We generate a model using the layers created above, and instantiate a
set of standard callbacks to display a progress bar during training,
and to save the model to a file, if one is specified in the command
line arguments. We then train the model on the dataset set up as
``train_set``, using the optimizer and cost functions defined
above. The number of epochs (complete passes over the entire training set)
to train for is also passed in through the arguments.

.. code-block:: python

    # initialize model object
    from neon.models import Model
    mlp = Model(layers=layers)

    # setup standard fit callbacks
    from neon.callbacks.callbacks import Callbacks
    callbacks = Callbacks(mlp, train_set, output_file=args.output_file,
                          progress_bar=args.progress_bar)

    # run fit
    mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost,
            callbacks=callbacks)


Evaluation Metric
'''''''''''''''''

Finally, we can evaluate the performance of our now trained model by examining
its misclassification rate on the held out test set.

.. code-block:: python

    from neon.transforms import  Misclassification
    print('Misclassification error = %.1f%%'
          % (mlp.eval(test_set, metric=Misclassification())*100))


Videos
------

SV Deep Learning Meetup, 2015/11/17
'''''''''''''''''''''''''''''''''''
Anil Thomas gives an introduction to convolutional neural networks and shows
how to apply them to the Kaggle right-wale detection challenge using neon.

Arjun Bansal also gives an introduction to Nervana.

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/WfuDrJA6JBE" frameborder="0"
    allowfullscreen></iframe>

.. raw:: html

    <iframe src="//www.slideshare.net/slideshow/embed_code/key/uLVbsqBj6kynGd"
    width="420" height="315" frameborder="0" marginwidth="0" marginheight="0"
    scrolling="no" style="border:1px solid #CCC; border-width:1px;
    margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe>

SD Deep Learning Meetup, 2015/12/02
'''''''''''''''''''''''''''''''''''
Urs Koster gives a hands on walkthrough of convolutional and recurrent neural
networks using neon, running under jupyter noteboks.

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/3lTti-RmeQQ" frameborder="0"
    allowfullscreen></iframe>

.. raw:: html

    <iframe src="//www.slideshare.net/slideshow/embed_code/key/Bg6SE0uWDlSDSG"
    width="420" height="315" frameborder="0" marginwidth="0" marginheight="0"
    scrolling="no" style="border:1px solid #CCC; border-width:1px;
    margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe>
