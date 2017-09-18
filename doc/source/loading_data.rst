Data loading
============

There are two components to working with data in neon. The first is a
data iterator (:py:class:`.NervanaDataIterator`), that feeds the model with
minibatches of data during training or evaluation. The second is a
dataset (:py:class:`Dataset`) class, which handles the loading and preprocessing
of the data. When working with your own data, the latter is optional
although highly recommended.

Data iterators are python iterables in that they implement the
``__iter__`` method, which returns a new minibatch of data with each
call.

* If your data is small enough to fit into memory:

    * For image data or other data in the form of numpy arrays, use |ArrayIterator|.
    * For specific modalities, neon includes specialized iterators (Text, Image Captioning, Q&A)

* If your data is too large:
    * For data in the HDF5 format, use the |HDF5Iterator| to load chunks of data to send to the model. This approach is flexible for
      any type of data.
    * For other types of data, use the macrobatching DataLoader, a specialized loader that loads macrobatches of data into memory, and then splits the macrobatches into minibatches to feed the model. This can be used for images, audio, video datasets and is recommended for large datasets or high-performance applications.

ArrayIterator
-------------

The |ArrayIterator| class provides for iteration over
minibatches of data that has been preloaded into memory as numpy arrays.
This iterator supports classification, regression, and autoencoder
tasks.

Classification
~~~~~~~~~~~~~~

Below is an example of a classification task with images where we load
in 10,000 images. Each image is 32x32 pixels with 3 color channels (R,
G, B), for a total of :math:`32\times32\times3=3,072` features.

.. code-block:: python

    from neon.data import ArrayIterator
    import numpy as np

    """
    X are the features and y are the labels.
    The data in X must have shape (# examples, feature size)
    """
    X = np.random.rand(10000,3072) # X.shape = (10000, 3072)
    """

    For classification, the labels y must have shape (# examples, 1). y must also
    consist of integers from 0 to nclass-1, where nclass is the number of categories.
    """
    y = np.random.randint(0,10,10000) # y.shape = (10000, )

    """
    The features X and labels y are passed to ArrayIterator be loaded into the backend
    nclass, the number of classes, is set to 10
    lshape, the local shape of the features, is set to (3,32,32) to represent
            the the image dimensions: 32x32 pixels with 3 channels
    """
    train = ArrayIterator(X=X, y=y, nclass=10, lshape=(3,32,32))

Importantly, the labels :math:`y` for classification should be integers from :math:`0` to :math:`K-1`,  where
:math:`K` is the number of classes. These labels are stored in the backend in a one-hot representation. This means that if we have :math:`N` labels with :math:`K` classes, the labels will be stored in a :math:`N \times K` binary matrix. Each column will be all zeros except at the :math:`k`-th element, which will be one. For example,

.. math::

   y = (0,0,1,3,2,2) \rightarrow \left( \begin{array}{cccccc}
   1 & 1 & 0 & 0 & 0 & 0\\
   0 & 0& 1 & 0 & 0 & 0 \\
   0 & 0& 0 & 0 & 1 & 1\\
   0 & 0& 0 & 1 & 0 & 0 \end{array}  \right).

Regression
~~~~~~~~~~

In regression, the model output for each training example is a vector :math:`\hat{y}` that is compared against a desired vector :math:`y` with a cost function (such as mean squared error). Below is a simple example implementing linear regression.

We first create the iterator. By default, ``ArrayIterator`` assumes
classification, so for regression we must set ``make_onehot = False`` to
turn off the one-hot representation.

.. code-block:: python

    from neon.data import ArrayIterator
    import numpy as np

    X = np.random.rand(1000, 1)
    y = 2*X + 1 + 0.01*np.random.randn(1000, 1)  # y = 2X+1 with some gaussian noise
    train = ArrayIterator(X=X, y=y, make_onehot=False)

We then fit a linear model with a bias term using stochastic gradient
descent:

.. code-block:: python

    from neon.initializers import Gaussian
    from neon.optimizers import GradientDescentMomentum
    from neon.layers import Linear, Bias
    from neon.layers import GeneralizedCost
    from neon.transforms import SumSquared
    from neon.models import Model
    from neon.callbacks.callbacks import Callbacks

    # Linear layer with one unit and a bias layer
    init_norm = Gaussian(loc=0.0, scale=0.01)
    layers = [Linear(1, init=init_norm), Bias(init=init_norm)]

    mlp = Model(layers=layers)

    # Loss function is the squared difference
    cost = GeneralizedCost(costfunc=SumSquared())

    # Learning rules
    optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

    # run fit
    mlp.fit(train, optimizer=optimizer, num_epochs=10, cost=cost,
            callbacks=Callbacks(mlp))

    # print weights
    slope = mlp.get_description(True)['model']['config']['layers'][0]['params']['W']
    print "slope = ", slope
    bias_weight = mlp.get_description(True)['model']['config']['layers'][1]['params']['W']
    print "bias = ", bias_weight

After training, the weights match what we expect:

.. code-block:: python

    slope =  [[ 2.01577163]]
    bias =  [[ 1.01664519]]

Autoencoders
~~~~~~~~~~~~

Autoencoders are a special case of regression where the desired outputs :math:`y` are the input features :math:`X`. For convenience, you can exclude passing the labels :math:`y` to the iterator:

.. code-block:: python

    # Example construction of ArrayIterator for Autoencoder task with MNIST
    from neon.data import MNIST
    from neon.data import ArrayIterator

    mnist = MNIST()

    # load the MNIST data
    (X_train, y_train), (X_test, y_test), nclass = mnist.load_data()

    # Set input and target to X_train
    train = ArrayIterator(X_train, lshape=(1, 28, 28))

For the full example, see ``examples/conv_autoencoder.py``.

Specialized ArrayIterators
~~~~~~~~~~~~~~~~~~~~~~~~~~

Neon includes specialized iterators that subclass from
``NervanaDataIterator`` for specific modalities where the entire dataset
can be directly loaded into memory.

.. csv-table::
   :header: "Name", "Description"
   :widths: 20, 40
   :escape: ~
   :delim: |

   :py:class:`neon.data.Text<neon.data.text.Text>` | Iterator for processing and feeding text data
   :py:class:`neon.data.ImageCaption<neon.data.imagecaption.ImageCaption>` | Iterator for feeding an image and a sentence for each training example
   :py:class:`neon.data.QA<neon.data.questionanswer.QA>` | Data iterator for taking a Q&A dataset, which has already been vectorized, and feeding data to training

For more information on usage of these iterators, see the API
documentation.

Sequence data
~~~~~~~~~~~~~
For sequence data, where data are fed to the model across multiple time steps, the shape
of the input data can depend on your usage.

* Often, data such as sentences are encoded as a vector sequence of integers, where each integer corresponds to a word in the vocabulary. This encoding is often used in conjunction with embedding layers. In this case, the input data should be formatted to have shape :math:`(T, N)`, where :math:`T` is the number of time steps and :math:`N` is the batch size. The embedding layer takes this input and provides as output to a subsequent recurrent neural network data of shape :math:`(F, T * N)`, where :math:`F` is the number of features (in this case, the embedding dimension). For an example, see `imdb_lstm.py <https://github.com/NervanaSystems/neon/blob/master/examples/imdb_lstm.py>`_.

* When the sequence data uses a one-hot encoding, the input data should be formatted to have shape :math:`(F, T*N)`. For example, if sentences use a one-hot encoding with 50 possible characters, and each sentence is 60-characters long, the input data will have shape :math:`(F=50, 60*N)`. See the :py:class:`.Text` class, or the `char_lstm.py <https://github.com/NervanaSystems/neon/blob/master/examples/char_lstm.py>`_ example.

* Time series data should be formatted to have shape :math:`(F, T * N)`, where :math:`F` is the number of features. For an example, see `timeseries_lstm.py <https://github.com/NervanaSystems/neon/blob/master/examples/timeseries_lstm.py>`_.

HDF5Iterator
-------------

For datasets that are too large to fit in memory the |HDF5Iterator| class can be used.  This uses
an HDF5 formatted data file to store the input and target data arrays so the data size is not limited
by on-host and/or on-device memory capacity.  To use the |HDF5Iterator|, the data arrays need to be
stored in an HDF5 file with the following format:

* The input data is in an HDF5 dataset named `input` and the target output, if needed, in a dataset named `output`. The data arrays are of the same format as the arrays used to initialize the |ArrayIterator| class.

* The `input` data class also requires an attribute named `lshape` which specifies the shape of the flattened input data array. For mean subtraction, an additional dataset named `mean` can be included in the HDF5 file which includes either a channel-wise mean vector or a complete mean image to subtract from the input data.

For alternate target label formats, such as converting the targets to a one-hot vector, or for autoencoder
data, the |HDF5IteratorOneHot| and |HDF5IteratorAutoencoder| subclasses are included.
These subclasses demonstrate how to extend the HDF5Iterator to handle different input and target data formats
or transformations.

See the example, `examples/mnist_hdf5.py`, for how to format the HDF5 data file
for use with the |HDF5Iterator| class.



Aeon DataLoader
---------------

If your data is too large to load directly into memory, use a
macrobatching approach. In macrobatching, the data is loaded in smaller
batches, then split further into minibatches to feed the model.
neon supports macrobatching with image, audio, and video datasets using
the ``AeonDataLoader`` class.

`Aeon <https://github.com/NervanaSystems/aeon>`_ is a new dataloader module we developed to load
macrobatches of data with ease and low latency. This module
uses a multithreaded library to hide the latency of decoding images,
applying augmentation and/or transformations, and transferring the resulting outputs to device memory
(if necessary). The module also adds optional functionality for applying
transformations (scale, flip, and rotation).

.. warning:: The old DataLoader and ImageLoader classes were recently deprecated. Documentation for these classes can be found `here <http://neon.nervanasys.com/docs/latest/previous_versions.html#neon-v1-5.4>`_.

.. warning:: In neon v2.2, we have moved to using a new version of aeon (v1.0+), which has a different manifest format, and also a different API in the provisioned data. We have provided a helper script in ``data/convert_manifest.py`` to assist in converting manifest files. In addition, by default the data loader object is wrapped with an adapter to convert the data from aeon v1.0 into the format expected the neon examples. See: ``AeonDataLoader` in ``data/aeon_shim.py`` and ``data/dataloaderadapter.py`` for more details.

Quick start guide
~~~~~~~~~~~~~~~~~

The user guide for aeon is found at http://aeon.nervanasys.com. Here we provide a quick start guide, but please consult the aeon user guide for important configurations and details.

Users interact with the aeon dataloader by providing two items:

1. Manifest file, a tab-separated file (*.tsv).
2. Configuration parameters, as a python dictionary.

Operations such as generating training/testing splits, or balancing labels for imbalanced datasets should be implemented outside of the dataloader by the user during **ingest** to create the appropriate manifest files. Several example ingest scripts are in the neon repository.

**Manifest files**

The manifest file contains UTF-8 text lines. Each line is one of header, comment, or record. For reference, please take a look at `aeon documentation <https://github.com/NervanaSystems/aeon/blob/rc1-master/doc/source/user_guide.rst#manifest-file>`_.

.. code-block:: bash

    @FILE FILE
    /image_dir/faces/naveen_rao.jpg	/labels/0.txt
    /image_dir/faces/arjun_bansal.jpg	/labels/0.txt
    /image_dir/faces/amir_khosrowshahi.jpg	 /labels/0.txt
    /image_dir/fruits/apple.jpg	/labels/1.txt
    /image_dir/fruits/pear.jpg	/labels/1.txt
    /image_dir/animals/lion.jpg	/labels/2.txt
    /image_dir/animals/tiger.jpg	/labels/2.txt
    ...
    /image_dir/vehicles/toyota.jpg	/labels/3.txt

**Configuration parameters**

Aeon is divided into separate providers for different modalities and problems. For image classification, we use the ``image,label`` provider. The configuration parameters include some base parameters for the dataloader itself, then a set of parameters for the input and target types of the provider. The configurations are provided as python dictionaries:

.. code-block:: python

    image_config = dict(height=40, width=50)
    label_config = dict(binary=False)

    config = dict(type="image,label",
                  image=image_config,
                  label=label_config,
                  manifest_filename='train.tsv',
                  minibatch_size=128)

For a full list of supported providers and their associated configurations, see documentation at: http://aeon.nervanasys.com.

**Dataloader Transformers**

Users often need to apply additonal transformations to the data being provided by aeon. Included in neon are several
:py:class:`.DataLoaderTransformer` classes that can be used to wrap the aeon dataloader. For example, we know that the ``image,label`` provider yields a pair of data ``(input, label)``. For classification tasks, to transform the label data into a one-hot representation (see Classification section above to learn about one-hot), we use the :py:class:`.OneHot` class:

.. code-block:: python
    from neon.data import AeonDataLoader
    from neon.data.dataloader_transformers import OneHot
    loader = AeonDataLoader(config, be)  # here be refers to the compute backend created with ``gen_backend`` function.
    loader = OneHot(loader, index=1, nclasses = 10)

During run-time, ``OneHot`` will apply the one-hot transformation to the data in ``index=1``. Neon includes several useful dataloader transformers for these purposes:

- :py:class:`.OneHot`: applies the one-hot transformation
- :py:class:`.PixelWiseOneHot`: applies the one-hot transformation on an image (e.g. image with HW -> HWK), where K is the number of classes.
- :py:class:`.TypeCast`: type cast a data to a different data type
- :py:class:`.BGRMeanSubtract`: Subtract pixel_mean from the data. Assumes data is in CHWN format, with C=3.


.. _aeon: https://github.com/NervanaSystems/aeon
.. |ArrayIterator| replace:: :py:class:`.ArrayIterator`
.. |HDF5Iterator| replace:: :py:class:`.HDF5Iterator`
.. |HDF5IteratorOneHot| replace:: :py:class:`.HDF5IteratorOneHot`
.. |HDF5IteratorAutoencoder| replace:: :py:class:`.HDF5IteratorAutoencoder`



















