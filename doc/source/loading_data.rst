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

* If your data is too large, use macrobatching, a specialized loader that loads macrobatches of data into memory, and then splits the macrobatches into minibatches to feed the model.

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
    from neon.data import ArrayIterator, load_mnist

    # load the MNIST data
    (X_train, y_train), (X_test, y_test), nclass = load_mnist()

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


Macrobatching
-------------

If your data is too large to load directly into memory, use a
macrobatching approach. In macrobatching, the data is loaded in smaller
batches, then split further into minibatches to feed the model.
Currently, neon only supports macrobatching with image datasets using
the :py:class:`.ImageLoader` class. However, future releases will
include a generic data loader for macrobatching of all data types (text,
video, images, etc.).

:py:class:`.ImageLoader` was created to provide a way to feed images
from disk to neon with minimal latency. The module takes advantage of
the high compressibility of images to conserve disk space and disk to
host memory IO. ImageLoader uses a multithreaded library to hide the
latency of decoding images, applying augmentation and/or
transformations, and transferring the resulting outputs to device memory
(if necessary). The module also adds optional functionality for applying
transformations (scale, flip, and rotation)

Writing macrobatches
~~~~~~~~~~~~~~~~~~~~

In order to use the :py:class:`.ImageLoader`, the images of the dataset
must be packaged into flat binary files which we refer to as
“macrobatches”. Macrobatches are simply archive files that package
together many data files (jpegs) to take advantage of disk locality. The
container for these macrobatches is designed to be compatible with the
GNU tool ``cpio``.

To generate macrobatches, use the ``neon.util.batch_writer.py`` script.
Macrobatch datasets can be generated with this script from four types of
raw image sources:

1. General directory structure

2. CSV Manifest file

3. ImageNet 1K tar files

4. CIFAR-10 numpy arrays (pickled)

General Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option presumes that your data is provided as a directory of
images, each with the same extension, ``<ext>``, that are organized in a
hierarchy as follows:

.. figure:: assets/image_loader_data_structure_v2.jpg

In this organization, there are :math:`K=4` categories, with each category containing a variable number of images.
The ``batch_writer.py`` utility will partition the data into train and
validation sets, and then write out CSV files mapping the file location
to an integer corresponding to the category label index. Then, the files
are copied into macrobatch files, with optional arguments for resizing
the images.

The following command illustrates how to invoke ``batch_writer.py`` in
this scenario:

.. code-block:: python

    python neon/data/batch_writer.py  --data_dir /usr/local/data/macrobatch_out \
                                      --image_dir /usr/local/data/raw_images \
                                      --set_type directory \
                                      --target_size 256 \
                                      --macro_size 5000 \
                                      --file_pattern "*.jpg"

In this command, the images will be loaded from
*/usr/local/data/raw_images* and the macrobatches written to
*/usr/local/data/macrobatch_out*. Images that are larger than the
``target_size=256`` will be scaled down (e.g. a 512x768 image will be
rescaled to 256x384, but a 128x128 will be untouched). Each macrobatch
will have at most ``macro_size=5000`` images.

CSV Manifest file
^^^^^^^^^^^^^^^^^

This user can provide training and validation *csv.gz* files, each
containing files and label indexes. The two required files are
*train_file.csv.gz* and *val_file.csv.gz*. They should each contain
one record per line, and be formatted as:

.. code-block:: bash

    <path_to_image_1>.<ext>,<label_1>
    <path_to_image_2>.<ext>,<label_2>
    ...
    <path_to_image_N>.<ext>,<label_N>

For example, the above images could be provided as:

.. code-block:: bash

    /image_dir/faces/naveen_rao.jpg, 0
    /image_dir/faces/arjun_bansal.jpg, 0
    /image_dir/faces/amir_khosrowshahi.jpg, 0
    /image_dir/fruits/apple.jpg, 1
    /image_dir/fruits/pear.jpg, 1
    /image_dir/animals/lion.jpg, 2
    /image_dir/animals/tiger.jpg, 2
    ...
    /image_dir/vehicles/toyota.jpg, 3

Note that the train file is not shuffled during batch creation, so the
user should take care to shuffle the lines when creating
*train_file.csv.gz*.

If the specified paths are not absolute (i.e. starts with ‘/’), then the
path will be assumed to be relative to the location of the csv file.

The batch writer can then be invoked by calling:

.. code-block:: bash

    python neon/data/batch_writer.py  --data_dir /usr/local/data/macrobatch_out \
                                      --image_dir /location/of/csv_files \
                                      --set_type csv

ImageNet 1K tar files
^^^^^^^^^^^^^^^^^^^^^

The ImageNet task is recognition task is described on the
`ILSVRC <http://www.image-net.org/challenges/LSVRC/>`__ website. The
1.3M training images, 50K validation images, and development kit are
provided as TAR archives. Because the images are organized in a way that
makes them unamenable to the generalized directory structure described
above, we provide some special handling to properly unpack the TARs and
correctly associate the category names to the integer labels. ImageNet
macrobatches can be created using the following command:

.. code-block:: bash

    python neon/data/batch_writer.py  --data_dir /usr/local/data/macrobatch_out \
                                      --image_dir /usr/local/data/I1K_tar_location \
                                      --set_type i1k

In this command, the ``file_pattern``, ``target_size``, and
``macro_size`` arguments are handled as defaults. The only difference
are the ``set_type`` argument and the ``image_dir`` argument. The
``image_dir`` should contain the three TAR files that are provided by
ILSVRC:

.. code-block:: bash

    ILSVRC2012_img_train.tar
    ILSVRC2012_img_val.tar
    ILSVRC2012_devkit_t12.tar.gz

Ensure that the disk where ``data_dir`` is located has sufficient space
to hold the resulting macrobatches as well as space for the unpacked
images (these can be deleted once the macrobatches have been written).
Since the dataset is relatively large, an SSD can greatly speed up the
batch writing process.

CIFAR-10 pickled numpy arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CIFAR10 dataset is provided as a pickled set of numpy arrays
containing the uncompressed pixel buffers of each image. This dataset is
small enough to easily fit in host memory. However, the
|ArrayIterator| module does not allow for random flipping, cropping,
or shuffling. We therefore added the ability to write out CIFAR10 data
as macrobatches to work with :py:class:`.ImageLoader` :

.. code-block:: bash

    python neon/data/batch_writer.py  --data_dir /usr/local/data/macrobatch_out \
                                      --set_type cifar10 \
                                      --target_size 40

CIFAR10 images are 32x32, so if the ``target_size`` argument is omitted,
then the images will be written out as 32x32. However, in many
scenarios, one might wish to zero-pad the images so that random cropping
can be done without further reducing the feature map size. Setting
``target_size`` to the desired padded image size instructs the batch
writer to center the image in the target feature map size and pad the
border with the means of that image along each channel. See
``numpy.pad`` for more details.

Because CIFAR images are so small, we have found that JPEG encoding of
the images can negatively impact the accuracy of classification
algorithms, so in this case we use lossless PNG encoding as the format
to dump into the macrobatches.

Metafile
~~~~~~~~

A required metafile named ``macrobatch_meta`` is automatically generated
by ``batch_writer.py``. This file instructs :py:class:`.ImageLoader`  on how many
batches to consider. The metafile is a plain text file with a different
attribute for each line. As an example, the metafile for the ImageNet
dataset would look like this:

.. code-block:: bash

    train_start 0
    train_nrec 1281167
    val_start 257
    val_nrec 50000
    nclass 1000
    item_max_size 1845130
    label_size 4
    R_mean 104.412277
    G_mean 119.213318
    B_mean 126.806091

Each of these attributes is described below:

 * ``train_nrec`` and ``val_nrec`` are the number of records for the train and validation sets, respectively. ``train_start`` and ``val_start`` are the index of the macrobatch where each of those partitions start (e.g. ``macrobatch_0`` through ``macrobatch_256`` contain training images, while ``macrobatch_257`` onwards contain validation images)
 * ``nclass`` is the number of distinct categories
 * ``item_max_size`` is the size (in bytes) of the largest encoded jpeg file
 * ``label_size`` is the size (in bytes) of the label format (for an integer, 4 bytes)
 * ``R_mean``, ``G_mean``, ``B_mean`` are the pixel means for the red, green, and blue channels, respectively.

Loading Images from macrobatches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once macrobatches have been created, the :py:class:`.ImageLoader` module, which
is an iterable, can be instantiated to load images in a pipelined
fashion while applying several types of image transformations or
augmentations.

See the documentation of arguments to the ImageLoader constructor for
explanation of the possible configurations. Below are several example
invocations.

Examples
^^^^^^^^

In the below examples, each macrobatch of images can be any size and any
aspect ratio. The ImageLoader takes care of rescaling the image and
cropping a region of interest. In general, the ``do_transforms`` flag is
used to switch on or off random transformations en masse, so even if
arguments are provided that indicate some range over which random values
should be picked, setting ``do_transforms`` to ``False`` will override
those ranges.

We will assume that the macrobatches are in the */usr/local/data/batches*
directory for simplicity. Note that the default value of
``do_transforms`` is ``True``, but we provide it explicitly in the
examples for clarity.

1. Scale the original image so that its short side is 256 pixels,
   randomly perform a horizontal reflection, then crop a randomly
   selected 100x100 region from the result.

   .. code-block:: python

       train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                           inner_size=100,
                           scale_range=256,
                           do_transforms=True)

2. Scale the original image so that its short side is 256 pixels, do not
   perform a horizontal reflection, then crop the center 100x100 region
   from the result.

   .. code-block:: python

       train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                           inner_size=100,
                           scale_range=256,
                           do_transforms=False)  # Overrides flipping/random cropping

3. Randomly scale the original image so that the short side is between
   100 and 200 pixels, randomly perform a horizontal reflection, then
   crop a randomly selected 80x80 region from the result.

   .. code-block:: python

       train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                           inner_size=80,
                           scale_range=(100, 200),
                           do_transforms=True)

4. Same as 3, but also randomly adjust the contrast to between 75% and
   125% of the original image.

   .. code-block:: python

       train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                           inner_size=80,
                           scale_range=(100, 200),
                           contrast_range=(75, 125),
                           do_transforms=True)

5. Same as 4, but also shuffle the order of images returned.

   .. code-block:: python

       train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                           inner_size=80,
                           scale_range=(100, 200),
                           contrast_range=(75, 125),
                           shuffle=True,
                           do_transforms=True)

6. Same as 5, but also randomly stretch the image horizontally or
   vertically (direction is also randomly determined) by a factor
   between 1 and 1.25.

   .. code-block:: python

       train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                           inner_size=80,
                           scale_range=(100, 200),
                           contrast_range=(75, 125),
                           aspect_ratio=125,
                           shuffle=True,
                           do_transforms=True)

7. Scale the original image so that the short side is 100 pixels, do not
   perform a horizontal reflection, do not adjust contrast, crop the
   center 80x80 region from the resulting image, and do not shuffle the
   order in which images are returned

   .. code-block:: python

       train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                           inner_size=80,
                           scale_range=(100, 200),
                           contrast_range=(75, 125),
                           shuffle=True,
                           do_transforms=False)  # Overrides all randomness

8. Force the original image to be scaled so that the entire image fits
   into a 100x100 region, regardless of aspect ratio distortion, and
   perform random horizontal reflections.

   .. code-block:: python

       train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                           inner_size=100,
                           scale_range=0,  # Force scaling to match inner_size
                           do_transforms=True)

Typical setup for ImageNet
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a typical setup for ImageNet training. Randomly select a 224x224
crop of an image randomly scaled so that its shortest side is between
256 and 480, randomly flipped, shuffled. For testing, scale to various
scales and take the whole image so that convolutional inference can be
performed.

.. code-block:: python

    train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                            inner_size=224,
                            scale_range=(256, 480),  # Force scaling to match inner_size
                            shuffle=True,
                            do_transforms=True)
    test256 = ImageLoader(repo_dir='/usr/local/data/batches', set_name='validation',
                          inner_size=256,
                          scale_range=0,  # Force scaling to match inner_size
                          do_transforms=False)
    test384 = ImageLoader(repo_dir='/usr/local/data/batches', set_name='validation',
                          inner_size=384,
                          scale_range=0,  # Force scaling to match inner_size
                          do_transforms=False)





.. |ArrayIterator| replace:: :py:class:`.ArrayIterator`
