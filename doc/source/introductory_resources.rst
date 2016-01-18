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

Using the ImageLoader module
----------------------------

The :py:class:`ImageLoader<neon.data.imageloader.ImageLoader>` module was created to provide a way
to feed images from disk to neon with minimal latency.  The module takes advantage of the high
compressibility of images to conserve diskspace and disk to host memory IO.  For small datasets,
images can stored in their decoded state in host memory, but for large datasets, that approach is
not practical.  ImageLoader uses a multithreaded library to hide the latency of decoding images,
applying augmentation and/or transformations, and transferring the resulting outputs to device
memory (if necessary).

Writing macrobatches
''''''''''''''''''''

In order to use the ImageLoader module, the images of the dataset must be packaged into flat binary
files which we refer to as "macrobatches".  Macrobatches are simply archive files that package
together many data files (jpegs) to take advantage of disk locality.  The container for these
macrobatches is designed to be compatible with the `GNU tool cpio
<http://www.gnu.org/software/cpio/manual/cpio.html>`_.  The ``neon.util.batch_writer.py``
illustrates how to generate macrobatch datasets from three types of raw image sources:

1.  General Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This option presumes that your data is provided as a directory of images, each with the same
extension, ``<ext>``, that are organized in a hierarchy as follows:

::

    image_dir/
    image_dir/<category_name1>/<img_1_1>.<ext>
    image_dir/<category_name1>/<img_1_2>.<ext>
    ...
    image_dir/<category_name1>/<img_1_N1>.<ext>
    image_dir/<category_name2>/<img_2_1>.<ext>
    image_dir/<category_name2>/<img_2_2>.<ext>
    ...
    image_dir/<category_name2>/<img_2_N2>.<ext>
    ...
    ...
    image_dir/<category_nameM>/<img_M_1>.<ext>
    image_dir/<category_nameM>/<img_M_2>.<ext>
    ...
    image_dir/<category_nameM>/<img_M_NM>.<ext>

With this organization, there are ``M`` categories, with each category containing a variable number
of images.  The ``batch_writer.py`` utility will partition the data into train and validation sets,
write out csv files pairing the file location to an integer corresponding to the category label
index, and then copy the files into macrobatch files, optionally resizing the images along the way.
The following command illustrates how to invoke the ``batch_writer.py`` command for this type of
scenario.

.. code-block:: bash

    python neon/data/batch_writer.py  --data_dir /usr/local/data/macrobatch_out \
                                      --image_dir /usr/local/data/raw_images \
                                      --set_type directory \
                                      --target_size 256 \
                                      --macro_size 5000 \
                                      --file_pattern "*.jpg"

In this command, ``/usr/local/data/macrobatch_out`` is the directory to which the macrobatches will
be written, ``/usr/local/data/raw_images`` is where the image subdirectories reside, ``directory``
indicates that we are using the general directory structure mode of ``batch_writer``,
``target_size`` indicates the pixel dimension that larger images will be scaled down to (e.g. a 512
x 768 image will be rescaled to 256 x 384, but a 128 x 128 image will be untouched), ``macro_size``
indicates the maximum number of images that will be packed per macrobatch file, and
``file_pattern`` indicates the extension of the files in the subdirectories that will be included
in the batch writing process.

2.  ILSVRC ImageNet 1K tar files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Imagenet task is recognition task is described on the `ILSVRC website <http://www.image-
net.org/challenges/LSVRC/>`_.  The 1.3M training images, 50K validation images, and development kit
are provided as TAR archives.  Because the images are organized in a way that makes them unamenable
to the generalized directory structure in the previous scenario, we provide some special handling
to properly unpack the TARs and correctly associate the category names (synsets) to the integer
labels using the provided development kit.  Imagenet macrobatches can be created using the
following command:

.. code-block:: bash

    python neon/data/batch_writer.py  --data_dir /usr/local/data/macrobatch_out \
                                      --image_dir /usr/local/data/I1K_tar_location \
                                      --set_type i1k

In this command, the ``file_pattern``, ``target_size``, and ``macro_size`` arguments are handled as
defaults, and the only difference from the previous example are the ``set_type`` argument and the
``image_dir`` argument.  The ``image_dir`` should contain the three TAR files that are provided by
ILSVRC:

- ILSVRC2012_img_train.tar
- ILSVRC2012_img_val.tar
- ILSVRC2012_devkit_t12.tar.gz

Furthermore, one should ensure that the disk where ``data_dir`` is pointing has sufficient space to
hold the resulting macrobatches as well as space for the unpacked images (these can be deleted once
the macrobatches have been written).  Since the dataset is relatively large, an SSD can greatly
speed up the batch_writing process.

3.  CIFAR10 python pickled numpy pixel buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CIFAR10 dataset is provided as a pickled set of numpy arrays containing the uncompressed pixel
buffers of each image.  This dataset is small enough to easily fit in host memory.  However, using
the :py:class:`DataIterator<neon.data.dataiterator.DataIterator>` module is limited in that it does
not allow for random flipping, cropping, or shuffling.  We therefore added the ability to write out
CIFAR10 data as macrobatches so that the ImageLoader module could be used with this command:

.. code-block:: bash

    python neon/data/batch_writer.py  --data_dir /usr/local/data/macrobatch_out \
                                      --set_type cifar10 \
                                      --target_size 40

CIFAR10 images are 32x32, so if the ``target_size`` argument is omitted, then the images will be
written out at this size.  However, in many scenarios, one might wish to zero-pad the images so that
random cropping can be done without further reducing the feature map size.  Setting ``target_size``
to the desired padded image size instructs the batch writer to center the image in the target
feature map size and pad the border with the means of that image along each channel.  See `numpy.pad
<http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.pad.html>`_ for more details.
Because CIFAR images are so small, we have found that JPEG encoding of the images can negatively
impact the accuracy of classification algorithms, so in this case we use lossless PNG encoding as
the format to dump into the macrobatches.

Metafile
~~~~~~~~

The macrobatch dataset directory must contain a metafile for instructing the dataset loader how
many batches to consider.  The metafile is a simply a plain text file with a different attribute
for each line.  As an example, the metafile for the Imagenet dataset would look like this:

::

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

- ``train_nrec`` and ``val_nrec`` are the number of records for the train and validation sets,
  respectively
- ``train_start`` and ``val_start`` are the index of the macrobatch where each of those partitions
  start (e.g. ``macrobatch_0`` through ``macrobatch_256`` contain training images, while
  ``macrobatch_257`` on contain validation images)
- ``nclass`` is the number of distinct categories
- ``item_max_size`` is the size (in bytes) of the largest encoded jpeg file
- ``label_size`` is the size (in bytes) of the label format (in this case it's an integer, which is
  4 bytes)
- ``R_mean``, ``G_mean``, ``B_mean`` are the pixel means for the red, green, and blue channels,
  respectively.

Command Line Macrobatch creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A final way that macrobatches can be created using the ``cpio`` command line utility is illustrated
here:

.. code-block:: bash

    # A list of image files
    file_list=../file_list

    # corresponding list of labels
    label_list=../label_list

    # Dump a blank header
    perl -e 'print pack("a4I2a8I5Q3", "MACR", 1, 1, "imgclass")' > cpiohdr

    # begin the macrobatch
    echo cpiohdr | cpio -o --quiet > macrobatch.cpio

    # now start adding in image, label pairs
    ndata=0; totalsz=0; maxsz=0;
    for i in $(paste -d ',' $file_list $label_list); do
        imgfile=cpiodtm$ndata
        lblfile=cpiotgt$((ndata++))

        # copy the image (could also be a resizing filter)
        cp `echo $i | cut -d ',' -f 1` $imgfile
        echo $imgfile | cpio -A -o --quiet -F macrobatch.cpio

        # Now we need to make a temporary file for the label
        label=`echo $i | cut -d ',' -f 2`
        perl -e 'print pack("I", $ARGV[0])' $label > $lblfile
        echo $lblfile | cpio -A -o --quiet -F macrobatch.cpio

        imgsz=`stat --printf "%s" $imgfile`
        totalsz=$((totalsz+imgsz))
        if (( "$maxsz" < "$imgsz" )); then
            maxsz=$imgsz
        fi

        rm $lblfile $imgfile
    done

    # Update the header with appropriate size information
    perl -e 'print pack('I5', @ARGV)' $ndata $maxsz 4 $totalsz $((ndata*4)) | dd conv=notrunc of=macrobatch.cpio bs=1 seek=54

Loading Images
''''''''''''''

Once macrobatches have been created, the :py:class:`ImageLoader<neon.data.imageloader.ImageLoader>`
module can be instantiated to load images in a pipelined fashion while applying several types of
image transformations or augmentations.  One can look at the documentation for the arguments to the
:py:class:`ImageLoader<neon.data.imageloader.ImageLoader>` constructor for an explanation of each of
the parameters, but here we give some example invocations.  In all of the examples that follow, each
batch of images on disk can be any size and any aspect ratio, depending on how they were written out
to the macrobatch.  The ImageLoader takes care of rescaling the image and cropping a region of
interest.  In general, the ``do_transforms`` flag is used to switch on or off random transformations
en masse, so even if arguments are provided that indicate some range over which random values can be
picked, setting ``do_transforms`` to ``False`` will override those ranges.

In the examples below, we will assume that the macrobatches are in the ``/usr/local/data/batches``
directory for simplicity.  Note that the default value of ``do_transforms`` is ``True``, but we
provide it explicitly in the examples for clarity.

1. Scale the original image so that its short side is ``256`` pixels, randomly perform a horizontal
reflection, then crop a randomly selected ``100x100`` region from the result.

.. code-block:: python

    train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                            inner_size=100,
                            scale_range=256,
                            do_transforms=True)


2. Scale the original image so that its short side is ``256`` pixels, *do not* perform a horizontal
reflection, then crop the *center* ``100x100`` region from the result.

.. code-block:: python

    train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                            inner_size=100,
                            scale_range=256,
                            do_transforms=False)  # Overrides flipping/random cropping

3. Randomly scale the original image so that the short side is between ``100`` and ``200`` pixels,
randomly perform a horizontal reflection, then crop a randomly selected ``80x80`` region from the
result.

.. code-block:: python

    train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                            inner_size=80,
                            scale_range=(100, 200),
                            do_transforms=True)


4. Same as 3, but also randomly adjust the contrast to between ``75%`` and ``125%`` of the original
image.

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

6. Scale the original image so that the short side is ``100`` pixels, *do not* perform a horizontal
   reflection, *do not* adjust contrast, crop the *center* ``80x80`` region from the resulting
   image, and *do not* shuffle the order in which images are returned

.. code-block:: python

    train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                            inner_size=80,
                            scale_range=(100, 200),
                            contrast_range=(75, 125),
                            shuffle=True,
                            do_transforms=False)  # Overrides all randomness

7. Force the original image to be scaled so that the entire image fits into a ``100x100`` region,
   regardless of aspect ratio distortion, and perform random horizontal reflections.

.. code-block:: python

    train_set = ImageLoader(repo_dir='/usr/local/data/batches', set_name='train',
                            inner_size=100,
                            scale_range=0,  # Force scaling to match inner_size
                            do_transforms=True)

8. Typical setup for imagenet training.  Randomly select a ``224x224`` crop of an image randomly
   scaled so that its shortest side is between ``256`` and ``480``, randomly flipped, shuffled.
   For testing, scale to various scales and take the whole image so that convolutional inference
   can be performed.

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

Adding a Custom Dataset Walk-through: bAbI
------------------------------------------

Neon provides many tools to facilitate adding new datasets. In this tutorial, we will walk-through
adding `Facebook’s bAbI dataset <https://research.facebook.com/researchers/1543934539189348>`_.
There are three main pieces:

1. File handler for the dataset
2. Pre-processing the data
3. Packaging the data in an iterator

We will walk-through each in turn.

1. File handler for the dataset
'''''''''''''''''''''''''''''''

The bAbI dataset comprises 20 tasks. Each task has both a small and large training and test split,
and comes in English, Hindi, and shuffled characters. The first step is to tell Neon where to find
the dataset, the size of the dataset, and a function handler ``load_babi``.

.. code-block:: python

    dataset_meta = {
       'babi': {
           'size': 11745123,
           'file': 'tasks_1-20_v1-2.tar.gz',
           'url': 'http://www.thespermwhale.com/jaseweston/babi',
           'func': load_babi
       }
    }

With this specification, we can now call the function ``load_dataset`` and call bAbI by name, in
addition to calling ``load_babi`` directly. Since bAbI comes with a number of tasks, different
languages, and different splits, the role of the ``load_babi`` function is to extract the correct
split. One useful helper function that Neon provides you is ``fetch_dataset`` which downloads data
from a URL in chunks.

.. code-block:: python

    def fetch_dataset(url, sourcefile, destfile, totalsz):
       """
       Download the file specified by the given URL.

       Args:
           url (str): Base URL of the file to be downloaded.
           sourcefile (str): Name of the source file.
           destfile (str): Path to the destination.
           totalsz (int): Size of the file to be downloaded.
       """

The ``load_babi`` function downloads the dataset, extracts the specified training and test splits,
and returns file handlers to the splits.

.. code-block:: python

    def load_babi(path=".", task='qa1_single-supporting-fact', subset='en'):
        """
        Fetch the Facebook bAbI dataset and load it to memory.

        Args:
            path (str, optional): Local directory in which to cache the raw
                                  dataset.  Defaults to current directory.
            task (str, optional): bAbI task to load
            subset (str, optional): Data comes in English, Hindi, or Shuffled
                                    characters. Options are 'en', 'hn', and
                                    'shuffled' for 1000 training and test
                                    examples or 'en-10k', 'hn-10k', and
                                    'shuffled-10k' for 10000 examples.

        Returns:
            tuple: training and test files are returned
        """
        babi = dataset_meta['babi']
        workdir, filepath = _valid_path_append(path, '', babi['file'])
        if not os.path.exists(filepath):
            fetch_dataset(babi['url'], babi['file'], filepath, babi['size'])

        babi_dir_name = babi['file'].split('.')[0]
        task = babi_dir_name + '/' + subset + '/' + task + '_{}.txt'
        train_file = os.path.join(workdir, task.format('train'))
        test_file = os.path.join(workdir, task.format('test'))

        if os.path.exists(train_file) is False or os.path.exists(test_file):
            with tarfile.open(filepath, 'r:gz') as f:
                f.extractall(workdir)

        return train_file, test_file

2. Pre-processing the data
''''''''''''''''''''''''''

bAbI is a question answering (QA) dataset. The examples consist of stories, questions, and answers.
Stories paint a sequence of actions and events, questions ask a basic fact or logical conclusion
based on the story, and answers are the targets of the example. In bAbI, these stories, questions,
and answers come in an interleaved format.

::

    1 John travelled to the hallway.
    2 Mary journeyed to the bathroom.
    3 Where is John?    hallway 1
    4 Daniel went back to the bathroom.
    5 John moved to the bedroom.
    6 Where is Mary?    bathroom    2
    7 John went to the hallway.
    8 Sandra journeyed to the kitchen.
    9 Where is Sandra?  kitchen 8
    10 Sandra travelled to the hallway.
    11 John went to the garden.
    12 Where is Sandra?     hallway 10
    13 Sandra went back to the bathroom.
    14 Sandra moved to the kitchen.
    15 Where is Sandra?     kitchen 14

Every line has a leftmost number indicating the position within the story. When this number restarts, it indicates a new story. Lines with a rightmost number consist of a question, an answer, and the number of the line in the story which provides evidence for the answer. An example of a story, question, and answer are:

::

    Story:
    John travelled to the hallway.
    Mary journeyed to the bathroom.
    Daniel went back to the bathroom.
    John moved to the bedroom.

    Question:
    Where is Mary?

    Answer:
    bathroom

We wrote a ``BABI`` class in ``neon/data/questionanswer.py`` to take care of this pre-processing.

.. code-block:: python

    class BABI(NervanaObject):
        """
        This class loads in the Facebook bAbI dataset and vectorizes them into stories,
        questions, and answers as described in:
        "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
        http://arxiv.org/abs/1502.05698

        """
        def __init__(self, path='.', task='qa1_single-supporting-fact', subset='en'):
            """
            Load bAbI dataset and extract text and read the stories
            For a particular task, the class will read both train and test files
            and combine the vocabulary.

            Args:
                path (str): Directory to store the dataset
                task (str): a particular task to solve (all bAbI tasks are train
                            and tested separately)
                subset (str): subset of the dataset to use:
                              {en, en-10k, shuffled, hn, hn-10k, shuffled-10k}
            """

An important additional pre-processing step we perform is tokenizing the text and vectorizing the
tokens. Rather than working with raw text, we create a dictionary that maps every token in the
vocabulary to an index. The vocabulary comprises every unique token in both the training and test
sets. A sentence is then a vector of integer indices. This step is specific to your dataset and you
should do any desired pre-processing and transformations.

3. Packaging the data in an iterator
''''''''''''''''''''''''''''''''''''

Neon requires a python iterator to traverse through datasets for training and evaluation. Luckily,
Neon comes with a QA class, which is a general purpose iterator for QA datasets such as bAbI. It
takes as input vectorized stories, queries, and answers. On each iteration it yields a minibatch of
inputs and outputs. We simply need to load our pre-processed bAbI data into a QA instance for
neural network use.

.. code-block:: python

    class QA(NervanaObject):
        """
        A general QA container to take Q&A dataset, which has already been
        vectorized and create a data iterator to feed data to training
        """
        def __init__(self, story, query, answer):

Neon comes with tools for other formats as well such as text, images, videos, among others.

Conclusion
''''''''''

Neon provides many tools for easily integrating custom datasets. To see bAbI dataset used in
action, please visit `here <https://gist.github.com/SNagappan/a7be6ce6e75c36c7406e>`_.


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
