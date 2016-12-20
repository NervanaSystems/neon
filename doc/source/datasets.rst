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

Datasets
========

:py:class:`~.Dataset` class is a base class for commonly-used
datasets. We recommend creating an object class for your dataset that
handles the loading and preprocessing of the data. Datasets should
implement :py:meth:`~.Dataset.gen_iterators`, which returns a dictionary data
iterator used for training and evaluation (see :doc:`Loading
data <loading_data>`).

Neon provides dataset objects for handling many stock datasets.

MNIST
-----

`MNIST <http://yann.lecun.com/exdb/mnist/>`__, is a dataset of
handwritten digits, consisting of 60,000 training samples and 10,000
test samples. Each image is 28x28 greyscale pixels.

MNIST can be fetched in the following manner:

.. code-block:: python

    from neon.data import MNIST

    mnist = MNIST(path='path/to/save/downloadeddata/')
    train_set = mnist.train_iter
    valid_set = mnist.valid_iter

The ``path`` argument desigates the directory to store
the downloaded dataset.  If the dataset already exists in that directory,
download will be skipped.  The default data path will be used if ``path``
is not provided.

CIFAR10
-------

`CIFAR10 <http://www.cs.toronto.edu/~kriz/cifar.html>`__, is a dataset
consisting of 50,000 training samples and 10,000 test samples. There are
10 categories and each sample is a 32x32 RGB color image.

CIFAR10 can be fetched in the following manner:

.. code-block:: python

    from neon.data import CIFAR10

    cifar10 = CIFAR10()
    train = cifar10.train_iter
    test = cifar10.valid_iter

ImageCaption
------------

This dataset uses precomputed CNN image features and caption sentences.
It works with the flickr8k, flickr30k, and COCO datasets and uses the
VGG image features and sentences from
http://cs.stanford.edu/people/karpathy/deepimagesent/ which have been
converted to python .pkl format. These datasets have 5 reference
sentences per image. For each sentence, the dataset converts each word
to its 1-hot representation so that each input batch of sentences is of
dimension (vocab_size, max_sentence_length * batch_size).

The image caption data can be fetched in the following manner:

.. code-block:: python

    from neon.data import Flickr8k

    # download dataset
    flickr8k = Flickr8k()  # Other set names are Flickr30k and Coco
    train_set = flickr8k.train_iter

Text
----

For existing datasets (e.g. Penn Treebank, Hutter Prize, and
Shakespeare), we have object classes for loading, and sometimes
pre-processing, the data. The online source are stored in the
``__init__`` method. Some datasets (such as Penn Treebank) also accept a
tokenizer (string) to parse the file. The tokenizer is a string which
matches the name of one of the tokenizers functions that are included in
the class definition.  For example, the method ``newline_tokenizer`` in
the ``PTB`` class replaces all newline characters (i.e. ``\n``) with
the string ``<eos>`` and splits the string.  These datasets use ``gen_iterators()``
to return a iterator (:py:class:`Text<neon.data.text.Text>`)

.. code-block:: python

    from neon.data import PTB

    # download Penn Treebank and parse at the word level
    ptb = PTB(time_steps, tokenizer="newline_tokenizer")
    train_set = ptb.train_iter

ImageNet
--------

The raw images need to be downloaded from ILSVRC as a tar file. Because
the data is too large to fit in memory, the data must be loaded from disk to host,
and then from host to device (if using a non-cpu backend), while being augmented
appropriately.  For this type of data, we use the `aeon` dataloader which is
described in :doc:`Loading data <loading_data>`.  Example of how to use `aeon`
with ImageNet in particular are shown in ``examples/imagenet``, with the data
preparation procedure (extracting from tar, resizing the images, generating manifest
files listing images and labels) encapsulated in the script ``examples/imagenet/data.py``.


QA and bAbI
-----------

A :py:class:`.bAbI` dataset object can be created by specifying which task and which
subset (20 tasks and 4 subsets in bAbI) to retrieve. The object will use
built-in metadata to get bAbI data from online sources, save and unzip
the files for that task locally, and then vectorize the
story-question-answer data. The training and test files are both needed
to build a vocabulary set.

A general question and answering container can take the
story-question-answer data from a bAbI data object and create a data
iterator for training.

.. code-block:: python

    from neon.data import BABI
    from neon.data import QA

    # get the bAbI data
    babi = BABI(path='.', task='qa15_basic-deduction', subset='en')

    # create a QA iterator
    train_set = QA(*babi.train)
    valid_set = QA(*babi.test)

Low level dataset operations
----------------------------

Some applications require access to the underlying data to generate more
complex data iterators. This can be done by using the ``load_data``
method of the DataSet class and its subclasses.  The method returns
the data arrays which are used to generate the data iterators. For
example, the code below shows how to generate a data iterator to
train an autoencoder on the MNIST dataset:

.. code-block:: python

    from neon.data import MNIST
    from neon.data import ArrayIterator

    mnist = MNIST()
    # get the raw data arrays, both train set and validation set
    (X_train, y_train), (X_test, y_test), nclass = mnist.load_data()

    # generate and ArrayIterator with no target data
    # this will return the image itself as the target
    train = ArrayIterator(X_train, lshape=(1, 28, 28))
