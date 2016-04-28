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

    (X_train, y_train), (X_test, y_test), nclass = MNIST().load_data()
    train_set = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(1, 28, 28))
    valid_set = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

CIFAR10
-------

`CIFAR10 <http://www.cs.toronto.edu/~kriz/cifar.html>`__, is a dataset
consisting of 50,000 training samples and 10,000 test samples. There are
10 categories and each sample is a 32x32 RGB color image.

CIFAR10 can be fetched in the following manner:

.. code-block:: python

    from neon.data import CIFAR10
    (X_train, y_train), (X_test, y_test), nclass = CIFAR10().load_data()
    train = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(3, 32, 32))
    test = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(3, 32, 32))

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

    # download dataset
    from neon.data import Flickr8k
    data_path = Flickr8k().load_data()  # Other setnames are Flickr30k and Coco

    # load data
    from neon.data import ImageCaption
    train_set = ImageCaption(path=data_path, max_images=-1)

Text
----

For existing datasets (e.g. Penn Treebank, Hutter Prize, and
Shakespeare), we have object classes for loading, and sometimes
pre-processing, the data. The online source are stored in the
``__init__`` method. Some datasets (such as Penn Treebank) also accept a
tokenizer (string) to parse the file. These datasets use ``gen_iterators()``
to return a iterator (:py:class:`Text<neon.data.text.Text>`)

.. code-block:: python

    from neon.data import PTB

    # download Penn Treebank and parse at the word level
    ptb = PTB(tokenizer="str.split")
    ptb.load_data()

    # create dict of iterators
    # iters['train'] is an iterator (neon.data.Text) for the training data
    # iters['test'] is an iterator for the testing data
    # iters['valid'] is an iterator for the validation data
    iters = ptb.gen_iterators()

ImageNet
--------

The raw images need to be downloaded from ILSVRC as a tar file. Because
the data is too large to fit in memory, the data must be loaded in
batches (called "macrobatches", see :doc:`Loading data <loading_data>`
). We first write the macrobatches with the
``batch_writer.py`` script. ``data_dir`` is where the
processed batches will be stored, and ``image_dir`` is where the
original tar files are saved.

.. code-block:: bash

    python neon/data/batch_writer.py  --data_dir /usr/local/data/tmp \
                                      --image_dir /usr/local/data/I1K/imagenet_orig \
                                      --set_type i1k

We then create the ImageNet dataset object and get the training data
iterator, which is of the :py:class:`.ImageLoader` class. :py:class:`.ImageLoader` allows
for fast loading and feeding of macrobatches to the model.

.. code-block:: python

    from neon.data import I1K

    # create the I1K object
    i1k = I1K(data_dir = args.data_dir, inner_size=224, subset_pct=100)

    # fetch a dict of iterators
    # iter['train'] is an iterator (neon.data.ImageLoader) for the training data
    # iter['val'] is an iterator for the validation data
    iters = i1k.gen_iterators()

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

    # get the bAbI data
    babi = BABI(path='.', task='qa15_basic-deduction', subset='en')

    # create a QA iterator
    train_set = QA(*babi.train)
    valid_set = QA(*babi.test)
