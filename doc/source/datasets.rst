.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.
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

Datasets represent the inputs to the model training or prediction process.  In
order to use a dataset in neon, it must conform to particular conventions so
that it can be correctly copied to the device being used.

We support loading datasets into host and device memory either all it once, or
in chunks (one macro batch at a time), useful if your dataset is too large to
fit in memory.  Note that macro batch loading is only supported in
:class:`neon.datasets.imageset.Imageset` currently.


Available Datasets
------------------

.. autosummary::
   :toctree: generated/

   neon.datasets.imageset.Imageset

   neon.datasets.cifar10.CIFAR10
   neon.datasets.cifar100.CIFAR100
   neon.datasets.housing.Housing
   neon.datasets.iris.Iris
   neon.datasets.mnist.MNIST
   neon.datasets.sparsenet.SPARSENET

   neon.datasets.mobydick.MOBYDICK

   neon.datasets.synthetic.UniformRandom
   neon.datasets.synthetic.ToyImages

.. _extending_dataset:


Basic YAML Parameters
---------------------

* ``repo_path``: Gives the base location in which datasets are stored on disk.
  Paths can contain environment variables, ``~`` for home directories, and be
  absolute or relative in nature.  Each type of dataset will be a subdirectory
  of this main repository path named according to the corresponding class name.
* ``sample_pct``: most datasets implement this, which if given a value < 100
  will be used to uniformly downsample dataset records to the specified
  percentage
* ``validation_pct``: Allows one to randomly select a portion of the original
  train set data points to construct a separate validation set.  Any existing
  validation set will be overwritten as a result.


Adding a new type of Dataset
----------------------------

Briefly, the process is as follows:

#. Subclass :class:`neon.datasets.dataset.Dataset` 
#. Write an implementation of :func:`neon.datasets.dataset.Dataset.load`.

* Datasets should have a single data point per row, and should either be in
  numpy ndarray format, or batched as such.
* Datasets are loaded and transformed by the appropriate backend via the
  :func:`neon.datasets.dataset.Dataset.format` call.
* If you have image data, have a look at the
  :class:`neon.datasets.imageset.Imageset` and instructions for working with it
  described in `Working with Imageset`.


To better understand the process, lets walk through adding a new dataset based
on the Kaggle sponsored
`National Data Science Bowl competition <https://www.kaggle.com/c/datasciencebowl/data>`_

This dataset consists of images of plankton split into a labelled 30k
example training set (organized into directories based on class), and a larger
unlabelled test dataset.  Given this setup, this data is well suited for
creating a new Imageset derived dataset, but for the moment let's set it up as
a general Dataset subclass to get a better feel for the process.

Let's begin by creating an initial template with some appropriate imports and
stubs for the functions we will need to implement:

.. code-block:: python
   :linenos:

    import cPickle
    import glob
    import logging
    import numpy as np
    import os
    from skimage import io, transform
    import zipfile

    from neon.datasets.dataset import Dataset

    class NDSB(Dataset):
        """
        Sets up an NDSB dataset.  See: https://www.kaggle.com/c/datasciencebowl

        Attributes:
            raw_train_url (str): where to download the source training set
            raw_test_url (str): where to download the source test set
        """
        raw_train_url = 'https://www.kaggle.com/c/datasciencebowl/download/train.zip'
        raw_test_url = 'https://www.kaggle.com/c/datasciencebowl/download/test.zip'

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def load(self):
            # TODO: we will fill this in

        def read_images(self, rootdir, leafdir, wildcard=''):
            # TODO: we will fill this in

So from this you see we create a new class called ``NDSB`` that is a child of
the ``Dataset`` class, create attributes that point at the URL's where we can
access the raw train and test source zipfiles, and define a (currently empty)
``load`` function that takes no parameters.

The goal of the ``load`` method is to populate an ``inputs`` and ``targets``
struct with appropriately formatted data.  For the NDSB data this means that
the images will be transformed and stored in ``inputs`` and we'll need to
extract directory names representing the required ground truth classes in
``targets``.  Because the NDSB data does not have a validation dataset nor
hold-out test set ground truth, we will end up populating 
``inputs['train'], inputs['test'], targets['train']``

One final note is that the URL's given won't work out of the box with Kaggle as
the user must first login to be presented with appropriate download links.  The
specified paths used in this example are simply for illustration.

Let's now start stepping through the implementation of the load function:

.. code-block:: python
   :linenos:

        def load(self):
            if self.inputs['train'] is not None:
                return
            if 'repo_path' not in self.__dict__:
                raise AttributeError('repo_path not specified in config')

            self.repo_path = os.path.expandvars(os.path.expanduser(self.repo_path))
            rootdir = os.path.join(self.repo_path, self.__class__.__name__)
            (self.inputs['train'], self.targets['train'], filetree,
             imgdims) = self.read_images(rootdir, 'train', '*')
            (self.inputs['test'], self.targets['test'], filetree,
             imgdims) = self.read_images(rootdir, 'test')
            self.format()

From the above, what we're doing is checking whether we even need to set
``inputs``, which we skip if already setup, then ensure that the YAML file
being used specifies the ``repo_path`` variable which we expand and setup the
``rootdir`` variable to point at a subdirectory called NDSB in the directory
specified by ``repo_path``.  Finally, we call the ``read_images`` helper
function (which we'll define below) to do the heavy lifting and actually load
the images which we assign to the appropriate ``inputs`` or ``targets`` item
based on whether we are parsing ``train`` or ``test`` data.  Finally we call
``self.format()`` which takes care of transforming and copying the training
inputs to device so the actual model training can proceed.

Finally lets look at the ``read_images`` implementation:

.. code-block:: python
   :linenos:

        def read_images(self, rootdir, leafdir, wildcard=''):
            logger.info('Reading images from %s', leafdir)
            repofile = os.path.join(rootdir, leafdir + '.zip')
            if not os.path.exists(repofile):
                if leafdir == 'train':
                   self.download_to_repo(self.raw_train_url, rootdir)
                else:
                   self.download_to_repo(self.raw_test_url, rootdir)
                infile = zipfile.ZipFile(repofile)
                infile.extractall(rootdir)
                infile.close()
            dirs = glob.glob(os.path.join(rootdir, leafdir, wildcard))
            dirs.sort()
            classind = 0
            imagecount = 0
            filetree = {}
            for dirname in dirs:
                filetree[classind] = []
                for walkresult in os.walk(dirname):
                    for filename in walkresult[2]:
                        if filename[-1] != 'g':
                            continue
                        filetree[classind].append(os.path.join(dirname, filename))
                        imagecount += 1
                filetree[classind].sort()
                classind += 1
            imagesize = self.nchannels * self.framesize
            nclasses = len(filetree)
            inputs = np.zeros((imagecount, imagesize), dtype=np.float32)
            targets = np.zeros((imagecount, 121), dtype=np.float32)
            imgdims = np.zeros(imagecount)
            imageind = 0
            for classind in range(nclasses):
                for filename in filetree[classind]:
                    img = io.imread(filename, as_grey=True)
                    imgdims[imageind] = np.mean(img.shape)
                    img = transform.resize(img, (self.image_width,
                                                 self.image_width))
                    img = np.float32(img)
                    # Invert the greyscale.
                    img = 1.0 - img
                    inputs[imageind][:self.framesize] = img.ravel()
                    inputs[imageind][self.framesize:] = self.whiten(filename, img).ravel()
                    targets[imageind, classind] = 1
                    imageind += 1
            return inputs, targets, filetree, imgdims

Breaking this file into chunks we see that the first 9 lines are used to
download and expand the raw zipfile into the appropriate ``repo_path``
subdirectory.

The next 14 lines are used to traverse the expanded zipfile directories to
build up the filetree data structure containing one key for each unique class
(directory name).  The values for each key are a list of image filenames.

The next 6 lines initialize a numpy buffers to hold the images and target
labels and setup sizes.

The double for loop spanning the final 14 lines is where the images actually
get loaded, and in this particular case we're utilizing some sklearn image
reading and transformation functions.  With the ``imread`` function we can take
the input jpeg images and convert them into (grayscale) 2D numpy matrices of
pixel intensities that lie between 0 and 1.  These images are resized,
inverted, flattened to 1D vectors then stored in whitened and un-whitened format
(minor pre-processing found to be useful for this particular dataset, stored as
separate channels).  Finally you can see ``inputs`` is updated where these
flattened pixel values are stored as a row vector indexed by each image.
Similarly ``targets`` is updated so that a 1 is placed in the column
representing the given class (a so called one-hot encoding takes place).  Note
that both ``inputs`` and ``targets`` are represented as one row vector per data
point.


Working with Imageset
---------------------
If you have a set of image files as input, consider using Imageset.  This
Dataset incorporates batching and pre-processing (cropping, normalization) in
an efficient, multi-threaded manner.  It can also take advantage of directory
subfolders to identify target labels.

Required Imageset constructor/YAML parameters:

* ``repo_path``: base path to where the raw data is kept.
* ``imageset``: Name of the subdirectory off of ``repo_path`` where raw image
                files live
* ``save_dir``: where to keep batched data objects and indices.  Will greatly
                speed up subsequent runs on this data.
* ``macro_size``: number of images to include in each macro batch
* ``cropped_image_size``: desired number of pixels along 1 dimension
                          (assumes square images)
* ``output_image_size``: original image number of pixels along 1 dimension
                         (assumes square images)

Optional Imageset parameters (mostly BatchWriter related):

* ``dotransforms``: carry out pre-processing transforms
* ``square_crop``: make cropped image square.  Default is False.
* ``mean_norm``: pixel intensenties are centered by having mean pixel intensity
                 subtracted from each value.  Note that this operation inhibits
                 asynchronous stream copying.  Default is False
* ``unit_norm``: pixel intensities are normalized to lie in range [0,1] (or
                 [-1, 1] if ``mean_norm`` is also set).  Default is False.
* ``tdims``: number of dimensions of each target.
* ``label_list``: array of label names
* ``num_channels``: number of image channels (ex. 3 for RGB images).  Defaults
                    to 3 if not set
* ``num_workers``: number of processes to spawn for batch writing.  Defaults to
                   6 if not set.
* ``backend_type``: element value type (for each image pixel).  Defaults to
                    ``np.float32`` if not set.

To see an example that uses Imageset, have a look at
:download:`ndsb_imageset.yaml <../../examples/convnet/ndsb.yaml>`
