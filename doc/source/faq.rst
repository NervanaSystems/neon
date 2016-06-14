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

Frequently Asked Questions
--------------------------


How do I generate or change the backend configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

neon can use either the CPU or a supported GPU as the backend for training
and running inference with models.  Currently GPUs with Pascal, Maxwell or
Kepler architecture are supported.  There is a utility function called
:py:func:`neon.backends.gen_backend` which is used to configure the backend.

This function allows users to select the CPU or GPU backend as well as
configure various settings such as the size of the data mini-batches,
default data precision (e.g. 32 bit float), random number generator seed,
and the GPU device to use for systems that have more than one GPU installed.
The size of the data batches must be set when generating the backend.

Furthermore, this function will handle the housekeeping tasks necessary when
switching backends dynamically.  This function will also handle clean up
tasks that are needed such as deleting the GPU context.

Example usage of the gen_backend function:

.. code:: python

    # generate a GPU backend using 16 bit floating point
    # running on GPU device with id = 1
    import numpy as np
    gen_backend(backend='gpu', batch_size=128,
                default_dtype=np.float16, device_id=1)

    #  use this backend for various work such as training a model

    # to switch to a different backend call gen_backend again
    gen_backend(backend='cpu', batch_size=128)


Why does the number of batches per epoch change sometimes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running model training the progress bar output will show the number of
data batches processed every epoch. Sometimes this number will vary from
epoch to epoch because the data is being handled in mini-batches and if the
size of the data set is not an integer multiple of the mini-batch size there
will be some differences from epoch to epoch in the data used in training.

For example, during the first epoch of training, if the end of the data set
is reached and there is not enough data for a full mini-batch, extra data
will be pulled from the beginning of the data set to fill up a
complete mini-batch. On the next epoch, the data used will start from where
the last epoch left off, not from the beginning of the data set. This process
will continue for each epoch of training and, since the next epoch may not
start from the beginning of the data set, it is possible that for some of the
epochs the last item of data will fall at the end of the mini-batch.  For such
a case, no data will need to be appended to the epoch and the mini-batch
count for that epoch may be smaller than the other epochs.


How is padding implemented?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For convolution, deconvolution and pooling layers, zero padding can be added
to the edges of the input data to a layer.  If this parameter is set to an
integer, a uniform zero pad of that length will be added to the top, bottom,
left and right of the input data.

If a dictionary with the keys ``pad_h`` and ``pad_w`` is used, then the height
dimension of the data will be padded with a zero padding of length ``pad_h``
and the width dimension with a padding of length ``pad_w``.  Note that this is
different from the cuda-convnet2 style of padding in that padding is added to
both ends of the dimension instead of just one end.


I'm getting an error loading a serialized model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting with release 1.1.0 there was a change to the way that the
output sizes of convolution and pooling layers are calculated.  This change
could make this new version incompatible with models saved using previous
versions of neon.  With release 1.1.0, the calculation of the output size
of convolution and pooling layers stops before the kernel runs over the edge
of the input data plus the padding.  The formula for this is:

output_size = floor(input_size + 2*pad - kernel_size)/stride + 1

This change in output sizes will alter the topology of some networks and cause
errors when trying to load weights saved with a previous version of neon.  For
example, in a model with strided convolution layers below a set of linear
layers, the size of the first linear layer to receive input from a convolution
layer may change in this new release. This size mismatch may cause errors when
trying to initialize a neon model using weights saved with an older version of
neon.


I'm getting an error when I try to use ImgMaster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In release 1.1.0, the format of the mean image saved in the
`dataset_cache.pkl` file has changed.  Previous versions of neon were storing
a mean value for each pixel of the input image whereas in the new version only
a single mean value is stored for each input channel.  So now, for an RGB
image, the mean image has just 3 values, one for each color channel.  We have
provided a utility script: update_dataset_cache.py_ to update the old cache
files to the new format. Make sure you have read and write privileges to
change the file before running the utility.

.. _update_dataset_cache.py: https://github.com/NervanaSystems/neon/blob/master/neon/util/update_dataset_cache.py
