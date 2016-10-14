# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Defines basic input datatset types.
"""
import os
import h5py
import logging
import numpy as np

from neon.data import ArrayIterator
logger = logging.getLogger(__name__)


class HDF5Iterator(ArrayIterator):
    """
    Data iterator which uses an HDF5 file as the source of the data, useful when
    the entire HDF5 dataset cannot fit into memory (for smaller datasets, use the ArrayIterator).

    To initialize the HDF5Iterator, simply call::

        train_set = HDF5Iterator('your_data_path.h5')

    The HDF5 file format must contain the following datasets:

    - `input` (ndarray):
                        Input data, which is a 2-D array (float or uint8) that
                        has size `(N, F)`, where `N` is the number of examples,
                        and `F` is the number of features. For images, `F = C*H*W` where
                        `C` is the number of channels and `H` and `W` are the height
                        and width of the image, respectively. This data must also
                        have the following attributes:
        - `lshape` (tuple):
                    Tuple of ints indicating the shape of each
                    input (for examples, image data may have
                    an lshape of `[C, H, W]`)
        - `mean` (ndarray):
                    The mean to subtract, either formatted as (F, 1) or
                    a mean for each channel with dimensions (C, 1)
    - `output` (ndarray):
                        An optional dataset which, if supplied, will be
                        used at the target/expected output of the network. the
                        array should have the shape `(N, M)` where `N` is the number
                        of items (must match the `N` dim of the input set)
                        and `M` is the size of the output data which must match
                        size of ouput from the output layer of the network.

    For cases where the output should be converted to a one-hot encoding (see Loading Data),
    use the `HDF5IteratorOneHot`. Or for autoencoder problems, use `HDFIteratorAutoencoder`.
    """
    def __init__(self, hdf_filename, name=None):
        """
        Args:
            hdf_filename (string): Path to the HDF5 datafile.
            name (string, optional): Name to assign this iterator. Defaults to None.
        """
        super(ArrayIterator, self).__init__(name=name)

        self.hdf_filename = hdf_filename

        if not os.path.isfile(hdf_filename):
            raise IOError('File not found %s' % hdf_filename)
        self.hdf_file = h5py.File(hdf_filename, mode='r', driver=None)

        # input data array
        self.inp = self.hdf_file['input']
        self.ndata = self.inp.shape[0]

        # must have at least 1 minibatch of data in the file
        assert self.ndata >= self.be.bsz
        self.start = 0

        # the input array unflattened size
        self.lshape = tuple(self.inp.attrs['lshape'])
        self.shape = self.lshape

        if 'output' in self.hdf_file:
            self.out = self.hdf_file['output']

        self.inpbuf = None
        self.outbuf = None
        self.allocated = False

    def allocate(self):
        """
        After the input and output (`self.inp` and `self.out)` have been
        set this function will allocate the host and device buffers
        for the mini-batches.

        The host buffer is referenced as `self.mini_batch_in` and `self.mini_batch_out`, and
        stored on device as `self.inbuf` and `self.outbuf`.
        """
        if not self.allocated:
            self.allocate_inputs()
            self.allocate_outputs()
            self.allocated = True

    def allocate_inputs(self):
        """
        Allocates the host and device input data buffers
        and any other associated storage.

        `self.inpbuf` is the on-device buffer for the input minibatch
        `self.mini_batch_in` is the on-host buffer for the input minibatch
        `self.mean` is the on-device buffer of the mean array
        """
        # on device minibatch_buffer (input)
        self.inpbuf = self.be.iobuf(self.inp.shape[1])

        # setup host buffer for a mini_batch
        self.mini_batch_in = np.zeros(self.inpbuf.shape)

        self.mean = None
        # the 'mean' dataset is the the mean values to subtract
        if 'mean' in self.hdf_file:
            mns_ = np.array(self.hdf_file['mean']).flatten()
            if mns_.size != self.inp.shape[1]:
                # channel by channel mean
                # there should be 1 element per channel
                assert mns_.size == self.lshape[0], 'mean image size mismatch'
                # need to have 2-d array for broadcasting
                mns_ = mns_.reshape((self.lshape[0], 1)).copy()
                # make channel-by-channel mean subtraction view
                self.meansub_view = self.inpbuf.reshape((self.lshape[0], -1))
            else:
                self.meansub_view = self.inpbuf

            self.mean = self.be.array(mns_)

    def allocate_outputs(self):
        """
        Allocates the host and device output data buffers
        and any other associated storage.

        `self.outbuf` is the on-device buffer for the output minibatch
        `self.mini_batch_out` is the on-host buffer for the output minibatch
        """
        self.outbuf = None
        if 'output' in self.hdf_file:
            self.outbuf = self.be.iobuf(self.out.shape[1])
            self.mini_batch_out = np.zeros(self.outbuf.shape)

    def gen_input(self, mini_batch):
        """
        Function to handle any preprocessing before pushing an input
        mini-batch to the device.  For example, mean subtraction etc.

        Arguments:
            mini_batch (ndarray): M-by-N array where M is the flatten
                                  input vector size and N is the batch size
        """
        self.inpbuf[:] = mini_batch
        # mean subtract
        if self.mean is not None:
            self.meansub_view[:] = -self.mean + self.meansub_view

    def gen_output(self, mini_batch):
        """
        Function to handle any preprocessing before pushing an output
        mini-batch to the device.  For example, one-hot generation.

        Arguments:
            mini_batch (ndarray): M-by-N array where M is the flatten
                                  output vector size and N is the batch size
        """
        self.outbuf[:] = mini_batch

    def __del__(self):
        self.cleanup()
        # needed for python3 to exit cleanly
        super(HDF5Iterator, self).__del__()

    def cleanup(self):
        """
        Closes the HDF file.
        """
        self.hdf_file.close()

    def reset(self):
        """
        Resets the index to zero.
        """
        self.start = 0

    def __iter__(self):
        """
        Defines a generator that can be used to iterate over this dataset.

        Yields:
            tuple: The next minibatch. A minibatch includes both features and
            labels.
        """
        if not self.allocated:
            self.allocate()
        full_shape = list(self.lshape)
        full_shape.append(-1)

        mini_batch_in = self.mini_batch_in
        if self.outbuf is not None:
            mini_batch_out = self.mini_batch_out
        for i1 in range(self.start, self.ndata, self.be.bsz):
            i2 = min(i1 + self.be.bsz, self.ndata)
            bsz = i2 - i1
            if i2 == self.ndata:
                self.start = self.be.bsz - bsz

            # load mini batch on host
            xdev = self.inp
            mini_batch_in[:, :bsz] = xdev[i1:i2, :].T.astype(np.float32)
            if self.be.bsz > bsz:
                mini_batch_in[:, bsz:] = xdev[:(self.be.bsz - bsz), :].T.astype(np.float32)

            # push to device
            self.gen_input(mini_batch_in)

            if self.outbuf is not None:
                mini_batch_out[:, :bsz] = self.out[i1:i2].T
                if self.be.bsz > bsz:
                    mini_batch_out[:, bsz:] = self.out[:(self.be.bsz - bsz)].T

                self.gen_output(mini_batch_out)

            inputs = self.inpbuf
            targets = self.outbuf
            yield (inputs, targets)


class HDF5IteratorOneHot(HDF5Iterator):
    """
    Extends the HDF5Iterator class to add one hot conversion of the
    target data. For example::

        train_set = HDF5IteratorOneHot('your_data_path.h5')

    The HDF5 file format must contain the following datasets:
    - `input` dataset: Input data, which is a 2-D array (float or uint8) that
                        has size `(N, C*H*W)`, where `N` is the number of examples,
                        `C` is the number of channels, and `H` and `W` are the height
                        and width of the image, respectively. This dataset must also
                        have the following attributes:
        - `lshape`: a tuple of int indicating the shape of each
                    input (for examples, image data may have
                    an lshape of `[C, H, W]`)
        - `mean`: the mean to subtract, either formatted as (C*H*W, 1) or
                  a mean for each channel with dimensions (C, 1)
    - `output` dataset: An optional dataset which, if supplied, will be
                        used at the target/expected output of the network. the
                        array should have the shape `(N, M)` where `N` is the number
                        of items (must match the `N` dim of the input set)
                        and `M` is the size of the output data which must match
                        size of ouput from the output layer of the network.

    The "output" dataset in the HDF5 (if present) must have the 'nclass'
    attribute specifying the number of total output classes which is needed
    for generating the one-hot encoding.
    """
    def __init__(self, hdf_filename, name=None):
        """
        Args:
            hdf_filename (string): Path to the HDF5 datafile.
            name (string, optional): Name to assign this iterator. Defaults to None.
        """
        super(HDF5IteratorOneHot, self).__init__(hdf_filename, name=name)
        if 'output' in self.hdf_file:
            assert 'nclass' in self.hdf_file['output'].attrs, 'Missing nclass attribute'
            self.nclass = int(self.hdf_file['output'].attrs['nclass'])
            self.out = np.array(self.hdf_file['output'], dtype=np.int32).reshape((-1, 1))

    def allocate_outputs(self):
        self.outbuf = None
        if 'output' in self.hdf_file:
            self.argmax_buf = self.be.iobuf(1, dtype=np.int32)
            self.mini_batch_out = np.zeros(self.argmax_buf.shape, dtype=np.int32)
            self.outbuf = self.be.iobuf(self.nclass)

    def gen_output(self, mini_batch):
        self.argmax_buf[:] = mini_batch
        self.be.onehot(self.argmax_buf, axis=0, out=self.outbuf)


class HDF5IteratorAutoencoder(HDF5Iterator):
    """
    Extends the base HDF5Iterator class for an autoencoder model.
    Returns the input data as the target.
    """
    def __iter__(self):
        """
        Defines a generator that can be used to iterate over this dataset.

        Yields:
            tuple: The next minibatch containing the inputs and the targets (here target=inputs)
        """
        for x, t in super(HDF5IteratorAutoencoder, self).__iter__():
            yield (x, x)
