#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from __future__ import division
import numpy as np
import math
from neon import NervanaObject


def rolling_window(a, lag):
    """
    Convert a into time-lagged vectors

    a    : (n, p)
    lag  : time steps used for prediction

    returns  (n-lag+1, lag, p)  array

    (Building time-lagged vectors is not necessary for neon.)
    """
    assert a.shape[0] > lag

    shape = [a.shape[0] - lag + 1, lag, a.shape[-1]]
    strides = [a.strides[0], a.strides[0], a.strides[-1]]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class SyntheticTimeSeries(object):

    def __init__(self, npoints=30, ncycles=3, divide=0.2, amplitude=1, curvetype='Lissajous1'):
        """
        curvetype (str, optional): 'Lissajous1' or 'Lissajous2'
        """
        self.nsamples = npoints * ncycles
        self.x = np.linspace(0, ncycles * 2 * math.pi, self.nsamples)

        if curvetype not in ('Lissajous1', 'Lissajous2'):
            raise NotImplementedError()

        sin_scale = 2 if curvetype == 'Lissajous1' else 1

        def y_x(x):
            return 4.0 / 5 * math.sin(x / sin_scale)

        def y_y(x):
            return 4.0 / 5 * math.cos(x / 2)

        self.data = np.zeros((self.nsamples, 2))
        self.data[:, 0] = np.asarray([y_x(xs)
                                      for xs in self.x]).astype(np.float32)
        self.data[:, 1] = np.asarray([y_y(xs)
                                      for xs in self.x]).astype(np.float32)

        L = len(self.data)
        c = int(L * (1 - divide))
        self.train = self.data[:c]
        self.test = self.data[c:]


class DataIteratorSequence(NervanaObject):

    """
    This class takes a sequence and returns an iterator providing data in batches suitable for RNN
    prediction.  Meant for use when the entire dataset is small enough to fit in memory.
    """

    def __init__(self, X, time_steps, forward=1, return_sequences=True):
        """
        Implements loading of given data into backend tensor objects. If the backend is specific
        to an accelerator device, the data is copied over to that device.

        Args:
            X (ndarray): Input sequence with feature size within the dataset.
                         Shape should be specified as (num examples, feature size]
            time_steps (int): The number of examples to be put into one sequence.
            forward (int, optional): how many forward steps the sequence should predict. default
                                     is 1, which is the next example
            return_sequences (boolean, optional): whether the target is a sequence or single step.
                                                  Also determines whether data will be formatted
                                                  as strides or rolling windows.
                                                  If true, target value be a sequence, input data
                                                  will be reshaped as strides.  If false, target
                                                  value will be a single step, input data will be
                                                  a rolling_window
        """
        self.seq_length = time_steps
        self.forward = forward
        self.batch_index = 0
        self.nfeatures = self.nclass = X.shape[1]
        self.nsamples = X.shape[0]
        self.shape = (self.nfeatures, time_steps)
        self.return_sequences = return_sequences

        target_steps = time_steps if return_sequences else 1
        # pre-allocate the device buffer to provide data for each minibatch
        # buffer size is nfeatures x (times * batch_size), which is handled by
        # backend.iobuf()
        self.X_dev = self.be.iobuf((self.nfeatures, time_steps))
        self.y_dev = self.be.iobuf((self.nfeatures, target_steps))

        if return_sequences is True:
            # truncate to make the data fit into multiples of batches
            extra_examples = self.nsamples % (self.be.bsz * time_steps)
            if extra_examples:
                X = X[:-extra_examples]

            # calculate how many batches
            self.nsamples -= extra_examples
            self.nbatches = self.nsamples // (self.be.bsz * time_steps)
            self.ndata = self.nbatches * self.be.bsz * time_steps  # no leftovers

            # y is the lagged version of X
            y = np.concatenate((X[forward:], X[:forward]))
            self.y_series = y
            # reshape this way so sequence is continuous along the batches
            self.X = X.reshape(self.be.bsz, self.nbatches,
                               time_steps, self.nfeatures)
            self.y = y.reshape(self.be.bsz, self.nbatches,
                               time_steps, self.nfeatures)
        else:
            self.X = rolling_window(X, time_steps)
            self.X = self.X[:-1]
            self.y = X[time_steps:]

            self.nsamples = self.X.shape[0]
            extra_examples = self.nsamples % (self.be.bsz)
            if extra_examples:
                self.X = self.X[:-extra_examples]
                self.y = self.y[:-extra_examples]

            # calculate how many batches
            self.nsamples -= extra_examples
            self.nbatches = self.nsamples // self.be.bsz
            self.ndata = self.nbatches * self.be.bsz
            self.y_series = self.y

            Xshape = (self.nbatches, self.be.bsz, time_steps, self.nfeatures)
            Yshape = (self.nbatches, self.be.bsz, 1, self.nfeatures)
            self.X = self.X.reshape(Xshape).transpose(1, 0, 2, 3)
            self.y = self.y.reshape(Yshape).transpose(1, 0, 2, 3)

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        """
        self.batch_index = 0

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuple : the next minibatch of data.
        """
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            # get the data for this batch and reshape to fit the device buffer
            # shape
            X_batch = self.X[:, self.batch_index].T.reshape(
                self.X_dev.shape).copy()
            y_batch = self.y[:, self.batch_index].T.reshape(
                self.y_dev.shape).copy()

            # make the data for this batch as backend tensor
            self.X_dev.set(X_batch)
            self.y_dev.set(y_batch)

            self.batch_index += 1

            yield self.X_dev, self.y_dev


# replicate neon's mse error metric
def err(y, t):
    feature_axis = 1
    return 0.5 * np.square(y - t).mean(axis=feature_axis).mean()
