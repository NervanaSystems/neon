# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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
from builtins import zip
import numpy as np

from neon import NervanaObject


class NervanaDataIterator(NervanaObject):
    """
    Abstract class for data iterators.

    For serialization, any data iterator should inherit from this class
    """
    def __init__(self, name=None):
        super(NervanaDataIterator, self).__init__(name)

    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        raise NotImplemented()

    def reset(self):
        """
        Reset the starting index of this dataset back to zero.
        """
        raise NotImplemented()

    def __iter__(self):
        """
        Define a generator that can be used to iterate over this dataset.
        """
        raise NotImplemented()


class ArrayIterator(NervanaDataIterator):

    """
    The ArrayIterator class iterates over minibatches of data that
    have been preloaded into memory in the form of numpy arrays. This may be used when
    the entire dataset (e.g. CIFAR-10 or MNIST) is small enough to fit in memory. For example::

        X = np.random.rand(10000, 3072)
        y = np.random.randint(0, 10, 10000)
        train = ArrayIterator(X=X, y=y, nclass=10, lshape=(3, 32, 32))

    The above will create the ArrayIterator object. This object implements python's __iter__
    method, and returns one minibatch of data, formatted as tuple of (input, label), with
    each iteration. The minibatch size is controlled by the generated backend.

    X should be an ndarray of shape (# example, # features). For images, the features should be
    formatted in (channel, height, width) order. The `lshape` keyword indicates the local shape of
    the images in (channel, height, width) format.

    For classification tasks, the labels `y` should be integers from 0 to K-1, where K is the total
    number of classes. When `y` is not provided, the input features themselves will be returned
    as the target values (e.g. autoencoder).

    In regression tasks, where `y` is not a categorical label, set `make_onehot` to `False`.
    For example::

        X = np.random.rand(1000, 1)
        y = 2*X + 1
        train = ArrayIterator(X=X, y=y, make_onehot=False)

    For more information, see the Loading data section of the documentation.
    """

    def __init__(self, X, y=None, nclass=None, lshape=None, make_onehot=True, name=None):
        """
        During initialization, the input data will be converted to backend tensor objects
        (e.g. CPUTensor or GPUTensor). If the backend uses the GPU, the data is copied over to the
        device.

        Args:
            X (ndarray, shape: [# examples, feature size]): Input features of the
                dataset.
            y (ndarray, shape:[# examples, 1 or feature size], optional): Labels corresponding to
                the input features. If absent, the input features themselves will be returned as
                target values (e.g. autoencoder)
            nclass (int, optional): The number of classes in labels. Not necessary if
                labels are not provided or where the labels are non-categorical.
            lshape (tuple, optional): Local shape for the input features
                (e.g. # channels, height, width)
            make_onehot (bool, optional): True if y is a categorical label that has to be converted
                to a one hot representation.

        """
        # Treat singletons like list so that iteration follows same syntax
        super(ArrayIterator, self).__init__(name=name)
        X = X if isinstance(X, list) else [X]
        self.ndata = len(X[0])
        assert self.ndata >= self.be.bsz
        self.start = 0
        self.nclass = nclass
        self.ybuf = None

        if make_onehot and nclass is None and y is not None:
            raise AttributeError('Must provide number of classes when creating onehot labels')

        # if labels provided, they must have same # examples as the features
        if y is not None:

            assert all([y.shape[0] == x.shape[0] for x in X]), \
                "Input features and labels must have equal number of examples."

            # for classifiction, the labels must be from 0 .. K-1, where K=nclass
            if make_onehot:
                assert y.max() <= nclass - 1 and y.min() >= 0, \
                    "Labels must range from 0 to {} (nclass-1).".format(nclass - 1)

                assert (np.floor(y) == y).all(), \
                    "Labels must only contain integers."

        # if local shape is provided, then the product of lshape should match the
        # number of features
        if lshape is not None:
            assert all([x.shape[1] == np.prod(lshape) for x in X]), \
                "product of lshape {} does not match input feature size".format(lshape)

        # store shape of the input data
        self.shape = [x.shape[1] if lshape is None else lshape for x in X]
        if len(self.shape) == 1:
            self.shape = self.shape[0]
            self.lshape = lshape

        # Helpers to make dataset, minibatch, unpacking function for transpose and onehot
        def transpose_gen(z):
            return (self.be.array(z), self.be.iobuf(z.shape[1]),
                    lambda _in, _out: self.be.copy_transpose(_in, _out))

        def onehot_gen(z):
            return (self.be.array(z.reshape((-1, 1)), dtype=np.int32), self.be.iobuf(nclass),
                    lambda _in, _out: self.be.onehot(_in, axis=0, out=_out))

        self.Xdev, self.Xbuf, self.unpack_func = list(zip(*[transpose_gen(x) for x in X]))

        # Shallow copies for appending, iterating
        self.dbuf, self.hbuf = list(self.Xdev), list(self.Xbuf)
        self.unpack_func = list(self.unpack_func)

        if y is not None:
            self.ydev, self.ybuf, yfunc = onehot_gen(y) if make_onehot else transpose_gen(y)
            self.dbuf.append(self.ydev)
            self.hbuf.append(self.ybuf)
            self.unpack_func.append(yfunc)

    @property
    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        return -((self.start - self.ndata) // self.be.bsz)

    def reset(self):
        """
        Resets the starting index of this dataset to zero. Useful for calling
        repeated evaluations on the dataset without having to wrap around
        the last uneven minibatch. Not necessary when data is divisible by batch size
        """
        self.start = 0

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.

        Yields:
            tuple: The next minibatch which includes both features and labels.
        """
        for i1 in range(self.start, self.ndata, self.be.bsz):
            bsz = min(self.be.bsz, self.ndata - i1)
            islice1, oslice1 = slice(0, bsz), slice(i1, i1 + bsz)
            islice2, oslice2 = None, None
            if self.be.bsz > bsz:
                islice2, oslice2 = slice(bsz, None), slice(0, self.be.bsz - bsz)
                self.start = self.be.bsz - bsz

            for buf, dev, unpack_func in zip(self.hbuf, self.dbuf, self.unpack_func):
                unpack_func(dev[oslice1], buf[:, islice1])
                if oslice2:
                    unpack_func(dev[oslice2], buf[:, islice2])

            inputs = self.Xbuf[0] if len(self.Xbuf) == 1 else self.Xbuf
            targets = self.ybuf if self.ybuf else inputs
            yield (inputs, targets)
