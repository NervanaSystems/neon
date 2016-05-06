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
import logging
import numpy as np

from neon import NervanaObject
logger = logging.getLogger(__name__)


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
    This generic class defines an interface to iterate over minibatches of
    data that has been preloaded into memory in the form of numpy arrays.
    This may be used when the entire dataset is small enough to fit within memory.
    """

    def __init__(self, X, y=None, nclass=None, lshape=None, make_onehot=True, name=None):
        """
        Implements loading of given data into backend tensor objects. If the
        backend is specific to an accelarator device, the data is copied over
        to that device.

        Args:
            X (ndarray, shape: [# examples, feature size]): Input features within the
                dataset.
            y (ndarray, shape:[# examples, 1], optional): Labels corresponding to the
                input features.
                If absent, the input features themselves will be returned as
                target values (AutoEncoder)
            nclass (int, optional): The number of possible types of labels.
                (not necessary if not providing labels)
            lshape (tuple, optional): Local shape for the input features
                (e.g. height, width, channel for images)
            make_onehot (bool, optional): True if y is a label that has to be converted to one hot
                            False if y doesn't need to be converted to one hot
                            (e.g. in a CAE)

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

        self.Xdev, self.Xbuf, self.unpack_func = zip(*[transpose_gen(x) for x in X])

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
        Reset the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        self.start = 0

    def __iter__(self):
        """
        Define a generator that can be used to iterate over this dataset.

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


class DataIterator(ArrayIterator):
    """
    This class has been renamed to ArrayIterator and deprecated.
    This is just a place holder until the class is removed.  Please
    use the ArrayIterator class.
    """
    def __init__(self, *args, **kwargs):
        logger.error('DataIterator class has been deprecated and renamed'
                     '"ArrayIterator" please use that name.')
        super(DataIterator, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    from neon.data import load_mnist
    (X_train, y_train), (X_test, y_test) = load_mnist()

    from neon.backends.nervanagpu import NervanaGPU
    ng = NervanaGPU(0, device_id=1)

    NervanaObject.be = ng
    ng.bsz = 128
    train_set = ArrayIterator(
        [X_test[:1000], X_test[:1000]], y_test[:1000], nclass=10)
    for i in range(3):
        for bidx, (X_batch, y_batch) in enumerate(train_set):
            print bidx, train_set.start
            pass
