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


class DataIterator(NervanaObject):

    """
    This generic class defines an interface to iterate over minibatches of
    data that has been preloaded into memory. This may be used when the
    entire dataset is small enough to fit within memory.
    """

    def __init__(self, X, y=None, nclass=None, lshape=None, make_onehot=True):
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
        X = X if isinstance(X, list) else [X]
        self.ndata = len(X[0])
        self.start = 0

        # on device tensor with full dataset
        self.Xdev = [self.be.array(x) for x in X]
        # mini-batch sized buffer
        self.Xbuf = [self.be.iobuf(x.shape[1]) for x in X]

        if lshape is not None:
            self.shape = [lshape for x in X]
        else:
            self.shape = [x.shape[1] for x in X]

        if len(self.shape) == 1:
            self.shape = self.shape[0]

            # store shape of the input data
            self.lshape = lshape

        assert self.ndata >= self.be.bsz

        self.ybuf = None
        self.make_onehot = make_onehot
        if y is not None:
            if make_onehot:
                assert nclass is not None
                self.ydev = self.be.array(y.reshape((-1, 1)), dtype=np.int32)
                self.ybuf = self.be.iobuf(nclass)
            else:
                self.ydev = self.be.array(y)
                self.ybuf = self.be.iobuf(y.shape[1])

    @property
    def nbatches(self):
        return -((self.start - self.ndata) // self.be.bsz)

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        self.start = 0

    def __iter__(self):
        """
        Defines a generator that can be used to iterate over this dataset.

        Yields:
            tuple: The next minibatch. A minibatch includes both features and
            labels.
        """
        for i1 in range(self.start, self.ndata, self.be.bsz):
            i2 = min(i1 + self.be.bsz, self.ndata)
            bsz = i2 - i1
            if i2 == self.ndata:
                self.start = self.be.bsz - bsz

            for xbuf, xdev in zip(self.Xbuf, self.Xdev):
                xbuf[:, :bsz] = xdev[i1:i2].T
                if self.be.bsz > bsz:
                    xbuf[:, bsz:] = xdev[:(self.be.bsz - bsz)].T

            if self.ybuf is not None:
                if self.make_onehot:
                    self.ybuf[:, :bsz] = self.be.onehot(
                        self.ydev[i1:i2], axis=0)
                    if self.be.bsz > bsz:
                        self.ybuf[:, bsz:] = self.be.onehot(
                            self.ydev[:(self.be.bsz - bsz)], axis=0)
                else:
                    self.ybuf[:, :bsz] = self.ydev[i1:i2].T
                    if self.be.bsz > bsz:
                        self.ybuf[:, bsz:] = self.ydev[:(self.be.bsz - bsz)].T

            inputs = self.Xbuf[0] if len(self.Xbuf) == 1 else self.Xbuf
            targets = self.ybuf if self.ybuf else inputs
            yield (inputs, targets)


if __name__ == '__main__':
    from neon.data import load_mnist
    (X_train, y_train), (X_test, y_test) = load_mnist()

    from neon.backends.nervanagpu import NervanaGPU
    ng = NervanaGPU(0, device_id=1)

    NervanaObject.be = ng
    ng.bsz = 128
    train_set = DataIterator(
        [X_test[:1000], X_test[:1000]], y_test[:1000], nclass=10)
    for i in range(3):
        for bidx, (X_batch, y_batch) in enumerate(train_set):
            print bidx, train_set.start
            pass
