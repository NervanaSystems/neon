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
Datasets with fake data for testing purposes.
"""

import logging
import numpy as np

from neon.datasets.dataset import Dataset
from neon.util.compat import range


logger = logging.getLogger(__name__)


class UniformRandom(Dataset):
    """
    Sets up a synthetic uniformly random dataset.

    Attributes:
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data
    """

    def __init__(self, ntrain, ntest, nin, nout, **kwargs):
        self.__dict__.update(kwargs)
        self.ntrain = ntrain
        self.ntest = ntest
        self.nin = nin
        self.nout = nout
        self.macro_batched = False
        np.random.seed(0)

    def load_data(self, shape):
        data = np.random.uniform(low=0.0, high=1.0, size=shape)
        labels = np.random.randint(low=0, high=self.nout, size=shape[0])
        onehot = np.zeros((len(labels), self.nout), dtype='float32')
        for col in range(self.nout):
            onehot[:, col] = (labels == col)
        return (data, onehot)

    def load(self):
        self.inputs['train'], self.targets['train'] = (
            self.load_data((self.ntrain, self.nin)))
        self.inputs['test'], self.targets['test'] = (
            self.load_data((self.ntest, self.nin)))
        self.format()


class ToyImages(Dataset):
    """
    Sets up a synthetic image classification dataset.

    Attributes:
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.macro_batched = False
        self.ntrain = 128
        self.ntest = 128
        self.ifmheight = 32
        self.ifmwidth = self.ifmheight
        self.maxrad = self.ifmwidth / 2
        self.minrad = self.ifmwidth / 8
        self.nifm = 3
        self.nin = self.nifm * self.ifmheight * self.ifmwidth
        self.nout = 2
        assert self.ifmheight % 2 == 0
        assert self.ifmwidth % 2 == 0
        self.center = (self.ifmwidth / 2, self.ifmheight / 2)
        np.random.seed(0)

    def ellipse(self, canvas, xrad, yrad):
        rcanvas = canvas.reshape((self.nifm, self.ifmheight, self.ifmwidth))
        smooth = 10
        angs = np.linspace(0, 2 * np.pi, smooth * 360)
        si = np.sin(angs)
        co = np.cos(angs)
        xvals = np.int32(xrad * co) + self.center[0]
        yvals = np.int32(yrad * si) + self.center[1]
        for fm in range(self.nifm):
            rcanvas[fm, xvals, yvals] = np.random.randint(256)

    def circle(self, canvas, rad):
        self.ellipse(canvas, rad, rad)

    def load_data(self, shape):
        data = np.zeros(shape, dtype='float32')
        labels = np.zeros(shape[0], dtype='float32')
        ncircles = shape[0] / 2

        for row in range(0, ncircles):
            # Make circles.
            rad = np.random.randint(self.minrad, self.maxrad)
            self.circle(data[row], rad)

        for row in range(ncircles, shape[0]):
            # Make ellipses.
            while True:
                xrad, yrad = np.random.randint(self.minrad, self.maxrad, 2)
                if xrad != yrad:
                    break
            self.ellipse(data[row], xrad, yrad)
            labels[row] = 1

        data /= 255
        onehot = np.zeros((len(labels), self.nout), dtype='float32')
        for col in range(self.nout):
            onehot[:, col] = (labels == col)
        return (data, onehot)

    def load(self):
        ntotal = self.ntrain + self.ntest
        inds = np.arange(ntotal)
        np.random.shuffle(inds)
        data, targets = self.load_data((ntotal, self.nin))
        self.inputs['train'] = data[inds[:self.ntrain]]
        self.targets['train'] = targets[inds[:self.ntrain]]
        self.inputs['test'] = data[inds[self.ntrain:]]
        self.targets['test'] = targets[inds[self.ntrain:]]
        self.format()
