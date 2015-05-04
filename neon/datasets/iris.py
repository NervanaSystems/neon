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
A self contained version of Fisher's Iris dataset.  This is a tiny 3 output
class, 5 input feature, 150 total instance (50 per outcome) dataset.  It
contains mesaures of sepal and petal length and width for 3 types of iris
plant.

The features (in order) are:
    1. Sepal Length
    2. Sepal Width
    3. Petal Length
    4. Petal Width

The class (species) is the target, it has been one hot encoded as a 3 column
binary vector as follows:
    1 0 0 - setosa
    0 1 0 - versicolor
    0 0 1 - virginica

More info at: https://archive.ics.uci.edu/ml/datasets/Iris
"""

import logging
import numpy

from neon.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class Iris(Dataset):
    """
    Sets up the iris plant dataset.

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """
    raw_inputs = numpy.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2],
                              [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
                              [5, 3.6, 1.4, 0.2],   [5.4, 3.9, 1.7, 0.4],
                              [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
                              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1],
                              [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
                              [4.8, 3, 1.4, 0.1],   [4.3, 3, 1.1, 0.1],
                              [5.8, 4, 1.2, 0.2],   [5.7, 4.4, 1.5, 0.4],
                              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3],
                              [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
                              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4],
                              [4.6, 3.6, 1, 0.2],   [5.1, 3.3, 1.7, 0.5],
                              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2],
                              [5, 3.4, 1.6, 0.4],   [5.2, 3.5, 1.5, 0.2],
                              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
                              [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
                              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2],
                              [4.9, 3.1, 1.5, 0.2], [5, 3.2, 1.2, 0.2],
                              [5.5, 3.5, 1.3, 0.2], [4.9, 3.6, 1.4, 0.1],
                              [4.4, 3, 1.3, 0.2],   [5.1, 3.4, 1.5, 0.2],
                              [5, 3.5, 1.3, 0.3],   [4.5, 2.3, 1.3, 0.3],
                              [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6],
                              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3],
                              [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
                              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2],
                              [7, 3.2, 4.7, 1.4],   [6.4, 3.2, 4.5, 1.5],
                              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3],
                              [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
                              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1],
                              [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4],
                              [5, 2, 3.5, 1],       [5.9, 3, 4.2, 1.5],
                              [6, 2.2, 4, 1],       [6.1, 2.9, 4.7, 1.4],
                              [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
                              [5.6, 3, 4.5, 1.5],   [5.8, 2.7, 4.1, 1],
                              [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
                              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3],
                              [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
                              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4],
                              [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7],
                              [6, 2.9, 4.5, 1.5],   [5.7, 2.6, 3.5, 1],
                              [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1],
                              [5.8, 2.7, 3.9, 1.2], [6, 2.7, 5.1, 1.6],
                              [5.4, 3, 4.5, 1.5],   [6, 3.4, 4.5, 1.6],
                              [6.7, 3.1, 4.7, 1.5], [6.3, 2.3, 4.4, 1.3],
                              [5.6, 3, 4.1, 1.3],   [5.5, 2.5, 4, 1.3],
                              [5.5, 2.6, 4.4, 1.2], [6.1, 3, 4.6, 1.4],
                              [5.8, 2.6, 4, 1.2],   [5, 2.3, 3.3, 1],
                              [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
                              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3],
                              [5.1, 2.5, 3, 1.1],   [5.7, 2.8, 4.1, 1.3],
                              [6.3, 3.3, 6, 2.5],   [5.8, 2.7, 5.1, 1.9],
                              [7.1, 3, 5.9, 2.1],   [6.3, 2.9, 5.6, 1.8],
                              [6.5, 3, 5.8, 2.2],   [7.6, 3, 6.6, 2.1],
                              [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
                              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5],
                              [6.5, 3.2, 5.1, 2],   [6.4, 2.7, 5.3, 1.9],
                              [6.8, 3, 5.5, 2.1],   [5.7, 2.5, 5, 2],
                              [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3],
                              [6.5, 3, 5.5, 1.8],   [7.7, 3.8, 6.7, 2.2],
                              [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5],
                              [6.9, 3.2, 5.7, 2.3], [5.6, 2.8, 4.9, 2],
                              [7.7, 2.8, 6.7, 2],   [6.3, 2.7, 4.9, 1.8],
                              [6.7, 3.3, 5.7, 2.1], [7.2, 3.2, 6, 1.8],
                              [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8],
                              [6.4, 2.8, 5.6, 2.1], [7.2, 3, 5.8, 1.6],
                              [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2],
                              [6.4, 2.8, 5.6, 2.2], [6.3, 2.8, 5.1, 1.5],
                              [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3],
                              [6.3, 3.4, 5.6, 2.4], [6.4, 3.1, 5.5, 1.8],
                              [6, 3, 4.8, 1.8],     [6.9, 3.1, 5.4, 2.1],
                              [6.7, 3.1, 5.6, 2.4], [6.9, 3.1, 5.1, 2.3],
                              [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3],
                              [6.7, 3.3, 5.7, 2.5], [6.7, 3, 5.2, 2.3],
                              [6.3, 2.5, 5, 1.9],   [6.5, 3, 5.2, 2],
                              [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]])

    raw_targets = numpy.array([1] * 50 + [2] * 50 + [3] * 50)
    raw_onehot_targets = numpy.zeros([len(raw_targets), 3])
    raw_onehot_targets[0:50, 0] = 1
    raw_onehot_targets[50:100, 1] = 1
    raw_onehot_targets[100:150, 2] = 1

    def __init__(self, **kwargs):
        self.macro_batched = False
        self.__dict__.update(kwargs)

    def load(self):
        if self.inputs['train'] is not None:
            return
        # split the dataset so that for each class we have 30 train, 10
        # validation, and 10 test instances.
        for name, l_idx, h_idx in (('train', 0, 30),
                                   ('validation', 30, 40),
                                   ('test', 40, 50)):
            logger.info('loading: %s data', name)
            s_idcs = slice(l_idx, h_idx)
            v_idcs = slice(l_idx + 50, h_idx + 50)
            c_idcs = slice(l_idx + 100, h_idx + 100)
            inputs = numpy.vstack((self.raw_inputs[s_idcs, :],
                                   self.raw_inputs[v_idcs, :],
                                   self.raw_inputs[c_idcs, :]))
            self.inputs[name] = inputs
            targets = numpy.vstack((self.raw_onehot_targets[s_idcs, :],
                                    self.raw_onehot_targets[v_idcs, :],
                                    self.raw_onehot_targets[c_idcs, :]))
            self.targets[name] = targets
        self.format()
