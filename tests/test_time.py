# ----------------------------------------------------------------------------
# Copyright 2015-2017 Nervana Systems Inc.
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

import numpy as np
import pytest

from neon import NervanaObject
from neon.backends import gen_backend
from neon.benchmark import Benchmark
from neon.data import ArrayIterator
from neon.initializers import Gaussian
from neon.layers import Conv, Affine, Pooling
from neon.models import Model

gen_backend(batch_size=16)
test_layers = [Conv((2, 2, 4), init=Gaussian(scale=0.01), padding=3, strides=4),
               Pooling(3, strides=2),
               Affine(nout=10, init=Gaussian(scale=0.01))]
NervanaObject.be.enable_winograd = 4
x = np.random.uniform(-1, 1, (16, 3 * 2 * 2))
y = np.random.randint(0, 9, (16, 10))
test_dataset = ArrayIterator(x, y, nclass=10, lshape=(3, 2, 2))


def test_empty_dataset():
    model = Model(test_layers)
    b = Benchmark(model=model)
    with pytest.raises(ValueError):
        b.time([], niterations=5, inference=True)


def test_fw_bw_no_cost_or_optimizer():
    model = Model(test_layers)
    model.initialize(test_dataset)
    b = Benchmark(model=model)
    with pytest.raises(RuntimeError):
        b.time(test_dataset, niterations=1)
