# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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

from neon.initializers import GlorotUniform
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, DeepBiRNN, RecurrentLast
from neon.models import Model
from neon.transforms import CrossEntropyBinary, Rectlin, Softmax


def create_network():
    init = GlorotUniform()
    layers = [
        Conv((3, 3, 128), init=init, activation=Rectlin(), strides=dict(str_h=1, str_w=2)),
        Conv((3, 3, 256), init=init, batch_norm=True, activation=Rectlin()),
        Pooling(2, strides=2),
        Conv((2, 2, 512), init=init, batch_norm=True, activation=Rectlin()),
        DeepBiRNN(256, init=init, activation=Rectlin(), reset_cells=True, depth=3),
        RecurrentLast(),
        Affine(32, init=init, batch_norm=True, activation=Rectlin()),
        Affine(nout=2, init=init, activation=Softmax())
    ]

    return Model(layers=layers), GeneralizedCost(costfunc=CrossEntropyBinary())
