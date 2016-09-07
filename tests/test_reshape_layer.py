#!/usr/bin/env python
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
import numpy as np
import itertools as itt
from neon.backends import gen_backend
from neon.initializers import Uniform, GlorotUniform
from neon.layers import (LSTM, Affine, LookupTable, RecurrentSum, Recurrent,
                         Conv, GeneralizedCost, Pooling, DeepBiRNN, Reshape)
from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyBinary
from neon.models import Model


def pytest_generate_tests(metafunc):
    if metafunc.config.option.all:
        bsz_rng = [32, 64]
    else:
        bsz_rng = [32]

    if 'fargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [10, 30]
            nout_rng = [5, 10]
        else:
            nin_rng = [30]
            nout_rng = [10]
        fargs = itt.product(nin_rng, nout_rng, bsz_rng)
        metafunc.parametrize('fargs', fargs)


def test_reshape_configure(backend_default):
    """
    test cases:
    - reshape with 0
    - reshape with -1
    - reshape with collapsing dimensions
    - reshape with expanding dimensions
    """
    bsz = backend_default.bsz

    reshape_0 = Reshape((0, 10, 10))
    reshape_0.configure((10, 2, 50))
    assert reshape_0.out_shape == (10, 10, 10)

    reshape_1 = Reshape((10, 25, -1))
    reshape_1.configure((10, 2, 50))
    assert reshape_1.out_shape == (10, 25, 4)

    reshape_2 = Reshape((5, -1))
    reshape_2.configure((10, 2, 25))
    assert reshape_2.out_shape == (5, 100)
    assert reshape_2.out_shape_t == (5, 100 * bsz)

    reshape_3 = Reshape((5, -1, 5))
    reshape_3.configure((10, 25))
    assert reshape_3.out_shape == (5, 10, 5)


def test_reshape_layer_model(backend_default, fargs):
    """
    test cases:
    - conv before RNNs
    - conv after RNNs
    - conv after LUT
    """
    np.random.seed(seed=0)

    nin, nout, bsz = fargs
    be = backend_default
    be.bsz = bsz
    input_size = (nin, be.bsz)

    init = Uniform(-0.1, 0.1)
    g_uni = GlorotUniform()

    inp_np = np.random.rand(nin, be.bsz)
    delta_np = np.random.rand(nout, be.bsz)

    inp = be.array(inp_np)
    delta = be.array(delta_np)

    conv_lut_1 = [
        LookupTable(vocab_size=2000, embedding_dim=400, init=init),
        Reshape(reshape=(4, 100, -1)),
        Conv((3, 3, 16), init=init),
        LSTM(64, g_uni, activation=Tanh(),
             gate_activation=Logistic(), reset_cells=True),
        RecurrentSum(),
        Affine(nout, init, bias=init, activation=Softmax())
    ]

    conv_lut_2 = [
        LookupTable(vocab_size=1000, embedding_dim=400, init=init),
        Reshape(reshape=(4, 50, -1)),
        Conv((3, 3, 16), init=init),
        Pooling(2, strides=2),
        Affine(nout=nout, init=init, bias=init, activation=Softmax()),
    ]

    conv_rnn_1 = [
        LookupTable(vocab_size=2000, embedding_dim=400, init=init),
        LSTM(64, g_uni, activation=Tanh(),
             gate_activation=Logistic(), reset_cells=True),
        Reshape(reshape=(4, 32, -1)),
        Conv((3, 3, 16), init=init),
        Affine(nout, init, bias=init, activation=Softmax())
    ]

    conv_rnn_2 = [
        LookupTable(vocab_size=2000, embedding_dim=400, init=init),
        Recurrent(64, g_uni, activation=Tanh(), reset_cells=True),
        Reshape(reshape=(4, -1, 32)),
        Conv((3, 3, 16), init=init),
        Affine(nout, init, bias=init, activation=Softmax())
    ]

    lut_sum_1 = [
        LookupTable(vocab_size=1000, embedding_dim=128, init=init),
        RecurrentSum(),
        Affine(nout=nout, init=init, bias=init, activation=Softmax()),
    ]

    lut_birnn_1 = [
        LookupTable(vocab_size=1000, embedding_dim=200, init=init),
        DeepBiRNN(32, init=GlorotUniform(), batch_norm=True, activation=Tanh(),
                  reset_cells=True, depth=1),
        Reshape((4, 32, -1)),
        Conv((3, 3, 16), init=init),
        Affine(nout=nout, init=init, bias=init, activation=Softmax())
    ]

    layers_test = [conv_lut_1, conv_lut_2, conv_rnn_1, conv_rnn_2, lut_sum_1, lut_birnn_1]

    for lg in layers_test:
        model = Model(layers=lg)
        cost = GeneralizedCost(costfunc=CrossEntropyBinary())
        model.initialize(input_size, cost)
        model.fprop(inp)
        model.bprop(delta)


if __name__ == '__main__':

    be = gen_backend(backend='gpu', batch_size=128)
    # test_reshape_layer_model(be, (30, 10, 2))
    test_reshape_configure(be)
