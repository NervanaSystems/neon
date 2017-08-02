# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
from builtins import zip
import numpy as np
import os

from neon.backends import gen_backend
from neon.data import ArrayIterator, MNIST, PTB
from neon.initializers import Gaussian, Constant
from neon.layers import (GeneralizedCost, Affine, DeepBiRNN, DeepBiLSTM, LSTM, GRU,
                         Dropout, Conv, Pooling, Sequential, MergeMultistream, Recurrent,
                         RecurrentMean)
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from utils import allclose_with_out
from neon import NervanaObject


def test_model_get_outputs_rnn(backend_default, data):

    dataset = PTB(50, path=data)
    dataiter = dataset.train_iter

    # weight initialization
    init = Constant(0.08)

    # model initialization
    layers = [
        Recurrent(150, init, activation=Logistic()),
        Affine(len(dataiter.vocab), init, bias=init, activation=Rectlin())
    ]

    model = Model(layers=layers)
    output = model.get_outputs(dataiter)

    assert output.shape == (dataiter.ndata, dataiter.seq_length, dataiter.nclass)

    # since the init are all constant and model is un-trained:
    # along the feature dim, the values should be all the same
    assert allclose_with_out(output[0, 0], output[0, 0, 0], rtol=0, atol=1e-4)
    assert allclose_with_out(output[0, 1], output[0, 1, 0], rtol=0, atol=1e-4)

    # along the time dim, the values should be increasing:
    assert np.alltrue(output[0, 2] > output[0, 1])
    assert np.alltrue(output[0, 1] > output[0, 0])


def test_model_N_S_setter(backend_default):

    # weight initialization
    init = Constant(0.08)

    # model initialization
    layers = [
        Recurrent(150, init, activation=Logistic()),
        Affine(100, init, bias=init, activation=Rectlin())
    ]

    model = Model(layers=layers)
    model.set_batch_size(20)
    model.set_seq_len(10)


def test_model_get_outputs(backend_default, data):
    dataset = MNIST(path=data)
    train_set = dataset.train_iter

    init_norm = Gaussian(loc=0.0, scale=0.1)

    layers = [Affine(nout=20, init=init_norm, bias=init_norm, activation=Rectlin()),
              Affine(nout=10, init=init_norm, activation=Logistic(shortcut=True))]
    mlp = Model(layers=layers)
    out_list = []
    mlp.initialize(train_set)
    for x, t in train_set:
        x = mlp.fprop(x)
        out_list.append(x.get().T.copy())
    ref_output = np.vstack(out_list)

    train_set.reset()
    output = mlp.get_outputs(train_set)
    assert allclose_with_out(output, ref_output[:output.shape[0], :])

    # test model benchmark inference
    mlp.benchmark(train_set, inference=True, niterations=5)


def test_model_serialize(backend_default, data):
    dataset = MNIST(path=data)
    (X_train, y_train), (X_test, y_test), nclass = dataset.load_data()
    train_set = ArrayIterator(
        [X_train, X_train], y_train, nclass=nclass, lshape=(1, 28, 28))

    init_norm = Gaussian(loc=0.0, scale=0.01)

    # initialize model
    path1 = Sequential([Conv((5, 5, 16), init=init_norm, bias=Constant(0), activation=Rectlin()),
                        Pooling(2),
                        Affine(nout=20, init=init_norm, bias=init_norm, activation=Rectlin())])
    path2 = Sequential([Affine(nout=100, init=init_norm, bias=Constant(0), activation=Rectlin()),
                        Dropout(keep=0.5),
                        Affine(nout=20, init=init_norm, bias=init_norm, activation=Rectlin())])
    layers = [MergeMultistream(layers=[path1, path2], merge="stack"),
              Affine(nout=20, init=init_norm, batch_norm=True, activation=Rectlin()),
              Affine(nout=10, init=init_norm, activation=Logistic(shortcut=True))]

    tmp_save = 'test_model_serialize_tmp_save.pickle'
    mlp = Model(layers=layers)
    mlp.optimizer = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)
    mlp.cost = GeneralizedCost(costfunc=CrossEntropyBinary())
    mlp.initialize(train_set, cost=mlp.cost)
    n_test = 3
    num_epochs = 3
    # Train model for num_epochs and n_test batches
    for epoch in range(num_epochs):
        for i, (x, t) in enumerate(train_set):
            x = mlp.fprop(x)
            delta = mlp.cost.get_errors(x, t)
            mlp.bprop(delta)
            mlp.optimizer.optimize(mlp.layers_to_optimize, epoch=epoch)
            if i > n_test:
                break

    # Get expected outputs of n_test batches and states of all layers
    outputs_exp = []
    pdicts_exp = [l.get_params_serialize() for l in mlp.layers_to_optimize]
    for i, (x, t) in enumerate(train_set):
        outputs_exp.append(mlp.fprop(x, inference=True))
        if i > n_test:
            break

    # Serialize model
    mlp.save_params(tmp_save, keep_states=True)

    # Load model
    mlp = Model(tmp_save)

    mlp.initialize(train_set)
    outputs = []
    pdicts = [l.get_params_serialize() for l in mlp.layers_to_optimize]
    for i, (x, t) in enumerate(train_set):
        outputs.append(mlp.fprop(x, inference=True))
        if i > n_test:
            break

    # Check outputs, states, and params are the same
    for output, output_exp in zip(outputs, outputs_exp):
        assert allclose_with_out(output.get(), output_exp.get())

    for pd, pd_exp in zip(pdicts, pdicts_exp):
        for s, s_e in zip(pd['states'], pd_exp['states']):
            if isinstance(s, list):  # this is the batch norm case
                for _s, _s_e in zip(s, s_e):
                    assert allclose_with_out(_s, _s_e)
            else:
                assert allclose_with_out(s, s_e)
        for p, p_e in zip(pd['params'], pd_exp['params']):
            assert type(p) == type(p_e)
            if isinstance(p, list):  # this is the batch norm case
                for _p, _p_e in zip(p, p_e):
                    assert allclose_with_out(_p, _p_e)
            elif isinstance(p, np.ndarray):
                assert allclose_with_out(p, p_e)
            else:
                assert p == p_e

    os.remove(tmp_save)


def test_conv_rnn(backend_default):
    train_shape = (1, 17, 142)

    be = NervanaObject.be
    inp = be.array(be.rng.randn(np.prod(train_shape), be.bsz))
    delta = be.array(be.rng.randn(10, be.bsz))

    init_norm = Gaussian(loc=0.0, scale=0.01)
    bilstm = DeepBiLSTM(128, init_norm, activation=Rectlin(), gate_activation=Rectlin(),
                        depth=1, reset_cells=True)
    birnn_1 = DeepBiRNN(128, init_norm, activation=Rectlin(),
                        depth=1, reset_cells=True, batch_norm=False)
    birnn_2 = DeepBiRNN(128, init_norm, activation=Rectlin(),
                        depth=2, reset_cells=True, batch_norm=False)
    bibnrnn = DeepBiRNN(128, init_norm, activation=Rectlin(),
                        depth=1, reset_cells=True, batch_norm=True)
    birnnsum = DeepBiRNN(128, init_norm, activation=Rectlin(),
                         depth=1, reset_cells=True, batch_norm=False, bi_sum=True)
    rnn = Recurrent(128, init=init_norm, activation=Rectlin(), reset_cells=True)
    lstm = LSTM(128, init_norm, activation=Rectlin(), gate_activation=Rectlin(), reset_cells=True)
    gru = GRU(128, init_norm, activation=Rectlin(), gate_activation=Rectlin(), reset_cells=True)

    rlayers = [bilstm, birnn_1, birnn_2, bibnrnn, birnnsum, rnn, lstm, gru]

    for rl in rlayers:
        layers = [
                    Conv((2, 2, 4), init=init_norm, activation=Rectlin(),
                         strides=dict(str_h=2, str_w=4)),
                    Pooling(2, strides=2),
                    Conv((3, 3, 4), init=init_norm, batch_norm=True, activation=Rectlin(),
                         strides=dict(str_h=1, str_w=2)),
                    rl,
                    RecurrentMean(),
                    Affine(nout=10, init=init_norm, activation=Rectlin()),
                ]
        model = Model(layers=layers)
        cost = GeneralizedCost(costfunc=CrossEntropyBinary())
        model.initialize(train_shape, cost)
        model.fprop(inp)
        model.bprop(delta)


if __name__ == '__main__':
    be_gpu = gen_backend(backend='gpu', batch_size=128)
    test_conv_rnn(be_gpu)
    be_cpu = gen_backend(backend='cpu', batch_size=128)
    test_conv_rnn(be_cpu)
    be_mkl = gen_backend(backend='mkl', batch_size=128)
    test_conv_rnn(be_mkl)
