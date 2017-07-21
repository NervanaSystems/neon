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
'''
Test of the BiLSTM layer
'''
from builtins import zip
import itertools as itt
import numpy as np

from neon import NervanaObject
from neon.initializers.initializer import GlorotUniform
from neon.layers.recurrent import BiLSTM, LSTM, get_steps
from neon.transforms import Logistic, Tanh
from numpy import concatenate as con
from utils import allclose_with_out


def pytest_generate_tests(metafunc):
    bsz_rng = [1, 2]

    if 'fargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            seq_rng = [2, 3, 4, 5]
            inp_rng = [3, 5, 10, 20]
            out_rng = [3, 5, 10, 20]
        else:
            seq_rng = [3]
            inp_rng = [5]
            out_rng = [10]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng)
        metafunc.parametrize('fargs', fargs)


def test_biLSTM_fprop_rnn(backend_default, fargs):

    # basic sanity check with 0 weights random inputs
    seq_len, input_size, hidden_size, batch_size = fargs
    in_shape = (input_size, seq_len)
    out_shape = (hidden_size, seq_len)
    NervanaObject.be.bsz = batch_size

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    bilstm = BiLSTM(hidden_size, gate_activation=Logistic(),
                    activation=Tanh(), init=init_glorot, reset_cells=True)
    bilstm.configure(in_shape)
    bilstm.prev_layer = True
    bilstm.allocate()

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    rnn = LSTM(hidden_size, gate_activation=Logistic(),
               activation=Tanh(), init=init_glorot, reset_cells=True)
    rnn.configure(in_shape)
    rnn.prev_layer = True
    rnn.allocate()

    # same weight for bi-rnn backward and rnn weights
    nout = hidden_size
    bilstm.W_input_b[:] = bilstm.W_input_f
    bilstm.W_recur_b[:] = bilstm.W_recur_f
    bilstm.b_b[:] = bilstm.b_f
    bilstm.dW[:] = 0
    rnn.W_input[:] = bilstm.W_input_f
    rnn.W_recur[:] = bilstm.W_recur_f
    rnn.b[:] = bilstm.b_f
    rnn.dW[:] = 0

    # inputs - random and flipped left-to-right inputs
    lr = np.random.random((input_size, seq_len * batch_size))
    lr_rev = list(reversed(get_steps(lr.copy(), in_shape)))

    rl = con(lr_rev, axis=1)
    inp_lr = bilstm.be.array(lr)
    inp_rl = bilstm.be.array(rl)
    inp_rnn = rnn.be.array(lr)

    # outputs
    out_lr = bilstm.fprop(inp_lr).get().copy()
    bilstm.h_buffer[:] = 0
    out_rl = bilstm.fprop(inp_rl).get()
    out_rnn = rnn.fprop(inp_rnn).get().copy()

    # views
    out_lr_f_s = get_steps(out_lr[:nout], out_shape)
    out_lr_b_s = get_steps(out_lr[nout:], out_shape)
    out_rl_f_s = get_steps(out_rl[:nout], out_shape)
    out_rl_b_s = get_steps(out_rl[nout:], out_shape)
    out_rnn_s = get_steps(out_rnn, out_shape)

    # asserts for fprop
    for x_rnn, x_f, x_b, y_f, y_b in zip(out_rnn_s, out_lr_f_s, out_lr_b_s,
                                         reversed(out_rl_f_s), reversed(out_rl_b_s)):
        assert allclose_with_out(x_f, y_b, rtol=0.0, atol=1.0e-5)
        assert allclose_with_out(x_b, y_f, rtol=0.0, atol=1.0e-5)
        assert allclose_with_out(x_rnn, x_f, rtol=0.0, atol=1.0e-5)
        assert allclose_with_out(x_rnn, y_b, rtol=0.0, atol=1.0e-5)


def test_biLSTM_fprop(backend_default, fargs):

    # basic sanity check with 0 weights random inputs
    seq_len, input_size, hidden_size, batch_size = fargs
    in_shape = (input_size, seq_len)
    out_shape = (hidden_size, seq_len)
    NervanaObject.be.bsz = batch_size

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    bilstm = BiLSTM(hidden_size, gate_activation=Logistic(), init=init_glorot,
                    activation=Tanh(), reset_cells=True)
    bilstm.configure(in_shape)
    bilstm.prev_layer = True
    bilstm.allocate()

    # same weight
    nout = hidden_size
    bilstm.W_input_b[:] = bilstm.W_input_f
    bilstm.W_recur_b[:] = bilstm.W_recur_f
    bilstm.b_b[:] = bilstm.b_f
    bilstm.dW[:] = 0

    # inputs - random and flipped left-to-right inputs
    lr = np.random.random((input_size, seq_len * batch_size))
    lr_rev = list(reversed(get_steps(lr.copy(), in_shape)))

    rl = con(lr_rev, axis=1)
    inp_lr = bilstm.be.array(lr)
    inp_rl = bilstm.be.array(rl)

    # outputs
    out_lr = bilstm.fprop(inp_lr).get().copy()

    bilstm.h_buffer[:] = 0
    out_rl = bilstm.fprop(inp_rl).get().copy()

    # views
    out_lr_f_s = get_steps(out_lr[:nout], out_shape)
    out_lr_b_s = get_steps(out_lr[nout:], out_shape)
    out_rl_f_s = get_steps(out_rl[:nout], out_shape)
    out_rl_b_s = get_steps(out_rl[nout:], out_shape)

    # asserts
    for x_f, x_b, y_f, y_b in zip(out_lr_f_s, out_lr_b_s,
                                  reversed(out_rl_f_s), reversed(out_rl_b_s)):
        assert allclose_with_out(x_f, y_b, rtol=0.0, atol=1.0e-5)
        assert allclose_with_out(x_b, y_f, rtol=0.0, atol=1.0e-5)


def test_biLSTM_bprop(backend_default, fargs, deltas_buffer):

    # basic sanity check with 0 weights random inputs
    seq_len, input_size, hidden_size, batch_size = fargs
    in_shape = (input_size, seq_len)
    out_shape = (hidden_size, seq_len)
    NervanaObject.be.bsz = batch_size

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    bilstm = BiLSTM(hidden_size, gate_activation=Logistic(),
                    activation=Tanh(), init=init_glorot, reset_cells=True)
    bilstm.configure(in_shape)
    bilstm.prev_layer = True
    bilstm.allocate()

    bilstm.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    bilstm.set_deltas(deltas_buffer)

    # same weight for bi-rnn backward and rnn weights
    nout = hidden_size
    bilstm.W_input_b[:] = bilstm.W_input_f
    bilstm.W_recur_b[:] = bilstm.W_recur_f
    bilstm.b_b[:] = bilstm.b_f
    bilstm.dW[:] = 0

    # inputs and views
    lr = np.random.random((input_size, seq_len * batch_size))
    lr_rev = list(reversed(get_steps(lr.copy(), in_shape)))
    rl = con(lr_rev, axis=1)

    # allocate gpu buffers
    inp_lr = bilstm.be.array(lr)
    inp_rl = bilstm.be.array(rl)

    # outputs
    out_lr_g = bilstm.fprop(inp_lr)
    out_lr = out_lr_g.get().copy()
    del_lr = bilstm.bprop(out_lr_g).get().copy()
    bilstm.h_buffer[:] = 0
    out_rl_g = bilstm.fprop(inp_rl)
    out_rl = out_rl_g.get().copy()
    del_rl = bilstm.bprop(out_rl_g).get().copy()

    # views
    out_lr_f_s = get_steps(out_lr[:nout], out_shape)
    out_lr_b_s = get_steps(out_lr[nout:], out_shape)
    out_rl_f_s = get_steps(out_rl[:nout], out_shape)
    out_rl_b_s = get_steps(out_rl[nout:], out_shape)

    # asserts
    for x_f, x_b, y_f, y_b in zip(out_lr_f_s, out_lr_b_s,
                                  reversed(out_rl_f_s), reversed(out_rl_b_s)):
        assert allclose_with_out(x_f, y_b, rtol=0.0, atol=1.0e-5)
        assert allclose_with_out(x_b, y_f, rtol=0.0, atol=1.0e-5)

    del_lr_s = get_steps(del_lr, in_shape)
    del_rl_s = get_steps(del_rl, in_shape)

    for (x, y) in zip(del_lr_s, reversed(del_rl_s)):
        assert allclose_with_out(x, y, rtol=0.0, atol=1.0e-5)
