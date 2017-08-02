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
Test of the BiRNN layer
'''

from builtins import zip
import itertools as itt
import numpy as np
from numpy import concatenate as con

from neon import NervanaObject
from neon.initializers.initializer import GlorotUniform
from neon.layers.recurrent import BiRNN, Recurrent, get_steps, BiSum, BiBNRNN
from neon.transforms import Rectlinclip
from utils import allclose_with_out


def pytest_generate_tests(metafunc):
    bsz_rng = [1, 4]
    if 'fargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            seq_rng = [2, 3, 4]
            inp_rng = [3, 5, 10]
            out_rng = [3, 5, 10, 1152]
        else:
            seq_rng = [3]
            inp_rng = [5]
            out_rng = [10, 1152]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng)
        metafunc.parametrize('fargs', fargs)


def test_biRNN_fprop_rnn(backend_default, fargs, deltas_buffer):

    # basic sanity check with 0 weights random inputs
    seq_len, input_size, hidden_size, batch_size = fargs
    in_shape = (input_size, seq_len)
    out_shape = (hidden_size, seq_len)
    NervanaObject.be.bsz = batch_size

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    birnn = BiRNN(hidden_size, activation=Rectlinclip(slope=0), init=init_glorot)
    birnn.configure(in_shape)
    birnn.prev_layer = True
    birnn.allocate()

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    rnn = Recurrent(hidden_size, activation=Rectlinclip(slope=0), init=init_glorot)
    rnn.configure(in_shape)
    rnn.prev_layer = True
    rnn.allocate()

    # same weight for bi-rnn backward and rnn weights
    nout = hidden_size
    birnn.W_input_b[:] = birnn.W_input_f
    birnn.W_recur_b[:] = birnn.W_recur_f
    birnn.b_b[:] = birnn.b_f
    birnn.dW[:] = 0
    rnn.W_input[:] = birnn.W_input_f
    rnn.W_recur[:] = birnn.W_recur_f
    rnn.b[:] = birnn.b_f
    rnn.dW[:] = 0

    # inputs - random and flipped left-to-right inputs
    lr = np.random.random((input_size, seq_len * batch_size))
    lr_rev = list(reversed(get_steps(lr.copy(), in_shape)))

    rl = con(lr_rev, axis=1)
    inp_lr = birnn.be.array(lr)
    inp_rl = birnn.be.array(rl)
    inp_rnn = rnn.be.array(lr)

    # outputs
    out_lr = birnn.fprop(inp_lr).get().copy()
    birnn.h_buffer[:] = 0
    out_rl = birnn.fprop(inp_rl).get()
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


def test_biRNN_fprop(backend_default, fargs):

    # basic sanity check with 0 weights random inputs
    seq_len, input_size, hidden_size, batch_size = fargs
    in_shape = (input_size, seq_len)
    out_shape = (hidden_size, seq_len)
    NervanaObject.be.bsz = batch_size

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    birnn = BiRNN(hidden_size, activation=Rectlinclip(slope=0), init=init_glorot)
    birnn.configure(in_shape)
    birnn.prev_layer = True
    birnn.allocate()

    # same weight
    nout = hidden_size
    birnn.W_input_b[:] = birnn.W_input_f
    birnn.W_recur_b[:] = birnn.W_recur_f
    birnn.b_b[:] = birnn.b_f
    birnn.dW[:] = 0

    # inputs - random and flipped left-to-right inputs
    lr = np.random.random((input_size, seq_len * batch_size))
    lr_rev = list(reversed(get_steps(lr.copy(), in_shape)))

    rl = con(lr_rev, axis=1)
    inp_lr = birnn.be.array(lr)
    inp_rl = birnn.be.array(rl)

    # outputs
    out_lr = birnn.fprop(inp_lr).get().copy()

    birnn.h_buffer[:] = 0
    out_rl = birnn.fprop(inp_rl).get().copy()

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


def test_biRNN_bprop(backend_default, fargs, deltas_buffer):

    # basic sanity check with 0 weights random inputs
    seq_len, input_size, hidden_size, batch_size = fargs
    in_shape = (input_size, seq_len)
    NervanaObject.be.bsz = batch_size

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    birnn = BiRNN(hidden_size, activation=Rectlinclip(slope=0), init=init_glorot)
    birnn.configure(in_shape)
    birnn.prev_layer = True
    birnn.allocate()

    birnn.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    birnn.set_deltas(deltas_buffer)

    # same weight for bi-rnn backward and rnn weights
    birnn.W_input_b[:] = birnn.W_input_f
    birnn.W_recur_b[:] = birnn.W_recur_f
    birnn.b_b[:] = birnn.b_f
    birnn.dW[:] = 0

    # same weight for bi-directional rnn
    init_glorot = GlorotUniform()
    rnn = Recurrent(hidden_size, activation=Rectlinclip(slope=0), init=init_glorot)
    rnn.configure(in_shape)
    rnn.prev_layer = True
    rnn.allocate()

    rnn.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    rnn.set_deltas(deltas_buffer)

    # inputs and views
    lr = np.random.random((input_size, seq_len * batch_size))
    lr_rev = list(reversed(get_steps(lr.copy(), in_shape)))
    rl = con(lr_rev, axis=1)

    # allocate gpu buffers
    inp_lr = birnn.be.array(lr)
    inp_rl = birnn.be.array(rl)

    # outputs
    out_lr_g = birnn.fprop(inp_lr)
    del_lr = birnn.bprop(out_lr_g).get().copy()
    birnn.h_buffer[:] = 0
    out_rl_g = birnn.fprop(inp_rl)
    del_rl = birnn.bprop(out_rl_g).get().copy()

    del_lr_s = get_steps(del_lr, in_shape)
    del_rl_s = get_steps(del_rl, in_shape)
    for (x, y) in zip(del_lr_s, reversed(del_rl_s)):
        assert allclose_with_out(x, y, rtol=0.0, atol=1.0e-5)


def test_biSum(backend_default, fargs, deltas_buffer):

    seq_len, input_size, hidden_size, batch_size = fargs
    input_size *= 2

    in_shape = (input_size, seq_len)
    NervanaObject.be.bsz = batch_size

    bisum = BiSum()
    bisum.configure(in_shape)
    bisum.prev_layer = True

    bisum.allocate()
    bisum.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    bisum.set_deltas(deltas_buffer)

    # inputs
    inp_np = np.random.random((input_size, seq_len * batch_size))
    inp_be = bisum.be.array(inp_np)

    # outputs
    out_be = bisum.fprop(inp_be)
    del_be = bisum.bprop(out_be)

    out_ref = bisum.be.empty_like(out_be)
    out_ref[:] = inp_be[:input_size // 2] + inp_be[input_size // 2:]
    assert out_be.shape[0] * 2 == inp_be.shape[0]
    assert allclose_with_out(out_be.get(), out_ref.get(), rtol=0.0, atol=1.0e-5)

    assert allclose_with_out(del_be[:input_size // 2].get(), out_be.get(), rtol=0.0, atol=1.0e-5)
    assert allclose_with_out(del_be[input_size // 2:].get(), out_be.get(), rtol=0.0, atol=1.0e-5)


def test_bibn(backend_default, fargs, deltas_buffer):

    seq_len, input_size, hidden_size, batch_size = fargs
    in_shape = (input_size, seq_len)
    NervanaObject.be.bsz = batch_size

    hidden_size = min(10, hidden_size)

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    birnn = BiBNRNN(hidden_size, activation=Rectlinclip(slope=0), init=init_glorot)
    birnn.configure(in_shape)
    birnn.prev_layer = True

    birnn.allocate()
    birnn.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    birnn.set_deltas(deltas_buffer)

    # test fprop

    # set the ff buffer
    inp_np = np.random.random(birnn.h_ff_buffer.shape)
    inp_be = birnn.be.array(inp_np)
    birnn.h_ff_buffer[:] = inp_np

    # compare the bn output with calling the backend bn
    xsum = birnn.be.zeros_like(birnn.xmean)
    xvar = birnn.be.zeros_like(birnn.xvar)
    gmean = birnn.be.zeros_like(birnn.gmean)
    gvar = birnn.be.zeros_like(birnn.gvar)
    gamma = birnn.be.ones(birnn.gamma.shape)
    beta = birnn.be.zeros_like(birnn.beta)
    grad_gamma = birnn.be.zeros_like(gamma)
    grad_beta = birnn.be.zeros_like(beta)
    out_ref = birnn.be.zeros_like(birnn.h_ff_buffer)

    xsum[:] = birnn.be.sum(birnn.h_ff_buffer, axis=1)

    birnn.be.compound_fprop_bn(
        birnn.h_ff_buffer, xsum, xvar, gmean, gvar,
        gamma, beta, out_ref, birnn.eps, birnn.rho, False,
        accumbeta=0, relu=False)

    # call the bibnrnn layer fprop_bn
    out_bn = birnn._fprop_bn(birnn.h_ff_buffer, inference=False)

    assert allclose_with_out(out_bn.get(), out_ref.get(), rtol=0.0, atol=1.0e-5)

    # test bprop
    err_np = np.random.random(birnn.h_ff_buffer.shape)
    err_be = birnn.be.array(err_np)

    err_out_ref = birnn.be.empty_like(err_be)
    birnn.be.compound_bprop_bn(err_out_ref, grad_gamma, grad_beta,
                               err_be,
                               inp_be, xsum, xvar, gamma,
                               birnn.eps)

    err_out_bn = birnn._bprop_bn(err_be, out_bn)

    assert allclose_with_out(err_out_bn.get(), err_out_ref.get(), rtol=0.0, atol=2.5e-5)
