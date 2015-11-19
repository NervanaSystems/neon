# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
import itertools as itt
import numpy as np

from neon import NervanaObject
from neon.initializers.initializer import GlorotUniform
from neon.layers.birecurrent import BiRNN
from neon.transforms import Logistic
from numpy import concatenate as con
from neon.layers import Recurrent


def pytest_generate_tests(metafunc):
    bsz_rng = [1]
    if 'refgruargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            seq_rng = [2, 3, 4]
            inp_rng = [3, 5, 10]
            out_rng = [3, 5, 10]
        else:
            seq_rng = [3]
            inp_rng = [5]
            out_rng = [10]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng)
        metafunc.parametrize('refgruargs', fargs)

    if 'gradgruargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            seq_rng = [2, 3]
            inp_rng = [5, 10]
            out_rng = [3, 5, 10]
        else:
            seq_rng = [3]
            inp_rng = [5]
            out_rng = [10]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng)
        metafunc.parametrize('gradgruargs', fargs)


def get_steps(x, shape):
    """
    Convert a (vocab_size, steps * batch_size) array
    into a [(vocab_size, batch_size)] * steps list of views
    """
    steps = shape[1]
    xs = x.reshape(shape + (-1,))
    return [xs[:, step, :] for step in range(steps)]


def test_biRNN_fprop_rnn(backend_default, refgruargs):

    # basic sanity check with 0 weights random inputs
    seq_len, input_size, hidden_size, batch_size = refgruargs
    in_shape = (input_size, seq_len)
    out_shape = (hidden_size, seq_len)
    NervanaObject.be.bsz = NervanaObject.be.bs = batch_size

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    birnn = BiRNN(hidden_size, activation=Logistic(), init=init_glorot)
    birnn.configure(in_shape)
    birnn.prev_layer = True
    birnn.allocate()
    birnn.set_deltas([birnn.be.iobuf(birnn.in_shape)])

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    rnn = Recurrent(hidden_size, activation=Logistic(), init=init_glorot)
    rnn.configure(in_shape)
    rnn.prev_layer = True
    rnn.allocate()
    rnn.set_deltas([rnn.be.iobuf(rnn.in_shape)])

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
        assert np.all(x_f == y_b)
        assert np.all(x_b == y_f)
        assert np.all(x_rnn == x_f)
        assert np.all(x_rnn == y_b)


def test_biRNN_fprop(backend_default, refgruargs):

    # basic sanity check with 0 weights random inputs
    seq_len, input_size, hidden_size, batch_size = refgruargs
    in_shape = (input_size, seq_len)
    out_shape = (hidden_size, seq_len)
    NervanaObject.be.bsz = NervanaObject.be.bs = batch_size

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    birnn = BiRNN(hidden_size, activation=Logistic(), init=init_glorot)
    birnn.configure(in_shape)
    birnn.prev_layer = True
    birnn.allocate()
    birnn.set_deltas([birnn.be.iobuf(birnn.in_shape)])

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
        assert np.all(x_f == y_b)
        assert np.all(x_b == y_f)


def test_biRNN_bprop(backend_default, refgruargs):

    # basic sanity check with 0 weights random inputs
    seq_len, input_size, hidden_size, batch_size = refgruargs
    in_shape = (input_size, seq_len)
    NervanaObject.be.bsz = NervanaObject.be.bs = batch_size

    # setup the bi-directional rnn
    init_glorot = GlorotUniform()
    birnn = BiRNN(hidden_size, activation=Logistic(), init=init_glorot)
    birnn.configure(in_shape)
    birnn.prev_layer = True
    birnn.allocate()
    birnn.set_deltas([birnn.be.iobuf(birnn.in_shape)])

    # same weight for bi-rnn backward and rnn weights
    birnn.W_input_b[:] = birnn.W_input_f
    birnn.W_recur_b[:] = birnn.W_recur_f
    birnn.b_b[:] = birnn.b_f
    birnn.dW[:] = 0

    # same weight for bi-directional rnn
    init_glorot = GlorotUniform()
    rnn = Recurrent(hidden_size, activation=Logistic(), init=init_glorot)
    rnn.configure(in_shape)
    rnn.prev_layer = True
    rnn.allocate()
    rnn.set_deltas([rnn.be.iobuf(rnn.in_shape)])

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
        assert np.all(x == y)
