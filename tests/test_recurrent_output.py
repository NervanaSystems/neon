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
Test of the recurrent outputs layers.
'''
import itertools as itt
import numpy as np

from neon.backends import gen_backend
from neon import NervanaObject
from neon.layers.recurrent import RecurrentSum, RecurrentMean, RecurrentLast
from utils import allclose_with_out


def pytest_generate_tests(metafunc):
    bsz_rng = [1]

    if 'refgruargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            seq_rng = [2, 3, 4]
            inp_rng = [3, 5, 10]
        else:
            seq_rng = [3]
            inp_rng = [5]
        fargs = itt.product(seq_rng, inp_rng, bsz_rng)
        metafunc.parametrize('refgruargs', fargs)


def test_recurrent_sum(backend_default, refgruargs, deltas_buffer):
    seq_len, nin, batch_size = refgruargs
    NervanaObject.be.bsz = batch_size

    in_shape = (nin, seq_len)
    layer = RecurrentSum()
    layer.configure(in_shape)
    layer.prev_layer = True
    layer.allocate()

    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    # zeros
    inp = layer.be.zeros((nin, seq_len * batch_size))
    out = layer.fprop(inp)
    err = layer.bprop(out).get()
    assert np.all(out.get() == np.zeros((nin, batch_size)))
    assert np.all(err == inp.get())

    # ones
    inp = layer.be.ones((nin, seq_len * batch_size))
    out = layer.fprop(inp)
    err = layer.bprop(out).get()
    assert np.all(out.get() == seq_len * np.ones((nin, batch_size)))
    assert np.all(err == seq_len * inp.get())

    # random batch
    rinp = np.random.random((nin, batch_size))
    inp = np.repeat(rinp, repeats=seq_len, axis=1)
    inp_g = layer.be.array(inp)
    out = layer.fprop(inp_g)
    err = layer.bprop(out)
    assert allclose_with_out(out.get(), seq_len * rinp)
    assert allclose_with_out(err.get(), seq_len * inp)

    # full random
    inp = np.random.random((nin, seq_len * batch_size))
    inp_g = layer.be.array(inp)
    out = layer.fprop(inp_g)
    err = layer.bprop(out)
    out_comp = np.zeros(out.shape)
    err_comp = np.zeros(inp.shape)
    for i in range(seq_len):
        out_comp[:] = out_comp + inp[:, i * batch_size:(i + 1) * batch_size]
        err_comp[:, i * batch_size:(i + 1) * batch_size] = out.get()
    assert allclose_with_out(out_comp, out.get())
    assert allclose_with_out(err_comp, err.get())


def test_recurrent_mean(backend_default, refgruargs, deltas_buffer):
    seq_len, nin, batch_size = refgruargs
    NervanaObject.be.bsz = batch_size

    in_shape = (nin, seq_len)
    layer = RecurrentMean()
    layer.configure(in_shape)
    layer.prev_layer = True
    layer.allocate()

    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    # zeros
    inp = layer.be.zeros((nin, seq_len * batch_size))
    out = layer.fprop(inp)
    err = layer.bprop(out).get()
    assert np.all(out.get() == np.zeros((nin, batch_size)))
    assert np.all(err == inp.get())

    # ones
    inp = layer.be.ones((nin, seq_len * batch_size))
    out = layer.fprop(inp)
    err = layer.bprop(out).get()
    assert np.all(out.get() == np.ones((nin, batch_size)))
    assert np.all(err == 1. / seq_len * inp.get())

    # random
    rinp = np.random.random((nin, batch_size))
    inp = np.repeat(rinp, repeats=seq_len, axis=1)
    inp_g = layer.be.array(inp)
    out = layer.fprop(inp_g)
    err = layer.bprop(out)
    assert allclose_with_out(out.get(), rinp)
    assert allclose_with_out(err.get(), 1. / seq_len * inp)

    # full random
    inp = np.random.random((nin, seq_len * batch_size))
    inp_g = layer.be.array(inp)
    out = layer.fprop(inp_g)
    err = layer.bprop(out)
    out_comp = np.zeros(out.shape)
    err_comp = np.zeros(inp.shape)
    for i in range(seq_len):
        out_comp[:] = out_comp + inp[:, i * batch_size:(i + 1) * batch_size]
        err_comp[:, i * batch_size:(i + 1) * batch_size] = out.get() / float(seq_len)
    out_comp[:] /= float(seq_len)

    assert allclose_with_out(out_comp, out.get())
    assert allclose_with_out(err_comp, err.get())


def test_recurrent_last(backend_default, refgruargs, deltas_buffer):
    seq_len, nin, batch_size = refgruargs
    NervanaObject.be.bsz = batch_size

    in_shape = (nin, seq_len)
    layer = RecurrentLast()
    layer.configure(in_shape)
    layer.prev_layer = True
    layer.allocate()

    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    # zeros
    inp = layer.be.zeros((nin, seq_len * batch_size))
    out = layer.fprop(inp)
    err = layer.bprop(out).get()
    assert np.all(out.get() == np.zeros((nin, batch_size)))
    assert np.all(err == inp.get())

    # ones
    inp = layer.be.ones((nin, seq_len * batch_size))
    out = layer.fprop(inp)
    err = layer.bprop(out).get()
    assert np.all(out.get() == np.ones((nin, batch_size)))
    assert np.all(err[:, -batch_size:] == inp.get()[:, -batch_size:])
    assert np.all(
        err[:, :-batch_size] == np.zeros((nin, (seq_len - 1) * batch_size)))

    # random
    rinp = np.random.random((nin, batch_size))
    inp = np.repeat(rinp, repeats=seq_len, axis=1)
    inp_g = layer.be.array(inp)
    out = layer.fprop(inp_g)
    err = layer.bprop(out)
    assert allclose_with_out(out.get(), rinp)
    assert allclose_with_out(err[:, -batch_size:].get(), rinp)

    # full random
    inp = np.random.random((nin, seq_len * batch_size))
    inp_g = layer.be.array(inp)
    out = layer.fprop(inp_g)
    err = layer.bprop(out)
    out_comp = np.zeros(out.shape)
    err_comp = np.zeros(inp.shape)
    out_comp[:] = inp[:, -batch_size:]
    err_comp[:, -batch_size:] = out.get()


if __name__ == '__main__':

    fargs = (2, 3, 1)
    be = gen_backend(backend='gpu')
    test_recurrent_sum(be, fargs)
