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
Test of a LookupTable layer, which is often being used for word embedding
'''
from builtins import zip
import itertools as itt
import numpy as np
from neon.backends import gen_backend
from neon import NervanaObject
from neon import logger as neon_logger
from neon.initializers.initializer import GlorotUniform
from neon.layers.layer import LookupTable
from utils import allclose_with_out


def pytest_generate_tests(metafunc):
    if metafunc.config.option.all:
        bsz_rng = [16, 32, 64]
    else:
        bsz_rng = [128]

    if 'basic_linargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [1, 2, 64, 128]
            nout_rng = [1, 4, 128, 64]
            vocab_size = [1, 4, 1000, 2000]
        else:
            nin_rng = [4, 32]
            nout_rng = [3, 33]
            vocab_size = [10, 34]
        fargs = itt.product(nin_rng, nout_rng, vocab_size, bsz_rng)
        neon_logger.display('{}'.format(fargs))
        metafunc.parametrize('basic_linargs', fargs)


def test_lookuptable_zeros_error(backend_default, basic_linargs, deltas_buffer):
    # basic sanity check with 0 weights random inputs
    nin, nout, batch_size, vocab_size = basic_linargs
    NervanaObject.be.bsz = batch_size

    dtypeu = np.float32

    init_glorot = GlorotUniform()
    layer = LookupTable(
        vocab_size=vocab_size, embedding_dim=nout, init=init_glorot)

    inp = np.random.random_integers(0, vocab_size - 1, size=nin * batch_size)
    layer.configure(nin)
    layer.allocate()

    layer.prev_layer = True  # Hack to force delta buffer allocation
    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    inputs = layer.be.array(inp.reshape((nin, batch_size)))
    out = layer.fprop(inputs).get()
    W = layer.W.get()
    for i in range(nin * batch_size):
        assert np.all(W[inp[i]].T == out[:, i])

    err = dtypeu(np.zeros((nout, nin * batch_size)))
    layer.bprop(layer.be.array(err)).get()

    dw = layer.dW.get()
    assert np.min(dw) == 0.0 and np.max(dw) == 0.0

    return


def test_lookuptable_ones_error(backend_default, basic_linargs, deltas_buffer):
    nin, nout, batch_size, vocab_size = basic_linargs
    NervanaObject.be.bsz = batch_size

    dtypeu = np.float32

    init_glorot = GlorotUniform()
    layer = LookupTable(
        vocab_size=vocab_size, embedding_dim=nout, init=init_glorot)

    inp = np.random.random_integers(0, vocab_size - 1, size=nin * batch_size)
    layer.configure(nin)
    layer.allocate()
    layer.prev_layer = True  # Hack to force delta buffer allocation

    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    inputs = layer.be.array(inp.reshape((nin, batch_size)))
    out = layer.fprop(inputs).get()
    W = layer.W.get()
    for i in range(nin * batch_size):
        assert np.all(W[inp[i]].T == out[:, i])

    err = dtypeu(np.ones((nout, nin * batch_size)))
    layer.bprop(layer.be.array(err)).get()

    dw = layer.dW.get()
    unqidx, count = np.unique(inp, return_counts=True)
    dw_exp = np.zeros((1, nout))
    for wrd_id, cnt in zip(unqidx, count):
        dw_exp = err[:, 0] * cnt
        assert np.all(dw_exp == dw[wrd_id, :])

    return


def test_lookuptable_rand_error(backend_default, basic_linargs, deltas_buffer):
    nin, nout, batch_size, vocab_size = basic_linargs
    NervanaObject.be.bsz = batch_size

    dtypeu = np.float32

    init_glorot = GlorotUniform()
    layer = LookupTable(
        vocab_size=vocab_size, embedding_dim=nout, init=init_glorot)

    inp = np.random.random_integers(0, vocab_size - 1, size=nin * batch_size)
    layer.configure(nin)
    layer.allocate()

    layer.prev_layer = True  # Hack to force delta buffer allocation
    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    inputs = layer.be.array(inp.reshape((nin, batch_size)))
    out = layer.fprop(inputs).get()
    W = layer.W.get()
    for i in range(nin * batch_size):
        assert np.all(W[inp[i]].T == out[:, i])

    err = dtypeu(np.random.random((nout, nin * batch_size)))
    layer.bprop(layer.be.array(err)).get()

    dw = layer.dW.get()
    unqidx, count = np.unique(inp, return_counts=True)
    dw_exp = np.zeros((1, nout))
    for wrd_id, cnt in zip(unqidx, count):
        dw_exp[:] = 0
        cnt_exp = 0
        for i, w_id in enumerate(inp):
            if w_id == wrd_id:
                dw_exp[:] = dw_exp[:] + err[:, i]
                cnt_exp += 1
        assert allclose_with_out(dw[wrd_id, :], dw_exp, atol=0, rtol=1e-4)
        assert allclose_with_out(dw_exp, dw[wrd_id, :], atol=0, rtol=1e-4)
        assert cnt == cnt_exp

    return


if __name__ == '__main__':

    fargs = [1, 128, 1, 1]

    be = gen_backend(backend='cpu',
                     datatype=np.float32,
                     batch_size=128,
                     rng_seed=0)

    test_lookuptable_zeros_error(be, fargs)
