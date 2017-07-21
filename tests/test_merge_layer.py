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
Test of the merge layer with linear layers
'''
from builtins import zip
import itertools as itt
import numpy as np

from neon import NervanaObject
from neon.initializers.initializer import Uniform
from neon.layers import Affine, MergeMultistream, Sequential
from utils import allclose_with_out


def pytest_generate_tests(metafunc):
    fargs = []
    eps = np.finfo(np.float32).eps
    # weight ranges
    w_rng = [[-1.0, 1.0]]
    rng_max = [eps]
    fargs = itt.product(w_rng, rng_max)
    metafunc.parametrize('allrand_args', fargs)


def test_concat_l1_l1(backend_default, allrand_args, deltas_buffer):
    # test two linear layers that are merged with concat
    dtypeu = np.float32
    w_rng, rngmax = allrand_args
    # Diff size inputs and outputs
    nins = [128, 1024]
    nouts = [64, 2048]
    batch_size = 16
    NervanaObject.be.bsz = batch_size
    be = NervanaObject.be

    init_unif = Uniform(low=w_rng[0], high=w_rng[1])
    layers = [Sequential(Affine(nout=nout, init=init_unif)) for nout in nouts]
    inputs = [be.array(dtypeu(np.random.random((nin, batch_size)))) for nin in nins]
    merge = MergeMultistream(layers, merge="stack")
    assert(len(inputs) == len(layers))
    merge.configure(inputs)
    merge.allocate()

    merge.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    merge.set_deltas(deltas_buffer)

    out = merge.fprop(inputs).get()

    sublayers = [s.layers[0] for s in layers]
    weights = [layer.W.get() for layer in sublayers]
    out_exp = np.concatenate([np.dot(w, inp.get()) for (w, inp) in zip(weights, inputs)])

    assert allclose_with_out(out, out_exp, atol=1e-3)

    err_lst = [dtypeu(np.random.random((nout, batch_size))) for nout in nouts]
    err_concat = np.concatenate(err_lst)
    merge.bprop(be.array(err_concat))
    dW_exp_lst = [np.dot(err, inp.get().T) for (err, inp) in zip(err_lst, inputs)]

    for layer, dW_exp in zip(sublayers, dW_exp_lst):
        assert allclose_with_out(layer.dW.get(), dW_exp)
    return


def test_concat_sequence_l1_l1(backend_default, allrand_args, deltas_buffer):
    # test two linear layers that are merged with concat
    dtypeu = np.float32
    w_rng, rngmax = allrand_args
    # Diff size input steps
    nin = 128
    steps = [32, 64]
    nout = 256
    batch_size = 16
    NervanaObject.be.bsz = batch_size
    be = NervanaObject.be

    init_unif = Uniform(low=w_rng[0], high=w_rng[1])
    layers = [Sequential(Affine(nout=nout, init=init_unif)) for _ in (0, 1)]
    inputs = [be.array(dtypeu(np.random.random((nin, batch_size * step))))
              for step in steps]
    merge = MergeMultistream(layers, merge="recurrent")
    assert(len(inputs) == len(layers))
    merge.configure(inputs)
    merge.allocate()

    merge.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    merge.set_deltas(deltas_buffer)

    out = merge.fprop(inputs).get()

    sublayers = [s.layers[0] for s in layers]
    weights = [layer.W.get() for layer in sublayers]
    out_exp = np.concatenate([np.dot(w, inp.get()) for (w, inp) in zip(weights, inputs)], axis=1)

    assert allclose_with_out(out, out_exp, atol=1e-3)

    err_lst = [dtypeu(np.random.random((nout, batch_size * step))) for step in steps]
    err_concat = be.array(np.concatenate(err_lst, axis=1))
    merge.bprop(err_concat)
    dW_exp_lst = [np.dot(err, inp.get().T) for (err, inp) in zip(err_lst, inputs)]

    for layer, dW_exp in zip(sublayers, dW_exp_lst):
        assert allclose_with_out(layer.dW.get(), dW_exp)
    return
