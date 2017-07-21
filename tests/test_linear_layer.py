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
Test of the mlp/linear layer
'''
import itertools as itt
import numpy as np

from neon import NervanaObject
from neon.initializers.initializer import Uniform
from neon.layers.layer import Linear
from utils import allclose_with_out


def pytest_generate_tests(metafunc):
    if metafunc.config.option.all:
        bsz_rng = [16, 32, 64]
    else:
        bsz_rng = [128]

    if 'basic_linargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [1, 2, 1023, 1024, 1025]
            nout_rng = [1, 4, 1023, 1024, 1025]
        else:
            nin_rng = [4, 32]
            nout_rng = [3, 33]
        fargs = itt.product(nin_rng, nout_rng, bsz_rng)
        metafunc.parametrize('basic_linargs', fargs)

    if 'allrand_args' in metafunc.fixturenames:
        fargs = []
        eps = np.finfo(np.float32).eps
        # weight ranges
        w_rng = [[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]
        if metafunc.config.option.all:
            rng_max = [eps, eps * 10, 1.0, 2048.0, 1.0e6, 1.0e10]
        else:
            rng_max = [eps, 1.0, 1.0e10]
        fargs = itt.product(w_rng, rng_max)
        metafunc.parametrize('allrand_args', fargs)


def test_linear_zeros(backend_default, basic_linargs, deltas_buffer):
    # basic sanity check with 0 weights random inputs
    nin, nout, batch_size = basic_linargs
    NervanaObject.be.bsz = batch_size

    dtypeu = np.float32

    init_unif = Uniform(low=0.0, high=0.0)
    layer = Linear(nout=nout, init=init_unif)
    inp = layer.be.array(dtypeu(np.random.random((nin, batch_size))))
    layer.configure(nin)
    layer.prev_layer = True  # Hack to force delta buffer allocation
    layer.allocate()

    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    out = layer.fprop(inp).get()

    assert np.min(out) == 0.0 and np.max(out) == 0.0

    err = dtypeu(np.zeros((nout, batch_size)))
    deltas = layer.bprop(layer.be.array(err)).get()
    assert np.min(deltas) == 0.0 and np.max(deltas) == 0.0

    dw = layer.dW.get()
    assert np.min(dw) == 0.0 and np.max(dw) == 0.0

    return


def test_linear_ones(backend_default, basic_linargs, deltas_buffer):
    # basic sanity check with all ones on the inputs
    # and weights, check that each row in output
    # is the sum of the weights for that output
    # this check will confirm that the correct number
    # of operations is being run
    nin, nout, batch_size = basic_linargs
    NervanaObject.be.bsz = batch_size

    dtypeu = np.float32

    init_unif = Uniform(low=1.0, high=1.0)
    layer = Linear(nout=nout, init=init_unif)
    inp = layer.be.array(dtypeu(np.ones((nin, batch_size))))
    layer.configure(nin)
    layer.prev_layer = True  # Hack to force delta buffer allocation
    layer.allocate()

    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    out = layer.fprop(inp).get()
    w = layer.W.get()
    sums = np.sum(w, 1).reshape((nout, 1)) * np.ones((1, batch_size))

    # for larger layers need to estimate numerical precision
    # atol = est_mm_prec(w, inp.get())
    assert allclose_with_out(sums, out, atol=0.0, rtol=0.0), \
        '%e' % np.max(np.abs(out - sums))
    return


def test_all_rand(backend_default, allrand_args, deltas_buffer):
    # test with random weights and random inputs
    dtypeu = np.float32
    w_rng, rngmax = allrand_args
    inp_rng = [0.0, rngmax]
    nin = 1024
    nout = 2048
    batch_size = 16
    NervanaObject.be.bsz = batch_size

    init_unif = Uniform(low=w_rng[0], high=w_rng[1])
    layer = Linear(nout=nout, init=init_unif)
    inp = np.random.random((nin, batch_size))
    inp *= inp_rng[1] - inp_rng[0]
    inp += inp_rng[0]
    inp = inp.astype(dtypeu)
    layer.configure(nin)
    layer.prev_layer = True  # Hack to force delta buffer allocation
    layer.allocate()

    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    out = layer.fprop(layer.be.array(inp)).get()
    w = layer.W.get()

    # the expected output using numpy
    out_exp = np.dot(w, inp)

    # for larger layers need to estimate numerical precision
    atol = 2 * est_mm_prec(w, inp, ntrials=1)
    assert allclose_with_out(out_exp, out, atol=atol, rtol=0.0), \
        '%e %e' % (np.max(np.abs(out - out_exp)), atol)

    err = np.random.random((nout, batch_size))
    err = err * (inp_rng[1] - inp_rng[0]) + inp_rng[0]
    err = err.astype(dtypeu)
    deltas = layer.bprop(layer.be.array(err)).get()
    dw = layer.dW.get()

    deltas_exp = np.dot(w.T, err)
    atol = 2 * est_mm_prec(w.T, err, ntrials=1)
    assert allclose_with_out(deltas_exp, deltas, atol=atol, rtol=0.0), \
        '%e %e' % (np.max(np.abs(deltas_exp - deltas)), atol)

    dw_exp = np.dot(err, inp.T)
    atol = 2 * est_mm_prec(err, inp.T, ntrials=1)
    assert allclose_with_out(dw_exp, dw, atol=atol, rtol=0.0), \
        '%e %e' % (np.max(np.abs(dw_exp - dw)), atol)

    return


# permute mm indicies to change order of computations
# to estimate numerical precision
# this is a rough estimate
def est_mm_prec(A, B, ntrials=1):
    A64 = np.float64(A)
    B64 = np.float64(B)
    gt = np.dot(A64, B64)
    max_err = -1.0
    for trial in range(ntrials):
        inds = np.random.permutation(A.shape[1])
        # this method gives better estimate of precision tolerances
        # but takes too long to run
        # for i in range(A.shape[0]):
        #    for j in range(B.shape[1]):
        #        c = np.sum(np.multiply(A[i,inds], B[inds,j]))
        #        max_err = max( max_err, np.abs(c-gt[i,j]))

        # need to scale this by 10 for comparison
        C = np.dot(A[:, inds], B[inds, :])
        dd = np.float32(gt - C)
        # just save the worst case from each iteration
        max_err = max(max_err, np.max(np.abs(dd)))
    # need to scale the np.dot results by 10 to
    # match the np.sum(np.multiply()) values
    max_err *= 10.0
    return max_err
