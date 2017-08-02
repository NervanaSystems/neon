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

"""
Generalized gradient testing applied to lrn layer
"""
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import LRN
from grad_funcs import general_gradient_comp


# add a reset methods to the layer classes
# this is used to reset the layer so that
# running fprop and bprop multiple times
# produces repeatable results
# some layers just need the function defined
class LRNWithReset(LRN):
    def reset(self):
        self.nglayer = None


def pytest_generate_tests(metafunc):
    # main test generator
    # generates the parameter combos for
    # the tests based on whether the
    # "--all" option is given to py.test
    # that option is added in conftest.py

    # global parameter
    if metafunc.config.option.all:
        bsz_rng = [8]  # [16, 32]
    else:
        bsz_rng = [8]

    if 'lrnargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [5, 6]
            nifm_rng = [1, 2, 4]
            fs_rng = [3, 5]
        else:
            nin_rng = [2]
            nifm_rng = [7, 20]
            fs_rng = [3, 5]
        fargs = itt.product(nin_rng, nifm_rng, fs_rng, bsz_rng)
        metafunc.parametrize("lrnargs", fargs)


def test_lrnorm(backend_cpu64, lrnargs):
    nin, nifm, fshape, batch_size = lrnargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1.0e-5
    # make sure perturbation can never change the max element
    inp = np.arange(sz) * 2.5 * epsilon
    # shuffle
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))

    lshape = (nifm, nin, nin)
    layer = LRNWithReset(depth=fshape, ascale=1.25e-4, bpower=0.75)

    pert_frac = 0.1  # test 10% of the inputs
    # select pert_frac fraction of inps to perturb
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]

    (max_abs, max_rel) = general_gradient_comp(layer,
                                               inp,
                                               epsilon=epsilon,
                                               lshape=lshape,
                                               pert_inds=pert_inds)
    assert max_abs < 1.0e-6


def test_lrn_large_inp(backend_cpu64, deltas_buffer):
    # adding an extra test with a large inp at 1 location=
    # LRN is not very sensitive to small inputs
    nin = 2
    nifm = 16
    depth = 5
    batch_size = 64

    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    be = NervanaObject.be

    shape = (nifm * nin * nin, batch_size)
    shape_full = (nifm, nin, nin, batch_size)

    inp_rng = 1.0e5
    epsilon = 10.0
    np.random.seed(1234)

    ind_pert = (8, 0, 0, 16)
    ind_pert2 = np.ravel_multi_index(ind_pert[0:3], shape_full[0:3])
    ind_pert = (ind_pert2, ind_pert[-1])

    inp = np.zeros(shape)
    inp[ind_pert] = inp_rng
    inpa = be.array(inp)

    lshape = shape_full[0:3]
    layer = LRNWithReset(depth=depth, ascale=1.25e-4, bpower=0.75)

    layer.configure(lshape)
    if layer.owns_delta:
        layer.prev_layer = True
    layer.allocate()

    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    loss_scale = np.ones(inpa.shape)
    layer.fprop(inpa).get()
    bprop_deltas = layer.bprop(be.array(loss_scale)).get()
    bprop_delta = bprop_deltas[ind_pert]

    # pos shift
    inp_p = inp.copy()
    inp_p[ind_pert] += epsilon
    inp_m = inp.copy()
    inp_m[ind_pert] -= epsilon

    out_p = layer.fprop(be.array(inp_p)).get()[ind_pert]
    out_m = layer.fprop(be.array(inp_m)).get()[ind_pert]

    grad_est = 0.5 / float(epsilon) * (out_p - out_m)
    assert np.abs(grad_est - bprop_delta) < 1e-12


@pytest.mark.xfail(reason="Precision differences with MKL backend. #914")
def test_lrnorm_mkl(backend_mkl, lrnargs):
    nin, nifm, fshape, batch_size = lrnargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1.0e-5
    # make sure perturbation can never change the max element
    inp = np.arange(sz) * 2.5 * epsilon
    # shuffle
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))

    lshape = (nifm, nin, nin)
    layer = LRNWithReset(depth=fshape, ascale=1.25e-4, bpower=0.75)

    pert_frac = 0.1  # test 10% of the inputs
    # select pert_frac fraction of inps to perturb
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]

    (max_abs, max_rel) = general_gradient_comp(layer,
                                               inp,
                                               epsilon=epsilon,
                                               lshape=lshape,
                                               pert_inds=pert_inds)
    assert max_abs < 1.0e-6


@pytest.mark.xfail(reason="Precision differences with MKL backend. #914")
def test_lrn_large_inp_mkl(backend_mkl, deltas_buffer):
    # adding an extra test with a large inp at 1 location=
    # LRN is not very sensitive to small inputs
    nin = 2
    nifm = 16
    depth = 5
    batch_size = 64

    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    be = NervanaObject.be

    shape = (nifm * nin * nin, batch_size)
    shape_full = (nifm, nin, nin, batch_size)

    inp_rng = 1.0e5
    epsilon = 10.0
    np.random.seed(1234)

    ind_pert = (8, 0, 0, 16)
    ind_pert2 = np.ravel_multi_index(ind_pert[0:3], shape_full[0:3])
    ind_pert = (ind_pert2, ind_pert[-1])

    inp = np.zeros(shape)
    inp[ind_pert] = inp_rng
    inpa = be.array(inp)

    lshape = shape_full[0:3]
    layer = LRNWithReset(depth=depth, ascale=1.25e-4, bpower=0.75)

    layer.configure(lshape)
    if layer.owns_delta:
        layer.prev_layer = True
    layer.allocate()

    layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    layer.set_deltas(deltas_buffer)

    loss_scale = np.ones(inpa.shape)
    layer.fprop(inpa).get()
    bprop_deltas = layer.bprop(be.array(loss_scale)).get()
    bprop_delta = bprop_deltas[ind_pert]

    # pos shift
    inp_p = inp.copy()
    inp_p[ind_pert] += epsilon
    inp_m = inp.copy()
    inp_m[ind_pert] -= epsilon

    out_p = layer.fprop(be.array(inp_p)).get()[ind_pert]
    out_m = layer.fprop(be.array(inp_m)).get()[ind_pert]

    grad_est = 0.5 / float(epsilon) * (out_p - out_m)
    assert np.abs(grad_est - bprop_delta) < 1e-12
