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
Generalized gradient testing applied to pooling layer
"""
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import Pooling
from grad_funcs import general_gradient_comp


# add a reset methods to the layer classes
# this is used to reset the layer so that
# running fprop and bprop multiple times
# produces repeatable results
# some layers just need the function defined
class PoolingWithReset(Pooling):
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
        bsz_rng = [16, 32]
    else:
        bsz_rng = [16]

    if 'poolargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [5, 8]
            nifm_rng = [1, 2, 4]
            fs_rng = [2, 3, 4]
        else:
            nin_rng = [10]
            nifm_rng = [1, 5]
            fs_rng = [2, 3]
        op_rng = ["max", "avg"]
        fargs = itt.product(nin_rng, nifm_rng, fs_rng, bsz_rng, op_rng)
        metafunc.parametrize("poolargs", fargs)


# -- pooling tests --
def test_pooling(backend_cpu64, poolargs):
    nin, nifm, fshape, batch_size, op = poolargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1.0e-5
    # make sure perturnationc an never change the max element
    inp = np.arange(sz) * 2.5 * epsilon
    # shuffle
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))

    lshape = (nifm, nin, nin)
    layer = PoolingWithReset(fshape, op=op)

    pert_frac = 0.1  # test 10% of the inputs
    # select pert_frac fraction of inps to perturb
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]

    (max_abs, max_rel) = general_gradient_comp(layer,
                                               inp,
                                               epsilon=epsilon,
                                               lshape=lshape,
                                               pert_inds=pert_inds,
                                               pooling=True)
    assert max_abs < 1.0e-7


@pytest.mark.xfail(reason="Known MKL bug. See #913. Also impacted by #914")
def test_pooling_mkl(backend_mkl, poolargs):
    nin, nifm, fshape, batch_size, op = poolargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    sz = nin * nin * nifm * batch_size
    epsilon = 1.0e-5
    # make sure perturnationc an never change the max element
    inp = np.arange(sz) * 2.5 * epsilon
    # shuffle
    np.random.shuffle(inp)
    inp = inp.reshape((nin * nin * nifm, batch_size))

    lshape = (nifm, nin, nin)
    layer = PoolingWithReset(fshape, op=op)

    pert_frac = 0.1  # test 10% of the inputs
    # select pert_frac fraction of inps to perturb
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]

    (max_abs, max_rel) = general_gradient_comp(layer,
                                               inp,
                                               epsilon=epsilon,
                                               lshape=lshape,
                                               pert_inds=pert_inds,
                                               pooling=True)
    assert max_abs < 1.0e-7
