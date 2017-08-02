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
Generalized gradient testing applied to mlp/linear layer
"""

from __future__ import print_function
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import Linear
from neon.initializers.initializer import Gaussian
from grad_funcs import general_gradient_comp


# add a reset methods to the layer classes
# this is used to reset the layer so that
# running fprop and bprop multiple times
# produces repeatable results
# some layers just need the function defined
class LinearWithReset(Linear):
    def reset(self):
        pass


def pytest_generate_tests(metafunc):
    # main test generator
    # generates the parameter combos for
    # the tests based on whether the
    # "--all" option is given to py.test
    # that option is added in conftest.py

    # global parameter
    if metafunc.config.option.all:
        bsz_rng = [16, 32, 64]
    else:
        bsz_rng = [16]

    # mlp tests
    if 'mlpargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            nin_rng = [1, 2, 3, 10]
            nout_rng = [1, 2, 3, 10]
        else:
            nin_rng = [1, 2]
            nout_rng = [3]

        # generate the params lists
        fargs = itt.product(nin_rng, nout_rng, bsz_rng)

        # parameterize the call for all test functions
        # with mlpargs as an argument
        metafunc.parametrize("mlpargs", fargs)


def test_mlp(backend_cpu64, mlpargs):
    nin, nout, batch_size = mlpargs
    # run the gradient check on an mlp
    batch_size = batch_size
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    init = Gaussian()
    layer = LinearWithReset(nout=nout, init=init)
    inp = np.random.randn(nin, batch_size)

    epsilon = 1.0e-5
    pert_frac = 0.1  # test 10% of the inputs
    # select pert_frac fraction of inps to perturb
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]

    (max_abs, max_rel) = general_gradient_comp(layer,
                                               inp,
                                               epsilon=epsilon,
                                               pert_inds=pert_inds)
    assert max_abs < 1.0e-7


@pytest.mark.xfail(reason="Precision differences with MKL backend. #914")
def test_mlp_mkl(backend_mkl, mlpargs):
    nin, nout, batch_size = mlpargs
    # run the gradient check on an mlp
    batch_size = batch_size
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    init = Gaussian()
    layer = LinearWithReset(nout=nout, init=init)
    inp = np.random.randn(nin, batch_size)

    epsilon = 1.0e-5
    pert_frac = 0.1  # test 10% of the inputs
    # select pert_frac fraction of inps to perturb
    pert_cnt = int(np.ceil(inp.size * pert_frac))
    pert_inds = np.random.permutation(inp.size)[0:pert_cnt]

    (max_abs, max_rel) = general_gradient_comp(layer,
                                               inp,
                                               epsilon=epsilon,
                                               pert_inds=pert_inds)
    assert max_abs < 1.0e-7
