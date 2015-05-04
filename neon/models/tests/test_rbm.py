# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
Tests for restricted boltzmann machine (RBM)

- create fake inputs (cudanet class with one small minibatch of 2D data)
- create a fake instance of the RBM class with the model structure from
  yaml replaced by some small weight init / nodes parameters
- precompute the output values we expect for a gradient update and
  numerically compare that we get them.

"""
from nose.plugins.attrib import attr
from nose.tools import nottest
import numpy as np

from neon.layers.boltzmann import RBMLayer
from neon.params import GaussianValGen
from neon.transforms.logistic import Logistic
from neon.transforms.sum_squared import SumSquaredDiffs
from neon.util.testing import assert_tensor_near_equal


@attr('cuda')
class TestCudaRBM:

    def setup(self):

        from neon.backends.cc2 import GPU, GPUTensor

        # TODO: remove randomness from expected target results
        self.be = GPU(rng_seed=0)

        # reusable fake data
        self.inputs = GPUTensor(np.ones((2, 100)))

        # create fake layer
        nin = 2
        conf = {'name': 'testlayer', 'num_nodes': 2,
                'weight_init': GaussianValGen(backend=self.be, loc=0.0,
                                              scale=0.01)}
        lr_params = {'learning_rate': 0.01}
        thislr = {'type': 'gradient_descent', 'lr_params': lr_params}
        activation = Logistic()
        self.layer = RBMLayer(name=conf['name'])
        # create fake cost
        self.cost = SumSquaredDiffs(olayer=self.layer)
        self.layer.initialize({'backend': self.be, 'batch_size': 100,
                               'lrule_init': thislr, 'nin': nin,
                               'nout': conf['num_nodes'],
                               'activation': activation,
                               'weight_init': conf['weight_init']})

    def test_cudanet_positive(self):
        self.layer.positive(self.inputs)
        target = np.array([0.50541031, 0.50804842],
                          dtype='float32')
        assert_tensor_near_equal(self.layer.p_hid_plus.asnumpyarray()[:, 0],
                                 target)

    def test_cudanet_negative(self):
        self.layer.positive(self.inputs)
        self.layer.negative(self.inputs)
        target = np.array([0.50274211,  0.50407821],
                          dtype='float32')
        assert_tensor_near_equal(self.layer.p_hid_minus.asnumpyarray()[:, 0],
                                 target)

    @nottest  # TODO: remove randomness
    def test_cudanet_cost(self):
        self.layer.positive(self.inputs)
        self.layer.negative(self.inputs)
        thecost = self.cost.apply_function(self.inputs)
        target = 106.588943481
        assert_tensor_near_equal(thecost, target)
