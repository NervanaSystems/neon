#!/usr/bin/env python
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

from nose.plugins.attrib import attr

from neon.backends.cpu import CPU
from neon.layers import FCLayer
from neon.params import IdentityValGen
from neon.util.testing import assert_tensor_equal

nin = 3
nout = 2
batch_size = 10


def check_fprop(layer, backend):
    inputs = backend.ones((nin, batch_size))
    output = backend.ones((nout, batch_size))
    layer.fprop(inputs)
    assert_tensor_equal(layer.output, output)


def check_bprop(layer, backend):
        errors = backend.ones((nout, batch_size))
        deltas = backend.zeros((nin, batch_size))
        deltas[:2] = backend.ones((nout, batch_size))

        # initialize deltas since they are not set
        # by the layer initialize method.
        layer.deltas = backend.ones((nin, batch_size))

        # layers should be refactored to remove references
        # to external layers. inputs can be cached during
        # fprop.
        class PreviousLayer(object):

            def __init__(self):
                self.is_data = True
                self.output = backend.ones((nin, batch_size))

        layer.prev_layer = PreviousLayer()
        layer.bprop(errors)
        assert_tensor_equal(layer.deltas, deltas)


class TestFullyConnectedLayer(object):

    def create_layer(self, backend):
        weight_init = IdentityValGen()
        layer = FCLayer(nin=nin,
                        nout=nout,
                        batch_size=batch_size,
                        weight_init=weight_init,
                        backend=backend)
        layer.set_weight_shape()
        layer.initialize([])
        return layer

    def test_cpu_fprop(self):
        backend = CPU(rng_seed=0)
        layer = self.create_layer(backend=backend)
        check_fprop(layer, backend)

    def test_cpu_bprop(self):
        backend = CPU(rng_seed=0)
        layer = self.create_layer(backend=backend)
        check_bprop(layer, backend)

    @attr('cuda')
    def test_gpu_fprop(self):
        from neon.backends.cc2 import GPU
        backend = GPU(rng_seed=0)
        layer = self.create_layer(backend=backend)
        check_fprop(layer, backend)

    @attr('cuda')
    def test_gpu_bprop(self):
        from neon.backends.cc2 import GPU
        backend = GPU(rng_seed=0)
        layer = self.create_layer(backend=backend)
        check_bprop(layer, backend)
