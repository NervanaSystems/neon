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


nin = 3
nout = 2
batch_size = 10


def check_fprop(layer, backend):
    inputs = backend.ones((nin, batch_size))
    layer.fprop(inputs)
    assert layer.output.shape == (nout, batch_size)


def check_bprop(layer, backend):
        errors = backend.ones((nout, batch_size))
        output = backend.ones((nin, batch_size))

        # initialize deltas since they are not set
        # by the layer initialize method.
        layer.deltas = output

        # layers should be refactored to remove references
        # to external layers. inputs can be cached during
        # fprop.
        class PreviousLayer(object):

            def __init__(self, output):
                self.is_data = True
                self.output = output

        layer.prev_layer = PreviousLayer(output)
        layer.bprop(errors)
        assert layer.deltas.shape == (nin, batch_size)


class TestFullyConnectedLayer(object):

    def create_layer(self, backend):
        layer = FCLayer(nin=nin,
                        nout=nout,
                        batch_size=batch_size,
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
