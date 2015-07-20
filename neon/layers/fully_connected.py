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
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging
from neon.layers.layer import WeightLayer
from neon.util.param import opt_param

logger = logging.getLogger(__name__)


class FCLayer(WeightLayer):
    """
    Fully connected feed-forward neural network layer.

    Attributes:
        nin (integer): number of input connections (from previous layer).
        nout (integer): number of output activations.
    """
    def initialize(self, kwargs):
        super(FCLayer, self).initialize(kwargs)

        self.allocate_output_bufs()

        # Change the weight shape for FC layers to use weird trick
        # Can't do this in set_weight_shape because backend not defined yet
        self.repl_inputs, self.frag_deltas = None, None
        self.share_tensors = None
        if self.backend.is_dist:
            assert self.nout % self.backend.num_dev == 0
            self.output.ptype = 'replica'
            self.weight_shape = (self.nout / self.backend.num_dev, self.nin)
            self.bias_shape = (self.nout / self.backend.num_dev, 1)
            self.set_split_share_tensors()
            if self.prev_layer.is_data:
                self.prev_layer.nodatapar = True

        self.allocate_param_bufs()

    def set_split_share_tensors(self):
        """
        Allocates space for transitioning between a local layer and a
        non-local layer (fragmented activations to replicated activations)
        Necessary because in-place transpose not currently implemented for
        multi-gpu backend
        """
        assert self.backend.is_dist
        if not self.prev_layer.is_local:
            return

        self.repl_inputs = self.backend.empty(
            self.prev_layer.out_shape, self.prev_layer.output_dtype)
        self.frag_deltas = self.backend.allocate_fragment(
            self.delta_shape, self.deltas_dtype)
        tmp1 = self.backend.empty(
            self.repl_inputs.shape[::-1], self.prev_layer.output_dtype)
        tmp2 = self.backend.empty(
            self.frag_deltas.shape[::-1], self.deltas_dtype)
        self.share_tensors = (tmp1, tmp2)

    def set_weight_shape(self):
        opt_param(self, ['weight_shape'], (self.nout, self.nin))
        opt_param(self, ['bias_shape'], (self.nout, 1))

    def make_mempool(self):
        """
        We need two buffers here, one for temp storage of updates during
        fprop, and one for bprop
        """
        tmp1 = self.backend.empty((self.weight_shape[0], self.out_shape[1]),
                                  self.output_dtype)
        tmp2 = self.backend.empty((self.weight_shape[1], self.delta_shape[1]),
                                  self.deltas_dtype)
        self.mempool = (tmp1, tmp2)

    def share_acts(self, inputs):
        if self.repl_inputs is None:
            return inputs
        # Weird trick, transitioning from fragments to replicas
        self.backend.share_activations(inputs, self.repl_inputs,
                                       self.share_tensors)
        return self.repl_inputs

    def get_deltas_buf(self):
        if self.frag_deltas is None or self.prev_layer.is_data:
            return self.deltas
        # Weird trick, transitioning from replicas to fragments
        self.backend.split_activations(self.deltas, self.frag_deltas,
                                       self.share_tensors)
        return self.frag_deltas

    def fprop(self, inputs):
        self.backend.fprop_fc(out=self.pre_act, inputs=inputs,
                              weights=self.weights, layer=self)
        if self.use_biases is True:
            self.backend.add_fc_bias(self.pre_act, self.biases)

        if self.batch_norm:
            self.bn.fprop_func(self.backend, self.pre_act, self.pre_act)
        self.activation.fprop_func(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        # Use the replica activations if they were previously shared
        if self.repl_inputs is not None:
            inputs = self.repl_inputs
        else:
            inputs = self.prev_layer.output

        self.activation.bprop_func(self.backend, self.pre_act, error,
                                   self.skip_act)

        upm = self.utemp if self.accumulate else self.updates
        u_idx = 0
        if self.batch_norm:
            self.bn.bprop_func(self.backend, self.pre_act, error,
                               self.skip_act)
            u_idx = 2

        if self.deltas is not None:
            self.backend.bprop_fc(out=self.deltas, weights=self.weights,
                                  deltas=error, layer=self)

        self.backend.update_fc(out=upm[u_idx], inputs=inputs,
                               deltas=error, layer=self)

        if self.use_biases is True:
            self.backend.update_fc_bias(error, out=upm[u_idx+1])

        if self.backend.is_dist:
            self.backend.synchronize()
            self.backend.redsynchronize()

        if self.accumulate:
            self.backend.add(upm[u_idx], self.updates[u_idx],
                             out=self.updates[u_idx])
            if self.use_biases is True:
                self.backend.add(upm[u_idx+1], self.updates[u_idx+1],
                                 out=self.updates[u_idx+1])
