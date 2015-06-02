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
Neural network layers involving the application of convolutional filters.
"""

import logging
from neon.backends.cpu import CPU
from neon.layers.layer import WeightLayer
from neon.util.param import opt_param, req_param

logger = logging.getLogger(__name__)


class ConvLayer(WeightLayer):

    """
    Convolutional layer.
    """

    def __init__(self, **kwargs):
        self.is_local = True
        super(ConvLayer, self).__init__(**kwargs)
        opt_param(self, ['local_conv'], False)

    def initialize(self, kwargs):
        super(ConvLayer, self).initialize(kwargs)
        self.initialize_local()
        if self.pad != 0 and isinstance(self.backend, CPU):
            raise NotImplementedError('pad != 0, for CPU backend in ConvLayer')

        self.allocate_output_bufs()

        opt_param(self, ['shared_bias'], True)
        if self.shared_bias:
            self.bias_shape = (self.nofm, 1)
            self.bias_expand = self.backend.empty((self.nout, 1),
                                                  dtype=self.weight_dtype)
        else:
            self.bias_shape = (self.nout, 1)

        if self.shared_bias or self.batch_norm:
            self.bias_expand_view = self.bias_expand.reshape(
                (self.nofm, self.ofmsize))
            self.pre_act_view = self.pre_act.reshape(
                (self.nofm, self.ofmsize * self.batch_size))

        self.allocate_param_bufs()

        opt_param(self, ['prodbuf', 'bpropbuf', 'updatebuf'], None)
        if isinstance(self.backend, CPU):
            self.prodbuf = self.backend.empty((self.nofm, self.batch_size))
            self.bpropbuf = self.backend.empty((self.fsize, self.batch_size))
            self.updatebuf = self.backend.empty(self.weights.shape)

        if self.backend.__module__ == 'neon.backends.gpu':
            self.conv_params = self.backend.ng.conv_layer(
                N=self.batch_size, C=self.nifm, K=self.nofm,
                D=1, H=self.ifmshape[0], W=self.ifmshape[1], T=1,
                R=self.fshape[0], S=self.fshape[1],
                pad_d=0, pad_h=self.pad, pad_w=self.pad,
                str_d=1, str_h=self.stride, str_w=self.stride,
                grid_P=0, grid_Q=0,
                dtype=self.weight_dtype)
            self.prodbuf = self.bpropbuf = self.updatebuf = self.conv_params

    def set_weight_shape(self):
        if hasattr(self, 'local_conv') and self.local_conv:
            weight_shape = (self.fsize * self.ofmsize, self.nofm)
        else:
            weight_shape = (self.fsize, self.nofm)
        opt_param(self, ['weight_shape'], weight_shape)

    def fprop(self, inputs):
        self.backend.fprop_conv(out=self.pre_act, inputs=inputs,
                                weights=self.weights, ofmshape=self.ofmshape,
                                ofmsize=self.ofmsize,
                                ofmlocs=self.ofmlocs, ifmshape=self.ifmshape,
                                links=self.links, nifm=self.nifm,
                                padding=self.negpad, stride=self.stride,
                                ngroups=1, fpropbuf=self.prodbuf,
                                local=self.local_conv)
        if self.use_biases is True:
            if self.shared_bias:
                self.backend.add(self.pre_act_view, self.biases,
                                 out=self.pre_act_view)
            else:
                self.backend.add(self.pre_act, self.biases, out=self.pre_act)

        if self.batch_norm:
            self.bn.fprop_func(self.backend,
                               self.pre_act_view, self.pre_act_view)

        self.activation.fprop_func(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        inputs = self.prev_layer.output
        self.activation.bprop_func(self.backend, self.pre_act, error,
                                   self.skip_act)

        upm = self.utemp if self.accumulate else self.updates
        u_idx = 0
        if self.batch_norm:
            error_view = error.reshape(self.pre_act_view.shape)
            self.bn.bprop_func(self.backend, self.pre_act_view, error_view,
                               self.skip_act)
            u_idx = 2

        if self.deltas is not None:
            self.backend.bprop_conv(out=self.deltas, weights=self.weights,
                                    deltas=error, ofmshape=self.ofmshape,
                                    ofmsize=self.ofmsize,
                                    ofmlocs=self.ofmlocs,
                                    ifmshape=self.ifmshape, links=self.links,
                                    padding=self.negpad, stride=self.stride,
                                    nifm=self.nifm, ngroups=1,
                                    bpropbuf=self.bpropbuf,
                                    local=self.local_conv)
        self.backend.update_conv(out=upm[u_idx], inputs=inputs,
                                 weights=self.weights, deltas=error,
                                 ofmshape=self.ofmshape,
                                 ofmsize=self.ofmsize,
                                 ofmlocs=self.ofmlocs,
                                 ifmshape=self.ifmshape, links=self.links,
                                 nifm=self.nifm, padding=self.negpad,
                                 stride=self.stride, ngroups=1,
                                 fwidth=self.fshape[-1],
                                 updatebuf=self.updatebuf,
                                 local=self.local_conv,
                                 layer=self)

        if self.use_biases is True:
            # We can't reshape the error buffer since it might be global buffer
            if self.shared_bias:
                self.backend.sum(error, axes=1, out=self.bias_expand)
                self.backend.sum(self.bias_expand_view, axes=1,
                                 out=upm[u_idx+1])
            else:
                self.backend.sum(error, axes=1, out=upm[u_idx+1])

        if self.accumulate:
            self.backend.add(upm[u_idx], self.updates[u_idx],
                             out=self.updates[u_idx])
            if self.use_biases is True:
                self.backend.add(upm[1], self.updates[1], out=self.updates[1])


class SubConvLayer(ConvLayer):
    """
    Convolutional layer with workaround for modulo 16 number of filters
    """
    def __init__(self, **kwargs):
        super(SubConvLayer, self).__init__(**kwargs)
        req_param(self, ['endidx'])

    def initialize(self, kwargs):
        super(SubConvLayer, self).initialize(kwargs)
        self.rowendidx = self.endidx * self.ofmsize
        self.bigoutput = self.output
        self.suboutput = self.backend.zeros(
            (self.rowendidx, self.batch_size), self.output_dtype)

    def fprop(self, inputs):
        self.output = self.bigoutput
        super(SubConvLayer, self).fprop(inputs)
        self.suboutput.fill(0.0)
        self.suboutput[:] = self.output[:self.rowendidx, :]
        self.output = self.suboutput

    def bprop(self, error):
        inputs = self.prev_layer.output
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act[:self.rowendidx],
                                  out=self.pre_act[:self.rowendidx])
        self.pre_act[self.rowendidx:] = 0.
        self.weights[self.endidx:] = 0.
        error = self.pre_act
        if self.deltas is not None:
            self.backend.bprop_conv(out=self.deltas, weights=self.weights,
                                    deltas=error, ofmshape=self.ofmshape,
                                    ofmsize=self.ofmsize,
                                    ofmlocs=self.ofmlocs,
                                    ifmshape=self.ifmshape, links=self.links,
                                    padding=self.pad, stride=self.stride,
                                    nifm=self.nifm, ngroups=1,
                                    bpropbuf=self.bpropbuf,
                                    local=self.local_conv)

        upm = self.utemp if self.accumulate else self.updates

        self.backend.update_conv(out=upm[0], inputs=inputs,
                                 weights=self.weights, deltas=error,
                                 ofmshape=self.ofmshape,
                                 ofmsize=self.ofmsize,
                                 ofmlocs=self.ofmlocs,
                                 ifmshape=self.ifmshape, links=self.links,
                                 nifm=self.nifm, padding=self.pad,
                                 stride=self.stride, ngroups=1,
                                 fwidth=self.fshape[-1],
                                 updatebuf=self.updatebuf,
                                 local=self.local_conv,
                                 layer=self)

        if self.use_biases is True:
            self.backend.sum(error, axes=1, out=upm[1])
        if self.accumulate:
            self.backend.add(upm[0], self.updates[0], out=self.updates[0])
            if self.use_biases is True:
                self.backend.add(upm[1], self.updates[1], out=self.updates[1])
