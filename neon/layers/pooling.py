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
Single neural network layer that performs a pooling or subsampling operation.
"""

import logging
from neon.backends.cpu import CPU
from neon.layers.layer import Layer, WeightLayer
from neon.util.param import req_param, opt_param

logger = logging.getLogger(__name__)


class PoolingLayer(Layer):
    """
    Generic pooling layer, with configurable operation type.

    Attributes:
        op (string): The type of pooling to perform.  We currently implement
                     "max", "avg", or "l2".
    """
    def __init__(self, **kwargs):
        self.is_local = True
        super(PoolingLayer, self).__init__(**kwargs)
        req_param(self, ['op'])
        opt_param(self, ['maxout'], 1)

    def initialize(self, kwargs):
        super(PoolingLayer, self).initialize(kwargs)
        self.pooling = True
        self.tempbuf = None
        self.initialize_local()
        self.allocate_output_bufs()
        if self.backend.__module__ == 'neon.backends.gpu':
            self.pool_params = self.backend.ng.pool_layer(
                dtype=self.output_dtype,
                op=self.op, N=self.batch_size, C=self.nifm,
                D=1, H=self.ifmshape[0], W=self.ifmshape[1], J=self.maxout,
                T=1, R=self.fshape[0], S=self.fshape[1],
                pad_d=0, pad_h=self.pad, pad_w=self.pad,
                str_d=1, str_h=self.stride, str_w=self.stride)
            self.outputbuf = self.deltasbuf = self.pool_params

    def fprop(self, inputs):
        self.backend.fprop_pool(out=self.output, inputs=inputs, op=self.op,
                                ofmshape=self.ofmshape,
                                ofmsize=self.ofmsize,
                                ofmlocs=self.tempbuf,
                                fshape=self.fshape, ifmshape=self.ifmshape,
                                links=self.links, nifm=self.nifm, padding=0,
                                stride=self.stride, fpropbuf=self.outputbuf)

    def bprop(self, error):
        inputs = self.prev_layer.output
        if self.deltas is not None:
            self.backend.bprop_pool(out=self.deltas, fouts=self.output,
                                    inputs=inputs, deltas=error, op=self.op,
                                    ofmshape=self.ofmshape,
                                    ofmsize=self.ofmsize,
                                    ofmlocs=self.tempbuf, fshape=self.fshape,
                                    fpsize=self.fpsize,
                                    ifmshape=self.ifmshape, links=self.links,
                                    nifm=self.nifm, padding=0,
                                    stride=self.stride,
                                    bpropbuf=self.deltasbuf)


class CrossMapPoolingLayer(WeightLayer):
    """
    Pool input feature maps by computing a weighted sum of
    corresponding spatial locations across maps. This is
    equivalent to a 1x1 convolution.
    """
    def __init__(self, **kwargs):
        self.is_local = True
        self.fshape = (1, 1)
        super(CrossMapPoolingLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        super(CrossMapPoolingLayer, self).initialize(kwargs)
        req_param(self, ['nofm'])

        self.initialize_local()
        self.allocate_output_bufs()
        self.allocate_param_bufs()
        opt_param(self, ['updatebuf'], None)
        if isinstance(self.backend, CPU):
            self.updatebuf = self.backend.empty((1, 1))

    def set_weight_shape(self):
        opt_param(self, ['weight_shape'], (self.nifm, self.nofm))

    def fprop(self, inputs):
        self.backend.fprop_cmpool(out=self.pre_act, inputs=inputs,
                                  weights=self.weights, ifmshape=self.ifmshape,
                                  ifmsize=self.ifmsize)
        self.activation.fprop_func(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        inputs = self.prev_layer.output
        self.activation.bprop_func(self.backend, self.pre_act, error)
        if self.deltas is not None:
            self.backend.bprop_cmpool(out=self.deltas, weights=self.weights,
                                      deltas=error, ifmshape=self.ifmshape,
                                      ifmsize=self.ifmsize)
        self.backend.update_cmpool(out=self.updates[0], inputs=inputs,
                                   deltas=error, ifmshape=self.ifmshape,
                                   ifmsize=self.ifmsize,
                                   updatebuf=self.updatebuf)
