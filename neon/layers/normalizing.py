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
Neural network layers that rescale or normalize their values (often across some
local neighborhood like neighboring feature maps of a convolutional network).
"""

import logging
from neon.backends.cpu import CPU
from neon.layers.layer import Layer
from neon.util.param import req_param

logger = logging.getLogger(__name__)


class CrossMapResponseNormLayer(Layer):
    """
    CrossMap response normalization.

    Calculates the normalization across feature maps at each pixel point.
    output will be same size as input

    The calculation is output(x,y,C) = input(x,y,C)/normFactor(x,y,C)

    where normFactor(x,y,C) is (1 + alpha * sum_ksize( input(x,y,k)^2 ))^beta

    ksize is the kernel size, so will run over the channel index with no
    padding at the edges of the feature map.  (so for ksize=5, at C=1, we will
    be summing the values of c=0,1,2,3)
    """
    def __init__(self, **kwargs):
        self.is_local = True
        self.stride = 1
        self.fshape = (1, 1)
        super(CrossMapResponseNormLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        req_param(self, ['ksize', 'alpha', 'beta'])
        self.alpha = self.alpha * 1.0 / self.ksize
        super(CrossMapResponseNormLayer, self).initialize(kwargs)
        self.nout = self.nin
        self.ofmshape, self.nofm = self.ifmshape, self.nifm
        self.allocate_output_bufs()
        self.tempbuf = None
        if isinstance(self.backend, CPU) and not self.prev_layer.is_data:
            self.tempbuf = self.backend.empty(
                (self.ifmshape[-2], self.ifmshape[-1], self.batch_size))

    def fprop(self, inputs):
        self.backend.fprop_cmrnorm(out=self.output, inputs=inputs,
                                   ifmshape=self.ifmshape, nifm=self.nifm,
                                   ksize=self.ksize, alpha=self.alpha,
                                   beta=self.beta)

    def bprop(self, error):
        inputs = self.prev_layer.output
        if self.deltas is not None:
            self.backend.bprop_cmrnorm(out=self.deltas, fouts=self.output,
                                       inputs=inputs, deltas=error,
                                       ifmshape=self.ifmshape, nifm=self.nifm,
                                       ksize=self.ksize, alpha=self.alpha,
                                       beta=self.beta, bpropbuf=self.tempbuf)


class LocalContrastNormLayer(CrossMapResponseNormLayer):
    """
    Local contrast normalization.
    """
    def initialize(self, kwargs):
        super(LocalContrastNormLayer, self).initialize(kwargs)
        self.meandiffs = self.backend.empty(self.output.shape)
        self.denoms = self.backend.empty(self.output.shape)

        # Note dividing again is INTENTIONAL, since this is normalized by an
        # area not just a linear dimension
        self.alpha = self.alpha * 1.0 / self.ksize
        if self.stride != 1:
            raise NotImplementedError('stride != 1, in LocalContrastNormLayer')
        if self.ifmshape[-2] != self.ifmshape[-1]:
            raise NotImplementedError('non-square inputs not supported')

    def fprop(self, inputs):
        self.backend.fprop_lcnnorm(out=self.output, inputs=inputs,
                                   meandiffs=self.meandiffs,
                                   denoms=self.denoms, ifmshape=self.ifmshape,
                                   nifm=self.nifm, ksize=self.ksize,
                                   alpha=self.alpha, beta=self.beta)

    def bprop(self, error):
        if self.deltas is not None:
            self.backend.bprop_lcnnorm(out=self.deltas, fouts=self.output,
                                       deltas=error, meandiffs=self.meandiffs,
                                       denoms=self.denoms,
                                       ifmshape=self.ifmshape, nifm=self.nifm,
                                       ksize=self.ksize, alpha=self.alpha,
                                       beta=self.beta)
