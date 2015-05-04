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
Contains neural network layers that are composed of multiple sublayer parts.
"""

import logging
from neon.layers.layer import Layer
from neon.util.compat import range
from neon.util.param import req_param

logger = logging.getLogger(__name__)


class CompositeLayer(Layer):
    """
    Generic parent for Branch and List layer that deals with sublayer list

    Attributes:
        sublayers (list of Layer): Individual layers that make up this
                                   compositional layer.
    """
    def initialize(self, kwargs):
        super(CompositeLayer, self).initialize(kwargs)
        req_param(self, ['sublayers'])
        self.has_params = True
        for subl in self.sublayers:
            subl.initialize(kwargs)

    def __str__(self):
        ret = '{} {}: {} nodes'.format(self.__class__.__name__,
                                       self.name, self.nout)
        ret += ':\n'
        for subl in self.sublayers:
            ret += '\t' + str(subl) + '\n'
        return ret

    def update(self, epoch):
        for subl in self.sublayers:
            subl.update(epoch)

    def set_train_mode(self, mode):
        for subl in self.sublayers:
            subl.set_train_mode(mode)

    def get_params(self):
        np_params = dict()
        for i, ll in enumerate(self.sublayers):
            if ll.has_params:
                lkey = self.name + '_' + ll.name + '_' + str(i)
                np_params[lkey] = ll.get_params()
        return np_params

    def set_params(self, params_dict):
        for i, ll in enumerate(self.sublayers):
            if ll.has_params:
                lkey = self.name + '_' + ll.name + '_' + str(i)
                ll.set_params(params_dict[lkey])


class BranchLayer(CompositeLayer):
    """
    Branch layer is composed of a list of other layers concatenated with one
    another.

    During fprop, it concatenates the component outputs and passes it on.
    During bprop, it splits the backward errors into the components and
    accumulates into common deltas
    """

    def __init__(self, **kwargs):
        super(BranchLayer, self).__init__(**kwargs)

    def set_previous_layer(self, pl):
        super(BranchLayer, self).set_previous_layer(pl)
        for subl in self.sublayers:
            subl.set_previous_layer(pl)
        self.nout = reduce(lambda x, y: x + y.nout, self.sublayers, 0)
        if pl is not None:
            self.nin = pl.nout

    def initialize(self, kwargs):
        super(BranchLayer, self).initialize(kwargs)

        self.startidx = [0] * len(self.sublayers)
        self.endidx = [0] * len(self.sublayers)
        self.endidx[0] = self.sublayers[0].nout
        for i in range(1, len(self.sublayers)):
            self.endidx[i] = self.endidx[i - 1] + self.sublayers[i].nout
            self.startidx[i] = self.endidx[i - 1]

        self.allocate_output_bufs()

    def set_deltas_buf(self, delta_pool, offset):
        if self.prev_layer is None:
            return
        if self.prev_layer.is_data:
            return

        self.deltas = self.backend.zeros(self.delta_shape, self.deltas_dtype)
        for subl in self.sublayers:
            subl.set_deltas_buf(delta_pool, offset)

    def fprop(self, inputs):
        for (subl, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            subl.fprop(inputs)
            self.output[si:ei] = subl.output

    def bprop(self, error):
        if self.deltas is not None:
            self.deltas.fill(0.0)
        for (subl, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            subl.bprop(error[si:ei])
            if self.deltas is not None:
                self.backend.add(self.deltas, subl.deltas, out=self.deltas)


class ListLayer(CompositeLayer):
    """
    List layer is composed of a list of other layers stacked on top of one
    another.

    During fprop and bprop, it simply operates along the chain.
    """
    def set_previous_layer(self, pl):
        super(ListLayer, self).set_previous_layer(pl)
        for subl in self.sublayers:
            subl.set_previous_layer(pl)
            pl = subl
        self.nout = self.sublayers[-1].nout

    def initialize(self, kwargs):
        super(ListLayer, self).initialize(kwargs)
        self.output = self.sublayers[-1].output
        if self.sublayers[-1].is_local is True:
            self.nofm = self.sublayers[-1].nofm
            self.ofmshape = self.sublayers[-1].ofmshape

    def set_deltas_buf(self, delta_pool, offset):
        self.ninmax = max(map(lambda x: x.nin, self.sublayers))
        delta_shape = (2 * self.ninmax, self.batch_size)
        assert len(self.sublayers) > 1
        delta_type = self.sublayers[1].deltas_dtype
        self.delta_pool = self.backend.zeros(delta_shape, delta_type)
        for idx, subl in enumerate(self.sublayers):
            offset = (idx % 2) * self.ninmax
            subl.set_deltas_buf(self.delta_pool, offset)

        if self.prev_layer is None:
            return
        if self.prev_layer.is_data:
            return
        self.deltas = self.sublayers[0].deltas

    def fprop(self, inputs):
        for subl in self.sublayers:
            subl.fprop(inputs)
            inputs = subl.output

    def bprop(self, error):
        for subl in reversed(self.sublayers):
            subl.bprop(error)
            error = subl.deltas
