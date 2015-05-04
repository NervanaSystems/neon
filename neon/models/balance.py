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
Contains code to train a balance network, containing both supervised and
unsupervised branches and multiple cost functions.
Requires model to specify prev layers at each layer to build the layer graph
For details, see http://arxiv.org/pdf/1412.6583.pdf
"""

import logging
from neon.models.mlp import MLP
from neon.util.param import req_param
from neon.util.compat import pickle


def my_pickle(filename, data):
    with open(filename, "w") as fo:
        pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)

logger = logging.getLogger(__name__)


class Balance(MLP):

    def __init__(self, **kwargs):
        self.accumulate = True
        super(Balance, self).__init__(**kwargs)
        req_param(self, ['classlayers', 'stylelayers'])
        self.cost_layer = self.classlayers[-1]
        self.out_layer = self.layers[-2]
        self.class_layer = self.classlayers[-2]
        self.branch_layer = self.stylelayers[-2]
        self.pathways = [self.layers, self.classlayers, self.stylelayers]
        self.kwargs = kwargs

    def initialize(self, backend, initlayer=None):
        super(Balance, self).initialize(backend, initlayer)
        self.kwargs['backend'] = self.backend
        for lp in [self.classlayers, self.stylelayers]:
            lp[-1].set_previous_layer(lp[-2])
            lp[-1].initialize(self.kwargs)

    def fprop(self):
        super(Balance, self).fprop()
        for ll in [self.classlayers[-1], self.stylelayers[-1]]:
            ll.fprop(ll.prev_layer.output)

    def bprop(self):
        for path, skip_act in zip(self.pathways, [False, True, False]):
            self.class_layer.skip_act = skip_act
            for ll, nl in zip(reversed(path), reversed(path[1:] + [None])):
                error = None if nl is None else nl.deltas
                ll.bprop(error)

    def get_reconstruction_output(self):
        return self.out_layer.output

    def generate_output(self, inputs):
        y = inputs
        for layer in self.layers[1:]:
            layer.fprop(y)
            y = layer.output
            if layer is self.branch_layer:
                y[self.zidx:] = self.zparam


class BalanceMP(MLP):
    def __init__(self, **kwargs):
        self.accumulate = True
        super(BalanceMP, self).__init__(**kwargs)
        req_param(self, ['costpaths'])
        # Append the prefix to the costpaths
        for ckey in self.costpaths.keys():
            self.costpaths[ckey] = self.prefixlayers + self.costpaths[ckey]

        self.cost_layer = self.costpaths['subject'][-1]
        self.branch_layer = self.costpaths['z'][-2]
        self.out_layer = self.layers[-2]

        softmaxlabels = filter(lambda x: x != 'z', self.costpaths.keys())
        self.softlayers = [self.costpaths[ck][-2] for ck in softmaxlabels]

        self.pathways = [self.layers, self.costpaths['z']]
        self.pathways += [self.costpaths[ck] for ck in softmaxlabels]
        self.path_skip_act = [False, False] + [True for ck in softmaxlabels]
        self.kwargs = kwargs

    def initialize(self, backend, initlayer=None):
        super(BalanceMP, self).initialize(backend, initlayer)
        self.kwargs['backend'] = self.backend
        for ckey in self.costpaths.keys():
            lp = self.costpaths[ckey]
            lp[-1].set_previous_layer(lp[-2])
            lp[-1].initialize(self.kwargs)

    def fprop(self):
        super(BalanceMP, self).fprop()
        for ckey in self.costpaths.keys():
            ll = self.costpaths[ckey][-1]
            ll.fprop(ll.prev_layer.output)

    def bprop(self):
        for path, skip_act in zip(self.pathways, self.path_skip_act):
            for cl in self.softlayers:
                cl.skip_act = skip_act
            for ll, nl in zip(reversed(path), reversed(path[1:] + [None])):
                error = None if nl is None else nl.deltas
                ll.bprop(error)
        for ll in self.layers:
            if ll.name == 'slice':
                my_pickle('bb.wts',
                          {'de': ll.deltas.asnumpyarray(),
                           'in': ll.prev_layer.output.asnumpyarray(),
                           'out': ll.output.asnumpyarray()})

    def get_reconstruction_output(self):
        return self.out_layer.output

    def generate_output(self, inputs):
        y = inputs
        for layer in self.layers[1:]:
            layer.fprop(y)
            y = layer.output
            if layer is self.branch_layer:
                y[self.zidx:] = self.zparam
