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
Visualizing histograms of various parameter ranges to diagnose underflows.
"""

import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa
matplotlib.rcParams['pdf.fonttype'] = 42  # ensure true type font

logger = logging.getLogger(__name__)


def print_param_stats(backend, logger, prefix):

    figname = 'param_ranges_' + prefix + ' ' + backend.__module__

    layers = backend.name_dict[0].keys()
    assert len(backend.name_dict[0]) > 0, "decorated function was never called"
    tensors = set(backend.name_dict[0][layers[0]])
    epochs = backend.raw_dict.keys()

    bins = {'ps_item': np.linspace(-0.25, 0.25, num=39),
            'vs_item': np.linspace(-.0001, .0001, num=39),
            'us_item': np.linspace(-.0001, .0001, num=39),
            'ratioup': np.linspace(-.01, .01, num=39)}

    for epoch in epochs:
        plt.figure(1, figsize=(12, 15), dpi=120, facecolor='w', edgecolor='k')
        fig = plt.gcf()
        fig.set_size_inches(12, 15)
        for k, layer in enumerate(layers):
            for j, tensor in enumerate(tensors):
                all_stats = np.vstack([backend.raw_dict[epoch][layer][i]
                                      for i, x
                                      in enumerate(backend.name_dict[0][layer])
                                      if x == tensor])
                ax1 = plt.subplot(len(layers), 4, j+1+4*k)
                plt.bar(left=bins[tensor[:7]],
                        height=np.sum(all_stats, axis=0),
                        alpha=.5, log=True,
                        width=bins[tensor[:7]][1]-bins[tensor[:7]][0])
                ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
                ax1.locator_params(axis='x', nbins=5)
                plt.xlim(bins[tensor[:7]][0], bins[tensor[:7]][-1])
                plt.xlabel(['weight' if 'ps' in tensor else
                            'velocity' if 'vs' in tensor else
                            'update or decay' if 'us' in tensor else
                            'update/weight' if 'ratio' in tensor else
                            'anon'][0])

            ax1 = plt.subplot(len(layers), 4, 1+4*k)
            plt.title(layer+" epoch "+str(epoch))

        plt.tight_layout()
        plt.savefig(figname+'_epoch'+str(epoch)+'.png', dpi=200)
        plt.clf()
