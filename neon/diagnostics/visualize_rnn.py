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
Visualization for recurrent neural networks
"""

import numpy as np
from neon.util.compat import range


class VisualizeRNN(object):
    """
    Visualzing weight matrices during training
    """
    def __init__(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt  # noqa
        matplotlib.rcParams['pdf.fonttype'] = 42  # ensure true type font

        self.plt = matplotlib.pyplot
        self.plt.interactive(1)

    def plot_weights(self, weights_in, weights_rec, weights_out):
        """
        Visizualize the three weight matrices after every epoch. Serves to
        check that weights are structured, not exploding, and get upated
        """
        self.plt.figure(2)
        self.plt.clf()
        self.plt.subplot(1, 3, 1)
        self.plt.imshow(weights_in.T, vmin=-1, vmax=1, interpolation='nearest')
        self.plt.title('input.T')
        self.plt.subplot(1, 3, 2)
        self.plt.imshow(weights_rec, vmin=-1, vmax=1, interpolation='nearest')
        self.plt.title('recurrent')
        self.plt.subplot(1, 3, 3)
        self.plt.imshow(weights_out, vmin=-1, vmax=1, interpolation='nearest')
        self.plt.title('output')
        self.plt.colorbar()
        self.plt.draw()
        self.plt.show()

    def plot_lstm_wts(self, lstm_layer, scale=1, fig=4):

        """
        Visizualize the three weight matrices after every epoch. Serves to
        check that weights are structured, not exploding, and get upated
        """
        self.plt.figure(fig)
        self.plt.clf()
        pltidx = 1
        for lbl, wts in zip(lstm_layer.param_names, lstm_layer.params[:4]):
            self.plt.subplot(2, 4, pltidx)
            self.plt.imshow(wts.asnumpyarray().T, vmin=-scale, vmax=scale,
                            interpolation='nearest')
            self.plt.title(lbl + ' Wx.T')
            pltidx += 1

        for lbl, wts, bs in zip(lstm_layer.param_names,
                                lstm_layer.params[4:8],
                                lstm_layer.params[8:12]):
            self.plt.subplot(2, 4, pltidx)
            self.plt.imshow(np.hstack((wts.asnumpyarray(),
                                      bs.asnumpyarray(),
                                      bs.asnumpyarray())).T,
                            vmin=-scale, vmax=scale, interpolation='nearest')
            self.plt.title(lbl + ' Wh.T')
            pltidx += 1

        self.plt.draw()
        self.plt.show()

    def plot_lstm_acts(self, lstm_layer, scale=1, fig=4):
        acts_lbl = ['i_t', 'f_t', 'o_t', 'g_t', 'net_i', 'c_t', 'c_t', 'c_phi']
        acts_stp = [0, 0, 0, 1, 0, 0, 1, 1]
        self.plt.figure(fig)
        self.plt.clf()
        for idx, lbl in enumerate(acts_lbl):
            act_tsr = getattr(lstm_layer, lbl)[acts_stp[idx]]
            self.plt.subplot(2, 4, idx+1)
            self.plt.imshow(act_tsr.asnumpyarray().T,
                            vmin=-scale, vmax=scale, interpolation='nearest')
            self.plt.title(lbl + '[' + str(acts_stp[idx]) + '].T')

        self.plt.draw()
        self.plt.show()

    def plot_error(self, suberror_list, error_list):
        self.plt.figure(1)
        self.plt.clf()
        self.plt.plot(np.arange(len(suberror_list)) /
                      np.float(len(suberror_list)) *
                      len(error_list), suberror_list)
        self.plt.plot(error_list, linewidth=2)
        self.plt.ylim((min(suberror_list), max(error_list)))
        self.plt.draw()
        self.plt.show()

    def plot_activations(self, pre1, out1, pre2, out2, targets):
        """
        Loop over tau unrolling steps, at each time step show the pre-acts
        and outputs of the recurrent layer and output layer. Note that the
        pre-acts are actually the g', so if the activation is linear it will
        be one.
        """

        self.plt.figure(3)
        self.plt.clf()
        for i in range(len(pre1)):  # loop over unrolling
            self.plt.subplot(len(pre1), 5, 5 * i + 1)
            self.plt.imshow(pre1[i].asnumpyarray(), vmin=-1, vmax=1,
                            interpolation='nearest')
            if i == 0:
                self.plt.title('pre1 or g\'1')
            self.plt.subplot(len(pre1), 5, 5 * i + 2)
            self.plt.imshow(out1[i].asnumpyarray(), vmin=-1, vmax=1,
                            interpolation='nearest')
            if i == 0:
                self.plt.title('out1')
            self.plt.subplot(len(pre1), 5, 5 * i + 3)
            self.plt.imshow(pre2[i].asnumpyarray(), vmin=-1, vmax=1,
                            interpolation='nearest')
            if i == 0:
                self.plt.title('pre2 or g\'2')
            self.plt.subplot(len(pre1), 5, 5 * i + 4)
            self.plt.imshow(out2[i].asnumpyarray(), vmin=-1, vmax=1,
                            interpolation='nearest')
            if i == 0:
                self.plt.title('out2')
            self.plt.subplot(len(pre1), 5, 5 * i + 5)
            self.plt.imshow(targets[i].asnumpyarray(),
                            vmin=-1, vmax=1, interpolation='nearest')
            if i == 0:
                self.plt.title('target')
        self.plt.draw()
        self.plt.show()

    def print_text(self, inputs, outputs):
        """
        Moved this here so it's legal to use numpy.
        """
        print("Prediction inputs")
        print(np.argmax(inputs, 0).asnumpyarray().astype(np.int8).view('c'))
        print("Prediction outputs")
        print(np.argmax(outputs, 0).asnumpyarray().astype(np.int8).view('c'))
