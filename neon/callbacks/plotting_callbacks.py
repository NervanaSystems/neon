# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
from __future__ import division
import logging
import numpy as np
from neon.callbacks.callbacks import Callback
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.exception("GANPlotCallback requires matplotlib to be installed")
    raise(e)


class GANPlotCallback(Callback):
    """
    Create PNG plots of samples from a GAN model.

    Arguments:
        filename (string): Filename prefix for output PNGs.
        hw (int): Height and width of the images.
        nchan (int): number of channels.
        num_saples (int): how many samples to show from traning and generated data.
    """
    def __init__(self, filename, hw=28, nchan=1, num_samples=5, sym_range=False, epoch_freq=1):
        super(GANPlotCallback, self).__init__(epoch_freq=epoch_freq)
        self.filename = filename
        self.hw = hw
        self.nchan = nchan
        self.num_samples = num_samples
        self.sym_range = sym_range

    def _make_plot(self, data, hw, nchan, title):
        im_args = dict(interpolation="nearest", vmin=0., vmax=1.)
        if nchan == 1:
            data = 1.-data.reshape(hw, hw)
            im_args['cmap'] = plt.get_cmap("gray")
        else:
            data = data.reshape(nchan, hw, hw)
            data = np.swapaxes(np.swapaxes(data, 0, 2), 0, 1)
            if self.sym_range:
                data = (data + 1.) / 2.
        plt.imshow(data, **im_args)
        plt.title(title)
        plt.axis('off')

    def on_epoch_end(self, callback_data, model, epoch):
        for i in range(self.num_samples):
            plt.subplot(2, self.num_samples, i+1)
            self._make_plot(model.data_batch[:, i].get(),
                            self.hw, self.nchan, 'data '+str(i))
            plt.subplot(2, self.num_samples, i+1+self.num_samples)
            self._make_plot(model.noise_batch[:, i].get(),
                            self.hw, self.nchan, 'noise'+str(i))
        plt.savefig(self.filename+str(epoch)+'.png')
