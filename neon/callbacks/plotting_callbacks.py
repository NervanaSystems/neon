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
from PIL import Image
from neon.callbacks.callbacks import Callback
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.exception("GANPlotCallback requires matplotlib to be installed")
    raise(e)
try:
    from scipy.signal import medfilt
except ImportError as e:
    logger.exception("GANPlotCallback requires scipy to be installed")
    raise(e)


class GANPlotCallback(Callback):
    """
    Create PNG plots of samples from a GAN model.

    Arguments:
        filename (string): Filename prefix for output PNGs.
        hw (int): Height and width of the images.
        nchan (int): number of channels.
        num_saples (int): how many samples to show from traning and generated data.
        sym_range (bool): pixel value [-1, 1] or [0, 1].
        padding (int): number of pixels to pad in output images.
        plot_width (int): width of output images.
        plot_height (int): height of output images.
        dpi (float): dots per inch.
        font_size (int): font size of labels.
        epoch_freq (int): number of epochs per plotting callback.
    """
    def __init__(self, filename, hw=28, nchan=1, num_samples=16, sym_range=False, padding=2,
                 plot_width=1200, plot_height=600, dpi=60., font_size=10, epoch_freq=1):
        super(GANPlotCallback, self).__init__(epoch_freq=epoch_freq)
        self.filename = filename
        self.hw = hw
        self.nchan = nchan
        self.num_samples = num_samples
        self.padding = padding
        self.sym_range = sym_range
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.dpi = dpi
        self.font_size = font_size

    def _value_transform(self, batch):
        if self.nchan == 1:
            batch = 1. - batch
        else:
            if self.sym_range:
                batch = (batch + 1.) / 2.
        return batch

    def _shape_transform(self, batch):
        assert self.nchan * self.hw * self.hw == batch.shape[0], "wrong image size specified"
        assert self.num_samples <= batch.shape[1], "number of samples must not exceed batch size"

        nrow = int(np.ceil(np.sqrt(self.num_samples)))
        ncol = int(np.ceil(1. * self.num_samples / nrow))
        width = ncol*(self.hw+self.padding)-self.padding
        height = nrow*(self.hw+self.padding)-self.padding

        batch = batch[:, :self.num_samples]
        batch = batch.reshape(self.nchan, self.hw, self.hw, self.num_samples)
        batch = np.swapaxes(np.swapaxes(batch, 0, 2), 0, 1)

        canvas = np.ones([height, width, self.nchan])
        for i in range(self.num_samples):
            irow, icol, step = i % nrow, i // nrow, self.hw + self.padding
            canvas[irow*step:irow*step+self.hw, icol*step:icol*step+self.hw, :] = \
                batch[:, :, ::-1, i]
        if self.nchan == 1:
            canvas = canvas.reshape(height, width)
        return canvas

    def on_epoch_end(self, callback_data, model, epoch):
        # convert to numpy arrays
        data_batch = model.data_batch.get()
        noise_batch = model.noise_batch.get()
        # value transform
        data_batch = self._value_transform(data_batch)
        noise_batch = self._value_transform(noise_batch)
        # shape transform
        data_canvas = self._shape_transform(data_batch)
        noise_canvas = self._shape_transform(noise_batch)
        # plotting options
        im_args = dict(interpolation="nearest", vmin=0., vmax=1.)
        if self.nchan == 1:
            im_args['cmap'] = plt.get_cmap("gray")
        fname = self.filename+'_data_'+'{:03d}'.format(epoch)+'.png'
        Image.fromarray(np.uint8(data_canvas*255)).convert('RGB').save(fname)
        fname = self.filename+'_noise_'+'{:03d}'.format(epoch)+'.png'
        Image.fromarray(np.uint8(noise_canvas*255)).convert('RGB').save(fname)

        # plot logged WGAN costs if logged
        if model.cost.costfunc.func == 'wasserstein':
            giter = callback_data['gan/gen_iter'][:]
            nonzeros = np.where(giter)
            giter = giter[nonzeros]
            cost_dis = callback_data['gan/cost_dis'][:][nonzeros]
            w_dist = medfilt(np.array(-cost_dis, dtype='float64'), kernel_size=101)
            plt.figure(figsize=(400/self.dpi, 300/self.dpi), dpi=self.dpi)
            plt.plot(giter, -cost_dis, 'k-', lw=0.25)
            plt.plot(giter, w_dist, 'r-', lw=2.)
            plt.title(self.filename, fontsize=self.font_size)
            plt.xlabel("Generator Iterations", fontsize=self.font_size)
            plt.ylabel("Wasserstein estimate", fontsize=self.font_size)
            plt.margins(0, 0, tight=True)
            plt.savefig(self.filename+'_training.png', bbox_inches='tight')
            plt.close()
