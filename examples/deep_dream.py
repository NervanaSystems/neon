#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
Deep Dream.

Reference:

    Inceptionism: Going Deeper into Neural Networks `[Mordvintsev2015]`_
..  _[Mordvintsev2015]: http://googleresearch.blogspot.ch/2015/06/\
inceptionism-going-deeper-into-neural.html

Usage:

    python examples/deep_dream.py <image> --output <output_location>

"""

from builtins import object, round, zip
import numpy as np
import os.path as osp
from PIL import Image
import sys
import warnings
from neon import logger as neon_logger

try:
    from scipy.ndimage import zoom
except ImportError as err:
    neon_logger.display("Running this example requires scipy packages.")
    neon_logger.display("try activating your virtualenv then: pip install scipy")
    sys.exit(1)

from neon.models import Model
from neon.layers import Activation
from neon.data.datasets import Dataset
from neon.layers import GeneralizedCost
from neon.transforms.cost import Cost
from neon.util.argparser import NeonArgparser

# force use of CPU backend since we require a batch size of 1
# (GPU needs a multiple of 32)
default_overrides = dict(backend='cpu', batch_size=1)
parser = NeonArgparser(__doc__, default_overrides=default_overrides)
parser.add_argument("image", help="Base image to create dream on.")
parser.add_argument("--dream_file", default='dream_out.png',
                    help="Save dream to named file.")
args = parser.parse_args()


# redirect the dream file to the path of output_file
if args.output_file is None:
    output_dir = parser.work_dir
elif osp.isdir(args.output_file):
    output_dir = args.output_file
else:
    output_dir = osp.dirname(args.output_file)

args.dream_file = osp.expanduser(
    osp.join(output_dir, osp.basename(args.dream_file)))
RGB_MEAN = np.array([104.4, 119.2, 126.8])[:, np.newaxis, np.newaxis]


class MaximizeActivations(Cost):
    def __init__(self):
        self.func = lambda y, t: self.be.sum(self.be.square(y), axis=0) / -2.
        self.funcgrad = lambda y, t: -y


class Dream(Activation):
    def __init__(self, name=None):
        super(Dream, self).__init__(name)
        self.owns_output = True

    def fprop(self, inputs, inference=False):
        return inputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        return error


class DreamModel(Model):
    def __init__(self, model_path):
        model_file = self.load_imagenet_weights(model_path)
        super(DreamModel, self).__init__(layers=model_file, weights_only=True)
        self.layers.layers.insert(0, Dream())

    def load_imagenet_weights(self, model_path):
        # download trained Alexnet weights
        url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/alexnet/'
        filename = 'alexnet_conv_ns.p'
        size = 20550623

        _, filepath = Dataset._valid_path_append(model_path, '', filename)
        if not osp.exists(filepath):
            Dataset.fetch_dataset(url, filename, filepath, size)

        return filepath

    def initialize(self, imgtensor, cost=None):
        self.initialized = False
        for l in self.layers.layers:
            l.outputs = None
            if hasattr(l, 'nglayer'):
                l.nglayer = None
        super(DreamModel, self).initialize(imgtensor.shape,
                                           cost=GeneralizedCost(costfunc=MaximizeActivations()))


class DeepImage(object):
    def __init__(self, inobj):
        # create from file
        if isinstance(inobj, str):
            self.image = np.uint8(Image.open(args.image))
            self.t_dirty, self.i_dirty = True, False
        elif isinstance(inobj, np.ndarray):
            self.tensor = inobj
            self.t_dirty, self.i_dirty = False, True

    def as_tensor(self):
        if self.t_dirty:
            self.tensor = (self.image.transpose(2, 0, 1) - RGB_MEAN)[::-1].astype(np.float32)
            self.t_dirty = False
        return self.tensor

    def as_image(self):
        if self.i_dirty:
            self.image = (
                self.tensor[::-1] + RGB_MEAN).astype(np.uint8).transpose(1, 2, 0)
            self.i_dirty = False
        return Image.fromarray(self.image.clip(0, 255).astype(np.uint8))

    @property
    def shape(self):
        return self.as_tensor().shape

    def take_step(self, model, step=1.5, jitter=32):
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        self.tensor = np.roll(np.roll(self.as_tensor(), ox, -1), oy, -2)

        image_buf = model.be.array(self.tensor)
        delta = model.cost.get_errors(model.fprop(image_buf), None)
        grad = model.bprop(delta).get().reshape(self.tensor.shape)

        imgnp = self.tensor - step * grad / np.abs(grad).mean()
        self.tensor = np.clip(imgnp, -RGB_MEAN[::-1], 255 - RGB_MEAN[::-1])
        self.tensor = np.roll(np.roll(self.tensor, -ox, -1), -oy, -2)
        self.i_dirty = True

    def save_image(self, filename):
        neon_logger.display("Saving {}".format(filename))
        self.as_image().save(filename)


def get_numbered_file(filename, index):
    base, ext = osp.splitext(filename)
    return "{}_{:03d}{}".format(base, index, ext)


def zoom_to(tsr, to_shape):
    if to_shape == tsr.shape:
        return tsr
    else:
        scale_factor = (ts / float(fs) for ts, fs in zip(to_shape, tsr.shape))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return zoom(tsr, scale_factor, order=1)


def deepdream(image, iter_n=10, octave_n=4, octave_scale=1.4, name="Deep Dream"):
    model = DreamModel(model_path=args.data_dir)
    detail = None
    scales = [octave_scale ** -o for o in reversed(list(range(octave_n)))]

    for o_idx, scale in enumerate(scales):
        octave_shape = (
            3, round(image.shape[1] * scale), round(image.shape[2] * scale))
        octave_base = zoom_to(image.as_tensor(), octave_shape)
        detail = np.zeros_like(octave_base) if detail is None else zoom_to(
            detail, octave_shape)

        dream = DeepImage(octave_base + detail)
        model.initialize(dream)

        for i in range(iter_n):
            dream.take_step(model)
            ofile = get_numbered_file(args.dream_file, o_idx * iter_n + i)
            dream.save_image(ofile)

        detail = dream.as_tensor() - octave_base

    return dream


image = DeepImage(args.image)
dream = deepdream(image, name=args.image)
