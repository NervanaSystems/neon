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
Deep Dream

Reference:
    http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html

Usage:
    python deep_dream.py <image> --output <output_location>

"""
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image
import scipy.ndimage as nd

from neon.backends import gen_backend
from neon.models import Model
from neon.util.persist import load_obj
from neon.data.datasets import Dataset
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Pooling, DataTransform, GeneralizedCost
from neon.transforms import Rectlin, Identity
from neon.transforms.cost import Cost
from neon.util.argparser import NeonArgparser, extract_valid_args

parser = NeonArgparser(__doc__)
parser.add_argument("image", help="Base image to create dream on.")
parser.add_argument("--output", default=None, help="File location to write dream.")
args = parser.parse_args(gen_be=False)
args.backend = 'cpu'
args.batch_size = 1
be = gen_backend(**extract_valid_args(args, gen_backend))


class Dream(DataTransform):
    def __init__(self, image, name=None):
        super(Dream, self).__init__(Identity(), name)
        self.W = image
        self.owns_output = True
        self.has_params = True

    def get_params(self):
        return self.dW

    def bprop(self, error, alpha=1.0, beta=0.0):
        if not self.deltas:
            self.deltas = error
        error[:] = self.transform.bprop(self.outputs) * error
        self.dW = error.reshape(self.inputs.shape)
        return error


class MaximizeActivations(Cost):
    def __init__(self):
        self.func = lambda y, t: self.be.sum(self.be.square(y), axis=0) / -2.
        self.funcgrad = lambda y, t: -y


def load_imagenet_weights(model, path):
    # download trained Alexnet weights
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/alexnet/'
    filename = 'alexnet_conv.p'
    size = 57824894

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    pdict = load_obj(filepath)

    param_layers = [l for l in model.layers.layers[1:]]
    param_dict_list = pdict['model']['config']['layers']
    for layer, ps in zip(param_layers, param_dict_list):
        layer.load_weights(ps, load_states=True)


def create_model(image):
    layers = [
        Dream(image),
        Conv((11, 11, 64), init=Gaussian(scale=0.01), bias=Constant(0),
            activation=Rectlin(), padding=3, strides=4),
        Pooling(3, strides=2),
        Conv((5, 5, 192), init=Gaussian(scale=0.01), bias=Constant(1),
            activation=Rectlin(), padding=2),
        Pooling(3, strides=2),
        Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0),
            activation=Rectlin(), padding=1),
        Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1),
            activation=Rectlin(), padding=1),
        Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1),
            activation=Rectlin(), padding=1)
    ]
    model = Model(layers=layers)

    model.initialize(image.shape)

    load_imagenet_weights(model, args.data_dir)

    model.cost = GeneralizedCost(costfunc=MaximizeActivations())
    model.cost.initialize(model.layers_to_optimize[-1])

    return model


def show_image(image, dtype=np.uint8, name='Deep Dream', first=False):
    image = deprocess(image, dtype)
    image = np.uint8(np.clip(image, 0, 255))
    plt.gcf().canvas.set_window_title(name)
    plt.imshow(PIL.Image.fromarray(image), interpolation='nearest')
    if first:
        plt.ion()
        plt.show()
    else:
        plt.draw()
    plt.pause(1)


def model_mean():
    return np.array([104.4, 119.2, 126.8])


def preprocess(image):
    return np.float32(np.rollaxis(image - model_mean(), 2)[::-1])


def deprocess(image, dtype=np.uint8):
    return np.array(np.dstack(image[::-1]) + model_mean(), dtype=dtype)


def make_step(image, model, step_size=1.5, jitter=32):
    ox, oy = np.random.randint(-jitter, jitter + 1, 2)
    image = np.roll(np.roll(image, ox, -1), oy, -2)
    image_buf = model.be.array(image)

    forward = model.fprop(image_buf)
    delta = model.cost.get_errors(forward, None)
    model.bprop(delta)

    grad = np.array(model.layers_to_optimize[0].get_params().get())
    image = np.array(image_buf.get()) - step_size/np.abs(grad).mean() * grad

    image = np.roll(np.roll(image, -ox, -1), -oy, -2)

    image = np.clip(image.transpose(), -model_mean(), 255 - model_mean()).transpose()

    return image


def deepdream(image, iter_n=10, octave_n=4, octave_scale=1.4, name="Deep Dream"):
    preprocessed_image = preprocess(image)
    show_image(preprocessed_image, image.dtype, name, first=True)

    octaves = [preprocessed_image]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale, 1.0/octave_scale), order=1))

    dream = None
    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, h/float(h1), w/float(w1)), order=1)

        dream = octave_base + detail
        model = create_model(dream)

        for i in xrange(iter_n):
            dream = make_step(dream, model)

            show_image(dream, image.dtype, name)
            print "Step: {}, {}".format(octave, i)

        detail = dream - octave_base

    return deprocess(dream)


image = np.float32(PIL.Image.open(args.image))
dream = deepdream(image, name=args.image)
if args.output is not None:
    PIL.Image.fromarray(dream).save(args.output)
