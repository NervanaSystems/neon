#!/usr/bin/env python
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
"""
Simple DCGAN implementation for generating MNIST images.
"""

import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback
from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.data.image import MNIST
from neon.initializers import Gaussian
from neon.layers import GeneralizedGANCost, Sequential, Conv, Deconv
from neon.layers.container import GenerativeAdversarial
from neon.models.model import GAN
from neon.optimizers import Adam
from neon.transforms import Rectlin, Logistic, GANCost
from neon.util.argparser import NeonArgparser
from neon.util.persist import ensure_dirs_exist

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--kbatch', type=int, default=1,
                    help='number of data batches per noise batch in training')
args = parser.parse_args()

# load up the mnist data set
dataset = MNIST(path=args.data_dir, size=27)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

# setup weight initialization function
init = Gaussian(scale=0.05)

# generator using "decovolution" layers
relu = Rectlin(slope=0)  # relu for generator
conv = dict(init=init, batch_norm=True, activation=relu)
convp1 = dict(init=init, batch_norm=True, activation=relu, padding=1)
convp2 = dict(init=init, batch_norm=True, activation=relu, padding=2)
convp1s2 = dict(init=init, batch_norm=True, activation=relu, padding=1, strides=2)
G_layers = [Deconv((1, 1, 16), name="G11", **conv),
            Deconv((3, 3, 192), name="G12", **convp1),
            Deconv((3, 3, 192), name="G21", **convp1s2),
            Deconv((3, 3, 192), name="G22", **convp1),
            Deconv((3, 3, 96), name="G31", **convp1s2),
            Deconv((3, 3, 96), name="G32", **conv),
            Deconv((3, 3, 1), name="G_out",
                   init=init, batch_norm=False, padding=1,
                   activation=Logistic(shortcut=False))]

# discriminiator using convolution layers
lrelu = Rectlin(slope=0.1)  # leaky relu for discriminator
conv = dict(init=init, batch_norm=True, activation=lrelu)
convp1 = dict(init=init, batch_norm=True, activation=lrelu, padding=1)
convp1s2 = dict(init=init, batch_norm=True, activation=lrelu, padding=1, strides=2)
D_layers = [Conv((3, 3, 96), name="D11", **convp1),
            Conv((3, 3, 96), name="D12", **convp1s2),
            Conv((3, 3, 192), name="D21", **convp1),
            Conv((3, 3, 192), name="D22", **convp1s2),
            Conv((3, 3, 192), name="D31", **convp1),
            Conv((1, 1, 16), name="D32", **conv),
            Conv((7, 7, 1), name="D_out",
                 init=init, batch_norm=False,
                 activation=Logistic(shortcut=False))]

layers = GenerativeAdversarial(generator=Sequential(G_layers, name="Generator"),
                               discriminator=Sequential(D_layers, name="Discriminator"))

# setup cost function as CrossEntropy
cost = GeneralizedGANCost(costfunc=GANCost(func="modified"))

# setup optimizer
optimizer = Adam(learning_rate=0.0005, beta_1=0.5)

# initialize model
noise_dim = (2, 7, 7)
gan = GAN(layers=layers, noise_dim=noise_dim, k=args.kbatch)

# configure callbacks
callbacks = Callbacks(gan, eval_set=valid_set, **args.callback_args)
fdir = ensure_dirs_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/'))
fname = os.path.splitext(os.path.basename(__file__))[0] +\
    '_[' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ']'
im_args = dict(filename=os.path.join(fdir, fname), hw=27,
               num_samples=args.batch_size, nchan=1, sym_range=True)
callbacks.add_callback(GANPlotCallback(**im_args))
callbacks.add_callback(GANCostCallback())

# run fit
gan.fit(train_set, optimizer=optimizer,
        num_epochs=args.epochs, cost=cost, callbacks=callbacks)
