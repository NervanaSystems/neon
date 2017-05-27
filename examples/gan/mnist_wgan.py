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
Train WGAN to generate MNIST images.
"""

import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback
from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.optimizers import RMSProp
from neon.util.argparser import NeonArgparser
from neon.util.persist import ensure_dirs_exist
from network_gan import create_model
from neon.data.image import MNIST

# parse the command line arguments
parser = NeonArgparser(__doc__, default_overrides={'epochs': 32, 'rng_seed': 0, 'batch_size': 64})
parser.add_argument('-D', '--dmodel', type=str, default='dc',
                    help='discriminator model type: dc or mlp, default dc')
parser.add_argument('-G', '--gmodel', type=str, default='dc',
                    help='generator model type: dc or mlp, default dc')
parser.add_argument('--n_dis_ftr', type=int, default=64,
                    help='base discriminator feature number, default 64')
parser.add_argument('--n_gen_ftr', type=int, default=64,
                    help='base generator feature number, default 64')
args = parser.parse_args()
random_seed = args.rng_seed if args.rng_seed else 0

# load up the mnist data set, padding images to size 32
dataset = MNIST(path=args.data_dir, sym_range=True, size=32, shuffle=True)
train = dataset.train_iter

# create a GAN
model, cost = create_model(dis_model=args.dmodel, gen_model=args.gmodel,
                           cost_type='wasserstein', noise_type='normal',
                           im_size=32, n_chan=1, n_noise=128,
                           n_gen_ftr=args.n_gen_ftr, n_dis_ftr=args.n_dis_ftr,
                           depth=4, n_extra_layers=4,
                           batch_norm=True, dis_iters=5,
                           wgan_param_clamp=0.01, wgan_train_sched=True)

# setup optimizer
optimizer = RMSProp(learning_rate=2e-4, decay_rate=0.99, epsilon=1e-8)

# configure callbacks
callbacks = Callbacks(model, **args.callback_args)
fdir = ensure_dirs_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/'))
fname = os.path.splitext(os.path.basename(__file__))[0] +\
    '_[' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ']'
im_args = dict(filename=os.path.join(fdir, fname), hw=32,
               num_samples=args.batch_size, nchan=1, sym_range=True)
callbacks.add_callback(GANPlotCallback(**im_args))
callbacks.add_callback(GANCostCallback())

# model fit
model.fit(train, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
