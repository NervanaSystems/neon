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
Train WGAN to generate LSUN bedroom images.
"""

import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback
from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.optimizers import RMSProp
from neon.util.argparser import NeonArgparser
from neon.util.persist import ensure_dirs_exist
from network_gan import create_model
from lsun_data import make_loader

# parse the command line arguments
train_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
config_files = [train_config] if os.path.exists(train_config) else []
parser = NeonArgparser(__doc__, default_config_files=config_files,
                       default_overrides={'rng_seed': 0, 'batch_size': 64})
parser.add_argument('-D', '--dmodel', type=str, default='dc',
                    help='discriminator model type: dc or mlp, default dc')
parser.add_argument('-G', '--gmodel', type=str, default='dc',
                    help='generator model type: dc or mlp, default dc')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage), default 100')
args = parser.parse_args()
random_seed = args.rng_seed if args.rng_seed else 0

# Check that the proper manifest sets have been supplied
assert 'train' in args.manifest, "Missing train manifest"

# create a GAN
model, cost = create_model(dis_model=args.dmodel, gen_model=args.gmodel,
                           cost_type='wasserstein',
                           im_size=64, n_chan=3, n_noise=100,
                           n_gen_ftr=64, n_dis_ftr=64,
                           depth=4, n_extra_layers=4,
                           batch_norm=True, dis_iters=5,
                           wgan_param_clamp=0.01, wgan_train_sched=True)

# setup optimizer
optimizer = RMSProp(learning_rate=5e-5, decay_rate=0.99, epsilon=1e-8)

# setup data provider
train = make_loader(args.manifest['train'], args.manifest_root, model.be,
                    args.subset_pct, random_seed)

# configure callbacks
callbacks = Callbacks(model, **args.callback_args)
fdir = ensure_dirs_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/'))
fname = os.path.splitext(os.path.basename(__file__))[0] +\
    '_[' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ']'
im_args = dict(filename=os.path.join(fdir, fname), hw=64,
               num_samples=args.batch_size, nchan=3, sym_range=True)
callbacks.add_callback(GANPlotCallback(**im_args))
callbacks.add_callback(GANCostCallback())

# model fit
model.fit(train, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
