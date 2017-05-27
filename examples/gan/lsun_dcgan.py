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
Simple DCGAN implementation for generating LSUN bedroom images.
"""

import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback
from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.util.argparser import NeonArgparser
from neon.optimizers import Adam
from neon.util.persist import ensure_dirs_exist
from network_gan import create_model
from lsun_data import make_loader

# parse the command line arguments
train_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
config_files = [train_config] if os.path.exists(train_config) else []
parser = NeonArgparser(__doc__, default_config_files=config_files,
                       default_overrides={'rng_seed': 0, 'batch_size': 64})
parser.add_argument("--original_cost", action='store_true',
                    help="generator cost = -log(D(G(z))) rather than log(1-D(G(z)))")
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()
random_seed = args.rng_seed if args.rng_seed else 0

# Check that the proper manifest sets have been supplied
assert 'train' in args.manifest, "Missing train manifest"

# create model and cost
model, cost = create_model(dis_model='dc', gen_model='dc',
                           cost_type='original' if args.original_cost else 'modified',
                           im_size=64, n_chan=3, n_noise=100, n_gen_ftr=64, n_dis_ftr=64,
                           n_extra_layers=0, batch_norm=True)

# setup optimizer
optimizer = Adam(learning_rate=2e-4, beta_1=0.5)

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
