#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
import os
from neon.util.argparser import NeonArgparser
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.transforms import TopKMisclassification
from neon.callbacks.callbacks import Callbacks

from data import make_alexnet_train_loader, make_validation_loader
from network_alexnet import create_network


# parse the command line arguments (generates the backend)
train_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
config_files = [train_config] if os.path.exists(train_config) else []

parser = NeonArgparser(__doc__, default_config_files=config_files)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

model, cost = create_network()
rseed = 0 if args.rng_seed is None else args.rng_seed

# setup data provider
assert 'train' in args.manifest, "Missing train manifest"
assert 'val' in args.manifest, "Missing validation manifest"
train = make_alexnet_train_loader(args.manifest['train'], args.manifest_root,
                                  model.be, args.subset_pct, rseed)
valid = make_validation_loader(args.manifest['val'], args.manifest_root,
                               model.be, args.subset_pct)

# drop weights LR by 1/250**(1/3) at epochs (23, 45, 66), drop bias LR by 1/10 at epoch 45
sched_weight = Schedule([22, 44, 65], 0.15874)
sched_biases = Schedule([44], 0.1)

opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=sched_weight)
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=sched_biases)
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

# configure callbacks
valmetric = TopKMisclassification(k=5)
callbacks = Callbacks(model, eval_set=valid, metric=valmetric, **args.callback_args)
model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
