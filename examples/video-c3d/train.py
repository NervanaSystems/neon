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
from neon import logger as neon_logger
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.transforms import Accuracy
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

from data import make_train_loader, make_test_loader
from network import create_network

# parse the command line arguments
train_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
config_files = [train_config] if os.path.exists(train_config) else []

parser = NeonArgparser(__doc__, default_config_files=config_files)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

random_seed = 0 if args.rng_seed is None else args.rng_seed
model, cost = create_network()

# setup data provider
assert 'train' in args.manifest, "Missing train manifest"
assert 'test' in args.manifest, "Missing validation manifest"

train = make_train_loader(args.manifest['train'], args.manifest_root, model.be, args.subset_pct,
                          random_seed)
valid = make_test_loader(args.manifest['test'], args.manifest_root, model.be, args.subset_pct)

# setup callbacks
callbacks = Callbacks(model, eval_set=valid, **args.callback_args)

# gradient descent with momentum, weight decay, and learning rate decay schedule
learning_rate_sched = Schedule(list(range(6, args.epochs, 6)), 0.1)
opt_gdm = GradientDescentMomentum(0.003, 0.9, wdecay=0.005, schedule=learning_rate_sched)
opt_biases = GradientDescentMomentum(0.006, 0.9, schedule=learning_rate_sched)
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

# train model
model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

# output accuracies
neon_logger.display('Train Accuracy = %.1f%%' % (model.eval(train, metric=Accuracy()) * 100))

neon_logger.display('Validation Accuracy = %.1f%%' % (model.eval(valid, metric=Accuracy()) * 100))
