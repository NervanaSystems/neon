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
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Misclassification
from neon.callbacks.callbacks import Callbacks, BatchNormTuneCallback
from neon.util.argparser import NeonArgparser

from network import create_network
from data import ingest_cifar10, make_train_loader, make_validation_loader, make_tuning_loader

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 9n+2)')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()
random_seed = args.rng_seed if args.rng_seed else 0

# perform ingest if it hasn't already been done (No op if already existing)
ingest_cifar10(padded_size=40, overwrite=False)

model, cost = create_network(args.depth)

# setup data provider
train = make_train_loader(model.be, args.subset_pct, random_seed)
test = make_validation_loader(model.be, args.subset_pct)

# tune batch norm parameters on subset of train set with no augmentations
tune_set = make_tuning_loader(model.be)

# configure callbacks
callbacks = Callbacks(model, eval_set=test, metric=Misclassification(), **args.callback_args)
callbacks.add_callback(BatchNormTuneCallback(tune_set), insert_pos=0)

# begin training
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001, schedule=Schedule([82, 124], 0.1))
model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
