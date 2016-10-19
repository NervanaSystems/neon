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
import os
from neon.util.argparser import NeonArgparser
from neon.optimizers import Adagrad
from neon.transforms import Misclassification
from neon.callbacks.callbacks import Callbacks

from data import make_train_loader, make_val_loader
from network import create_network


train_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
config_files = [train_config] if os.path.exists(train_config) else []

parser = NeonArgparser(__doc__, default_config_files=config_files)
args = parser.parse_args()

assert 'train' in args.manifest, "Missing train manifest"
assert 'val' in args.manifest, "Missing validation manifest"

model, cost = create_network()

# setup data provider
train = make_train_loader(args.manifest['train'], args.manifest_root, model.be)
val = make_val_loader(args.manifest['val'], args.manifest_root, model.be)

opt = Adagrad(learning_rate=0.01)
metric = Misclassification()
callbacks = Callbacks(model, eval_set=val, metric=metric, **args.callback_args)

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

print('Misclassification error = %.1f%%' % (model.eval(val, metric=metric)*100))
