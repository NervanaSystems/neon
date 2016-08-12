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
from neon.util.argparser import NeonArgparser
from neon.optimizers import Adagrad
from neon.transforms import Misclassification
from neon.callbacks.callbacks import Callbacks

from data import ingest_genre_data, make_train_loader, make_val_loader
from network import create_network

parser = NeonArgparser(__doc__)
parser.add_argument('--tar_file', default=None, help='Input tar filename')
args = parser.parse_args()

train_idx, val_idx = ingest_genre_data(args.tar_file)

model, cost = create_network()

# setup data provider
train = make_train_loader(train_idx, model.be)
val = make_val_loader(val_idx, model.be)

opt = Adagrad(learning_rate=0.01)
metric = Misclassification()
callbacks = Callbacks(model, eval_set=val, metric=metric, **args.callback_args)

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

print('Misclassification error = %.1f%%' % (model.eval(val, metric=metric)*100))
