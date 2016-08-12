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
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.optimizers import Adadelta
from neon.transforms import Misclassification
from neon.callbacks.callbacks import Callbacks
from network import create_network
from data import make_train_loader, make_val_loader, make_test_loader


parser = NeonArgparser(__doc__)
parser.add_argument('--submission_mode', action='store_true',
                    help='whether to run in submission mode (all data) or validation mode')
args = parser.parse_args()

model, cost_obj = create_network()

if args.submission_mode:
    train = make_train_loader('all', model.be)
    test = make_test_loader('test', model.be)
    val, metric = None, None  # These aren't used in submission mode
else:
    train = make_train_loader('train', model.be)
    val = make_val_loader('val', model.be)
    metric = Misclassification()

model.fit(dataset=train,
          cost=cost_obj,
          optimizer=Adadelta(),
          num_epochs=args.epochs,
          callbacks=Callbacks(model, eval_set=val, metric=metric, **args.callback_args))

if args.submission_mode:
    preds = model.get_outputs(test)
    np.savetxt('subm.txt', preds[:, 1], fmt='%.5f')
else:
    print('Misclassification error = %.1f%%' % (model.eval(val, metric=Misclassification())*100))
