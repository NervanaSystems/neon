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
import numpy as np
from neon import logger as neon_logger
from neon.util.argparser import NeonArgparser
from neon.optimizers import Adadelta
from neon.callbacks.callbacks import Callbacks
from network import create_network
from data import make_train_loader, make_test_loader

subm_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'whale_subm.cfg')
config_files = [subm_config] if os.path.exists(subm_config) else []
parser = NeonArgparser(__doc__, default_config_files=config_files)
parser.add_argument('--submission_file', help='where to write prediction output')
args = parser.parse_args()

model, cost_obj = create_network()

assert 'all' in args.manifest, "Missing train manifest"
assert 'test' in args.manifest, "Missing test manifest"
assert args.submission_file is not None, "Must supply a submission file to output scores to"

neon_logger.display('Performing train and test in submission mode')
train = make_train_loader(args.manifest['all'], args.manifest_root, model.be,
                          noise_file=args.manifest.get('noise'))
test = make_test_loader(args.manifest['test'], args.manifest_root, model.be)

model.fit(dataset=train,
          cost=cost_obj,
          optimizer=Adadelta(),
          num_epochs=args.epochs,
          callbacks=Callbacks(model, **args.callback_args))

preds = model.get_outputs(test)
np.savetxt(args.submission_file, preds[:, 1], fmt='%.5f')
