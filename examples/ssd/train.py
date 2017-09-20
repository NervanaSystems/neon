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
from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser
from neon.util.persist import get_data_cache_dir
from neon.models.model import Model
from neon.callbacks.callbacks import Callbacks
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from ssd_container import SSD, load_vgg_weights
from ssd_dataloader import build_dataloader
from callbacks import MAP_Callback, ssd_image_callback
from mboxloss import MBoxLoss
from collections import OrderedDict
import json

"""
Trains the SSD model on the provided dataset.
The ingest scripts for datasets are stored in the /datasets directory. We
include ingest scripts for PASCALVOC, KITTI, and SPACENET datasets.
"""
# Parse the command line arguments
arg_defaults = {'batch_size': 0}

parser = NeonArgparser(__doc__, default_overrides=arg_defaults)
parser.add_argument('--height', type=int, help='image height')
parser.add_argument('--width', type=int, help='image width')
parser.add_argument('--subset_pct', type=float, default=100.0,
                    help='fraction of full training data set to use')
parser.add_argument('--ssd_config', action='append', required=True, help='path to ssd json file')
parser.add_argument('--lr_scale', type=float, default=1.0, help='scale lr by this amount')
parser.add_argument('--image_sample_dir', type=str, help='path to save image samples')
parser.add_argument('--num_images', type=int, help='number of images to save')
parser.add_argument('--lr_step', type=int, action='append', help='epochs to step lr')

args = parser.parse_args(gen_be=False)
if args.ssd_config:
    args.ssd_config = {k: v for k, v in [ss.split(':') for ss in args.ssd_config]}

# directory to store VGG weights
cache_dir = get_data_cache_dir(args.data_dir, subdir='ssd_cache')
train_config = json.load(open(args.ssd_config['train']), object_pairs_hook=OrderedDict)
val_config = json.load(open(args.ssd_config['val']), object_pairs_hook=OrderedDict)

if args.batch_size == 0:
    args.batch_size = train_config["batch_size"]

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=args.batch_size,
                 device_id=args.device_id,
                 compat_mode='caffe',
                 rng_seed=1,
                 deterministic_update=True,
                 deterministic=True, max_devices=args.max_devices)
be.enable_winograd = 0

# build dataloaders
train_config["manifest_filename"] = args.manifest['train']
train_set = build_dataloader(train_config, args.manifest_root, be.bsz, args.subset_pct)

val_config["manifest_filename"] = args.manifest['val']
val_set = build_dataloader(val_config, args.manifest_root, be.bsz, args.subset_pct)

model = Model(layers=SSD(ssd_config=train_config['ssd_config'], dataset=train_set))

cost = MBoxLoss(num_classes=train_set.num_classes)

if args.model_file is None:
    load_vgg_weights(model, cache_dir)
else:
    model.load_params(args.model_file)

if args.lr_step is None:
    args.lr_step = [40, 80, 120]

base_lr = 0.0001 * be.bsz * args.lr_scale
schedule = Schedule(args.lr_step, 0.1)
opt_w = GradientDescentMomentum(base_lr, momentum_coef=0.9, wdecay=0.0005, schedule=schedule)
opt_b = GradientDescentMomentum(base_lr, momentum_coef=0.9, schedule=schedule)
opt = MultiOptimizer({'default': opt_w, 'Bias': opt_b})

# hijack the eval callback arg here
eval_freq = args.callback_args.pop('eval_freq')
callbacks = Callbacks(model, **args.callback_args)
callbacks.add_callback(MAP_Callback(eval_set=val_set, epoch_freq=eval_freq))

if args.image_sample_dir is not None:
    callbacks.add_callback(ssd_image_callback(eval_set=val_set, image_dir=args.image_sample_dir,
                                              epoch_freq=eval_freq, num_images=args.num_images,
                                              classes=val_config['class_names']))

model.fit(train_set, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
