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
"""
Train a Faster-RCNN model to do object detection using PASCAL VOC dataset.
This training currently runs 1 image at a time. All parameters are from the
reference. The training will get to the reference performance after 7 epochs.

At the end of the a training process, the model is serialized with the bounding box
regression layer normalized.

Reference:
    "Faster R-CNN"
    http://arxiv.org/abs/1506.01497
    https://github.com/rbgirshick/py-faster-rcnn

"""
from __future__ import division

from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, StepSchedule
from neon.callbacks.callbacks import Callbacks
from neon.util.persist import save_obj, get_data_cache_dir
from objectlocalization import PASCALVOC
from neon.transforms import CrossEntropyMulti, SmoothL1Loss
from neon.layers import Multicost, GeneralizedCostMask
import util
import faster_rcnn
import os

train_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pascalvoc.cfg')
config_files = [train_config] if os.path.exists(train_config) else []

# parse the command line arguments
parser = NeonArgparser(__doc__, default_config_files=config_files)
parser.add_argument('--width', type=int, default=1000, help='Width of input image')
parser.add_argument('--height', type=int, default=1000, help='Height of input image')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args(gen_be=False)

# hyperparameters
assert args.batch_size is 1, "Faster-RCNN only supports batch size 1"
assert 'train' in args.manifest

rpn_rois_per_img = 256  # number of rois to sample to train rpn
frcn_rois_per_img = 128  # number of rois to sample to train frcn

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))
be.enable_winograd = 4  # default to winograd 4 for fast autotune

# directory to store VGG weights
cache_dir = get_data_cache_dir(args.data_dir, subdir='pascalvoc_cache')

# build data loader
# get config file for PASCALVOC
config = PASCALVOC(args.manifest['train'], args.manifest_root,
                   width=args.width, height=args.height,
                   rois_per_img=rpn_rois_per_img, inference=False)
config['subset_fraction'] = float(args.subset_pct / 100.0)

train_set = faster_rcnn.build_dataloader(config, frcn_rois_per_img)

# build the Faster-RCNN model
model = faster_rcnn.build_model(train_set, frcn_rois_per_img, inference=False)

# set up cost different branches, respectively
weights = 1.0 / (rpn_rois_per_img)
roi_w = 1.0 / (frcn_rois_per_img)

frcn_tree_cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti(), weights=roi_w),
                                  GeneralizedCostMask(costfunc=SmoothL1Loss(), weights=roi_w)
                                  ], weights=[1, 1])

cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti(), weights=weights),
                        GeneralizedCostMask(costfunc=SmoothL1Loss(sigma=3.0), weights=weights),
                        frcn_tree_cost,
                        ],
                 weights=[1, 1, 1])

# setup optimizer
schedule_w = StepSchedule(step_config=[10], change=[0.001 / 10])
schedule_b = StepSchedule(step_config=[10], change=[0.002 / 10])

opt_w = GradientDescentMomentum(0.001, 0.9, wdecay=0.0005, schedule=schedule_w)
opt_b = GradientDescentMomentum(0.002, 0.9, wdecay=0.0005, schedule=schedule_b)
opt_skip = GradientDescentMomentum(0.0, 0.0)

optimizer = MultiOptimizer({'default': opt_w, 'Bias': opt_b,
                            'skip': opt_skip, 'skip_bias': opt_skip})

# if training a new model, seed the image model conv layers with pre-trained weights
# otherwise, just load the model file
if args.model_file is None:
    util.load_vgg_all_weights(model, cache_dir)

callbacks = Callbacks(model, eval_set=train_set, **args.callback_args)

model.fit(train_set, optimizer=optimizer, cost=cost, num_epochs=args.epochs, callbacks=callbacks)

# Scale the bbox regression branch linear layer weights before saving the model
model = util.scale_bbreg_weights(model, [0.0, 0.0, 0.0, 0.0],
                                 [0.1, 0.1, 0.2, 0.2], train_set.num_classes)

if args.save_path is not None:
    save_obj(model.serialize(keep_states=True), args.save_path)
