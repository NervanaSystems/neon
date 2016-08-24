#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
This training currently runs 1 image at a time.

Reference:
    "Faster R-CNN"
    http://arxiv.org/abs/1506.01497
    https://github.com/rbgirshick/py-faster-rcnn

Usage:
    python examples/faster-rcnn/train.py -r0 -e 16 -s frcn_model.pkl -vv \
    --epoch_step 5 --roi_branch_scale --lr_scale 1.0 -H 16

"""
from __future__ import division

from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, StepSchedule
from neon.callbacks.callbacks import Callbacks, TrainMulticostCallback
from neon.util.persist import save_obj
from objectlocalization import PASCAL
from neon.transforms import CrossEntropyMulti, SmoothL1Loss
from neon.layers import Multicost, GeneralizedCostMask

import util

# parse the command line arguments
parser = NeonArgparser(__doc__, default_overrides={'batch_size': 1})
parser.add_argument('--lr_scale', type=float, help='learning rate scale', default=1.0)
parser.add_argument('--lr_step', type=float, help="step for learning schedule", default=10.0)
parser.add_argument('--epoch_step', type=int, help="epoch to step the learning rate", default=5)
parser.add_argument('--roi_branch_scale', action='store_true',
                    help="Scale ROI branchs by rois_per_img.")

args = parser.parse_args(gen_be=False)
args.roi_branch_scale = True

# hyperparameters
assert args.batch_size is 1, "Faster-RCNN only supports batch size 1"

n_mb = None
rpn_rois_per_img = 256  # number of rois to sample to train rpn
frcn_rois_per_img = 128  # number of rois to sample to train frcn
lr_scale = 1.0 / float(args.lr_scale)

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))
be.enable_winograd = 4

year = '2007'

train_set = PASCAL('trainval', year, path=args.data_dir, n_mb=n_mb,
                   rpn_rois_per_img=rpn_rois_per_img, frcn_rois_per_img=frcn_rois_per_img,
                   add_flipped=True, shuffle=True, rebuild_cache=True)

# build the Faster-RCNN model
model = util.build_model(train_set, frcn_rois_per_img, inference=False)

# set up cost different branches, respectively
weights = 1.0 / (rpn_rois_per_img)

if args.roi_branch_scale is True:
    roi_w = 1.0 / (frcn_rois_per_img)
else:
    roi_w = 1.0

frcn_tree_cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti(), weights=roi_w),
                                  GeneralizedCostMask(costfunc=SmoothL1Loss(), weights=roi_w)
                                  ], weights=[1, 1])

cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti(), weights=weights),
                        GeneralizedCostMask(costfunc=SmoothL1Loss(sigma=3.0), weights=weights),
                        frcn_tree_cost,
                        ],
                 weights=[1, 1, 1])

# setup optimizer
schedule_w = StepSchedule(step_config=[args.epoch_step],
                          change=[0.001 * lr_scale / args.lr_step])
schedule_b = StepSchedule(step_config=[args.epoch_step],
                          change=[0.002 * lr_scale / args.lr_step])

opt_w = GradientDescentMomentum(0.001 * lr_scale, 0.9, wdecay=0.0005, schedule=schedule_w)
opt_b = GradientDescentMomentum(0.002 * lr_scale, 0.9, wdecay=0.0005, schedule=schedule_b)
opt_skip = GradientDescentMomentum(0.0, 0.0)

optimizer = MultiOptimizer({'default': opt_w, 'Bias': opt_b,
                            'skip': opt_skip, 'skip_bias': opt_skip})

# if training a new model, seed the image model conv layers with pre-trained weights
# otherwise, just load the model file
if args.model_file is None:
    util.load_vgg_all_weights(model, args.data_dir)

callbacks = Callbacks(model, eval_set=train_set, **args.callback_args)
callbacks.add_callback(TrainMulticostCallback())

# model.benchmark(train_set, optimizer=optimizer, cost=cost)
model.fit(train_set, optimizer=optimizer, cost=cost, num_epochs=args.epochs, callbacks=callbacks)

# Scale the bbox regression branch linear layer weights before saving the model
model = util.scale_bbreg_weights(model, [0.0, 0.0, 0.0, 0.0],
                                 [0.1, 0.1, 0.2, 0.2], train_set.num_classes)

if args.save_path is not None:
    save_obj(model.serialize(keep_states=True), args.save_path)
