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
from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.callbacks.callbacks import Callbacks, TrainMulticostCallback
from neon.util.persist import save_obj
from objectlocalization import PASCAL
from neon.initializers import Gaussian, Constant, Xavier, GlorotUniform
from neon.transforms import Rectlin, Identity, Softmax, CrossEntropyMulti, SmoothL1Loss, PixelwiseSoftmax
from neon.layers import Conv, Pooling, Affine, BranchNode, Tree, Multicost, GeneralizedCost, GeneralizedCostMask, Dropout
from neon.models import Model
from roi_pooling import RoiPooling
from proposal_layer import ProposalLayer
import util
import os
# main script

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--lr_scale', help='learning rate scale', default=16.0)
args = parser.parse_args(gen_be=False)

if args.data_dir is None:
    args.data_dir = '/usr/local/data/'

# Override save path if None
if args.save_path is None:
    args.save_path = 'frcn_vgg.pkl'

if args.callback_args['save_path'] is None:
    args.callback_args['save_path'] = args.save_path

if args.callback_args['serialize'] is None:
    args.callback_args['serialize'] = min(args.epochs, 10)

# hyperparameters
args.batch_size = 1

num_epochs = args.epochs
n_mb = None
img_per_batch = args.batch_size
rpn_rois_per_img = 256  # number of rois to sample to train rpn
frcn_rois_per_img = 128 # number of rois to sample to train frcn
frcn_fine_tune = False
learning_rate_scale = 1.0 / float(args.lr_scale)

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

train_set = PASCAL('trainval', '2007', path=args.data_dir, n_mb=n_mb,
                   img_per_batch=img_per_batch, rpn_rois_per_img=rpn_rois_per_img,
                   frcn_rois_per_img=frcn_rois_per_img, add_flipped=True, shuffle=True,
                   rebuild_cache=False)

# Faster-RCNN contains three models: VGG, the Region Proposal Network (RPN),
# and the Classification Network (ROI-pooling + Fully Connected layers), organized
# as a tree. Tree has 4 branches:
#
# VGG -> b1 -> Conv (3x3) -> b2 -> Conv (1x1) -> CrossEntropyMulti (objectness label)
#                            b2 -> Conv (1x1) -> SmoothL1Loss (bounding box targets)
#        b1 -> PropLayer -> ROI -> Affine -> Affine -> b3 -> Affine -> CrossEntropyMulti (category)
#                                                      b3 -> Affine -> SmoothL1Loss (bounding box)
#

# define the branch points
b1 = BranchNode(name="conv_branch")
b2 = BranchNode(name="rpn_branch")
b3 = BranchNode(name="roi_branch")

# define VGG
VGG = util.add_vgg_layers()

# define RPN
rpn_init = dict(init=Gaussian(scale=0.01), bias=Constant(0))
# these references are passed to the ProposalLayer.
RPN_3x3 = Conv((3, 3, 512), activation=Rectlin(), padding=1, strides=1, **rpn_init)
RPN_1x1_obj = Conv((1, 1, 18), activation=PixelwiseSoftmax(c=2), padding=0, strides=1, **rpn_init)
RPN_1x1_bbox = Conv((1, 1, 36), activation=Identity(), padding=0, strides=1, **rpn_init)

# define ROI classification network
ROI = [ProposalLayer([RPN_1x1_obj, RPN_1x1_bbox],
                      train_set.get_global_buffers(),
                      num_rois=frcn_rois_per_img),
       RoiPooling(HW=(7, 7)),
       Affine(nout=4096, init=Gaussian(scale=0.005),
              bias=Constant(.1), activation=Rectlin()),
       Dropout(keep=0.5),
       Affine(nout=4096, init=Gaussian(scale=0.005),
              bias=Constant(.1), activation=Rectlin()),
       Dropout(keep=0.5)]

ROI_category = Affine(nout=21, init=Gaussian(scale=0.01), bias=Constant(0), activation=Softmax())
ROI_bbox = Affine(nout=84, init=Gaussian(scale=0.001), bias=Constant(0), activation=Identity())

# build the model
# the four branches of the tree mirror the branches listed above
frcn_tree = Tree([ROI + [b3, ROI_category],
                [b3, ROI_bbox]
                ])

model = Model(layers=Tree([VGG + [b1, RPN_3x3, b2, RPN_1x1_obj],
                           [b2, RPN_1x1_bbox],
                           [b1] + [frcn_tree],
                           ]))

# set up cost different branches, respectively
weights = 1.0 / (rpn_rois_per_img)

frcn_tree_cost = Multicost(costs=[
                                  GeneralizedCostMask(costfunc=CrossEntropyMulti()),
                                  GeneralizedCostMask(costfunc=SmoothL1Loss())
                                  ],
                           weights=[1, 1])

cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti(), weights=weights),
                        GeneralizedCostMask(costfunc=SmoothL1Loss(sigma=3.0), weights=weights),
                        frcn_tree_cost,
                        ],
                 weights=[1, 1, 1])

# setup optimizer
schedule_w = Schedule(step_config=[6], change=[0.0001])
schedule_b = Schedule(step_config=[6], change=[0.0002])
opt_w = GradientDescentMomentum(0.001 * learning_rate_scale, 0.9, wdecay=0.0005, schedule=schedule_w)
opt_b = GradientDescentMomentum(0.002 * learning_rate_scale, 0.9, wdecay=0.0005, schedule=schedule_b)
optimizer = MultiOptimizer({'default': opt_w, 'Bias': opt_b})

# if training a new model, seed the image model conv layers with pre-trained weights
# otherwise, just load the model file
if args.model_file is None:
    util.load_vgg_weights(model, '~/private-neon/examples/faster-rcnn/')

callbacks = Callbacks(model, eval_set=train_set, **args.callback_args)
callbacks.add_callback(TrainMulticostCallback())


model.fit(train_set, optimizer=optimizer,
          num_epochs=num_epochs, cost=cost, callbacks=callbacks)

# Fast R-CNN model requires scale the bbox regression branch linear layer weights
# before saving the model
model = scale_bbreg_weights(model, train_set.bbtarget_means, train_set.bbtarget_stds)

save_obj(model.serialize(keep_states=True), args.save_path)

# print 'running eval...'
# metric_train = model.eval(train_set, metric=ObjectDetection())
# print 'Train: label accuracy - {}%, object deteciton logloss - {}'.format(metric_train[0]*100,
#                                                                           metric_train[1])

# metric_test = model.eval(test_set, metric=ObjectDetection())
# print 'Test: label accuracy - {}%, object deteciton logloss - {}'.format(metric_test[0]*100,
#                                                                          metric_test[1])
