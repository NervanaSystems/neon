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
Trains a Fast-RCNN model on PASCAL VOC dataset.
This Fast-RCNN is based on VGG16 that was pre-trained using ImageI1K.

By default, the script will download the pre-trained VGG16 from neon model zoo
and seed the convolution and pooling layers. And Fast R-CNN starts training from
that. If the script is given --model_file, it will continue training the
Fast R-CNN from the given model file.

Reference:
    "Fast R-CNN"
    http://arxiv.org/pdf/1504.08083v2.pdf
    https://github.com/rbgirshick/fast-rcnn

Usage:
    python examples/fast-rcnn/train.py -e 20 --save_path frcn_vgg.pkl

Notes:

    1. For VGG16 based Fast R-CNN model, we can support training/testing with small
    batch size such as, 2 or 3 images per batch. The model training will converge
    around 20 epochs. With 3 images per batch, and 64 ROIs per image, the training
    consumes about 11G memory.

    2. The original caffe model goes through 40000 iteration (mb) of training, with
    2 images per minibatch.

    3. The dataset will cache the preprocessed file and re-use that if the same
    configuration of the dataset is used again. The cached file by default is in
    ~/nervana/data/VOCDevkit/VOC<year>/train_< >.pkl or
    ~/nervana/data/VOCDevkit/VOC<year>/inference_< >.pkl

"""

from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.callbacks.callbacks import Callbacks, TrainMulticostCallback
from neon.util.persist import save_obj
from objectlocalization import PASCAL
from neon.initializers import Gaussian, Constant, Xavier, GlorotUniform
from neon.transforms import Rectlin, Softmax, Identity, Logistic, CrossEntropyMulti, SmoothL1Loss, PixelwiseSoftmax
from neon.layers import Conv, Pooling, Affine, BranchNode, Tree, Multicost, GeneralizedCostMask
from neon.models import Model
import os
import util
# main script

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--lr_scale', help='learning rate scale', default=16.0)
args = parser.parse_args(gen_be=False)

if args.save_path is None:
    args.save_path = 'frcn_vgg.pkl'

if args.callback_args['save_path'] is None:
    args.callback_args['save_path'] = args.save_path

if args.callback_args['serialize'] is None:
    args.callback_args['serialize'] = min(args.epochs, 10)

if args.data_dir is None:
    args.data_dir = '/usr/local/data/'

# hyperparameters
args.batch_size = 1

num_epochs = args.epochs
n_mb = None
img_per_batch = args.batch_size
rois_per_img = 256
frcn_fine_tune = False
learning_rate_scale = 1.0/float(args.lr_scale)

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))


train_set = PASCAL('trainval', '2007', path=args.data_dir, n_mb=n_mb,
                   img_per_batch=img_per_batch, rois_per_img=rois_per_img,
                   add_flipped=True, shuffle=True, rebuild_cache=False)

# test_set = PASCAL('test', '2007', path=args.data_dir, n_mb=n_mb,
#                    img_per_batch=img_per_batch, rois_per_img=rois_per_img)
# setup model
# add VGG layers
conv_layers = util.add_vgg_layers()

b1 = BranchNode(name="rpn_branch")

# add 3x3 Conv, and the first branch for the bbox objectness
conv_layers += [
     Conv((3, 3, 512), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(),
          padding=1, strides=1),
     b1,
     Conv((1, 1, 18), init=Gaussian(scale=0.01), bias=Constant(0), activation=PixelwiseSoftmax(c=2),
          padding=0, strides=1)
]

# add second branch for the bbox_regression
bbox_regression = [b1, Conv((1, 1, 36), init=Gaussian(scale=0.01), bias=Constant(0),
                   activation=Identity(), padding=0, strides=1)]

model = Model(layers=Tree([conv_layers, bbox_regression]))


# set up cost for the objectness and regression branchs, respectively
weights = 1.0/(256)
cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti(), weights=weights),
                        GeneralizedCostMask(costfunc=SmoothL1Loss(sigma=3.0), weights=weights)])

# setup optimizer
# schedule = Schedule(step_config=[6], change=[0.1])
opt_w = GradientDescentMomentum(0.001 * learning_rate_scale, 0.9, wdecay=0.0005)
opt_b = GradientDescentMomentum(0.002 * learning_rate_scale, 0.9, wdecay=0.0005)
optimizer = MultiOptimizer({'default': opt_w, 'Bias': opt_b})


# if training a new model, seed the image model conv layers with pre-trained weights
# otherwise, just load the model file
if args.model_file is None:
    util.load_vgg_weights(model, "~/private-neon/examples/rpn/")

callbacks = Callbacks(model, eval_set=train_set, **args.callback_args)
callbacks.add_callback(TrainMulticostCallback())
#
# outputs = model.get_outputs(train_set)

model.fit(train_set, optimizer=optimizer,
          num_epochs=num_epochs, cost=cost, callbacks=callbacks)
#
# # Fast R-CNN model requires scale the bbox regression branch linear layer weights
# # before saving the model
# model = scale_bbreg_weights(model, train_set.bbtarget_means, train_set.bbtarget_stds)
#
# save_obj(model.serialize(keep_states=True), args.save_path)
#
# print 'running eval...'
# metric_train = model.eval(train_set, metric=ObjectDetection())
# print 'Train: label accuracy - {}%, object deteciton logloss - {}'.format(metric_train[0]*100,
#                                                                           metric_train[1])
#
# metric_test = model.eval(test_set, metric=ObjectDetection())
# print 'Test: label accuracy - {}%, object deteciton logloss - {}'.format(metric_test[0]*100,
#                                                                          metric_test[1])
