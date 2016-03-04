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
This Fast-RCNN is based on Alexnet that was pre-trained in ImageI1K using neon.

Reference:
    "Fast R-CNN"
    http://arxiv.org/pdf/1504.08083v2.pdf
    https://github.com/rbgirshick/fast-rcnn

Usage:
    python examples/fast_rcnn_alexnet.py -e 100 --save_path frcn_alexnet.pickle

Notes:

    1. Neon currently has to process images with batch size being multiple of 32,
    and this model uses different learning rate, the training will converge to
    the level of caffe model around 100 epochs.
    The original caffe model goes through 40000 iteration (mb) of training,
    with 2 images per minibatch.

    2. The caffe model we used for comparison is based on a pre-trained Alexnet
    we trained in Neon and converted into Caffe format. The resulted training
    error running in Caffe is close to the one from running the published
    Fast-RCNN model using CaffeNet.
    Neon support of Fast-RCNN based on VGG16 is coming soon.

    3. This example demonstrates the Fast-RCNN training process. Neon support of
    the inference pipeline is coming soon.

"""
import os

from neon.backends import gen_backend
from neon.data import PASCALVOC
from neon.data.datasets import Dataset
from neon.initializers import Gaussian, Constant
from neon.transforms import (Rectlin, Softmax, Identity, CrossEntropyMulti,
                             SmoothL1Loss, ObjectDetection)
from neon.models import Model
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.optimizers import GradientDescentMomentum, MultiOptimizer
from neon.layers import (Conv, Pooling, Affine, Dropout, RoiPooling,
                         BranchNode, Multicost, GeneralizedCost,
                         GeneralizedCostMask, Tree)
from neon.callbacks.callbacks import Callbacks
from neon.util.persist import load_obj


# functions

def load_imagenet_weights(model, path):
    # load a pre-trained Alexnet from Neon model zoo to the local
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/alexnet/'
    filename = 'alexnet.p'
    size = 488808400

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    print 'De-serializing the pre-trained Alexnet using ImageNet I1K ...'
    pdict = load_obj(filepath)

    param_layers = [l for l in model.layers.layers[0].layers[0].layers]
    param_dict_list = pdict['model']['config']['layers']
    for layer, ps in zip(param_layers, param_dict_list):
        print layer.name, ps['config']['name']
        layer.load_weights(ps, load_states=True)
        if ps['config']['name'] == 'Pooling_2':
            print 'Only load the pre-trained weights up to conv5 layer of Alexnet'
            break


# main script

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

# Override save path if None
if args.save_path is None:
    args.save_path = 'frcn_alexnet.pickle'

if args.callback_args['save_path'] is None:
    args.callback_args['save_path'] = args.save_path

if args.callback_args['serialize'] is None:
    args.callback_args['serialize'] = min(args.epochs, 10)

num_epochs = args.epochs

# hyperparameters
args.batch_size = 32
n_mb = None
img_per_batch = args.batch_size
rois_per_img = 64
frcn_fine_tune = False
learning_rate_scale = 1.0/10

if frcn_fine_tune is True:
    learning_rate_scale = 1.0/16

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# setup training dataset
train_set = PASCALVOC('trainval', '2007', path=args.data_dir, output_type=0,
                      n_mb=n_mb, img_per_batch=img_per_batch, rois_per_img=rois_per_img)

# setup layers

b1 = BranchNode(name="b1")

imagenet_layers = [
    Conv((11, 11, 64), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(),
         padding=3, strides=4),
    Pooling(3, strides=2),
    Conv((5, 5, 192), init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin(),
         padding=2),
    Pooling(3, strides=2),
    Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(),
         padding=1),
    Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1), activation=Rectlin(),
         padding=1),
    Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1), activation=Rectlin(),
         padding=1),
    # The following layers are used in Alexnet, but not being used for Fast-RCNN
    # Pooling(3, strides=2),
    # Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
    # Dropout(keep=0.5),
    # Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
    # Dropout(keep=0.5),
    # Affine(nout=1000, init=Gaussian(scale=0.01), bias=Constant(-7), activation=Softmax())
]

class_score = Affine(
    nout=21, init=Gaussian(scale=0.01), bias=Constant(0), activation=Softmax())
bbox_pred = Affine(
    nout=84, init=Gaussian(scale=0.001), bias=Constant(0), activation=Identity())

frcn_layers = [
    RoiPooling(layers=imagenet_layers, HW=(6, 6), bprop_enabled=frcn_fine_tune),
    Affine(nout=4096, init=Gaussian(scale=0.005),
           bias=Constant(.1), activation=Rectlin()),
    Dropout(keep=0.5),
    Affine(nout=4096, init=Gaussian(scale=0.005),
           bias=Constant(.1), activation=Rectlin()),
    Dropout(keep=0.5),
    b1,
    class_score
]
bb_layers = [
    b1,
    bbox_pred,
]


# setup optimizer
opt_w = GradientDescentMomentum(0.001 * learning_rate_scale, 0.9, wdecay=0.0005)
opt_b = GradientDescentMomentum(0.002 * learning_rate_scale, 0.9)

optimizer = MultiOptimizer({'default': opt_w, 'Bias': opt_b})

# setup model

model = Model(layers=Tree([frcn_layers, bb_layers]))

# if training a new model, seed the Alexnet conv layers with pre-trained weights
# otherwise, just load the model file
if args.model_file is None:
    load_imagenet_weights(model, args.data_dir)

cost = Multicost(costs=[GeneralizedCost(costfunc=CrossEntropyMulti()),
                        GeneralizedCostMask(costfunc=SmoothL1Loss())],
                 weights=[1, 1])

callbacks = Callbacks(model, **args.callback_args)

model.fit(train_set, optimizer=optimizer,
          num_epochs=num_epochs, cost=cost, callbacks=callbacks)


print 'running eval on the training set...'
metric_train = model.eval(train_set, metric=ObjectDetection())
print 'Train: label accuracy - {}%, object detection SmoothL1Loss - {}'.format(
    metric_train[0]*100,
    metric_train[1])
