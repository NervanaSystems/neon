#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
Train a Fast-RCNN model on the PASCAL VOC dataset.

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

from neon import logger as neon_logger
from neon.backends import gen_backend
from neon.data import PASCALVOCTrain
from neon.transforms import CrossEntropyMulti, SmoothL1Loss, ObjectDetection
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.optimizers import GradientDescentMomentum, MultiOptimizer
from neon.callbacks.callbacks import Callbacks
from neon.layers import Multicost, GeneralizedCostMask
from neon.util.persist import save_obj
from util import load_vgg_weights, create_frcn_model, scale_bbreg_weights

# main script

# parse the command line arguments
parser = NeonArgparser(__doc__, default_overrides=dict(batch_size=4))
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args(gen_be=False)

# Override save path if None
if args.save_path is None:
    args.save_path = 'frcn_vgg.pkl'

if args.callback_args['save_path'] is None:
    args.callback_args['save_path'] = args.save_path

if args.callback_args['serialize'] is None:
    args.callback_args['serialize'] = min(args.epochs, 10)

# hyperparameters
args.batch_size = 4

num_epochs = args.epochs
n_mb = None
img_per_batch = args.batch_size
rois_per_img = 64
frcn_fine_tune = False
learning_rate_scale = 1.0 / 10

if frcn_fine_tune is True:
    learning_rate_scale = 1.0 / 16

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

if be.gpu_memory_size < 11 * 1024 * 1024 * 1024:
    exit("ERROR: This model requires at least 11GB GPU memory to be run.")

# setup training dataset
train_set = PASCALVOCTrain('trainval', '2007', path=args.data_dir, n_mb=n_mb,
                           img_per_batch=img_per_batch, rois_per_img=rois_per_img,
                           rois_random_sample=True,
                           add_flipped=False, subset_pct=args.subset_pct)
test_set = PASCALVOCTrain('test', '2007', path=args.data_dir, n_mb=n_mb,
                          img_per_batch=img_per_batch, rois_per_img=rois_per_img,
                          rois_random_sample=True,
                          add_flipped=False)

# setup model
model = create_frcn_model(frcn_fine_tune)

# setup optimizer
opt_w = GradientDescentMomentum(
    0.001 * learning_rate_scale, 0.9, wdecay=0.0005)
opt_b = GradientDescentMomentum(0.002 * learning_rate_scale, 0.9)

optimizer = MultiOptimizer({'default': opt_w, 'Bias': opt_b})

# if training a new model, seed the image model conv layers with pre-trained weights
# otherwise, just load the model file
if args.model_file is None:
    load_vgg_weights(model, args.data_dir)

cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti()),
                        GeneralizedCostMask(costfunc=SmoothL1Loss())],
                 weights=[1, 1])

callbacks = Callbacks(model, eval_set=test_set, **args.callback_args)

model.fit(train_set, optimizer=optimizer,
          num_epochs=num_epochs, cost=cost, callbacks=callbacks)

# Fast R-CNN model requires scale the bbox regression branch linear layer weights
# before saving the model
model = scale_bbreg_weights(
    model, train_set.bbtarget_means, train_set.bbtarget_stds)

save_obj(model.serialize(keep_states=True), args.save_path)

neon_logger.display('running eval...')

metric_train = model.eval(train_set, metric=ObjectDetection())
neon_logger.display(
    'Train: label accuracy - {}%, object detection logloss - {}'.format(metric_train[0] * 100,
                                                                        metric_train[1]))

metric_test = model.eval(test_set, metric=ObjectDetection())
neon_logger.display(
    'Test: label accuracy - {}%, object detection logloss - {}'.format(metric_test[0] * 100,
                                                                       metric_test[1]))
