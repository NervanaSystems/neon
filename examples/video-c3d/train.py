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
Trains C3D on UCF-101 dataset.

Reference:
    "Learning Spatiotemporal Features with 3D Convolutional Networks"
    http://arxiv.org/pdf/1412.0767.pdf

Usage:
    Run neon/examples/video-c3d/preprocess.py to preprocess videos into clips
    for training and test splits. Point the --data_dir arg to the directory
    containing the preprocessed train and test folders. Try a smaller batch
    size if memory is limited.

    python examples/video-c3d/train.py --data_dir <preprocessed_video_dir>
                                       --batch_size 32
                                       --epochs 18
                                       --save_path UCF101-C3D.p
"""

import os
import numpy as np
from neon.layers import GeneralizedCost
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.transforms import CrossEntropyMulti, Accuracy
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.data import DataLoader, VideoParams, ImageParams


from network import create_network

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

# setup data provider
traindir = os.path.join(args.data_dir, 'train1')
testdir = os.path.join(args.data_dir, 'test1')

shape = dict(channel_count=3, height=112, width=112, scale_min=128, scale_max=128)

trainParams = VideoParams(frame_params=ImageParams(center=False, flip=True, **shape),
                          frames_per_clip=16)
testParams = VideoParams(frame_params=ImageParams(center=True, flip=False, **shape),
                         frames_per_clip=16)

common = dict(target_size=1, nclasses=101, datum_dtype=np.uint8)

train = DataLoader(set_name='train', repo_dir=traindir, media_params=trainParams,
                   shuffle=True, subset_percent=args.subset_pct, **common)
test = DataLoader(set_name='val', repo_dir=testdir, media_params=testParams,
                  shuffle=False, **common)

# model creation
model = create_network()

# setup callbacks
callbacks = Callbacks(model, eval_set=test, **args.callback_args)

# gradient descent with momentum, weight decay, and learning rate decay schedule
learning_rate_sched = Schedule(range(6, args.epochs, 6), 0.1)
opt_gdm = GradientDescentMomentum(0.003, 0.9, wdecay=0.005, schedule=learning_rate_sched)
opt_biases = GradientDescentMomentum(0.006, 0.9, schedule=learning_rate_sched)
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

# train model
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

# output accuracies
print('Train Accuracy = %.1f%%' % (model.eval(train, metric=Accuracy()) * 100))
print('Test Accuracy = %.1f%%' % (model.eval(test, metric=Accuracy()) * 100))
