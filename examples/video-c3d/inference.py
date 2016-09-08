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
Test C3D on UCF-101 dataset. Computes video level accuracy by averaging
predictions from 10 random 16-frame clips per each video.

Reference:
    "Learning Spatiotemporal Features with 3D Convolutional Networks"
    http://arxiv.org/pdf/1412.0767.pdf

Usage:
    First train a video-c3d model using train.py. Then run testing with:

    python examples/video-c3d/inference.py --data_dir <preprocessed_video_dir>
                                           --batch_size 32
                                           --model_file UCF101-C3D.p
"""

import os
import random
import numpy as np
from neon import logger as neon_logger
from neon.util.argparser import NeonArgparser
from neon.data import DataLoader, VideoParams, ImageParams

from network import create_network

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()
assert args.model_file is not None, "need a model file for testing"

# setup data provider
testdir = os.path.join(args.data_dir, 'test1')

shape = dict(channel_count=3, height=112, width=112, scale_min=128, scale_max=128)

testParams = VideoParams(frame_params=ImageParams(center=True, flip=False, **shape),
                         frames_per_clip=16)

common = dict(target_size=1, nclasses=101, datum_dtype=np.uint8)

test_set = DataLoader(set_name='val', repo_dir=testdir, media_params=testParams,
                      shuffle=False, **common)


def get_model_pred(model_file, dataset):
    # model creation
    model = create_network()
    model.load_params(model_file)
    model.initialize(dataset=dataset)
    # pred will have shape (num_clips, num_classes) and contain class probabilities
    pred = model.get_outputs(test_set)
    return pred


def accumulate_video_pred(pred):
    #  Index file will look like:
    #  filename,label1
    #  WritingOnBoard/v_WritingOnBoard_g05_c06_6.avi,99
    #  video_name will be WritingOnBoard/v_WritingOnBoard_g05
    video_pred = {}
    with open(test_set.index_file, 'r') as index_file:
        # Skip header
        index_file.readline()
        for i, clip_file in enumerate(index_file):
            video_name = '_'.join(clip_file.split('_')[:2])
            label = int(clip_file.split(',')[-1])
            if video_name not in video_pred:
                video_pred[video_name] = (label, [pred[i, :]])
            else:
                video_pred[video_name][1].append(pred[i, :])
    return video_pred

pred = get_model_pred(args.model_file, test_set)
video_pred = accumulate_video_pred(pred)

top1acc, top5acc = 0.0, 0.0
for video_name, (label, prob_list) in list(video_pred.items()):
    # Sample 10 random clips per each video and average probabilities
    sample = random.sample(prob_list, 10)
    avg_prob = np.sum(sample, axis=0) / len(sample)
    # Get top 5 predictions
    top5pred = np.argsort(avg_prob)[-5:]
    if label == top5pred[-1]:
        top1acc += 1
    if label in set(top5pred):
        top5acc += 1

neon_logger.display("Top 1 Accuracy: {} Top 5 Accuracy: {}".format(top1acc / len(video_pred),
                                                                   top5acc / len(video_pred)))
