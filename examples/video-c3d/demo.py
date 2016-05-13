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
Demos C3D on UCF-101 dataset.

Reference:
    "Learning Spatiotemporal Features with 3D Convolutional Networks"
    http://arxiv.org/pdf/1412.0767.pdf

Usage:
    python examples/video-c3d/demo.py --data_dir <video_dir>
                                      --class_ind_file <label_index_file_map>
                                      --model_weights <trained_pickle_file>
"""

from builtins import map, range, zip
import cv2
import numpy as np
import os
from neon.backends import gen_backend
from neon.data import DataLoader, VideoParams, ImageParams
from neon.util.argparser import NeonArgparser, extract_valid_args
from network import create_network

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--class_ind_file',
                    help='Path of two column file mapping integer'
                         'class labels to their canonical names.')
parser.add_argument('--model_weights', help='Pickle file of trained model weights.')
args = parser.parse_args(gen_be=False)

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))
be.bsz = 32

# setup data provider
shape = dict(channel_count=3, height=112, width=112, scale_min=128, scale_max=128)

testParams = VideoParams(frame_params=ImageParams(center=True, flip=False, **shape),
                         frames_per_clip=16)

common = dict(target_size=1, nclasses=101, datum_dtype=np.uint8)

videos = DataLoader(set_name='val', repo_dir=args.data_dir, media_params=testParams,
                    shuffle=False, **common)

# initialize model
model = create_network()
model.load_params(args.model_weights)
model.initialize(dataset=videos)

# read label index file into dictionary
label_index = {}
with open(args.class_ind_file) as label_index_file:
    for line in label_index_file:
        index, label = line.split()
        label_index[int(index) - 1] = label


def print_label_on_image(frame, top_labels):
    labels = [(label_index[index], "{0:.2f}".format(prob)) for (index, prob) in top_labels]

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    rect_color = (0, 0, 0)
    text_color = (255, 255, 255)
    font_scale = 0.45
    thickness = 1
    start_pt = (10, 10)
    extra_space = (4, 10)

    label_offset = 0
    label_num = 0
    for label, prob in labels:
        if label_num > 0:
            font_scale = .3
        rect_pt = (start_pt[0], start_pt[1] + label_offset)
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        prob_size = cv2.getTextSize(prob, font, font_scale, thickness)[0]
        prob_offset = (prob_size[0] + extra_space[0], 0)
        text_top = tuple(map(sum, list(zip(rect_pt, extra_space))))
        rect_ops_pt = tuple(map(sum, list(zip(text_top, text_size, extra_space, prob_offset))))
        text_bot = (text_top[0], rect_ops_pt[1] - extra_space[1])
        prob_bot = (text_top[0] + text_size[0] + extra_space[0], text_bot[1])
        cv2.rectangle(frame, rect_pt, rect_ops_pt, rect_color, thickness=cv2.cv.CV_FILLED)
        cv2.putText(frame, label, text_bot, font, font_scale, text_color, thickness)
        cv2.putText(frame, prob, prob_bot, font, font_scale, text_color, thickness)
        label_offset += rect_ops_pt[1] - rect_pt[1]
        label_num += 1

    return frame

#  predict label on each video clip then display
original_videos = []
for root, dirs, files in os.walk(os.path.expanduser(args.data_dir)):
    if len(dirs) != 0:
        continue
    for f in files:
        original_videos.append(os.path.join(root, f))

codec = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
vw = cv2.VideoWriter('output.avi', codec, 17, (128, 171))
correct = 0
batch_num = 0

topk = 1

for video, label in videos:
    probs_mb = model.fprop(x=video, inference=True).get()
    for batch_idx in range(be.bsz):
        if batch_num * be.bsz + batch_idx >= len(original_videos):
            break

        probs_labels = probs_mb[:, batch_idx]
        if np.argmax(probs_labels) == np.argmax(label.get()[:, batch_idx]):
            correct += 1
        else:
            continue

        top_probs = np.argpartition(probs_labels, -topk)[-topk:]
        sorted_top_indexes = top_probs[np.argsort(probs_labels[top_probs])]
        top_labels = [(idx, probs_labels[idx]) for idx in sorted_top_indexes]
        top_labels.reverse()

        original_video = cv2.VideoCapture(original_videos[batch_num * be.bsz + batch_idx])

        while True:
            ret, frame = original_video.read()
            if not ret:
                break
            frame = print_label_on_image(frame, top_labels)
            vw.write(frame)
    batch_num += 1
cv2.destroyAllWindows()
