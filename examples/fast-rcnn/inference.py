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
Test a trained Fast-RCNN model to do object detection using PASCAL VOC dataset.
This test currently runs 1 image at a time.

Reference:
    "Fast R-CNN"
    http://arxiv.org/pdf/1504.08083v2.pdf
    https://github.com/rbgirshick/fast-rcnn

Usage:
    python examples/fast-rcnn/test.py --model_file frcn_vgg.pkl

Notes:
    1. For VGG16 based Fast R-CNN model, we can support testing with batch size as 1
    images. The testing consumes about 7G memory.

    2. During testing/inference, all the selective search ROIs will be used to go
    through the network, so the inference time varies based on how many ROIs in each
    image. For PASCAL VOC 2007, the average number of SelectiveSearch ROIs is around
    2000.

    3. The dataset will cache the preprocessed file and re-use that if the same
    configuration of the dataset is used again. The cached file by default is in
    ~/nervana/data/VOCDevkit/VOC<year>/train_< >.pkl or
    ~/nervana/data/VOCDevkit/VOC<year>/inference_< >.pkl

The mAP evaluation script is adapted from:
https://github.com/rbgirshick/py-faster-rcnn/commit/45e0da9a246fab5fd86e8c96dc351be7f145499f
"""

import sys
import os
import numpy as np
import heapq
from neon import logger as neon_logger
from neon.data.pascal_voc import PASCAL_VOC_CLASSES, PASCALVOCInference
from neon.util.argparser import NeonArgparser
from neon.util.compat import xrange
from util import create_frcn_model, run_voc_eval

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()
assert args.model_file is not None, "need a model file to do Fast R-CNN testing"

# hyperparameters
args.batch_size = 1
n_mb = None
img_per_batch = args.batch_size
rois_per_img = 5403

# setup dataset
image_set = 'test'
year = '2007'
valid_set = PASCALVOCInference(image_set, year, path=args.data_dir,
                               n_mb=n_mb, rois_per_img=rois_per_img)

# setup models
model = create_frcn_model()
model.load_params(args.model_file)
model.initialize(dataset=valid_set)

# set up the detection params
num_images = valid_set.num_images if n_mb is None else n_mb
num_classes = valid_set.num_classes
image_index = valid_set.image_index
# heuristic: keep an average of 40 detections per class per images prior
# to NMS
max_per_set = 40 * num_images
# heuristic: keep at most 100 detection per class per image prior to NMS
max_per_image = 100
# detection thresold for each class (this is adaptively set based on the
# max_per_set constraint)
thresh = -np.inf * np.ones(num_classes)
# top_scores will hold one minheap of scores per class (used to enforce
# the max_per_set constraint)
top_scores = [[] for _ in xrange(num_classes)]
# all detections are collected into:
#    all_boxes[cls][image] = N x 5 array of detections in
#    (x1, y1, x2, y2, score)
all_boxes = [[[] for _ in xrange(num_images)]
             for _ in xrange(num_classes)]

NMS_THRESH = 0.3

neon_logger.display('total batches {}'.format(valid_set.nbatches))

last_strlen = 0
# iterate through minibatches of the dataset
for mb_idx, (x, db) in enumerate(valid_set):

    # print testing progress
    prt_str = "Finished: {} / {}".format(mb_idx, valid_set.nbatches)
    sys.stdout.write('\r' + ' ' * last_strlen + '\r')
    sys.stdout.write(prt_str)
    last_strlen = len(prt_str)
    sys.stdout.flush()

    if hasattr(valid_set, 'actual_seq_len'):
        model.set_seq_len(valid_set.actual_seq_len)

    outputs = model.fprop(x, inference=True)

    scores, boxes = valid_set.post_processing(outputs, db)

    # Skip the background class, start processing from class 1
    for cls in PASCAL_VOC_CLASSES[1:]:

        # pick out scores and bboxes replated to this class
        cls_ind = PASCAL_VOC_CLASSES.index(cls)
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[cls_ind]
        # only keep that ones with high enough scores
        # and use gt_class being 0 as the ss ROIs, not the gt ones
        keep = np.where((cls_scores.reshape(-1, 1) >
                         thresh[cls_ind]) & (db['gt_classes'] == 0))[0]
        if len(keep) == 0:
            continue

        # with these, do nonmaximum suppression
        cls_boxes = cls_boxes[keep]
        cls_scores = cls_scores[keep]
        top_inds = np.argsort(-cls_scores)[:max_per_image]
        cls_scores = cls_scores[top_inds]
        cls_boxes = cls_boxes[top_inds]
        # push new scores onto the minheap
        for val in cls_scores:
            heapq.heappush(top_scores[cls_ind], val)
        # if we've collected more than the max number of detection,
        # then pop items off the minheap and update the class threshold
        if len(top_scores[cls_ind]) > max_per_set:
            while len(top_scores[cls_ind]) > max_per_set:
                heapq.heappop(top_scores[cls_ind])
            thresh[cls_ind] = top_scores[cls_ind][0]

        all_boxes[cls_ind][mb_idx] = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

for j in xrange(1, num_classes):
    for i in xrange(num_images):
        if len(all_boxes[j][i]) > 0:
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

neon_logger.display('\nApplying NMS to all detections')
all_boxes = valid_set.apply_nms(all_boxes, NMS_THRESH)

neon_logger.display('Evaluating detections')
output_dir = 'frcn_output'
annopath, imagesetfile = valid_set.evaluation(
    all_boxes, os.path.join(args.data_dir, output_dir))
run_voc_eval(annopath, imagesetfile, year, image_set, PASCAL_VOC_CLASSES,
             os.path.join(args.data_dir, output_dir))
