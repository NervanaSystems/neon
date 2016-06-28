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
from neon.backends import gen_backend
from neon.initializers import Gaussian, Constant, Xavier, GlorotUniform
from neon.transforms import Rectlin, Identity, Softmax, CrossEntropyMulti, SmoothL1Loss, PixelwiseSoftmax
from neon.layers import Conv, Pooling, Affine, BranchNode, Tree, Multicost, GeneralizedCost, GeneralizedCostMask, Dropout
from neon.models import Model
from roi_pooling import RoiPooling
from proposal_layer import ProposalLayer
from objectlocalization import PASCALInference
from neon.util.argparser import NeonArgparser, extract_valid_args
import util
import sys
import os
import numpy as np
import heapq

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()
assert args.model_file is not None, "need a model file to do Faster R-CNN testing"

# hyperparameters
args.batch_size = 1
n_mb = None
img_per_batch = args.batch_size
rois_per_img = 300
frcn_rois_per_img = 300
post_nms_N = 300

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# setup dataset
image_set = 'test'
year = '2007'
valid_set = PASCALInference(image_set, year, path=args.data_dir,
                               n_mb=n_mb, rpn_rois_per_img=rois_per_img,
                               frcn_rois_per_img=frcn_rois_per_img)

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
                      valid_set.get_global_buffers(),
                      num_rois=frcn_rois_per_img,
                      post_nms_N=post_nms_N),
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
model = Model(layers=Tree([VGG + [b1, RPN_3x3, b2, RPN_1x1_obj],
                           [b2, RPN_1x1_bbox],
                           [b1] + ROI + [b3, ROI_category],
                           [b3, ROI_bbox],
                           ]))


model.load_params(args.model_file)
model.initialize(dataset=valid_set)

# set up the detection params
num_images = valid_set.num_image_entries if n_mb is None else n_mb
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

print 'total batches {}'.format(valid_set.nbatches)

PASCAL_VOC_CLASSES = valid_set.CLASSES

last_strlen = 0
# iterate through minibatches of the dataset
for mb_idx, (x, db) in enumerate(valid_set):

    # print testing progress
    prt_str = "Finished: {} / {}".format(mb_idx, valid_set.nbatches)
    sys.stdout.write('\r' + ' '*last_strlen + '\r')
    sys.stdout.write(prt_str.encode('utf-8'))
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
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[cls_ind]
        # only keep that ones with high enough scores
        # and use gt_class being 0 as the ss ROIs, not the gt ones
        keep = np.where((cls_scores.reshape(-1, 1) > thresh[cls_ind]) & (db['gt_classes'] == 0))[0]
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

print '\nApplying NMS to all detections'
all_boxes = valid_set.apply_nms(all_boxes, NMS_THRESH)

print 'Evaluating detections'
output_dir = 'frcn_output'
annopath, imagesetfile = valid_set.evaluation(all_boxes, os.path.join(args.data_dir, output_dir))
util.run_voc_eval(annopath, imagesetfile, year, image_set, PASCAL_VOC_CLASSES,
             os.path.join(args.data_dir, output_dir))
