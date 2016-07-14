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
from neon.initializers import Gaussian, Constant
from neon.transforms import Rectlin, Identity, Softmax, PixelwiseSoftmax
from neon.layers import Conv, Affine, BranchNode, Tree, Dropout
from neon.models import Model
from roi_pooling import RoiPooling
from proposal_layer import ProposalLayer
from objectlocalization import PASCAL
from neon.util.argparser import NeonArgparser, extract_valid_args
from bbox_transform import bbox_transform_inv, clip_boxes
import util
import sys
import os
import numpy as np

SCALE_BBTARGETS = True

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()
assert args.model_file is not None, "need a model file to do Faster R-CNN testing"

# hyperparameters
args.batch_size = 1
n_mb = None
img_per_batch = args.batch_size
rois_per_img = 256
frcn_rois_per_img = 128
rpn_rois_per_img = 256

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

image_set = 'test'
year = '2007'
valid_set = PASCAL(image_set, year, path=args.data_dir, n_mb=n_mb,
                   img_per_batch=img_per_batch, rpn_rois_per_img=rpn_rois_per_img,
                   frcn_rois_per_img=frcn_rois_per_img, add_flipped=False, shuffle=False,
                   rebuild_cache=False)

num_classes = valid_set.num_classes
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

# create proposal layer
proposalLayer = ProposalLayer([RPN_1x1_obj, RPN_1x1_bbox],
                              valid_set.get_global_buffers(),
                              num_rois=frcn_rois_per_img, inference=True)

# define ROI classification network
ROI = [proposalLayer,
       RoiPooling(HW=(7, 7)),
       Affine(nout=4096, init=Gaussian(scale=0.005),
              bias=Constant(.1), activation=Rectlin()),
       Dropout(keep=0.5),
       Affine(nout=4096, init=Gaussian(scale=0.005),
              bias=Constant(.1), activation=Rectlin()),
       Dropout(keep=0.5)]

ROI_category = Affine(nout=num_classes, init=Gaussian(scale=0.01),
                      bias=Constant(0), activation=Softmax())
ROI_bbox = Affine(nout=4 * num_classes, init=Gaussian(scale=0.001),
                  bias=Constant(0), activation=Identity())

# build the model
# the four branches of the tree mirror the branches listed above
frcn_tree = Tree([ROI + [b3, ROI_category],
                 [b3, ROI_bbox]
                  ])

model = Model(layers=Tree([VGG + [b1, RPN_3x3, b2, RPN_1x1_obj],
                           [b2, RPN_1x1_bbox],
                           [b1] + [frcn_tree],
                           ]))


# load parameters and initialize model
model.load_params(args.model_file)
model.initialize(dataset=valid_set)

if SCALE_BBTARGETS:
    model = util.scale_bbreg_weights(model, [0.0, 0.0, 0.0, 0.0],
                                     [0.1, 0.1, 0.2, 0.2], num_classes)

# run inference

# detection parameters
num_images = valid_set.num_image_entries if n_mb is None else n_mb
max_per_image = 100   # maximum detections per image
thresh = 0.001  # minimum threshold on score
nms_thresh = 0.4  # threshold used for non-maximum supression

# all detections are collected into:
#    all_boxes[cls][image] = N x 5 array of detections in
#    (x1, y1, x2, y2, score)
all_boxes = [[[] for _ in xrange(num_images)]
             for _ in xrange(num_classes)]

last_strlen = 0
for mb_idx, (x, y) in enumerate(valid_set):

    prt_str = "Finished: {} / {}".format(mb_idx, valid_set.nbatches)
    sys.stdout.write('\r' + ' '*last_strlen + '\r')
    sys.stdout.write(prt_str.encode('utf-8'))
    last_strlen = len(prt_str)
    sys.stdout.flush()

    outputs = model.fprop(x, inference=True)

    # retrieve image metadata
    im_shape = valid_set.im_shape.get()
    im_scale = valid_set.im_scale.get()

    # retrieve region proposals generated by the model
    (proposals, num_proposals) = proposalLayer.get_proposals()
    proposals = proposals.get()[:num_proposals, :]  # remove padded proposals
    boxes = proposals[:, 1:5] / im_scale  # scale back to real image space

    # obtain bounding box corrections from the frcn layers
    scores = outputs[2][0].get()[:, :num_proposals].T
    bbox_deltas = outputs[2][1].get()[:, :num_proposals].T

    # apply bounding box corrections to the region proposals
    pred_boxes = bbox_transform_inv(boxes, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape)

    # Skip the background class, start processing from class 1
    for j in xrange(1, valid_set.num_classes):
        inds = np.where(scores[:, j] > thresh)[0]

        # obtain class-speciic boxes and scores
        cls_scores = scores[inds, j]
        cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

        # apply non-max supression
        keep = util.nms(cls_dets, nms_thresh)
        cls_dets = cls_dets[keep, :]

        # store results
        all_boxes[j][mb_idx] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:

        # obtain flattened list of all image scores
        image_scores = np.hstack([all_boxes[j][mb_idx][:, -1]
                                  for j in xrange(1, valid_set.num_classes)])

        if len(image_scores) > max_per_image:
            # compute threshold needed to keep the top max_per_image
            image_thresh = np.sort(image_scores)[-max_per_image]

            # apply threshold
            for j in xrange(1, valid_set.num_classes):
                keep = np.where(all_boxes[j][mb_idx][:, -1] >= image_thresh)[0]
                all_boxes[j][mb_idx][keep, :]

print 'Evaluating detections'
output_dir = 'frcn_output'
annopath, imagesetfile = valid_set.evaluation(all_boxes, os.path.join(args.data_dir, output_dir))
util.run_voc_eval(annopath, imagesetfile, year, image_set, valid_set.CLASSES,
                  os.path.join(args.data_dir, output_dir))
