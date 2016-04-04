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
Demo a trained Fast-RCNN model to do object detection using PASCAL VOC dataset.
This demo currently runs 1 image at a time.

Reference:
    "Fast R-CNN"
    http://arxiv.org/pdf/1504.08083v2.pdf
    https://github.com/rbgirshick/fast-rcnn

Usage:
    python examples/fast-rcnn/demo.py --model_file frcn_vgg.pkl

Notes:
    1. For VGG16 based Fast R-CNN model, we can support testing with batch size as 1
    images. The testing consumes about 7G memory.

    2. During demo, all the selective search ROIs will be used to go through the network,
    so the inference time varies based on how many ROIs in each image.
    For PASCAL VOC 2007, the average number of SelectiveSearch ROIs is around 2000.

    3. The dataset will cache the preprocessed file and re-use that if the same
    configuration of the dataset is used again. The cached file by default is in
    ~/nervana/data/VOCDevkit/VOC<year>/train_< >.pkl or
    ~/nervana/data/VOCDevkit/VOC<year>/inference_< >.pkl

"""
import os
import numpy as np
from PIL import Image
from neon.backends import gen_backend
from neon.data.pascal_voc import PASCAL_VOC_CLASSES
from neon.data import PASCALVOCInference
from neon.util.argparser import NeonArgparser
from util import create_frcn_model

do_plots = True
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
except ImportError:
    print('matplotlib needs to be installed manually to generate plots needed '
          'for this example.  Skipping plot generation')
    do_plots = False

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--img_prefix', type=str,
                    help='prefix for the saved image file names. If None, use '
                         'the model file name')
args = parser.parse_args(gen_be=True)
assert args.model_file is not None, "need a model file to do Fast R-CNN testing"

if args.img_prefix is None:
    args.img_prefix = os.path.splitext(os.path.basename(args.model_file))[0]

output_dir = os.path.join(args.data_dir, 'frcn_output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# hyperparameters
args.batch_size = 1
n_mb = 40
img_per_batch = args.batch_size
rois_per_img = 5403

# setup dataset
image_set = 'test'
image_year = '2007'
valid_set = PASCALVOCInference(image_set, image_year, path=args.data_dir, n_mb=n_mb,
                               rois_per_img=rois_per_img, shuffle=False)

# setup model

model = create_frcn_model()
model.load_params(args.model_file)
model.initialize(dataset=valid_set)

CONF_THRESH = 0.8
NMS_THRESH = 0.3

# iterate through minibatches of the dataset
for mb_idx, (x, db) in enumerate(valid_set):

    im = np.array(Image.open(db['img_file']))  # This is RGB order
    print db['img_id']

    outputs = model.fprop(x, inference=True)

    scores, boxes = valid_set.post_processing(outputs, db)

    # Visualize detections for each class
    if do_plots:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

    for cls in PASCAL_VOC_CLASSES[1:]:

        # pick out scores and bboxes replated to this class
        cls_ind = PASCAL_VOC_CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[cls_ind]
        # only keep that ones with high enough scores
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        if len(keep) == 0:
            continue

        # with these, do nonmaximum suppression
        cls_boxes = cls_boxes[keep]
        cls_scores = cls_scores[keep]

        keep = valid_set.nonmaximum_suppression(cls_boxes, cls_scores, NMS_THRESH)

        # keep these after nms
        cls_boxes = cls_boxes[keep]
        cls_scores = cls_scores[keep]

        # Draw detected bounding boxes
        inds = np.where(cls_scores >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue

        print 'detect {}'.format(cls)

        if do_plots:
            for i in inds:
                bbox = cls_boxes[i]
                score = cls_scores[i]

                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=3.5)
                    )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(cls, score),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')

            plt.axis('off')
            plt.tight_layout()

    if do_plots:
        fname = os.path.join(output_dir, '{}_{}_{}_{}.png'.format(
            args.img_prefix, image_set,
            image_year, db['img_id']))

        plt.savefig(fname)
        plt.close()
