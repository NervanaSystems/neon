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
from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, StepSchedule
from neon.callbacks.callbacks import Callbacks, TrainMulticostCallback
from neon.util.persist import save_obj
from objectlocalization import PASCAL
from neon.transforms import CrossEntropyMulti, SmoothL1Loss
from neon.layers import Multicost, GeneralizedCostMask
import util
import sys
import os
import faster_rcnn
# main script

# parse the command line arguments
parser = NeonArgparser(__doc__, default_overrides={'batch_size': 1})
parser.add_argument('--lr_scale', type=float, help='learning rate scale', default=16.0)
parser.add_argument('--lr_step', type=float, help="step for learning schedule", default=10.0)
parser.add_argument('--epoch_step', type=float, help="epoch to step the learning rate", default=6)
parser.add_argument('--evaluate', action='store_true', help="evaluate mAP on test set")
parser.add_argument('--output_dir', default='frcn_output',
                    help='Directory to save AP metric results. Path is relative to data_dir.')
args = parser.parse_args(gen_be=False)

# hyperparameters
assert args.batch_size is 1, "Faster-RCNN only supports batch size 1"

n_mb = None
rpn_rois_per_img = 256  # number of rois to sample to train rpn
frcn_rois_per_img = 128  # number of rois to sample to train frcn
lr_scale = 1.0 / float(args.lr_scale)

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))
be.enable_winograd = 4

year = '2007'

train_set = PASCAL('trainval', year, path=args.data_dir, n_mb=n_mb,
                   rpn_rois_per_img=rpn_rois_per_img, frcn_rois_per_img=frcn_rois_per_img,
                   add_flipped=True, shuffle=True, rebuild_cache=True)

# build the Faster-RCNN model
model = faster_rcnn.build_model(train_set, frcn_rois_per_img, inference=False)

# set up cost different branches, respectively
weights = 1.0 / (rpn_rois_per_img)

frcn_tree_cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti()),
                                  GeneralizedCostMask(costfunc=SmoothL1Loss())
                                  ], weights=[1, 1])

cost = Multicost(costs=[GeneralizedCostMask(costfunc=CrossEntropyMulti(), weights=weights),
                        GeneralizedCostMask(costfunc=SmoothL1Loss(sigma=3.0), weights=weights),
                        frcn_tree_cost,
                        ],
                 weights=[1, 1, 1])

# setup optimizer
schedule_w = StepSchedule(step_config=[args.epoch_step],
                          change=[0.001 * lr_scale / args.lr_step])
schedule_b = StepSchedule(step_config=[args.epoch_step],
                          change=[0.002 * lr_scale / args.lr_step])

opt_w = GradientDescentMomentum(0.001 * lr_scale, 0.9, wdecay=0.0005, schedule=schedule_w)
opt_b = GradientDescentMomentum(0.002 * lr_scale, 0.9, wdecay=0.0005, schedule=schedule_b)
opt_skip = GradientDescentMomentum(0.0, 0.0)

optimizer = MultiOptimizer({'default': opt_w, 'Bias': opt_b,
                            'skip': opt_skip, 'skip_bias': opt_skip})

# if training a new model, seed the image model conv layers with pre-trained weights
# otherwise, just load the model file
if args.model_file is None:
    util.load_vgg_weights(model, args.data_dir)

callbacks = Callbacks(model, eval_set=train_set, **args.callback_args)
callbacks.add_callback(TrainMulticostCallback())

model.fit(train_set, optimizer=optimizer, cost=cost, num_epochs=args.epochs, callbacks=callbacks)

# Scale the bbox regression branch linear layer weights
# before saving the model
model = util.scale_bbreg_weights(model, [0.0, 0.0, 0.0, 0.0],
                                 [0.1, 0.1, 0.2, 0.2], train_set.num_classes)

if args.save_path is not None:
    save_obj(model.serialize(keep_states=True), args.save_path)

if args.evaluate is True:
    assert args.save_path is not None, "For inference, model weights must be saved to save_path."

    #  remove the model used for training since inference requires changes to the model structure
    be.cleanup_backend()

    valid_set = PASCAL('test', year, path=args.data_dir, n_mb=n_mb,
                       rpn_rois_per_img=rpn_rois_per_img, frcn_rois_per_img=frcn_rois_per_img,
                       add_flipped=False, shuffle=False, rebuild_cache=True)

    # detection parameters
    num_images = valid_set.num_image_entries if n_mb is None else n_mb

    # build model with inference=True
    (model, proposalLayer) = faster_rcnn.build_model(valid_set, frcn_rois_per_img, inference=True)

    # load parameters and initialize model
    model.load_params(args.save_path)
    model.initialize(dataset=valid_set)

    # all detections are collected into:
    #    all_boxes[image][cls] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(valid_set.num_classes)]
                 for _ in xrange(num_images)]

    last_strlen = 0
    for mb_idx, (x, y) in enumerate(valid_set):

        prt_str = "Finished: {} / {}".format(mb_idx, valid_set.nbatches)
        sys.stdout.write('\r' + ' '*last_strlen + '\r')
        sys.stdout.write(prt_str.encode('utf-8'))
        last_strlen = len(prt_str)
        sys.stdout.flush()

        # perform forward pass
        outputs = model.fprop(x, inference=True)

        # retrieve image metadata
        im_shape = valid_set.im_shape.get()
        im_scale = valid_set.im_scale.get()

        # retrieve region proposals generated by the model
        (proposals, num_proposals) = proposalLayer.get_proposals()

        # convert outputs to bounding boxes
        boxes = faster_rcnn.get_bboxes(outputs, proposals, num_proposals, valid_set.num_classes,
                                       im_shape, im_scale)

        all_boxes[mb_idx] = boxes

    print 'Evaluating detections'
    output_dir = 'frcn_output'
    annopath, imagesetfile = valid_set.evaluation(all_boxes, os.path.join(args.data_dir, output_dir))
    util.run_voc_eval(annopath, imagesetfile, year, 'test', valid_set.CLASSES,
                      os.path.join(args.data_dir, output_dir))
