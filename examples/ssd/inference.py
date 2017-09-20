#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
from neon.util.argparser import NeonArgparser
from neon.models.model import Model
import numpy as np
from ssd_container import SSD
from util.voc_eval import voc_eval
from tqdm import tqdm
import json
from ssd_dataloader import build_dataloader
import pickle
import os
from sys import exit
from util.util import plot_image
from collections import OrderedDict

"""
Runs inference using a trained model to compute the MAP score.

Example usage:

python inference.py --config [config file] --model_file trained_ssd.prm
"""


def get_boxes(model, val_set, num_images=0, image_dir=None, score_threshold=0.6):

    n = 0
    all_boxes = []
    all_gt_boxes = []
    with tqdm(total=val_set.ndata) as pbar:  # progress bar

        for (img, (gt_boxes, gt_classes, num_gt_boxes, difficult, im_shape)) in val_set:

            outputs = model.fprop(img, inference=True)

            for (k, boxes) in enumerate(outputs):
                pbar.update(1)

                all_boxes.append(boxes)

                ngt = num_gt_boxes[0, k]
                gtb = gt_boxes[:, k].reshape((-1, 4))

                # retrieve gt boxes
                # we add a extra column to track detections during the AP calculation
                detected = np.array([False] * ngt)
                gtb = np.hstack([gtb[:ngt],
                                 gt_classes[:ngt, k][:, np.newaxis],
                                 difficult[:ngt, k][:, np.newaxis], detected[:, np.newaxis]])

                all_gt_boxes.append(gtb)

                # plot images if needed
                if(n < num_images):
                    gt_boxes = np.copy(gt_boxes.reshape((-1, 4, val_set.be.bsz)))
                    boxes = np.copy(boxes)
                    ngt = num_gt_boxes[0, k]

                    img = plot_image(img=img[:, k].get(), im_shape=im_shape[:, k],
                                     gt_boxes=gt_boxes[:ngt, :, k], boxes=boxes,
                                     score_threshold=score_threshold)

                    file_name = os.path.join(image_dir, 'image_{}.jpg'.format(n))
                    img.save(file_name)
                    n = n + 1

    return (all_boxes, all_gt_boxes)

if __name__ == '__main__':
    """

    Simple example of using the dataloader with pre-generated augmentation data

    """
    arg_defaults = {'batch_size': 0}

    parser = NeonArgparser(__doc__, default_overrides=arg_defaults)
    parser.add_argument('--ssd_config', action='append', required=True, help='ssd json file path')
    parser.add_argument('--height', type=int, help='image height')
    parser.add_argument('--width', type=int, help='image width')
    parser.add_argument('--num_images', type=int, default=0, help='number of images to plot')
    parser.add_argument('--image_dir', type=str, help='folder to save sampled images')
    parser.add_argument('--score_threshold', type=float, help='threshold for predicted scores.')
    parser.add_argument('--output', type=str, help='file to save detected boxes.')
    args = parser.parse_args(gen_be=False)
    if args.model_file is None:
        parser.print_usage()
        exit('You need to specify model file to evaluate.')

    if args.ssd_config:
        args.ssd_config = {k: v for k, v in [ss.split(':') for ss in args.ssd_config]}

    config = json.load(open(args.ssd_config['val']), object_pairs_hook=OrderedDict)

    if args.batch_size == 0:
        args.batch_size = config["batch_size"]

    # setup backend
    be = gen_backend(backend=args.backend, batch_size=args.batch_size,
                     device_id=args.device_id, compat_mode='caffe', rng_seed=1,
                     deterministic_update=True, deterministic=True)
    be.enable_winograd = 0

    config["manifest_root"] = args.manifest_root
    config["manifest_filename"] = args.manifest['val']
    config["batch_size"] = be.bsz

    val_set = build_dataloader(config, args.manifest_root, args.batch_size)

    model = Model(layers=SSD(ssd_config=config['ssd_config'], dataset=val_set))
    model.initialize(dataset=val_set)
    model.load_params(args.model_file)

    if args.num_images > 0:
        assert args.image_dir is not None, "Specify ---image_dir for path to save sampled images."

        if os.path.exists(args.image_dir):
            print("Warning, {} already exists, overwriting.".format(args.image_dir))
        else:
            os.mkdir(args.image_dir)

    (all_boxes, all_gt_boxes) = get_boxes(model, val_set,
                                          args.num_images, args.image_dir,
                                          args.score_threshold)

    if args.output is not None:
        pickle.dump(all_boxes, open(args.output, 'wb'))

    print('Evaluating detections')

    # all_boxes should be (num_boxes, 6) with (xmin, ymin, xmax, ymax, SCORE, CLASS)
    # gt_boxes should be (num_boxes, 7) with (xmin, ymin, xmax, ymax, CLASS, DIFFICULT, DETECTED)
    avg_precision = voc_eval(all_boxes, all_gt_boxes, val_set.CLASSES, use_07_metric=True)
