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
from __future__ import print_function
import os
import glob
import json
import numpy as np
from PIL import Image
import math
from tqdm import tqdm
from collections import OrderedDict
import ingest_utils as util
from neon.util.persist import get_data_cache_or_nothing


def get_ssd_config(img_reshape, inference=False):
    ssd_config = OrderedDict()
    ssd_config['batch_size'] = 32
    if inference:
        ssd_config['batch_size'] = 1
    ssd_config['block_size'] = 50
    ssd_config['cache_directory'] = get_data_cache_or_nothing(subdir='kitti_cache')
    ssd_config["etl"] = [{
                    "type": "localization_ssd",
                    "height": img_reshape[0],
                    "width": img_reshape[1],
                    "max_gt_boxes": 500,
                    "class_names": ['__background__', 'Car', 'Van', 'Truck', 'Pedestrian',
                                    'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
                }, {
                    "type": "image",
                    "height": img_reshape[0],
                    "width": img_reshape[1],
                    "channels": 3
                }]
    if not inference:
        ssd_config["augmentation"] = [{
            "type": "image",
            "batch_samplers":
            [
                {
                    "max_sample": 1,
                    "max_trials": 1
                },
                {
                    "max_sample": 1,
                    "max_trials": 50,
                    "sampler": {"scale": [0.3, 1.0], "aspect_ratio": [0.5, 2.0]},
                    "sample_constraint": {"min_jaccard_overlap": 0.1}
                },
                {
                    "max_sample": 1,
                    "max_trials": 50,
                    "sampler": {"scale": [0.3, 1.0], "aspect_ratio": [0.5, 2.0]},
                    "sample_constraint": {"min_jaccard_overlap": 0.3}
                },
                {
                    "max_sample": 1,
                    "max_trials": 50,
                    "sampler": {"scale": [0.3, 1.0], "aspect_ratio": [0.5, 2.0]},
                    "sample_constraint": {"min_jaccard_overlap": 0.5}
                },
                {
                    "max_sample": 1,
                    "max_trials": 50,
                    "sampler": {"scale": [0.3, 1.0], "aspect_ratio": [0.5, 2.0]},
                    "sample_constraint": {"min_jaccard_overlap": 0.7}
                },
                {
                    "max_sample": 1,
                    "max_trials": 50,
                    "sampler": {"scale": [0.3, 1.0], "aspect_ratio": [0.5, 2.0]},
                    "sample_constraint": {"min_jaccard_overlap": 0.9}
                },
                {
                    "max_sample": 1,
                    "max_trials": 50,
                    "sampler": {"scale": [0.3, 1.0], "aspect_ratio": [0.5, 2.0]},
                    "sample_constraint": {"max_jaccard_overlap": 1.0, "min_jaccard_overlap": 0.1}
                }
            ]
        }]

    ssd_config['ssd_config'] = OrderedDict(
                 [('conv4_3', {'min_sizes': 30.0,  'max_sizes': 60.0,
                   'aspect_ratios': 2.0,        'step': 8, 'normalize': True}),
                  ('fc7',     {'min_sizes': 60.0,  'max_sizes': 111.0,
                   'aspect_ratios': (2.0, 3.0), 'step': 16}),
                  ('conv6_2', {'min_sizes': 111.0, 'max_sizes': 162.0,
                   'aspect_ratios': (2.0, 3.0), 'step': 32}),
                  ('conv7_2', {'min_sizes': 162.0, 'max_sizes': 213.0,
                   'aspect_ratios': (2.0, 3.0), 'step': 64}),
                  ('conv8_2', {'min_sizes': 213.0, 'max_sizes': 264.0,
                   'aspect_ratios': 2.0,        'step': 100}),
                  ('conv9_2', {'min_sizes': 264.0, 'max_sizes': 315.0,
                   'aspect_ratios': 2.0,        'step': {'step_h': 300, 'step_w': 100}})])

    return ssd_config


def convert_annot_to_json(path, im_path, out_path, difficult, img_reshape=None):
    """
    Converts the KITTI annotations to json file.

    Uses the below reference for the KITTI dataset:

    OO representation of label format used in Kitti dataset.

    Description of fields from Kitti dataset dev kit: (link)[]
    The label files contain the following information, which can be read and
    written using the matlab tools (readLabels.m, writeLabels.m) provided within
    this devkit. All values (numerical or strings) are separated via spaces,
    each row corresponds to one object. The 15 columns represent:
    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.

    Arguments:
        path (string): path to KITTI annotation file
        im_path (string): path to image
        out_path (string): path to save the json file
        difficult (bool): include difficult objects
        img_reshape (tuple of int): if a tuple of H,W values is given, image will be reshaped
    """
    with open(path) as f:
        labels = f.readlines()

    # start empty dictionary
    annot = {'object': []}

    # load image
    im = np.array(Image.open(im_path))
    scale, (h, w) = util.get_image_scale(im.shape[:2], img_reshape)
    c = im.shape[2]
    annot['size'] = {'depth': c, 'height': h, 'width': w}

    for label in labels:
        vals = label.split()
        typeid = vals[0]
        truncated = float(vals[1])
        occluded = int(vals[2])
        bbox = [float(x) for x in vals[4:8]]
        bbox = util.scale_boxes(bbox, scale)
        bbox_int = tuple([int(math.floor(x)) for x in bbox])

        if typeid == 'DontCare':
            assert truncated == -1
            assert occluded == -1
        else:
            assert occluded in (0, 1, 2, 3)

        diff = truncated > 0.5 or occluded == 2

        # add object to annotation
        obj = {'bndbox': {'xmin': bbox_int[0], 'ymin': bbox_int[1],
               'xmax': bbox_int[2], 'ymax': bbox_int[3]},
               'difficult': diff,
               'name': typeid,
               'truncated': truncated > 0.5,
               'occluded': occluded
               }

        if not diff or difficult:
            annot['object'].append(obj)

    with open(out_path, 'w') as f:
        json.dump(annot, f, indent=4)


def ingest_kitti(input_dir, out_dir, img_reshape=(300, 994),
                 train_percent=90, overwrite=False, skip_unzip=False):
    """
    Ingests the KITTI dataset. Peforms the following ops:
    0. Unzips the files into output directory.
    1. Reshapes image to lower resolution (default reshape of 300x994 maintains KITTI image AR)
    1. Convert annotations to json format
    2. Split the training data into train and validation sets
    3. Write manifest file
    4. Write configuration file

    Arguments:
        input_dir (string): path to folder with KITTI zip files.
        out_dir (string): path to unzip KITTI data
        img_reshape (tuple of int): size to reshape image (default = (300, 994))
        train_percent (float): percent of data to use for training.
        overwrite (bool): overwrite existing files
    """

    assert img_reshape is not None, "Target image reshape required."
    hw = '{}x{}'.format(img_reshape[0], img_reshape[1])

    zip_files = ['data_object_image_2.zip', 'data_object_label_2.zip']

    root_dir = os.path.join(out_dir, 'kitti')
    train_manifest = os.path.join(root_dir, 'train_{}.csv'.format(hw))
    val_manifest = os.path.join(root_dir, 'val_{}.csv'.format(hw))

    if os.path.exists(train_manifest) and os.path.exists(val_manifest) and not overwrite:
        print("Manifest files already found, skipping ingest.")
        print("Use --overwrite flag to force re-ingest.")
        return

    util.make_dir(root_dir)

    tags = {'trainval': [], 'test': []}

    if skip_unzip is False:
        util.unzip_files(zip_files, input_dir, root_dir)

    img_folder = os.path.join(root_dir, 'training', 'image_2')
    annot_folder = os.path.join(root_dir, 'training', 'label_2')

    target_img_folder = os.path.join(root_dir, 'training', 'image_2-converted')
    target_annot_folder = os.path.join(root_dir, 'training', 'label_2-json')

    tags = glob.glob(os.path.join(img_folder, '*.png'))
    tags = [os.path.basename(os.path.splitext(tag)[0]) for tag in tags]
    assert len(tags) > 0, "No images found in {}".format(img_folder)

    util.make_dir(target_img_folder)
    util.make_dir(target_annot_folder)

    manifest = []

    for tag in tqdm(tags):

        image = os.path.join(img_folder, tag + '.png')
        annot = os.path.join(annot_folder, tag + '.txt')
        assert os.path.exists(image), "{} not found.".format(image)
        assert os.path.exists(annot), "{} not found.".format(annot)

        target_image = os.path.join(target_img_folder, tag + '.png')
        target_annot = os.path.join(target_annot_folder, tag + '.json')

        convert_annot_to_json(annot, image, target_annot, difficult=True, img_reshape=None)
        util.resize_image(image, target_image, img_reshape=None)

        manifest.append((target_image, target_annot))

    # shuffle files and split into training and validation set.
    np.random.seed(0)
    np.random.shuffle(manifest)

    train_count = (len(manifest) * train_percent) // 100
    train = manifest[:train_count]
    val = manifest[train_count:]

    util.create_manifest(train_manifest, train, root_dir)
    util.create_manifest(val_manifest, val, root_dir)

    # write SSD CONFIG
    ssd_config = get_ssd_config(img_reshape)
    ssd_config_path = os.path.join(root_dir, 'kitti_ssd_{}.cfg'.format(hw))
    util.write_ssd_config(ssd_config, ssd_config_path, True)

    # write SSD VAL CONFIG
    ssd_config_val = get_ssd_config(img_reshape, True)
    ssd_config_path_val = os.path.join(root_dir, 'kitti_ssd_{}_val.cfg'.format(hw))
    util.write_ssd_config(ssd_config_val, ssd_config_path_val, True)

    config_path = os.path.join(root_dir, 'kitti_{}.cfg'.format(hw))
    config = {'manifest': '[train:{}, val:{}]'.format(train_manifest, val_manifest),
              'manifest_root': root_dir,
              'epochs': 100,
              'height': img_reshape[0],
              'width': img_reshape[0],
              'ssd_config': '[train:{}, val:{}]'.format(ssd_config_path, ssd_config_path_val)
              }

    util.write_config(config, config_path)


if __name__ == '__main__':
    from configargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='path to dir with KITTI zip files.')
    parser.add_argument('--output_dir', required=True, help='path to unzip data.')
    parser.add_argument('--overwrite', action='store_true', help='overwrite files')
    parser.add_argument('--training_pct', default=90, help='fraction of data used for training.')
    parser.add_argument('--skip_unzip', action='store_true', help='skip unzip')

    args = parser.parse_args()

    ingest_kitti(args.input_dir, args.output_dir, train_percent=args.training_pct,
                 overwrite=args.overwrite, skip_unzip=args.skip_unzip)
