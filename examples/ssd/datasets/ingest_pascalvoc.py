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
from argparse import ArgumentParser
from convert_xml_to_json import convert_xml_to_json
import numpy as np
import os
import tarfile
import ingest_utils as util
from collections import OrderedDict
from tqdm import tqdm
from neon.util.persist import get_data_cache_or_nothing


def get_ssd_config(img_reshape, inference=False):
    ssd_config = OrderedDict()
    ssd_config['batch_size'] = 32
    ssd_config['shuffle_enable'] = True
    ssd_config['shuffle_manifest'] = True
    if inference:
        ssd_config['batch_size'] = 1
    ssd_config['block_size'] = 50
    ssd_config['cache_directory'] = get_data_cache_or_nothing(subdir='pascalvoc_cache')
    ssd_config["etl"] = [{
                "type": "localization_ssd",
                "height": img_reshape[0],
                "width": img_reshape[1],
                "max_gt_boxes": 500,
                "class_names": ["__background__", "aeroplane", "bicycle", "bird", "boat",
                                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                "dog", "horse", "motorbike", "person", "pottedplant",
                                "sheep", "sofa", "train", "tvmonitor"]
                }, {
                    "type": "image",
                    "height": img_reshape[0],
                    "width": img_reshape[1],
                    "channels": 3
                }]
    if not inference:
        ssd_config["augmentation"] = [{
            "type": "image",
            "crop_enable": False,
            "flip_enable": True,
            "expand_ratio": [1., 4.],
            "expand_probability": 0.5,
            # "emit_constraint_type": "center", TODO: enable when adds support for no gt boxes
            "brightness": [0.9, 1.1],
            "hue": [-18, 18],
            "saturation": [0.9, 1.1],
            "contrast": [0.9, 1.1],
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

    if img_reshape == (300, 300):
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
                       'aspect_ratios': 2.0,        'step': 300})])

    elif img_reshape == (512, 512):
        ssd_config['ssd_config'] = OrderedDict(
                     [('conv4_3', {'min_sizes': 35.84, 'max_sizes': 76.80,
                       'aspect_ratios': 2.0,        'step': 8, 'normalize': True}),
                      ('fc7',     {'min_sizes': 76.80, 'max_sizes': 153.6,
                       'aspect_ratios': (2.0, 3.0), 'step': 16}),
                      ('conv6_2', {'min_sizes': 153.6, 'max_sizes': 230.4,
                       'aspect_ratios': (2.0, 3.0), 'step': 32}),
                      ('conv7_2', {'min_sizes': 230.4, 'max_sizes': 307.2,
                       'aspect_ratios': (2.0, 3.0), 'step': 64}),
                      ('conv8_2', {'min_sizes': 307.2, 'max_sizes': 384.0,
                       'aspect_ratios': 2.0,        'step': 128}),
                      ('conv9_2', {'min_sizes': 384.0, 'max_sizes': 460.8,
                       'aspect_ratios': 2.0,        'step': 256}),
                      ('conv10_2', {'min_sizes': 460.8, 'max_sizes': 537.8,
                       'aspect_ratios': 2.0,        'step': 512})])
    else:
        raise ValueError("Image shape of {} not supported.".format(img_reshape))

    return ssd_config


def extract_tarfiles(tarfiles, out_dir):
    for file in tarfiles:
        with tarfile.open(file, 'r') as t:
            print("Extracting {} to {}".format(file, out_dir))
            t.extractall(out_dir)


def get_tag_list(index_file):
    with open(index_file) as f:
        tag_list = [tag.rstrip(os.linesep) for tag in f]

    return tag_list


def ingest_pascal(data_dir, out_dir, img_reshape=(300, 300), overwrite=False, skip_untar=False):

    assert img_reshape is not None, "Target image reshape required."
    hw = '{}x{}'.format(img_reshape[0], img_reshape[1])

    datasets = ['VOC2007', 'VOC2012']
    tar_files = {'VOC2007': ['VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar'],
                 'VOC2012': ['VOCtrainval_11-May-2012.tar']}

    index_name = {'trainval': 'trainval.txt', 'test': 'test.txt'}
    manifest = {'trainval': [], 'test': []}

    root_dir = os.path.join(out_dir, 'VOCdevkit')

    train_manifest = os.path.join(root_dir, 'train_{}.csv'.format(hw))
    val_manifest = os.path.join(root_dir, 'val_{}.csv'.format(hw))

    if os.path.exists(train_manifest) and os.path.exists(val_manifest) and not overwrite:
        print("Manifest files already found, skipping ingest.")
        print("Use --overwrite flag to force re-ingest.")
        return

    for year in datasets:
        tags = {'trainval': [], 'test': []}

        # define paths
        if skip_untar is False:
            tarfiles = [os.path.join(data_dir, tar) for tar in tar_files[year]]
            extract_tarfiles(tarfiles, out_dir)

        # read the index files and build a list of tags to process
        # in PASCALVOC, each tag (e.g. '000032') refers to an image (000032.jpg)
        # and an annotation XML file (000032.xml)
        for sets in index_name.keys():
            index_file = os.path.join(root_dir, year, 'ImageSets', 'Main', index_name[sets])
            if os.path.exists(index_file):
                tag_list = get_tag_list(index_file)
                tags[sets].extend(tag_list)
                print('Found {} images in {}'.format(len(tag_list), index_file))

        img_folder = os.path.join(root_dir, year, 'JPEGImages')
        annot_folder = os.path.join(root_dir, year, 'Annotations')

        # create data folders to save converted images and annotations
        target_img_folder = os.path.join(root_dir, year, 'JPEGImages-converted')
        target_annot_folder = os.path.join(root_dir, year, 'Annotations-json')

        print('Processing {}'.format(year))

        util.make_dir(target_img_folder)
        util.make_dir(target_annot_folder)

        all_tags = tags['trainval'] + tags['test']  # process all the tags in our index files.

        for tag in tqdm(all_tags):

            image = os.path.join(img_folder, tag + '.jpg')
            annot = os.path.join(annot_folder, tag + '.xml')
            assert os.path.exists(image)
            assert os.path.exists(annot)

            target_image = os.path.join(target_img_folder, tag + '.jpg')
            target_annot = os.path.join(target_annot_folder, tag + '.json')

            # convert the annotations to json, including difficult objects
            convert_xml_to_json(annot, target_annot, difficult=True, img_reshape=None)
            util.resize_image(image, target_image, img_reshape=None)

            if tag in tags['trainval']:
                manifest['trainval'].append((target_image, target_annot))
            elif tag in tags['test']:
                manifest['test'].append((target_image, target_annot))

    np.random.seed(0)
    np.random.shuffle(manifest['trainval'])

    util.create_manifest(train_manifest, manifest['trainval'], root_dir)
    util.create_manifest(val_manifest, manifest['test'], root_dir)

    # write SSD CONFIG
    ssd_config = get_ssd_config(img_reshape)
    ssd_config_path = os.path.join(root_dir, 'pascalvoc_ssd_{}.cfg'.format(hw))
    util.write_ssd_config(ssd_config, ssd_config_path, True)

    # write SSD VAL CONFIG
    ssd_config_val = get_ssd_config(img_reshape, True)
    ssd_config_path_val = os.path.join(root_dir, 'pascalvoc_ssd_{}_val.cfg'.format(hw))
    util.write_ssd_config(ssd_config_val, ssd_config_path_val, True)

    config_path = os.path.join(root_dir, 'pascalvoc_{}.cfg'.format(hw))

    config = {'manifest': '[train:{}, val:{}]'.format(train_manifest, val_manifest),
              'manifest_root': root_dir,
              'epochs': 230,
              'height': img_reshape[0],
              'width': img_reshape[1],
              'ssd_config': '[train:{}, val:{}]'.format(ssd_config_path, ssd_config_path_val)
              }

    util.write_config(config, config_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='path to directory with vocdevkit data')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--overwrite', action='store_true', help='overwrite files')
    parser.add_argument('--height', type=int, default=300, help='height of reshaped image')
    parser.add_argument('--width', type=int, default=300, help='width of reshape image')
    parser.add_argument('--skip_untar', action='store_true',
                        help='skip the untar. Use if unzipped files already exist.')

    args = parser.parse_args()

    ingest_pascal(args.input_dir, args.output_dir, img_reshape=(args.height, args.width),
                  overwrite=args.overwrite, skip_untar=args.skip_untar)
