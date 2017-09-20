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
import pickle
import glob
import os
import json
import spacenet_utils
from PIL import Image, ImageDraw
from collections import OrderedDict
import ingest_utils as util
import numpy as np
from tqdm import tqdm
import warnings
from configargparse import ArgumentParser
from neon.util.persist import get_data_cache_or_nothing


def get_ssd_config(img_reshape, inference=False):
    ssd_config = OrderedDict()
    ssd_config['batch_size'] = 32
    if inference:
        ssd_config['batch_size'] = 1
    ssd_config['block_size'] = 50
    ssd_config['cache_directory'] = get_data_cache_or_nothing(subdir='spacenet_cache')
    ssd_config["etl"] = [{
                "type": "localization_ssd",
                "height": img_reshape[0],
                "width": img_reshape[1],
                "max_gt_boxes": 500,
                "class_names": ['__background__', 'building']
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

    return ssd_config


def is_eligible_example(annotation, percent_blank=0.5):
    num_objects = len(annotation['object'])
    blank = annotation['num_blank'] / float(annotation['num_total'])

    return num_objects > 0 and blank < percent_blank


def ensure_within_bounds(box, size):
    box['xmin'] = np.minimum(np.maximum(box['xmin'], 0), size['width']-1)
    box['xmax'] = np.minimum(np.maximum(box['xmax'], 0), size['width']-1)
    box['ymin'] = np.minimum(np.maximum(box['ymin'], 0), size['height']-1)
    box['ymax'] = np.minimum(np.maximum(box['ymax'], 0), size['height']-1)

    return box


def shrink_box(box, size, scale=0.8):

    # center coordinates
    xc = float(box['xmax'] - box['xmin']) / 2 + box['xmin']
    yc = float(box['ymax'] - box['ymin']) / 2 + box['ymin']

    box['xmax'] = int(xc + scale * (box['xmax'] - xc))
    box['xmin'] = int(xc + scale * (box['xmin'] - xc))
    box['ymax'] = int(yc + scale * (box['ymax'] - yc))
    box['ymin'] = int(yc + scale * (box['ymin'] - yc))

    return box


def rescale_box(box, scales, size):

    box['xmax'] = int(box['xmax'] * scales[1])
    box['xmin'] = int(box['xmin'] * scales[1])
    box['ymax'] = int(box['ymax'] * scales[0])
    box['ymin'] = int(box['ymin'] * scales[0])

    return box


def plot_image(im_path, json_path, save_path):
    im = Image.open(im_path)
    img = np.array(im).astype(np.float32)

    im = Image.fromarray(img.astype(np.uint8))
    bboxes = json.load(open(json_path))

    draw = ImageDraw.Draw(im)

    for obj in bboxes['object']:
        bbox = obj['bndbox']
        draw.rectangle([bbox['xmin'], bbox['ymin'],
                        bbox['xmax'], bbox['ymax']], outline=(255, 0, 0))

    im.save(save_path)


def convert_image_annot(image_path, annot_path, target_image, target_annot,
                        width, height, box_shrink=0.8, min_size=0.01,
                        debug_dir=None):

    boxes = spacenet_utils.get_bounding_boxes(image_path, annot_path)

    # save as target image
    image_array = spacenet_utils.load_as_uint8(image_path)
    (c, h, w) = image_array.shape

    image = Image.fromarray(image_array.transpose([1, 2, 0]))
    image = image.resize((width, height), Image.ANTIALIAS)
    image.save(target_image)
    # print('Image saved to {}'.format(target_image))

    (scale, _) = util.get_image_scale((h, w), (height, width))
    h = height
    w = width
    assert c == 3, 'Only 3-band data supported.'

    annot = {'object': [],
             'filename': target_image,
             'annot_filename': target_annot,
             'size': {'depth': c, 'height': h, 'width': w},
             'num_blank': np.sum(np.mean(image_array, axis=0) == 0),
             'num_total': h * w}

    h_threshold = annot['size']['height'] * min_size
    w_threshold = annot['size']['width'] * min_size

    for box in boxes:
        box = {'xmin': box[0], 'xmax': box[1], 'ymin': box[2], 'ymax': box[3]}

        box = rescale_box(box, scale, annot['size'])  # resize boxes to match image resize
        box = shrink_box(box, annot['size'], box_shrink)  # shrink boxes by box_shrink
        box = ensure_within_bounds(box, annot['size'])  # make sure box within bounds of image
        box = {key: box[key].astype(int) for key in box.keys()}  # convert to integers

        obj = {'bndbox': box,
               'difficult': False,
               'name': "building"
               }

        if (obj['bndbox']['xmax'] - obj['bndbox']['xmin']) > w_threshold and \
           (obj['bndbox']['ymax'] - obj['bndbox']['ymin']) > h_threshold:
            annot['object'].append(obj)

    with open(target_annot, 'w') as f:
        json.dump(annot, f, indent=4)

    # print('Annotations saved to {}'.format(target_annot))

    # plot test image for inspection
    if debug_dir is not None:
        test_image_save = os.path.join(debug_dir, os.path.basename(target_image))
        plot_image(target_image, target_annot, test_image_save)

    return annot


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print('Creating directory: {}'.format(directory))


def _warning(message, category=UserWarning, filename='', lineno=-1):
    print('WARNING: {}'.format(message))


def ingest_spacenet(data_dir, cities, width, height, overwrite=False,
                    train_fraction=0.9, percent_blank=0.5, annot_save=None):

    warnings.showwarning = _warning  # monkeypatch to not emit code
    hw = '{}x{}'.format(height, width)
    ext = '.png'
    data = {}

    train_manifest = os.path.join(data_dir, 'train_{}.csv'.format(hw))
    val_manifest = os.path.join(data_dir, 'val_{}.csv'.format(hw))

    if os.path.exists(train_manifest) and os.path.exists(val_manifest) and not overwrite:
        print("Manifest files already found, skipping ingest.")
        print("Use --overwrite flag to force re-ingest.")
        return

    for city in cities:

        if city == 'AOI_1_Rio':  # Rio has different dataset structure
            img_folder = os.path.join(data_dir, city, 'processedData', '3band')
            annot_folder = os.path.join(data_dir, city, 'processedData', 'vectorData', 'geoJson')

            target_img_folder = os.path.join(data_dir, city, 'processedData',
                                             '3band-{}'.format(hw))
            target_annot_folder = os.path.join(data_dir, city, 'processedData',
                                               'json-{}'.format(hw))
            test_img_folder = os.path.join(data_dir, city,
                                           'processedData', '3band-{}-gt'.format(hw))

            # helper function for converting image files to their corresponding annotation file
            # e.g. 3band_013022223133_Public_img3593.tif -> 013022223133_Public_img3593_Geo.geojson
            def img_to_annot(x): x.replace('3band_', '').replace('.tif', '_Geo.geojson')

        else:
            prefix = 'RGB-PanSharpen'
            img_folder = os.path.join(data_dir, city, prefix)
            annot_folder = os.path.join(data_dir, city, 'geojson', 'buildings')

            # create data folders to save converted images and annotations
            target_img_folder = os.path.join(data_dir, city, '{}-{}'.format(prefix, hw))
            target_annot_folder = os.path.join(data_dir, city, 'json-{}'.format(hw))
            test_img_folder = os.path.join(data_dir, city, '{}-{}-gt'.format(prefix, hw))

            # helper function for converting image files to their corresponding annotation file
            # e.g. RGB-PanSharpen_AOI_2_Vegas_img9.tif -> buildings_AOI_2_Vegas_img9.geojson
            def img_to_annot(x): x.replace(prefix, 'buildings').replace('.tif', '.geojson')

        print('Processing {}'.format(city))

        make_dir(target_img_folder)
        make_dir(target_annot_folder)
        make_dir(test_img_folder)

        images = glob.glob(os.path.join(img_folder, "*.tif"))
        assert len(images) > 0, 'No Images found in {}'.format(img_folder)

        data[city] = {'manifest': [], 'annotation': [],
                      'img_folder': img_folder,
                      'annot_folder': annot_folder}

        for image in tqdm(images):

            img_file = os.path.basename(image)
            annot_file = img_to_annot(img_file)
            annot = os.path.join(annot_folder, annot_file)
            assert os.path.exists(annot)

            # target image has extension=ext, and target_annot has extension JSON
            target_image = os.path.join(target_img_folder, os.path.splitext(img_file)[0] + ext)
            target_annot = os.path.join(target_annot_folder,
                                        os.path.splitext(annot_file)[0] + '.json')

            if not os.path.exists(target_image) or not os.path.exists(target_annot) or overwrite:

                annotation = convert_image_annot(image_path=image, annot_path=annot,
                                                 target_image=target_image,
                                                 target_annot=target_annot,
                                                 width=512, height=512, box_shrink=0.8,
                                                 debug_dir=test_img_folder)
            else:
                warnings.warn(
                    'File for {} already exists, skipping processing.Use --overwrite to force.'.
                    format(city), Warning)
                annotation = json.load(open(target_annot))

            # filter on percent_blank, as well as presence of any objects
            if is_eligible_example(annotation, percent_blank):
                data[city]['annotation'].append(annotation)
                data[city]['manifest'].append((target_image, target_annot))

    # write manifest files

    # build manifest list from each city's manifest
    manifest = []
    for city in cities:
        manifest.extend(data[city]['manifest'])

    ntrain = int(np.round(len(manifest) * train_fraction))

    np.random.seed(0)
    np.random.shuffle(manifest)

    util.create_manifest(train_manifest, manifest[:ntrain], data_dir)
    util.create_manifest(val_manifest, manifest[ntrain:], data_dir)

    # write SSD CONFIG
    ssd_config = get_ssd_config((height, width))
    ssd_config_path = os.path.join(data_dir, 'spacenet_ssd_{}.cfg'.format(hw))
    util.write_ssd_config(ssd_config, ssd_config_path, True)

    # write SSD VAL CONFIG
    ssd_config_val = get_ssd_config((height, width), True)
    ssd_config_path_val = os.path.join(data_dir, 'spacenet_ssd_{}_val.cfg'.format(hw))
    util.write_ssd_config(ssd_config_val, ssd_config_path_val, True)

    config_path = os.path.join(data_dir, 'spacenet_{}.cfg'.format(hw))

    config = {'manifest': '[train:{}, val:{}]'.format(train_manifest, val_manifest),
              'manifest_root': data_dir,
              'epochs': 230,
              'height': height,
              'width': width,
              'ssd_config': '[train:{}, val:{}]'.format(ssd_config_path, ssd_config_path_val)
              }

    util.write_config(config, config_path)

    # write annotation pickle
    if annot_save is not None:
        pickle.dump(data, open(annot_save, 'w'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='path to directory with vocdevkit data')
    parser.add_argument('--overwrite', action='store_true', help='overwrite files')
    parser.add_argument('--height', type=int, default=512, help='height of reshaped image')
    parser.add_argument('--width', type=int, default=512, help='width of reshape image')
    parser.add_argument('--train_fraction', type=float, default=0.9, help='width of reshape image')
    parser.add_argument('--annot_save', type=str, default=None,
                        help='separately save annotations to this file.')

    args = parser.parse_args()

    cities = ['AOI_1_Rio', 'AOI_2_Vegas_Train',
              'AOI_3_Paris_Train', 'AOI_4_Shanghai_Train', 'AOI_5_Khartoum_Train']

    ingest_spacenet(cities=cities, data_dir=args.data_dir, height=args.height, width=args.width,
                    overwrite=args.overwrite, annot_save=args.annot_save)
