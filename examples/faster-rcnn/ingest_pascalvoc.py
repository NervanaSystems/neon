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
from configargparse import ArgumentParser
from convert_xml_to_json import convert_xml_to_json
import numpy as np
import os
import tarfile


def ingest_pascal(data_dir, out_dir, year='2007', overwrite=False):

    # define paths
    root_dir = os.path.join(out_dir, 'VOCdevkit', 'VOC' + year)
    manifest_train = os.path.join(root_dir, 'trainval.csv')
    manifest_inference = os.path.join(root_dir, 'val.csv')

    # write configuration file
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pascalvoc.cfg')
    with open(config_path, 'w') as f:
        f.write('manifest = [train:{}, val:{}]\n'.format(manifest_train, manifest_inference))
        f.write('manifest_root = {}\n'.format(out_dir))
        f.write('epochs = 14\n')
        f.write('height = 1000\n')
        f.write('width = 1000\n')
        f.write('batch_size = 1\n')
        f.write('rng_seed = 0')
    print("Wrote config file to: {}".format(config_path))

    if not overwrite and os.path.exists(manifest_train) and os.path.exists(manifest_inference):
        print("""Found existing manfiest files, skipping ingest,
              Use --overwrite to rerun ingest anyway.""")
        return (manifest_train, manifest_inference)

    # untar data to output directory
    tarfiles = [os.path.join(data_dir, tar) for
                tar in ['VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar']]

    for file in tarfiles:
        with tarfile.open(file, 'r') as t:
            print("Extracting {} to {}".format(file, out_dir))
            t.extractall(out_dir)

    # convert the annotations to json, exclude difficult objects (used for training)
    input_path = os.path.join(root_dir, 'Annotations')
    annot_path = os.path.join(root_dir, 'Annotations-json')
    print("Reading PASCAL XML files from {}".format(input_path))
    print("Converting XML files to json format, writing to: {}".format(annot_path))
    convert_xml_to_json(input_path, annot_path, difficult=False)

    # convert the annotations to json, including difficult objects (used for inference)
    annot_path_difficult = os.path.join(root_dir, 'Annotations-json-difficult')
    print("Converting XML files to json format (including objects with difficult flag),")
    print("writing to: {}".format(annot_path_difficult))
    convert_xml_to_json(input_path, annot_path_difficult, difficult=True)

    img_dir = os.path.join(root_dir, 'JPEGImages')

    # write manifest file for training
    index_path = os.path.join(root_dir, 'ImageSets', 'Main', 'trainval.txt')
    create_manifest(manifest_train, index_path, annot_path, img_dir, out_dir)

    # write manifest file for testing
    # for testing, we inlude the difficult objects, so use annot_path_difficult
    index_path = os.path.join(root_dir, 'ImageSets', 'Main', 'test.txt')
    create_manifest(manifest_inference, index_path, annot_path_difficult, img_dir, out_dir)


def create_manifest(manifest_path, index_file, annot_dir, image_dir, root_dir):
    """
    Based on a PASCALVOC index file, creates a manifest csv file.
    If the manifest file already exists, this function will skip writing, unless the
    overwrite argument is set to True.

    Arguments:
        manifest_path (string): path to save the manifest file
        index (string or list): list of images.
        annot_dir (string): directory of annotations
        img_dir (string): directory of images
        root_dir (string): paths will be made relative to this directory
        ext (string, optional): image extension (default=.jpg)
    """
    records = [('@FILE', 'FILE')]
    with open(index_file) as f:
        for img in f:
            tag = img.rstrip(os.linesep)

            image = os.path.join(image_dir, tag + '.jpg')
            annot = os.path.join(annot_dir, tag + '.json')

            assert os.path.exists(image), 'Path {} not found'.format(image)
            assert os.path.exists(annot), 'Path {} not found'.format(annot)

            records.append((os.path.relpath(image, root_dir),
                            os.path.relpath(annot, root_dir)))

    np.savetxt(manifest_path, records, fmt='%s\t%s')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='path to directory with vocdevkit data')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--overwrite', action='store_true', help='overwrite files')
    args = parser.parse_args()

    ingest_pascal(args.input_dir, args.output_dir, overwrite=args.overwrite)
