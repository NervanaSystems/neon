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
import json
import glob
import collections
import os
from os.path import join
import xml.etree.ElementTree as et
from collections import defaultdict
import argparse


# http://stackoverflow.com/questions/7684333/converting-xml-to-dictionary-using-elementtree
def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def validate_metadata(jobj, file):
    boxlist = jobj['object']
    if not isinstance(boxlist, collections.Sequence):
        print('{0} is not a sequence').format(file)
        return False

    index = 0
    for box in boxlist:
        if 'part' in box:
            parts = box['part']
            if not isinstance(parts, collections.Sequence):
                print('parts {0} is not a sequence').format(file)
                return False
        index += 1
    return True


def convert_xml_to_json(input_path, output_path, difficult):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    onlyfiles = glob.glob(join(input_path, '*.xml'))
    onlyfiles.sort()
    for file in onlyfiles:
        outfile = join(output_path, os.path.basename(file))
        outfile = os.path.splitext(outfile)[0] + '.json'
        trimmed = parse_single_file(join(input_path, file), difficult)
        if validate_metadata(trimmed, file):
            result = json.dumps(trimmed, sort_keys=True, indent=4, separators=(',', ': '))
            f = open(outfile, 'w')
            f.write(result)
        else:
            print('error parsing metadata {0}').format(file)


def parse_single_file(path, difficult):
    tree = et.parse(path)
    root = tree.getroot()
    d = etree_to_dict(root)
    trimmed = d['annotation']
    olist = trimmed['object']
    if not isinstance(olist, collections.Sequence):
        trimmed['object'] = [olist]
        olist = trimmed['object']
    size = trimmed['size']

    # Add version number to json
    trimmed['version'] = {'major': 1, 'minor': 0}

    # convert all numbers from string representation to number so json does not quote them
    # all of the bounding box numbers are one based so subtract 1
    size['width'] = int(size['width'])
    size['height'] = int(size['height'])
    size['depth'] = int(size['depth'])
    width = trimmed['size']['width']
    height = trimmed['size']['height']
    for obj in olist:
        obj['difficult'] = int(obj['difficult']) != 0
        obj['truncated'] = int(obj['truncated']) != 0
        box = obj['bndbox']
        box['xmax'] = int(box['xmax']) - 1
        box['xmin'] = int(box['xmin']) - 1
        box['ymax'] = int(box['ymax']) - 1
        box['ymin'] = int(box['ymin']) - 1
        if 'part' in obj:
            for part in obj['part']:
                box = part['bndbox']
                box['xmax'] = float(box['xmax']) - 1
                box['xmin'] = float(box['xmin']) - 1
                box['ymax'] = float(box['ymax']) - 1
                box['ymin'] = float(box['ymin']) - 1
        xmax = box['xmax']
        xmin = box['xmin']
        ymax = box['ymax']
        ymin = box['ymin']
        if xmax > width - 1:
            print('xmax {0} exceeds width {1}').format(xmax, width)
        if xmin < 0:
            print('xmin {0} exceeds width {1}').format(xmin, width)
        if ymax > height - 1:
            print('ymax {0} exceeds width {1}').format(ymax, height)
        if ymin < 0:
            print('ymin {0} exceeds width {1}').format(ymin, height)

    # exclude difficult objects
    if not difficult:
        trimmed['object'] = [o for o in trimmed['object'] if not o['difficult']]

    return trimmed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert xml to json for pascalvoc dataset")
    parser.add_argument('-i, --input', dest='input', help='input directory with xml files.')
    parser.add_argument('-o, --output', dest='output', help='output directory of json files.')
    parser.add_argument('-p, --parse', dest='parse', help='parse a single xml file.')
    parser.add_argument('--difficult', dest='difficult', action='store_true',
                        help='include objects with the difficult tag. Default is to exclude.')

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    parse_file = args.parse

    if parse_file:
        print(parse_file)
        parsed = parse_single_file(parse_file, args.difficult)
        json1 = json.dumps(parsed, sort_keys=True, indent=4, separators=(',', ': '))
        print(json1)
    elif input_path:
        convert_xml_to_json(input_path, output_path, args.difficult)
