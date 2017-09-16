#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
from builtins import str, zip
from configargparse import ArgParser
from itertools import repeat
from neon import logger as neon_logger
from neon.util.persist import ensure_dirs_exist
from PIL import Image
import logging
import multiprocessing
import numpy as np
import os
import re
import shutil
import tarfile
import tqdm
import zlib


def process_i1k_tar_subpath(args):
    """
    Process a single subpath in a I1K tar. By process:
        optionally untar recursive tars (only on 'train')
        resize/copy images
    Returns a list of [(fname, label), ...]
    """
    target_size, toptar, img_dir, setn, label_dict, subpath = args
    name_slice = slice(None, 9) if setn == 'train' else slice(15, -5)
    label = label_dict[subpath.name[name_slice]]
    outpath = os.path.join(img_dir, str(label))
    if setn == 'train':
        tf = tarfile.open(toptar)
        subtar = tarfile.open(fileobj=tf.extractfile(subpath))
        file_list = subtar.getmembers()
        return process_files_in_tar(target_size, label, subtar, file_list, outpath)
    elif setn == 'val':
        tf = tarfile.open(toptar)
        file_list = [subpath]
        return process_files_in_tar(target_size, label, tf, file_list, outpath)


def process_files_in_tar(target_size, label, tar_handle, file_list, outpath):
    pair_list = []
    if not os.path.exists(outpath):
        # This avoids race conditions that sometimes happen when doing these
        # checks in parallel
        try:
            os.makedirs(outpath)
        except OSError:
            pass
    for fobj in file_list:
        fname = os.path.join(outpath, fobj.name)
        if not os.path.exists(fname):
            transform_and_save(target_size, tar_handle, fobj, fname)
        pair_list.append((fname, label))
    return pair_list


def transform_and_save(target_size, tar_handle, img_object, output_filename):
    """
    Takes a tar file handle and a TarInfo object inside that tarfile and
    optionally transforms it and then writes it out to output_filename
    """
    img_handle = tar_handle.extractfile(img_object)
    img = Image.open(img_handle)
    width, height = img.size

    # Take the smaller image dimension down to target_size
    # while retaining aspect_ration. Otherwise leave it alone
    if width < height:
        if width > target_size:
            scale_factor = float(target_size) / width
            width = target_size
            height = int(height*scale_factor)
    else:
        if height > target_size:
            scale_factor = float(target_size) / height
            height = target_size
            width = int(width*scale_factor)
    if img.size[0] != width or img.size[1] != height:
        img = img.resize((width, height), resample=Image.LANCZOS)
        img.save(output_filename, quality=95)
    else:
        # Avoid recompression by saving file out directly without transformation
        dname, fname = os.path.split(output_filename)
        tar_handle.extract(img_object, path=dname)
        if fname != img_object.name:
            # Rename if name inside of tar is different than what we want it
            # called on the outside
            shutil.move(os.path.join(dname, img_object.name), output_filename)
    assert(os.stat(output_filename).st_size > 0), "{} has size 0".format(output_filename)


class IngestI1K(object):
    def __init__(self, input_dir, out_dir, target_size=256, overwrite=False):
        np.random.seed(0)

        self.orig_out_dir = out_dir
        self.out_dir = os.path.join(out_dir, 'i1k-extracted')
        self.input_dir = os.path.expanduser(input_dir) if input_dir is not None else None
        self.devkit = os.path.join(self.input_dir, 'ILSVRC2012_devkit_t12.tar.gz')
        self.overwrite = overwrite

        self.manifests, self.tars = dict(), dict()
        for setn in ('train', 'val'):
            self.manifests[setn] = os.path.join(self.out_dir, '{}-index.csv'.format(setn))
            self.tars[setn] = os.path.join(self.input_dir, 'ILSVRC2012_img_{}.tar'.format(setn))

        self.target_size = target_size
        self._target_filenames = {}

    def _target_filename(self, target):
        """
        Return a filename of a file containing a binary representation of
        target.  If no such file exists, make one.
        """
        target_filename = self._target_filenames.get(target)
        if target_filename is None:
            target_filename = os.path.join(self.out_dir, 'labels', str(target) + '.txt')
            ensure_dirs_exist(target_filename)
            np.savetxt(target_filename, [target], '%d')
            self._target_filenames[target] = target_filename

        return target_filename

    def extract_labels(self, setn):
        if not os.path.exists(self.devkit):
            raise IOError(("Metadata file {} not found. Ensure you have ImageNet downloaded"
                           ).format(self.devkit))

        with tarfile.open(self.devkit, "r:gz") as tf:
            synsetfile = 'ILSVRC2012_devkit_t12/data/meta.mat'
            valfile = 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

            if setn == 'train':
                # get the synset mapping by hacking around matlab's terrible compressed format
                meta_buff = tf.extractfile(synsetfile).read()
                decomp = zlib.decompressobj()
                self.synsets = re.findall(re.compile('n\d+'), decomp.decompress(meta_buff[136:]))
                return {s: i for i, s in enumerate(self.synsets)}
            elif setn == 'val':
                # get the ground truth validation labels and offset to zero
                return {"%08d" % (i + 1): int(x) - 1 for i, x in
                        enumerate(tf.extractfile(valfile))}
            else:
                raise ValueError("Unknown set name: {}".format(setn))

    def train_or_val_pairs(self, setn):
        """
        untar imagenet tar files into directories that indicate their label.

        returns [(filename, label), ...] for train or val set partitions
        """
        img_dir = os.path.join(self.out_dir, setn)

        neon_logger.display("Extracting %s files" % (setn))
        root_tf_path = self.tars[setn]
        if not os.path.exists(root_tf_path):
            raise IOError(("tar file {} not found. Ensure you have ImageNet downloaded"
                           ).format(root_tf_path))

        try:
            root_tf = tarfile.open(root_tf_path)
        except tarfile.ReadError as e:
            raise ValueError('ReadError opening {}: {}'.format(root_tf_path, e))

        label_dict = self.extract_labels(setn)
        subpaths = root_tf.getmembers()
        arg_iterator = zip(repeat(self.target_size), repeat(root_tf_path), repeat(img_dir),
                           repeat(setn), repeat(label_dict), subpaths)
        pool = multiprocessing.Pool()

        pairs = []
        for pair_list in tqdm.tqdm(pool.imap_unordered(process_i1k_tar_subpath, arg_iterator),
                                   total=len(subpaths)):
            pairs.extend(pair_list)
        pool.close()
        pool.join()
        root_tf.close()

        return pairs

    def run(self):
        """
        extract and resize images then write manifest files to disk.
        """
        cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
        log_file = os.path.join(self.orig_out_dir, 'train.log')
        manifest_list_cfg = ', '.join([k+':'+v for k, v in self.manifests.items()])

        with open(cfg_file, 'w') as f:
            f.write('manifest = [{}]\n'.format(manifest_list_cfg))
            f.write('manifest_root = {}\n'.format(self.out_dir))
            f.write('log = {}\n'.format(log_file))
            f.write('epochs = 90\nrng_seed = 0\nverbose = True\neval_freq = 1\n')

        if (all([os.path.exists(manifest) for manifest in self.manifests.values()])
                and not self.overwrite):
            print("Found manfiest files, skipping ingest, use --overwrite to overwrite them.")
            return

        for setn, manifest in self.manifests.items():
            pairs = self.train_or_val_pairs(setn)
            records = [(os.path.relpath(fname, self.out_dir), int(tgt))
                       for fname, tgt in pairs]
            records.insert(0, ('@FILE', 'STRING'))
            np.savetxt(manifest, records, fmt='%s\t%s')

if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument('--input_dir', required=True,
                        help='Directory to find input tars', default=None)
    parser.add_argument('--out_dir', required=True,
                        help='Directory to write ingested files', default=None)
    parser.add_argument('--target_size', type=int, default=256,
                        help='Size in pixels to scale shortest side DOWN to (0 means no scaling)')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite files')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    bw = IngestI1K(input_dir=args.input_dir, out_dir=args.out_dir, target_size=args.target_size,
                   overwrite=args.overwrite)

    bw.run()
