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
Process macro batches of data in a pipelined fashion.
"""

import logging

from glob import glob
import functools
import gzip
from multiprocessing import Pool
import numpy as np
import os
import tarfile
import struct
from PIL import Image as PILImage
from neon.util.compat import range, StringIO
from neon.util.persist import load_obj, save_obj
from neon.data import load_i1kmeta
from neon.util.argparser import NeonArgparser


# NOTE: We have to leave this helper function out of the class to use multiprocess pool.map
def proc_img(target_size, squarecrop, is_string=False, imgfile=None):
    imgfile = StringIO(imgfile) if is_string else imgfile
    im = PILImage.open(imgfile)

    scale_factor = target_size / np.float32(min(im.size))
    if scale_factor == 1 and im.size[0] == im.size[1] and is_string is False:
        return np.fromfile(imgfile, dtype=np.uint8)

    (wnew, hnew) = map(lambda x: int(round(scale_factor * x)), im.size)
    if scale_factor != 1:
        filt = PILImage.BICUBIC if scale_factor > 1 else PILImage.ANTIALIAS
        im = im.resize((wnew, hnew), filt)

    if squarecrop is True:
        (cx, cy) = map(lambda x: (x - target_size) // 2, (wnew, hnew))
        im = im.crop((cx, cy, cx+target_size, cy+target_size))

    buf = StringIO()
    im.save(buf, format='JPEG', subsampling=0, quality=95)
    return buf.getvalue()


class BatchWriter(object):

    def __init__(self, out_dir, image_dir, target_size=256, squarecrop=True, validation_pct=0.2,
                 class_samples_max=None, file_pattern='*.jpg', macro_size=3072):
        np.random.seed(0)
        self.out_dir = os.path.expanduser(out_dir)
        self.image_dir = os.path.expanduser(image_dir)
        self.macro_size = macro_size
        self.num_workers = 8
        self.target_size = target_size
        self.squarecrop = squarecrop
        self.file_pattern = file_pattern
        self.class_samples_max = class_samples_max
        self.validation_pct = validation_pct
        self.train_file = os.path.join(self.out_dir, 'train_file.csv.gz')
        self.val_file = os.path.join(self.out_dir, 'val_file.csv.gz')
        self.meta_file = os.path.join(self.out_dir, 'dataset_cache.pkl')
        self.global_mean = np.array([0, 0, 0]).reshape((3, 1))
        self.batch_prefix = 'data_batch_'

    def write_csv_files(self):
        # Get the labels as the subdirs
        subdirs = glob(os.path.join(self.image_dir, '*'))
        self.label_names = sorted(map(lambda x: os.path.basename(x), subdirs))

        indexes = range(len(self.label_names))
        self.label_dict = {k: v for k, v in zip(self.label_names, indexes)}

        tlines = []
        vlines = []
        for subdir in subdirs:
            subdir_label = self.label_dict[os.path.basename(subdir)]
            files = glob(os.path.join(subdir, self.file_pattern))
            if self.class_samples_max is not None:
                files = files[:self.class_samples_max]
            lines = [(filename, subdir_label) for filename in files]
            v_idx = int(self.validation_pct * len(lines))
            tlines += lines[v_idx:]
            vlines += lines[:v_idx]
        np.random.shuffle(tlines)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        for ff, ll in zip([self.train_file, self.val_file], [tlines, vlines]):
            with gzip.open(ff, 'wb') as f:
                f.write('filename,l_id\n')
                for tup in ll:
                    f.write('{},{}\n'.format(*tup))

        self.train_nrec = len(tlines)
        self.ntrain = -(-self.train_nrec // self.macro_size)
        self.train_start = 0

        self.val_nrec = len(vlines)
        self.nval = -(-self.val_nrec // self.macro_size)
        if self.ntrain == 0:
            self.val_start = 0
        else:
            self.val_start = 10 ** int(np.log10(self.ntrain * 10))

    def parse_file_list(self, infile):
        lines = np.loadtxt(infile, delimiter=',', skiprows=1, dtype={'names': ('fname', 'l_id'),
                                                                     'formats': (object, 'i4')})
        imfiles = [l[0] for l in lines]
        labels = {'l_id': [l[1] for l in lines]}
        self.nclass = {'l_id': (max(labels['l_id']) + 1)}
        return imfiles, labels

    def write_batches(self, name, offset, labels, imfiles):
        pool = Pool(processes=self.num_workers)
        npts = -(-len(imfiles) // self.macro_size)
        starts = [i * self.macro_size for i in range(npts)]
        is_tar = isinstance(imfiles[0], tarfile.ExFileObject)
        proc_img_func = functools.partial(proc_img, self.target_size, self.squarecrop, is_tar)
        imfiles = [imfiles[s:s + self.macro_size] for s in starts]
        labels = [{k: v[s:s + self.macro_size] for k, v in labels.iteritems()} for s in starts]

        print("Writing %s batches..." % (name))
        for i, jpeg_file_batch in enumerate(imfiles):
            if is_tar:
                jpeg_file_batch = [j.read() for j in jpeg_file_batch]
            jpeg_strings = pool.map(proc_img_func, jpeg_file_batch)
            bfile = os.path.join(self.out_dir, '%s%d' % (self.batch_prefix, offset + i))
            self.write_binary(jpeg_strings, labels[i], bfile)
            print("Writing batch %d" % (i))
        pool.close()

    def write_binary(self, jpegs, labels, ofname):
        num_imgs = len(jpegs)
        keylist = ['l_id']
        with open(ofname, 'wb') as f:
            f.write(struct.pack('I', num_imgs))
            f.write(struct.pack('I', len(keylist)))

            for key in keylist:
                ksz = len(key)
                f.write(struct.pack('L' + 'B' * ksz, ksz, *bytearray(key)))
                f.write(struct.pack('I' * num_imgs, *labels[key]))

            for i in range(num_imgs):
                jsz = len(jpegs[i])
                bin = struct.pack('I' + 'B' * jsz, jsz, *bytearray(jpegs[i]))
                f.write(bin)

    def save_meta(self):
        save_obj({'ntrain': self.ntrain,
                  'nval': self.nval,
                  'train_start': self.train_start,
                  'val_start': self.val_start,
                  'macro_size': self.macro_size,
                  'batch_prefix': self.batch_prefix,
                  'global_mean': self.global_mean,
                  'label_dict': self.label_dict,
                  'label_names': self.label_names,
                  'val_nrec': self.val_nrec,
                  'train_nrec': self.train_nrec,
                  'img_size': self.target_size,
                  'nclass': self.nclass}, self.meta_file)

    def run(self):
        self.write_csv_files()
        if self.validation_pct == 0:
            namelist = ['train']
            filelist = [self.train_file]
            startlist = [self.train_start]
        elif self.validation_pct == 1:
            namelist = ['validation']
            filelist = [self.val_file]
            startlist = [self.val_start]
        else:
            namelist = ['train', 'validation']
            filelist = [self.train_file, self.val_file]
            startlist = [self.train_start, self.val_start]
        for sname, fname, start in zip(namelist, filelist, startlist):
            print("%s %s %s" % (sname, fname, start))
            if fname is not None and os.path.exists(fname):
                imgs, labels = self.parse_file_list(fname)
                self.write_batches(sname, start, labels, imgs)
            else:
                print("Skipping %s, file missing" % (sname))
        self.save_meta()


class BatchWriterImagenet(BatchWriter):

    # code below adapted from Alex Krizhevsky's cuda-convnet2 library,
    # make-data.py
    # Copyright 2014 Google Inc. All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #    http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    ##########################################################################

    def run(self):
        load_dir = self.image_dir
        train_tar = os.path.join(load_dir, 'ILSVRC2012_img_train.tar')
        validation_tar = os.path.join(load_dir, 'ILSVRC2012_img_val.tar')

        for infile in (train_tar, validation_tar):
            if not os.path.exists(infile):
                raise IOError(infile + " not found. Please ensure you have ImageNet downloaded."
                              "More info here: http://www.image-net.org/download-imageurls")
        # download our version of the metadata
        meta_dir = load_i1kmeta(self.out_dir)
        meta_file = os.path.join(meta_dir, 'neon_ILSVRC2012_devmeta.pkl')
        self.meta = load_obj(meta_file)
        self.__dict__.update(self.meta)  # get label_dict, label_names, global_mean from meta
        self.global_mean = np.mean(self.global_mean.reshape(3, -1), axis=1).reshape(3, 1)[::-1]

        np.random.seed(0)
        with tarfile.open(train_tar) as tf:
            s_sets = tf.getmembers()
            s_tars = [tarfile.open(fileobj=tf.extractfile(s)) for s in s_sets]
            print('Building trainset list from synset tars.')
            t_jpegfiles = []
            totalsz = len(s_tars)
            for i, st in enumerate(s_tars):
                if i % 100 == 0:
                    print("%d%% ..." % (int(round((100.0 * i) / totalsz))))
                t_jpegfiles += [st.extractfile(m) for m in st.getmembers()]
                st.close()
            print("Done loading")
            np.random.shuffle(t_jpegfiles)
            train_labels = [[self.label_dict[j.name[:9]]] for j in t_jpegfiles]
            self.train_nrec = len(t_jpegfiles)
            self.ntrain = -(-self.train_nrec // self.macro_size)
            self.nclass = {'l_id': 1000}
            self.train_start = 0
            train_labels = {'l_id': np.array(train_labels, dtype=np.int32)}
            self.write_batches('train', self.train_start, train_labels, t_jpegfiles)

        with tarfile.open(validation_tar) as tf:
            jpegfiles = sorted([tf.extractfile(m) for m in tf.getmembers()], key=lambda x: x.name)
            self.val_nrec = len(jpegfiles)
            self.nval = -(-self.val_nrec // self.macro_size)
            self.val_start = 10 ** int(np.log10(self.ntrain) + 1)
            val_labels = {'l_id': np.array(self.val_ground_truth, dtype=np.int32)}
            self.write_batches('val', self.val_start, val_labels, jpegfiles)
        self.save_meta()


if __name__ == "__main__":
    parser = NeonArgparser(__doc__)
    parser.add_argument('--set_type', help='(i1k|directory)', required=True,
                        choices=['i1k', 'directory'])
    parser.add_argument('--image_dir', help='Directory to find images', required=True)
    parser.add_argument('--target_size', type=int, default=256,
                        help='Size in pixels to scale images (Must be 256 for i1k dataset)')
    parser.add_argument('--macro_size', type=int, default=5000, help='Images per processed batch')
    parser.add_argument('--file_pattern', default='*.jpg', help='Image extension to include in'
                        'directory crawl')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    # Supply dataset type and location
    if args.set_type == 'i1k':
        bw = BatchWriterImagenet(out_dir=args.data_dir, image_dir=args.image_dir,
                                 macro_size=args.macro_size)
    else:
        bw = BatchWriter(out_dir=args.data_dir, image_dir=args.image_dir,
                         target_size=args.target_size, macro_size=args.macro_size,
                         file_pattern=args.file_pattern)

    bw.run()
