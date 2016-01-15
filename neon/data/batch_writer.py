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
import gzip
import numpy as np
import os
import tarfile
import ctypes as ct

from neon.util.compat import range
from neon.util.argparser import NeonArgparser

logger = logging.getLogger(__name__)


class BatchWriter(object):

    def __init__(self, out_dir, image_dir, target_size=256, validation_pct=0.2,
                 class_samples_max=None, file_pattern='*.jpg', macro_size=3072,
                 pixel_mean=(0, 0, 0)):

        libpath = os.path.dirname(os.path.realpath(__file__))
        try:
            self.writerlib = ct.cdll.LoadLibrary(
                os.path.join(libpath, 'loader', 'loader.so'))
            self.writerlib.write_batch.restype = None
            self.writerlib.read_max_item.restype = ct.c_int
        except:
            logger.error('Unable to load loader.so. Ensure that '
                         'this file has been compiled')

        np.random.seed(0)
        self.out_dir = os.path.expanduser(out_dir)
        self.image_dir = os.path.expanduser(image_dir)
        self.macro_size = macro_size
        self.target_size = target_size
        self.file_pattern = file_pattern
        self.class_samples_max = class_samples_max
        self.validation_pct = validation_pct
        self.train_file = os.path.join(self.out_dir, 'train_file.csv.gz')
        self.val_file = os.path.join(self.out_dir, 'val_file.csv.gz')
        self.batch_prefix = 'macrobatch_'
        self.meta_file = os.path.join(self.out_dir, self.batch_prefix + 'meta')
        self.pixel_mean = pixel_mean
        self.item_max_size = 25000  # reasonable default max image size
        self.post_init()

    def post_init(self):
        pass

    def write_csv_files(self):
        # Get the labels as the subdirs
        subdirs = glob(os.path.join(self.image_dir, '*'))
        self.label_names = sorted([os.path.basename(x) for x in subdirs])

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
        self.val_start = self.ntrain

    def parse_file_list(self, infile):
        lines = np.loadtxt(infile, delimiter=',', skiprows=1, dtype={'names': ('fname', 'l_id'),
                                                                     'formats': (object, 'i4')})
        imfiles = [l[0] for l in lines]
        labels = {'l_id': [l[1] for l in lines]}
        self.nclass = {'l_id': (max(labels['l_id']) + 1)}
        return imfiles, labels

    def write_individual_batch(self, batch_file, label_batch, jpeg_file_batch):
        ndata = len(jpeg_file_batch)
        jpgfiles = (ct.c_char_p * ndata)()
        jpgfiles[:] = jpeg_file_batch

        # This interface to the batchfile.hpp allows you to specify
        # destination file, number of input jpg files, list of jpg files,
        # and corresponding list of integer labels
        self.writerlib.write_batch(ct.c_char_p(batch_file),
                                   ct.c_int(ndata),
                                   jpgfiles,
                                   (ct.c_int * ndata)(*label_batch),
                                   ct.c_int(self.target_size))

    def write_batches(self, offset, labels, imfiles):
        npts = -(-len(imfiles) // self.macro_size)
        starts = [i * self.macro_size for i in range(npts)]
        imfiles = [imfiles[s:s + self.macro_size] for s in starts]
        labels = [{k: v[s:s + self.macro_size] for k, v in labels.iteritems()} for s in starts]

        for i, jpeg_file_batch in enumerate(imfiles):
            bfile = os.path.join(self.out_dir, '%s%d.cpio' % (self.batch_prefix, offset + i))
            label_batch = labels[i]['l_id']
            if os.path.exists(bfile):
                print("File %s exists, skipping..." % (bfile))
            else:
                self.write_individual_batch(bfile, label_batch, jpeg_file_batch)
                print("Wrote batch %d" % (i))

            # Check the batchfile for the max item value
            batch_max_item = self.writerlib.read_max_item(ct.c_char_p(bfile))
            if batch_max_item == 0:
                raise ValueError("Batch file %s probably empty or corrupt" % (bfile))

            self.item_max_size = max(batch_max_item, self.item_max_size)

    def save_meta(self):
        with open(self.meta_file, 'w') as f:
            for settype in ('train', 'val'):
                f.write('n%s %d\n' % (settype, getattr(self, 'n' + settype)))
                f.write('%s_start %d\n' % (settype, getattr(self, settype + '_start')))
                f.write('%s_nrec %d\n' % (settype, getattr(self, settype + '_nrec')))
            f.write('nclass %d\n' % (self.nclass['l_id']))
            f.write('item_max_size %d\n' % (self.item_max_size))
            f.write('label_size %d\n' % (4))
            f.write('R_mean      %f\n' % self.pixel_mean[0])
            f.write('G_mean      %f\n' % self.pixel_mean[1])
            f.write('B_mean      %f\n' % self.pixel_mean[2])

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
            print("Writing %s %s %s" % (sname, fname, start))
            if fname is not None and os.path.exists(fname):
                imgs, labels = self.parse_file_list(fname)
                self.write_batches(start, labels, imgs)
            else:
                print("Skipping %s, file missing" % (sname))
        # Get the max item size and store it for meta file
        self.save_meta()


class BatchWriterI1K(BatchWriter):

    def post_init(self):
        import zlib
        import re

        load_dir = self.image_dir
        self.train_tar = os.path.join(load_dir, 'ILSVRC2012_img_train.tar')
        self.val_tar = os.path.join(load_dir, 'ILSVRC2012_img_val.tar')
        self.devkit = os.path.join(load_dir, 'ILSVRC2012_devkit_t12.tar.gz')

        for infile in (self.train_tar, self.val_tar, self.devkit):
            if not os.path.exists(infile):
                raise IOError(infile + " not found. Please ensure you have ImageNet downloaded."
                              "More info here: http://www.image-net.org/download-imageurls")

        with tarfile.open(self.devkit, "r:gz") as tf:
            synsetfile = 'ILSVRC2012_devkit_t12/data/meta.mat'
            valfile = 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

            # get the synset mapping by hacking around matlab's terrible compressed format
            meta_buff = tf.extractfile(synsetfile).read()
            decomp = zlib.decompressobj()
            self.synsets = re.findall(re.compile('n\d+'), decomp.decompress(meta_buff[136:]))
            self.train_labels = {s: i for i, s in enumerate(self.synsets)}

            # get the ground truth validation labels and offset to zero
            self.val_labels = {"%08d" % (i + 1): int(x) - 1 for i, x in
                               enumerate(tf.extractfile(valfile))}
        self.validation_pct = None

        self.train_nrec = 1281167
        self.ntrain = -(-self.train_nrec // self.macro_size)
        self.train_start = 0

        self.val_nrec = 50000
        self.nval = -(-self.val_nrec // self.macro_size)
        self.val_start = self.ntrain
        self.pixel_mean = [104.41227722, 119.21331787, 126.80609131]

    def extract_images(self, overwrite=False):
        for setn in ('train', 'val'):
            img_dir = os.path.join(self.out_dir, setn)

            print("Extracting %s files" % (setn))
            toptar = getattr(self, setn + '_tar')
            label_dict = getattr(self, setn + '_labels')
            name_slice = slice(None, 9) if setn == 'train' else slice(15, -5)
            with tarfile.open(toptar) as tf:
                for s in tf.getmembers():
                    label = label_dict[s.name[name_slice]]
                    subpath = os.path.join(img_dir, str(label))
                    if not os.path.exists(subpath):
                        os.makedirs(subpath)
                    if setn == 'train':
                        tarfp = tarfile.open(fileobj=tf.extractfile(s))
                        file_list = tarfp.getmembers()
                    else:
                        tarfp = tf
                        file_list = [s]

                    for fobj in file_list:
                        fname = os.path.join(subpath, fobj.name)
                        if not os.path.exists(fname) or overwrite:
                            with open(fname, 'wb') as jf:
                                jf.write(tarfp.extractfile(fobj).read())

    def write_csv_files(self, overwrite=False):
        self.extract_images()
        for setn in ('train', 'val'):
            img_dir = os.path.join(self.out_dir, setn)
            csvfile = getattr(self, setn + '_file')
            print("Getting %s file list" % (setn))
            if os.path.exists(csvfile) and not overwrite:
                print("File %s exists, not overwriting" % (csvfile))
                continue
            flines = []

            subdirs = glob(os.path.join(img_dir, '*'))
            for subdir in subdirs:
                subdir_label = os.path.basename(subdir)  # This is the int label
                files = glob(os.path.join(subdir, self.file_pattern))
                flines += [(filename, subdir_label) for filename in files]

            if setn == 'train':
                np.random.seed(0)
                np.random.shuffle(flines)

            with gzip.open(csvfile, 'wb') as f:
                f.write('filename,l_id\n')
                for tup in flines:
                    f.write('{},{}\n'.format(*tup))


class BatchWriterCIFAR10(BatchWriterI1K):

    def post_init(self):
        self.pad_size = (self.target_size - 32) / 2 if self.target_size > 32 else 0
        self.pad_width = ((0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size))

        self.validation_pct = None

        self.train_nrec = 50000
        self.ntrain = -(-self.train_nrec // self.macro_size)
        self.train_start = 0

        self.val_nrec = 10000
        self.nval = -(-self.val_nrec // self.macro_size)
        self.val_start = self.ntrain

    def extract_images(self, overwrite=False):
        from neon.data import load_cifar10
        from PIL import Image
        dataset = dict()
        dataset['train'], dataset['val'], _ = load_cifar10(self.out_dir, normalize=False)

        for setn in ('train', 'val'):
            data, labels = dataset[setn]

            img_dir = os.path.join(self.out_dir, setn)
            ulabels = np.unique(labels)
            for ulabel in ulabels:
                subdir = os.path.join(img_dir, str(ulabel))
                if not os.path.exists(subdir):
                    os.makedirs(subdir)

            for idx in range(data.shape[0]):
                im = np.pad(data[idx].reshape((3, 32, 32)), self.pad_width, mode='mean')
                im = np.uint8(np.transpose(im, axes=[1, 2, 0]).copy())
                im = Image.fromarray(im)
                path = os.path.join(img_dir, str(labels[idx][0]), str(idx) + '.png')
                im.save(path, format='PNG')

            if setn == 'train':
                self.pixel_mean = list(data.mean(axis=0).reshape(3, -1).mean(axis=1))
                self.pixel_mean.reverse()  # We will see this in BGR order b/c of opencv


if __name__ == "__main__":
    parser = NeonArgparser(__doc__)
    parser.add_argument('--set_type', help='(i1k|cifar10|directory)', required=True,
                        choices=['i1k', 'cifar10', 'directory'])
    parser.add_argument('--image_dir', help='Directory to find images', required=True)
    parser.add_argument('--target_size', type=int, default=0,
                        help='Size in pixels to scale shortest side DOWN to (0 means no scaling)')
    parser.add_argument('--macro_size', type=int, default=5000, help='Images per processed batch')
    parser.add_argument('--file_pattern', default='*.jpg', help='Image extension to include in'
                        'directory crawl')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    if args.set_type == 'i1k':
        args.target_size = 256  # (maybe 512 for Simonyan's methodology?)
        bw = BatchWriterI1K(out_dir=args.data_dir, image_dir=args.image_dir,
                            target_size=args.target_size, macro_size=args.macro_size,
                            file_pattern="*.JPEG")
    elif args.set_type == 'cifar10':
        bw = BatchWriterCIFAR10(out_dir=args.data_dir, image_dir=args.image_dir,
                                target_size=args.target_size, macro_size=args.macro_size,
                                file_pattern="*.png")
    else:
        bw = BatchWriter(out_dir=args.data_dir, image_dir=args.image_dir,
                         target_size=args.target_size, macro_size=args.macro_size,
                         file_pattern=args.file_pattern)

    bw.run()
