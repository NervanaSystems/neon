# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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

import argparse as argp
import functools
from glob import glob
import gzip
import imgworker
from multiprocessing import Pool
import numpy as np
import os
import sys
import tarfile
from time import time
import yaml

from neon.util.compat import range, StringIO
from neon.util.param import opt_param
from neon.util.persist import serialize

TARGET_SIZE = None
SQUARE_CROP = True

logger = logging.getLogger(__name__)


# NOTE: We have to leave this helper function out of the class and use the
#       global variable hack so that we can use multiprocess pool.map
def proc_img(imgfile, is_string=False):
    from PIL import Image
    if is_string:
        imgfile = StringIO(imgfile)
    im = Image.open(imgfile)

    # This part does the processing
    scale_factor = TARGET_SIZE / np.float32(min(im.size))
    (wnew, hnew) = map(lambda x: int(round(scale_factor * x)), im.size)
    if scale_factor != 1:
        filt = Image.BICUBIC if scale_factor > 1 else Image.ANTIALIAS
        im = im.resize((wnew, hnew), filt)

    if SQUARE_CROP is True:
        (cx, cy) = map(lambda x: (x - TARGET_SIZE) // 2, (wnew, hnew))
        im = im.crop((cx, cy, cx+TARGET_SIZE, cy+TARGET_SIZE))

    buf = StringIO()
    im.save(buf, format='JPEG')
    return buf.getvalue()


class BatchWriter(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.out_dir = os.path.expanduser(self.save_dir)
        self.in_dir = os.path.expanduser(self.image_dir)
        self.batch_size = self.macro_size
        global TARGET_SIZE, SQUARE_CROP
        TARGET_SIZE = self.output_image_size
        SQUARE_CROP = self.square_crop
        opt_param(self, ['file_pattern'], '*.jpg')
        opt_param(self, ['validation_pct'], 0.2)
        opt_param(self, ['num_workers'], 5)
        opt_param(self, ['class_samples_max'])
        self.train_file = os.path.join(self.out_dir, 'train_file.csv.gz')
        self.val_file = os.path.join(self.out_dir, 'val_file.csv.gz')
        self.stats = os.path.join(self.out_dir, 'dataset_cache.pkl')
        self.val_mean = np.zeros((self.output_image_size,
                                 self.output_image_size,
                                 self.num_channels), dtype=np.uint8)
        self.train_mean = np.zeros((self.output_image_size,
                                   self.output_image_size,
                                   self.num_channels), dtype=np.uint8)

    def __str__(self):
        pairs = map(lambda a: a[0] + ': ' + a[1],
                    zip(self.__dict__.keys(),
                        map(str, self.__dict__.values())))
        return "\n".join(pairs)

    def write_csv_files(self):
        # Get the labels as the subdirs
        subdirs = glob(os.path.join(self.in_dir, '*'))
        labels = sorted(map(lambda x: os.path.basename(x), subdirs))
        indexes = range(len(labels))
        self.labels_dict = {k: v for k, v in zip(labels, indexes)}

        tlines = []
        vlines = []
        for subdir in subdirs:
            subdir_label = self.labels_dict[os.path.basename(subdir)]
            files = glob(os.path.join(subdir, self.file_pattern))
            np.random.shuffle(files)
            if self.class_samples_max is not None:
                files = files[:self.class_samples_max]
            lines = [(filename, subdir_label) for filename in files]
            v_idx = int(self.validation_pct * len(lines))
            tlines += lines[v_idx:]
            vlines += lines[:v_idx]

        np.random.shuffle(tlines)
        np.random.shuffle(vlines)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        for ff, ll in zip([self.train_file, self.val_file], [tlines, vlines]):
            with gzip.open(ff, 'wb') as f:
                f.write('filename,l_id\n')
                for tup in ll:
                    f.write('{},{}\n'.format(*tup))
            f.close()

        # Write out cached stats for this data
        self.ntrain = (len(tlines) + self.batch_size - 1) // self.batch_size
        self.train_nrec = len(tlines)
        self.nval = (len(vlines) + self.batch_size - 1) // self.batch_size
        self.val_nrec = len(vlines)
        self.train_start = 0
        self.val_start = 10 ** int(np.log10(self.ntrain * 10))

    def parse_file_list(self, infile):
        import pandas as pd
        compression = 'gzip' if infile.endswith('.gz') else None
        df = pd.read_csv(infile, compression=compression)

        lk = filter(lambda x: x.startswith('l'), df.keys())
        tk = filter(lambda x: x.startswith('t'), df.keys())

        labels = {ll: np.array(df[ll].values, np.int32) for ll in lk}
        targets = np.array(df[tk].values, np.float32) if len(tk) > 0 else None
        imfiles = df['filename'].values

        self.nclass = {ll: (max(df[ll].values) + 1) for ll in lk}
        return imfiles, labels, targets

    def write_batches(self, name, start, labels, imfiles, targets=None,
                      is_tar=False):
        pool = Pool(processes=self.num_workers)
        psz = self.batch_size
        osz = self.output_image_size
        npts = (len(imfiles) + psz - 1) // psz

        imfiles = [imfiles[i*psz: (i+1)*psz] for i in range(npts)]

        if targets is not None:
            targets = [targets[i*psz: (i+1)*psz].T.copy() for i in range(npts)]

        labels = [{k: v[i*psz: (i+1)*psz] for k, v in labels.iteritems()}
                  for i in range(npts)]

        accum_buf = np.zeros((osz, osz, self.num_channels), dtype=np.int32)
        batch_mean = np.zeros(accum_buf.shape, dtype=np.uint8)
        logger.info("Writing %s batches...", name)
        for i, jpeg_file_batch in enumerate(imfiles):
            t = time()
            if is_tar:
                jpeg_file_batch = [j.read() for j in jpeg_file_batch]
            jpeg_strings = pool.map(
                functools.partial(proc_img, is_string=is_tar), jpeg_file_batch)
            targets_batch = None if targets is None else targets[i]
            labels_batch = labels[i]
            bfile = os.path.join(self.out_dir, 'data_batch_%d' % (start + i))
            serialize({'data': jpeg_strings,
                       'labels': labels_batch,
                       'targets': targets_batch},
                      bfile)
            logger.info("Wrote to %s (%s batch %d of %d) (%.2f sec)",
                        self.out_dir, name, i + 1, len(imfiles), time() - t)

            # get the means and accumulate
            imgworker.calc_batch_mean(jpglist=jpeg_strings, tgt=batch_mean,
                                      orig_size=osz, rgb=self.rgb,
                                      nthreads=self.num_workers)

            # scale for the case where we have an undersized batch
            if len(jpeg_strings) < self.batch_size:
                batch_mean *= len(jpeg_strings) / self.batch_size
            accum_buf += batch_mean
        pool.close()
        mean_buf = self.train_mean if name == 'train' else self.val_mean
        mean_buf[:] = accum_buf / len(imfiles)

    def save_meta(self):
        serialize({'ntrain': self.ntrain,
                   'nval': self.nval,
                   'train_start': self.train_start,
                   'val_start': self.val_start,
                   'macro_size': self.batch_size,
                   'train_mean': self.train_mean,
                   'val_mean': self.val_mean,
                   'labels_dict': self.labels_dict,
                   'val_nrec': self.val_nrec,
                   'train_nrec': self.train_nrec,
                   'nclass': self.nclass}, self.stats)

    def run(self):
        self.write_csv_files()
        namelist = ['train', 'validation']
        filelist = [self.train_file, self.val_file]
        startlist = [self.train_start, self.val_start]
        for sname, fname, start in zip(namelist, filelist, startlist):
            logger.info("%s %s %s", sname, fname, start)
            if fname is not None and os.path.exists(fname):
                imgs, labels, targets = self.parse_file_list(fname)
                self.write_batches(sname, start, labels, imgs, targets)
            else:
                logger.info('Skipping %s, file missing', sname)
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
        bsz = self.batch_size

        load_dir = self.in_dir
        # load_dir = os.path.join(os.path.expandvars(
        #                 os.path.expanduser(self.in_dir)), 'I1K')
        train_tar = os.path.join(load_dir, 'ILSVRC2012_img_train.tar')
        validation_tar = os.path.join(load_dir, 'ILSVRC2012_img_val.tar')
        devkit_tar = os.path.join(load_dir, 'ILSVRC2012_devkit_t12.tar.gz')
        self.url = "http://www.image-net.org/download-imageurls"
        for infile in (train_tar, validation_tar, devkit_tar):
            if not os.path.exists(infile):
                raise IOError(infile + " not found. Please ensure you have"
                              "ImageNet downloaded. More info here: " +
                              self.url)
        labels_dict, label_names, val_labels = self.parse_dev_meta(devkit_tar)
        self.labels_dict = labels_dict
        np.random.seed(0)
        with self.open_tar(train_tar, 'training tar') as tf:
            s_sets = tf.getmembers()
            s_tars = [tarfile.open(fileobj=tf.extractfile(s)) for s in s_sets]

            logger.info("Loaded synset tars.")
            logger.info('Building trainset list ( can take a while)...')

            t_jpegfiles = []
            for i, st in enumerate(s_tars):
                if i % 100 == 0:
                    pct_done = int(round((100.0 * i) / len(s_tars)))
                    logger.info("%d%% ...", pct_done)
                t_jpegfiles += [st.extractfile(m) for m in st.getmembers()]
                st.close()

            np.random.shuffle(t_jpegfiles)
            train_labels = [[labels_dict[j.name[:9]]] for j in t_jpegfiles]
            num_train_files = len(t_jpegfiles)
            logger.info("created list of jpg files")
            logger.info("Number of training files = %d", num_train_files)

            self.ntrain = (num_train_files + bsz - 1) // bsz
            self.train_nrec = num_train_files
            self.nclass = {'l_id': 1000}
            self.train_start = 0
            train_labels = {'l_id': np.array(train_labels, dtype=np.int32)}
            self.write_batches('train', self.train_start, train_labels,
                               t_jpegfiles, targets=None, is_tar=True)

        with self.open_tar(validation_tar, 'validation tar') as tf:
            v_jpegfiles = sorted([tf.extractfile(m) for m in tf.getmembers()],
                                 key=lambda x: x.name)
            num_val_files = len(v_jpegfiles)

            self.nval = (num_val_files + bsz - 1) // bsz
            self.val_nrec = num_val_files
            self.val_start = 10 ** int(np.log10(self.ntrain) + 1)
            val_labels = {'l_id': np.array(val_labels, dtype=np.int32)}
            self.write_batches('validation', self.val_start, val_labels,
                               v_jpegfiles, targets=None, is_tar=True)
        self.save_meta()

    def open_tar(self, path, name):
        if not os.path.exists(path):
            logger.error("ILSVRC 2012 %s not found at %s.",
                         "Make sure to set ILSVRC_SRC_DIR correctly at the",
                         "top of this file (%s)." % (name, path, sys.argv[0]))
            sys.exit(1)
        return tarfile.open(path)

    def parse_dev_meta(self, ilsvrc_devkit_tar):
        tf = self.open_tar(ilsvrc_devkit_tar, 'devkit tar')
        fmeta = tf.extractfile(
            tf.getmember('ILSVRC2012_devkit_t12/data/meta.mat'))
        import scipy.io
        meta_mat = scipy.io.loadmat(StringIO(fmeta.read()))
        labels_dic = dict(
            (m[0][1][0], m[0][0][0][0] - 1) for m in meta_mat['synsets']
            if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
        label_names_dic = dict(
            (m[0][1][0], m[0][2][0]) for m in meta_mat['synsets']
            if (m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000))
        label_names = [tup[1] for tup in sorted(
            [(v, label_names_dic[k]) for k, v in labels_dic.items()],
            key=lambda x:x[0])]

        fvgtruth = tf.extractfile(tf.getmember(
            'ILSVRC2012_devkit_t12/data/' +
            'ILSVRC2012_validation_ground_truth.txt'))
        vgtruth = [[int(line.strip()) - 1] for line in fvgtruth.readlines()]
        tf.close()
        return labels_dic, label_names, vgtruth


if __name__ == "__main__":
    parser = argp.ArgumentParser()
    parser.add_argument('--config', help='Configuration File', required=True)
    parser.add_argument('--dataset', help='Dataset name', required=True)

    args = parser.parse_args()
    with open(args.config) as f:
        ycfg = yaml.load(f)[args.dataset]
    bw = BatchWriterImagenet(**ycfg)
    bw.run()
