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
Generic image-like dataset able to be processed in macro batches.
"""

import logging
import numpy as np
import os
import sys
from threading import Thread

from neon.datasets.dataset import Dataset
from neon.util.param import opt_param, req_param
from neon.util.persist import deserialize

logger = logging.getLogger(__name__)


class MacrobatchDecodeThread(Thread):
    """
    Load and decode a macrobatch of images in a separate thread,
    double buffering.

    Hide the time to transpose and convert (astype).
    """

    def __init__(self, ds):
        Thread.__init__(self)
        self.ds = ds

    def run(self):
        import imgworker
        bsz = self.ds.batch_size
        b_idx = self.ds.macro_decode_buf_idx
        jdict = self.ds.get_macro_batch()
        betype = self.ds.backend_type

        # This macrobatch could be smaller than macro_size for last macrobatch
        mac_sz = len(jdict['data'])
        self.ds.tgt_macro[b_idx] = \
            jdict['targets'] if 'targets' in jdict else None
        lbl_macro = {k: jdict['labels'][k] for k in self.ds.label_list}

        img_macro = np.zeros((self.ds.macro_size, self.ds.npixels),
                             dtype=np.uint8)

        imgworker.decode_list(jpglist=jdict['data'],
                              tgt=img_macro[:mac_sz],
                              orig_size=self.ds.output_image_size,
                              crop_size=self.ds.cropped_image_size,
                              center=self.ds.predict, flip=True,
                              rgb=self.ds.rgb,
                              nthreads=self.ds.num_workers)
        if mac_sz < self.ds.macro_size:
            img_macro[mac_sz:] = 0
        # Leave behind the partial minibatch
        self.ds.minis_per_macro[b_idx] = mac_sz / bsz

        self.ds.lbl_one_hot[b_idx] = \
            {lbl: [None for mini_idx in range(self.ds.minis_per_macro[b_idx])]
                for lbl in self.ds.label_list}

        self.ds.img_mini_T[b_idx] = \
            [None for mini_idx in range(self.ds.minis_per_macro[b_idx])]

        for mini_idx in range(self.ds.minis_per_macro[b_idx]):
            s_idx = mini_idx * bsz
            e_idx = (mini_idx + 1) * bsz
            self.ds.img_mini_T[b_idx][mini_idx] = \
                img_macro[s_idx:e_idx].T.astype(betype, order='C')

            if self.ds.img_mini_T[b_idx][mini_idx].shape[1] < bsz:
                tmp = self.ds.img_mini_T[b_idx][mini_idx].shape[0]
                mb_residual = self.ds.img_mini_T[b_idx][mini_idx].shape[1]
                filledbatch = np.vstack((img_macro[s_idx:e_idx],
                                         np.zeros((bsz - mb_residual, tmp))))
                self.ds.img_mini_T[b_idx][mini_idx] = \
                    filledbatch.T.astype(betype, order='C')
            for lbl in self.ds.label_list:
                hl = np.squeeze(lbl_macro[lbl][s_idx:e_idx])
                self.ds.lbl_one_hot[b_idx][lbl][mini_idx] = \
                    np.eye(self.ds.nclass[lbl])[hl].T.astype(betype, order='C')

        return


class Imageset(Dataset):

    """
    Sets up a macro batched imageset dataset.

    Assumes you have the data already partitioned and in macrobatch format

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """
    def __init__(self, **kwargs):

        opt_param(self, ['preprocess_done'], False)
        opt_param(self, ['dotransforms', 'square_crop'], False)
        opt_param(self, ['mean_norm', 'unit_norm'], False)

        opt_param(self, ['tdims'], 0)
        opt_param(self, ['label_list'], ['l_id'])
        opt_param(self, ['num_channels'], 3)

        opt_param(self, ['num_workers'], 6)
        opt_param(self, ['backend_type'], 'np.float32')

        self.__dict__.update(kwargs)

        if self.backend_type in ['float16', 'np.float16', 'numpy.float16']:
            self.backend_type = np.float16
        elif self.backend_type in ['float32', 'np.float32', 'numpy.float32']:
            self.backend_type = np.float32
        else:
            raise ValueError('Datatype not understood')
        logger.warning("Imageset initialized with dtype %s", self.backend_type)
        req_param(self, ['cropped_image_size', 'output_image_size',
                         'imageset', 'save_dir', 'repo_path', 'macro_size'])

        opt_param(self, ['image_dir'], os.path.join(self.repo_path,
                                                    self.imageset))

        self.rgb = True if self.num_channels == 3 else False
        self.norm_factor = 128. if self.mean_norm else 256.

    def __getstate__(self):
        """
        Defines what and how we go about serializing an instance of this class.
        """
        self.macro_decode_thread = None
        return self.__dict__

    def __setstate__(self, state):
        """
        Defines how we go about deserializing into an instance of this class.
        """
        self.__dict__.update(state)

    def load(self):
        bdir = os.path.expanduser(self.save_dir)
        cachefile = os.path.join(bdir, 'dataset_cache.pkl')
        if not os.path.exists(cachefile):
            logger.warning("Batch dir cache not found in %s:", cachefile)
            # response = 'Y'
            response = raw_input("Press Y to create, otherwise exit: ")
            if response == 'Y':
                from neon.util.batch_writer import (BatchWriter,
                                                    BatchWriterImagenet)

                if self.imageset.startswith('I1K'):
                    self.bw = BatchWriterImagenet(**self.__dict__)
                else:
                    self.bw = BatchWriter(**self.__dict__)
                self.bw.run()
                logger.warning('Done writing batches - please rerun to train.')
            else:
                logger.warning('Exiting...')
            sys.exit()
        cstats = deserialize(cachefile, verbose=False)
        if cstats['macro_size'] != self.macro_size:
            raise NotImplementedError("Cached macro size %d different from "
                                      "specified %d, delete save_dir %s "
                                      "and try again.",
                                      cstats['macro_size'],
                                      self.macro_size,
                                      self.save_dir)
        # Set the max indexes of batches for each from the cache file
        self.maxval = cstats['nval'] + cstats['val_start'] - 1
        self.maxtrain = cstats['ntrain'] + cstats['train_start'] - 1

        # Make sure only those properties not by yaml are updated
        cstats.update(self.__dict__)
        self.__dict__.update(cstats)
        # Should also put (in addition to nclass), number of train/val images
        req_param(self, ['ntrain', 'nval', 'train_start', 'val_start',
                         'train_mean', 'val_mean', 'labels_dict'])

    def get_macro_batch(self):
        self.macro_idx = (self.macro_idx + 1 - self.startb) \
            % self.nmacros + self.startb
        fname = os.path.join(self.save_dir,
                             'data_batch_{:d}'.format(self.macro_idx))
        return deserialize(os.path.expanduser(fname), verbose=False)

    def del_mini_batch_producer(self):
        if self.macro_decode_thread is not None:
            self.macro_decode_thread.join()

    def init_mini_batch_producer(self, batch_size, setname, predict=False):
        # local shortcuts
        sbe = self.backend.empty
        betype = self.backend_type
        sn = 'val' if (setname == 'validation') else setname
        osz = self.output_image_size
        csz = self.cropped_image_size
        self.npixels = csz * csz * self.num_channels

        self.startb = getattr(self, sn + '_start')
        self.nmacros = getattr(self, 'n' + sn)
        self.maxmacros = getattr(self, 'max' + sn)

        if self.startb + self.nmacros - 1 > self.maxmacros:
            self.nmacros = self.maxmacros - self.startb + 1
            logger.warning("Truncating n%s to %d", sn, self.nmacros)

        self.endb = self.startb + self.nmacros - 1
        if self.endb == self.maxmacros:
            nrecs = getattr(self, sn + '_nrec') % self.macro_size + \
                (self.nmacros - 1) * self.macro_size
        else:
            nrecs = self.nmacros * self.macro_size
        num_batches = nrecs / batch_size

        self.mean_img = getattr(self, sn + '_mean')
        self.mean_img.shape = (self.num_channels, osz, osz)
        pad = (osz - csz) / 2
        self.mean_crop = self.mean_img[:, pad:(pad + csz), pad:(pad + csz)]
        self.mean_be = sbe((self.npixels, 1), dtype=betype)
        self.mean_be.copy_from(self.mean_crop.reshape(
            (self.npixels, 1)).astype(np.float32))

        # Control params for macrobatch decoding thread
        self.macro_active_buf_idx = 0
        self.macro_decode_buf_idx = 0
        self.macro_num_decode_buf = 2
        self.macro_decode_thread = None

        self.batch_size = batch_size
        self.predict = predict
        self.minis_per_macro = [self.macro_size / batch_size
                                for i in range(self.macro_num_decode_buf)]

        if self.macro_size % batch_size != 0:
            raise ValueError('self.macro_size not divisible by batch_size')

        self.macro_idx = self.endb
        self.mini_idx = -1

        # Allocate space for host side image, targets and labels
        self.img_mini_T = [None for i in range(self.macro_num_decode_buf)]
        self.tgt_macro = [None for i in range(self.macro_num_decode_buf)]
        self.lbl_one_hot = [None for i in range(self.macro_num_decode_buf)]

        # Allocate space for device side buffers
        inp_shape = (self.npixels, self.batch_size)
        self.inp_be = sbe(inp_shape, dtype=betype)

        lbl_shape = {lbl: (self.nclass[lbl], self.batch_size)
                     for lbl in self.label_list}
        self.lbl_be = {lbl: sbe(lbl_shape[lbl], dtype=betype)
                       for lbl in self.label_list}

        # Allocate space for device side targets if necessary
        tgt_shape = (self.tdims, self.batch_size)
        self.tgt_be = sbe(tgt_shape, dtype=betype) if self.tdims != 0 else None

        return num_batches

    def get_mini_batch(self, batch_idx):
        b_idx = self.macro_active_buf_idx
        self.mini_idx = (self.mini_idx + 1) % self.minis_per_macro[b_idx]

        # Decode macrobatches in a background thread,
        # except for the first one which blocks
        if self.mini_idx == 0:
            if self.macro_decode_thread is not None:
                # No-op unless all mini finish faster than one macro
                self.macro_decode_thread.join()
            else:
                # special case for first run through
                self.macro_decode_thread = MacrobatchDecodeThread(self)
                self.macro_decode_thread.start()
                self.macro_decode_thread.join()

            # usual case for kicking off a background macrobatch thread
            self.macro_active_buf_idx = self.macro_decode_buf_idx
            self.macro_decode_buf_idx = \
                (self.macro_decode_buf_idx + 1) % self.macro_num_decode_buf
            self.macro_decode_thread = MacrobatchDecodeThread(self)
            self.macro_decode_thread.start()

        # All minibatches except for the 0th just copy pre-prepared data
        b_idx = self.macro_active_buf_idx
        s_idx = self.mini_idx * self.batch_size
        e_idx = (self.mini_idx + 1) * self.batch_size

        # See if we are a partial minibatch
        self.inp_be.copy_from(self.img_mini_T[b_idx][self.mini_idx])

        # Try to avoid this if possible as it inhibits async stream copy
        if self.mean_norm:
            self.backend.subtract(self.inp_be, self.mean_be, self.inp_be)

        if self.unit_norm:
            self.backend.divide(self.inp_be, self.norm_factor, self.inp_be)

        for lbl in self.label_list:
            self.lbl_be[lbl].copy_from(
                self.lbl_one_hot[b_idx][lbl][self.mini_idx])

        if self.tgt_be is not None:
            self.tgt_be.copy_from(
                self.tgt_macro[b_idx][:, s_idx:e_idx]
                    .astype(self.backend_type))

        return self.inp_be, self.tgt_be, self.lbl_be

    def has_set(self, setname):
        return True if (setname in ['train', 'validation']) else False
