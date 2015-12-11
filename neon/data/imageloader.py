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

import ctypes as ct
import logging
import numpy as np
import os
import atexit

from neon import NervanaObject
from neon.util.persist import load_obj

logger = logging.getLogger(__name__)


class ImageLoader(NervanaObject):
    """
    Encapsulates the data loader library and exposes a backward-compatible API
    to iterate over minibatches of images.
    """

    def __init__(self, repo_dir, inner_size, do_transforms=True,
                 rgb=True, shuffle=False, set_name='train', subset_pct=100,
                 nlabels=1, macro=True, dtype=np.float32):
        if not rgb:
            raise ValueError('Non-RGB images are currently not supported')
        self.configure(repo_dir, inner_size, do_transforms,
                       rgb, shuffle, set_name, subset_pct, macro)
        libpath = os.path.dirname(os.path.realpath(__file__))
        try:
            self.loaderlib = ct.cdll.LoadLibrary(
                os.path.join(libpath, 'loader', 'loader.so'))
            self.loaderlib.start.restype = ct.c_void_p
            self.loaderlib.next.argtypes = [ct.c_void_p]
            self.loaderlib.stop.argtypes = [ct.c_void_p]
            self.loaderlib.reset.argtypes = [ct.c_void_p]
        except:
            logger.error('Unable to load loader.so. Ensure that '
                         'this file has been compiled')
        self.npix = 3 * self.inner_size * self.inner_size
        ishape = (3, self.inner_size, self.inner_size)
        self.bsz = self.be.bsz
        self.shape = ishape
        self.idx = 0
        self.nlabels = nlabels

        self.data = self.be.iobuf(self.npix, dtype=dtype)
        # View for subtracting the mean.
        self.data_view = self.data.reshape((ishape[0], -1))
        self.buffers = []
        self.labels = []
        for i in range(2):
            self.buffers.append(self.be.empty(self.data.shape, dtype=np.uint8))
            self.labels.append(self.be.iobuf(nlabels, dtype=np.int32))
        self.onehot_labels = self.be.iobuf(self.nclass, dtype=dtype)

        if self.global_mean is not None:
            self.mean = self.be.array(self.global_mean, dtype=dtype)
        else:
            # Just center uint8 values if missing global mean.
            self.mean = 127.
        self.start()
        atexit.register(self.stop)

    def configure(self, repo_dir, inner_size, do_transforms,
                  rgb, shuffle, set_name, subset_pct, macro):
        """
        Set up all dataset config options.
        """
        assert (subset_pct > 0 and subset_pct <= 100), (
            'subset_pct must be between 0 and 100')
        assert(set_name in ['train', 'validation'])
        self.set_name = set_name if set_name == 'train' else 'val'

        self.repo_dir = repo_dir
        self.inner_size = inner_size
        self.minibatch_size = self.be.bsz

        self.center = not do_transforms
        self.flip = do_transforms
        self.rgb = rgb
        self.shuffle = shuffle
        self.start_idx = 0
        self.macro = macro

        if not macro:
            self.filename = os.path.join(repo_dir, 'filelist.txt')
            if not os.path.exists(self.filename):
                raise IOError('Cannot find %s' % self.filename)
            filelist = np.genfromtxt(self.filename, dtype=str)
            self.ndata = int(len(filelist) * subset_pct / 100.)
            assert self.ndata != 0
            self.macro_start = 0
            self.nlabels = 1
            self.nclass = 1
            self.global_mean = None
            self.img_size = inner_size
            return

        # Load from repo dataset_cache:
        try:
            cache_filepath = os.path.join(repo_dir, 'dataset_cache.pkl')
            dataset_cache = load_obj(cache_filepath)
        except IOError:
            raise IOError("Cannot find '%s/dataset_cache.pkl'. Run "
                          "batch_writer to preprocess the data and create "
                          "batch files for imageset" % (repo_dir))

        # Should have following defined:
        req_attributes = ['global_mean', 'nclass', 'val_start', 'ntrain',
                          'label_names', 'train_nrec', 'img_size', 'nval',
                          'train_start', 'val_nrec', 'label_dict',
                          'batch_prefix']

        for r in req_attributes:
            if r not in dataset_cache:
                raise ValueError(
                    'Dataset cache missing required attribute %s' % (r))

        if dataset_cache['global_mean'].shape != (3, 1):
            raise ValueError('Dataset cache global mean is not in the proper '
                             'format. Run neon/util/update_dataset_cache.py '
                             'utility on %s.' % cache_filepath)

        self.__dict__.update(dataset_cache)
        self.filename = os.path.join(repo_dir, self.batch_prefix)

        self.label = 'l_id'
        if isinstance(self.nclass, dict):
            self.nclass = self.nclass[self.label]

        self.recs_available = getattr(self, self.set_name + '_nrec')
        self.macro_start = getattr(self, self.set_name + '_start')
        self.macros_available = getattr(self, 'n' + self.set_name)
        self.ndata = int(self.recs_available * subset_pct / 100.)

    @property
    def nbatches(self):
        return -((self.start_idx - self.ndata) // self.bsz)  # ceildiv

    def init_batch_provider(self):
        """
        For backward compatibility.
        """
        pass

    def exit_batch_provider(self):
        """
        For backward compatibility.
        """
        pass

    def start(self):
        """
        Launch background threads for loading the data.
        """
        DataBufferPair = (ct.POINTER(ct.c_ubyte)) * 2
        LabelBufferPair = (ct.POINTER(ct.c_int)) * 2

        class DeviceParams(ct.Structure):
            _fields_ = [('type', ct.c_int),
                        ('id', ct.c_int),
                        ('data', DataBufferPair),
                        ('labels', LabelBufferPair)]

        if self.be.device_type == 0:
            data_buffers = DataBufferPair(
                self.buffers[0].get().ctypes.data_as(ct.POINTER(ct.c_ubyte)),
                self.buffers[1].get().ctypes.data_as(ct.POINTER(ct.c_ubyte)))
            label_buffers = LabelBufferPair(
                self.labels[0].get().ctypes.data_as(ct.POINTER(ct.c_int)),
                self.labels[1].get().ctypes.data_as(ct.POINTER(ct.c_int)))
        else:
            data_buffers = DataBufferPair(
                ct.cast(int(self.buffers[0].gpudata), ct.POINTER(ct.c_ubyte)),
                ct.cast(int(self.buffers[1].gpudata), ct.POINTER(ct.c_ubyte)))
            label_buffers = LabelBufferPair(
                ct.cast(int(self.labels[0].gpudata), ct.POINTER(ct.c_int)),
                ct.cast(int(self.labels[1].gpudata), ct.POINTER(ct.c_int)))
        params = DeviceParams(self.be.device_type, self.be.device_id,
                              data_buffers, label_buffers)
        self.loader = self.loaderlib.start(ct.c_int(self.img_size),
                                           ct.c_int(self.inner_size),
                                           ct.c_bool(self.center),
                                           ct.c_bool(self.flip),
                                           ct.c_bool(self.rgb),
                                           ct.c_bool(self.shuffle),
                                           ct.c_int(self.minibatch_size),
                                           ct.c_char_p(self.filename),
                                           ct.c_int(self.macro_start),
                                           ct.c_uint(self.ndata),
                                           ct.c_int(self.nlabels),
                                           ct.c_bool(self.macro),
                                           ct.POINTER(DeviceParams)(params))
        assert self.start_idx % self.bsz == 0

    def stop(self):
        """
        Clean up and exit background threads.
        """
        self.loaderlib.stop(self.loader)

    def reset(self):
        """
        Restart data from index 0
        """
        # Reset local state
        self.idx = 0
        self.start_idx = 0
        self.loaderlib.reset(self.loader)

    def __iter__(self):
        for start in range(self.start_idx, self.ndata, self.bsz):
            end = min(start + self.bsz, self.ndata)
            if end == self.ndata:
                self.start_idx = self.bsz - (self.ndata - start)
            self.loaderlib.next(self.loader)
            # Separating these steps to avoid possible casting error
            self.data[:] = self.buffers[self.idx]
            self.data_view[:] = self.data_view - self.mean

            # Expanding out the labels on device
            self.onehot_labels[:] = self.be.onehot(self.labels[self.idx],
                                                   axis=0)
            self.idx = 1 if self.idx == 0 else 0
            yield self.data, self.onehot_labels


if __name__ == '__main__':
    from timeit import default_timer
    from neon.backends import gen_backend
    from neon.util.argparser import NeonArgparser
    parser = NeonArgparser(__doc__)
    args = parser.parse_args()

    be = gen_backend(backend='gpu', rng_seed=100)
    NervanaObject.be.bsz = 128

    master = ImageLoader(repo_dir=args.data_dir, set_name='train',
                         inner_size=224, subset_pct=10)
    t0 = default_timer()
    total_time = 0

    for epoch in range(3):
        for x, t in master:
            print '****', epoch, master.start, master.idx, master.ndata
            print t.get().argmax(axis=0)[:17]
    master.stop()
