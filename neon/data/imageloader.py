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
from neon.data.datasets import Dataset
from neon.data.dataiterator import NervanaDataIterator

logger = logging.getLogger(__name__)


class ImageLoader(NervanaDataIterator):
    """
    Encapsulates the data loader library and exposes a backward-compatible API
    to iterate over minibatches of images.

    Arguments:
        repo_dir (str): Directory to find image batches to load
        inner_size (int): Side dimension of image to return from the loader.  The spatial
                          dimensions of each image datum will be inner_size x inner_size
        scale_range (int, tuple): Scale range to scale the short side of a given input image.
                                  If an image is 100 x 200, for example, scale_range is 256,
                                  and inner_size is 224, then the image will be first scaled to
                                  256 x 512, and then a random crop of size 224 x 224 will be
                                  taken from the result.  (If do_transforms is False, the center
                                  crop will be taken).  If scale_range is a tuple like (256, 300)
                                  then the resize dimension will be randomly selected between
                                  256 and 300 (unless do_transforms is False, in which case the
                                  lower value, 256, will always be used).  If scale_range is 0,
                                  then the entire image will be used, without regard to aspect
                                  ratio.  For the 100 x 200 image, the entire image will be used
                                  and rescaled into an inner_size x inner_size output.
        do_transforms (boolean, optional): whether to apply transformations (scaling, flipping,
                                           random cropping) or not.  If False, no flipping or
                                           center cropping will be taken.  If False, the shuffle
                                           argument will also be ignored.  Defaults to True.
        rgb (boolean, optional): whether to use rgb channel input or not (for now, purely
                                 grayscale is not supported).  Defaults to True.
        shuffle (boolean, optional): whether to shuffle the order of images as they are loaded.
                                     Useful for batch normalization.  Defaults to False.
        subset_pct (float, optional): value between 0 and 100 indicating what percentage of the
                                      dataset partition to use.  Defaults to 100
        set_name (str, optional): Which dataset partition to use.  Either 'train' or 'validation'.
                                  Defaults to 'train'
        nlabels (int, optional): how many labels exist per image.  Defaults to 1.
        macro (boolean, optional): whether to use macrobatches as input.  If False, uses an input
                                   list of files from which to read images. Useful for debugging.
                                   Defaults to True.
        contrast_range (tuple, optional): specified as (contrast_min, contrast_max), which are
                                          percentage values indicating the range over which to
                                          randomly vary the contrast of the image.  No contrast
                                          variation is applied if contrast_min == contrast_max.
                                          Defaults to (100, 100).
        aspect_ratio (int, optional): if non-zero, then this will be interpreted as a pct to
                                      to randomly stretch the image in either horizontal or
                                      vertical direction by some amount between 100 and
                                      aspect_ratio.  For example, aspect_ratio = 133 implies that
                                      the square crop will be stretched in the horizontal or
                                      vertical direction (randomly determined) by some range
                                      between 1.0 and 1.33 (4/3).  If set to <= 100, or
                                      do_transforms is False, no random stretching will occur.
                                      Defaults to 0.
    """

    def __init__(self, repo_dir, inner_size, scale_range, do_transforms=True,
                 rgb=True, shuffle=False, set_name='train', subset_pct=100,
                 nlabels=1, macro=True, dtype=np.float32,
                 contrast_range=(100, 100), aspect_ratio=0):
        super(ImageLoader, self).__init__(name=set_name)
        if not rgb:
            raise ValueError('Non-RGB images are currently not supported')
        self.configure(repo_dir, inner_size, scale_range, do_transforms,
                       rgb, shuffle, set_name, subset_pct, macro,
                       contrast_range, aspect_ratio)
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
        # Find a shape that's fast for ew broadcast
        image_dim = self.data.reshape((ishape[0], -1)).shape[1]
        fast_dim = [i for i in range(1, 257) if image_dim % i == 0][-1]
        self.data_view = self.data.reshape((ishape[0], image_dim//fast_dim, fast_dim))

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

    def configure(self, repo_dir, inner_size, scale_range, do_transforms,
                  rgb, shuffle, set_name, subset_pct, macro,
                  contrast_range, aspect_ratio):
        """
        Set up all dataset config options.
        """
        assert (subset_pct > 0 and subset_pct <= 100), (
            'subset_pct must be between 0 and 100')
        assert(set_name in ['train', 'validation'])
        self.set_name = set_name if set_name == 'train' else 'val'

        self.repo_dir = repo_dir
        self.inner_size = inner_size
        if isinstance(scale_range, int):
            self.scale_range = (scale_range, scale_range)
        else:
            self.scale_range = scale_range
        self.minibatch_size = self.be.bsz

        self.center = not do_transforms
        self.flip = do_transforms
        self.contrast_range = contrast_range if do_transforms else (100, 100)
        self.aspect_ratio = aspect_ratio if do_transforms else 0
        if not do_transforms:
            self.scale_range = (self.scale_range[0], self.scale_range[0])
        if do_transforms:
            assert (self.aspect_ratio == 0 or self.aspect_ratio > 100), (
                'bad value for aspect_ratio augmentation')

        self.shuffle = shuffle if do_transforms else False

        self.rgb = rgb
        self.start_idx = 0
        self.macro = macro
        self.batch_prefix = "macrobatch_"

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
        cache_filepath = os.path.join(repo_dir, self.batch_prefix + 'meta')
        try:
            dataset_cache = dict()
            with open(cache_filepath, 'r') as f:
                for line in f:
                    (k, v) = line.split()
                    dataset_cache[k] = float(v) if k.endswith('mean') else int(v)
            rgbmean = [[dataset_cache[c + '_mean']] for c in ['B', 'G', 'R']]
            dataset_cache['global_mean'] = np.array(rgbmean, dtype=np.float32)
        except IOError:
            raise IOError("Cannot find '%s'. Run batch_writer to preprocess the "
                          "data and create batch files for imageset" % (cache_filepath))

        # Should have following defined:
        req_attributes = ['global_mean', 'nclass', 'val_start', 'train_start',
                          'train_nrec', 'val_nrec',
                          'item_max_size', 'label_size']

        for r in req_attributes:
            if r not in dataset_cache:
                raise ValueError(
                    'Dataset cache missing required attribute %s' % (r))

        self.__dict__.update(dataset_cache)
        self.filename = os.path.join(repo_dir, self.batch_prefix)

        self.label = 'l_id'
        if isinstance(self.nclass, dict):
            self.nclass = self.nclass[self.label]

        self.recs_available = getattr(self, self.set_name + '_nrec')
        self.macro_start = getattr(self, self.set_name + '_start')
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
        self.loader = self.loaderlib.start(ct.c_int(self.inner_size),
                                           ct.c_bool(self.center),
                                           ct.c_bool(self.flip),
                                           ct.c_bool(self.rgb),
                                           ct.c_int(self.scale_range[0]),
                                           ct.c_int(self.scale_range[1]),
                                           ct.c_int(self.contrast_range[0]),
                                           ct.c_int(self.contrast_range[1]),
                                           ct.c_int(0), ct.c_int(0),  # ignored rotation params
                                           ct.c_int(self.aspect_ratio),
                                           ct.c_int(self.minibatch_size),
                                           ct.c_char_p(self.filename),
                                           ct.c_int(self.macro_start),
                                           ct.c_uint(self.ndata),
                                           ct.c_int(self.nlabels),
                                           ct.c_bool(self.macro),
                                           ct.c_bool(self.shuffle),
                                           ct.c_int(self.item_max_size),
                                           ct.c_int(self.label_size),
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

            if type(self.mean) is float:
                self.data_view[:] = self.data_view - self.mean
            else:
                # hack this up for now to get decent performnace on this op
                # the real fix is 3d broadcast support in ew
                for c in range(self.data_view.shape[0]):
                    self.data_view[c] = self.data_view[c] - self.mean[c]

            # Expanding out the labels on device
            self.onehot_labels[:] = self.be.onehot(self.labels[self.idx],
                                                   axis=0)
            self.idx = 1 if self.idx == 0 else 0
            yield self.data, self.onehot_labels


class I1K(Dataset):
    def __init__(self, data_dir, inner_size=224, subset_pct=100):
        self.data_dir = data_dir
        self.inner_size = inner_size
        self.subset_pct = subset_pct

    def load_data(self):
        assert os.path.isdir(self.data_dir)

    def gen_iterators(self):
        img_set_options = dict(repo_dir=self.data_dir,
                               inner_size=self.inner_size,
                               subset_pct=self.subset_pct)
        train = ImageLoader(set_name='train', do_transforms=False, **img_set_options)
        val = ImageLoader(set_name='validation', do_transforms=False, **img_set_options)

        self.data_dict = {'train': train,
                          'valid': val}
        return self.data_dict


if __name__ == '__main__':
    from timeit import default_timer
    from neon.backends import gen_backend
    from neon.util.argparser import NeonArgparser
    parser = NeonArgparser(__doc__)
    args = parser.parse_args()

    be = gen_backend(backend='gpu', rng_seed=100)
    NervanaObject.be.bsz = 128

    master = ImageLoader(repo_dir=args.data_dir, set_name='train', scale_range=256,
                         inner_size=224, subset_pct=10)
    t0 = default_timer()
    total_time = 0

    for epoch in range(3):
        for x, t in master:
            print '****', epoch, master.start, master.idx, master.ndata
            print t.get().argmax(axis=0)[:17]
    master.stop()
