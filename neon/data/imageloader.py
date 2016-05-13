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
import logging
import os

from .dataloader import DataLoader
from .media import ImageParams

logger = logging.getLogger(__name__)


class ImageLoader(DataLoader):
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
                 nlabels=1, macro=True,
                 contrast_range=(100, 100), aspect_ratio=0):
        if not rgb:
            raise ValueError('Non-RGB images are currently not supported')
        assert macro is True
        if do_transforms:
            assert (aspect_ratio == 0 or aspect_ratio > 100), (
                'bad value for aspect_ratio augmentation')
        if type(scale_range) == int:
            scale_min = scale_max = scale_range
        else:
            scale_min, scale_max = scale_range

        self.repo_dir = repo_dir
        shape = dict(channel_count=3, height=inner_size, width=inner_size)
        media_params = ImageParams(center=not do_transforms, scale_min=scale_min,
                                   scale_max=scale_max,
                                   contrast_min=contrast_range[0],
                                   contrast_max=contrast_range[1],
                                   aspect_ratio=aspect_ratio,
                                   **shape)
        self.configure(repo_dir, set_name, subset_pct)
        super(ImageLoader, self).__init__(set_name=self.set_name,
                                          repo_dir=repo_dir,
                                          media_params=media_params,
                                          target_size=1, reshuffle=shuffle,
                                          nclasses=self.nclass,
                                          subset_percent=subset_pct)

    def configure(self, repo_dir, set_name, subset_pct):
        """
        Set up all dataset config options.

        Arguments:
            repo_dir (str): repository directory.
            set_name (str): One of "train" or "validation".
            subset_pct (int): Percentage of dataset to use.
        """
        assert (subset_pct > 0 and subset_pct <= 100), ('subset_pct must be between 0 and 100')
        assert(set_name in ['train', 'validation'])
        self.set_name = set_name if set_name == 'train' else 'val'

        self.archive_prefix = 'macrobatch_'
        # Load from repo dataset_cache:
        cache_filepath = os.path.join(repo_dir, self.archive_prefix + 'meta')
        try:
            dataset_cache = dict()
            with open(cache_filepath, 'r') as f:
                for line in f:
                    (k, v) = line.split()
                    dataset_cache[k] = float(v) if k.endswith('mean') else int(v)
        except IOError:
            raise IOError("Cannot find '%s'. Run batch_writer to preprocess the "
                          "data and create batch files for imageset" % (cache_filepath))

        # Should have following defined:
        req_attributes = ['nclass', 'val_start', 'train_start', 'train_nrec', 'val_nrec']

        for r in req_attributes:
            if r not in dataset_cache:
                raise ValueError('Dataset cache missing required attribute %s' % (r))

        self.__dict__.update(dataset_cache)

        self.label = 'l_id'
        if isinstance(self.nclass, dict):
            self.nclass = self.nclass[self.label]

        self.recs_available = getattr(self, self.set_name + '_nrec')
        self.ndata = int(self.recs_available * subset_pct / 100.)

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
        self.item_count.value = self.ndata
        self.archive_dir = self.repo_dir
        self.archive_prefix = 'macrobatch_'
        self.macro_start = getattr(self, self.set_name + '_start')
        super(ImageLoader, self).start()
