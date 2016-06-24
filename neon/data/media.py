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

"""
This must be kept in sync with loader/media.hpp.
"""

from __future__ import division
from future.utils import iteritems
import math
import logging
import numpy as np
import ctypes as ct

logger = logging.getLogger(__name__)


class MediaType(object):
    unknown = -1
    image = 0
    video = 1
    audio = 2
    text = 3


class MediaParams(ct.Structure):
    _fields_ = [('mtype', ct.c_int)]

    def get_shape(self):
        raise NotImplementedError

    def datum_size(self):
        return np.prod(self.get_shape())

    def alloc(self, loader):
        pass

    def process(self, loader, data, targets, meta):
        return data, targets


class ImageParams(MediaParams):
    """
    Used to provide image specific parameters while loading data.

    Arguments:
        channel_count (int):
            The number of channels in the image.
        height (int):
            The height to crop the image to in pixels.
        width (int):
            The width to crop the iamge to  in pixels.
        center (boolean):
            Whether to center the crop.  If this is set to False, random
            cropping is performed.
        flip (boolean):
            Whether to flip the image randomly.
        scale_min (int):
            This and the scale_max parameter specify the range to scale
            the short side of a given input image.
            If an image is 100 x 200, for example, scale_min and scale_max are
            (256, 256) and height and width are given as 224, then the image
            will be first scaled to 256 x 512, and then a random crop of size
            224 x 224 will be taken from the result.  (If center is True, the
            center crop will be taken).  If scale_min and scale_max are
            (256, 300) then the resize dimension will be randomly selected
            between 256 and 300 (unless center is True, in which case the
            lower value, 256, will always be used).  If scale_min and scale_max
            are (0, 0), then the entire image will be used, without regard to
            aspect ratio.  For the 100 x 200 image, the entire image will be
            used and rescaled into a height x width output.
        scale_max (int):
            See scale_min.
        contrast_min (int):
            This and the contrast_max parameter are percentage values
            indicating the range over which to randomly vary the contrast of
            the image.  No contrast variation is applied if
            contrast_min == contrast_max.  Defaults to (100, 100).
        contrast_max (int):
            See contrast_min.
        rotate_min (int):
            This and the rotate_max parameter specify the minimum and maximum
            angle (in degrees) to randomly rotate the input image.
        rotate_max (int):
            See rotate_max.
        aspect_ratio (int):
            If non-zero, then this will be interpreted as the percentage to
            randomly stretch the image in either horizontal or vertical
            direction by some amount between 100 and aspect_ratio.  For example,
            aspect_ratio = 133 implies that the square crop will be stretched in
            the horizontal or vertical direction (randomly determined) by some
            range between 1.0 and 1.33 (4/3).  If set to <= 100, no random
            stretching will occur.
        subtract_mean (boolean):
            Whether to subtract mean values from pixel values.
        blue_mean (int):
            The mean of blue pixel values.
        green_mean (int):
            The mean of green pixel values.
        red_mean (int):
            The mean of red pixel values.
        gray_mean (int):
            The mean of gray pixel values.
"""
    _fields_ = [('channel_count', ct.c_int),
                ('height', ct.c_int),
                ('width', ct.c_int),
                ('center', ct.c_bool),
                ('flip', ct.c_bool),
                ('scale_min', ct.c_int),
                ('scale_max', ct.c_int),
                ('contrast_min', ct.c_int),
                ('contrast_max', ct.c_int),
                ('rotate_min', ct.c_int),
                ('rotate_max', ct.c_int),
                ('aspect_ratio', ct.c_int),
                ('subtract_mean', ct.c_bool),
                ('blue_mean', ct.c_int),
                ('green_mean', ct.c_int),
                ('red_mean', ct.c_int),
                ('gray_mean', ct.c_int),
                ('color_noise_std', ct.c_float)]
    _defaults_ = {'center': True,
                  'flip': False,
                  'scale_min': 0,
                  'scale_max': 0,
                  'contrast_min': 100,
                  'contrast_max': 100,
                  'rotate_min': 0,
                  'rotate_max': 0,
                  'aspect_ratio': 0,
                  'subtract_mean': True,
                  'blue_mean': 127,
                  'green_mean': 119,
                  'red_mean': 104,
                  'gray_mean': 127,
                  'color_noise_std': 0}

    def __init__(self, **kwargs):
        for key in kwargs:
            if not hasattr(self, (key)):
                raise ValueError('Unknown argument %s' % key)
        for key, value in self._defaults_.items():
            setattr(self, key, value)
        super(ImageParams, self).__init__(mtype=MediaType.image, **kwargs)
        for key in ['color_noise_std']:
            if getattr(self, key) != self._defaults_[key]:
                raise ValueError('Argument %s must not be specified' % key)
        self.color_noise_std = (self.contrast_max - 100) / 400.

    def get_shape(self):
        return (self.channel_count, self.height, self.width)

    def process(self, loader, data, targets, meta):
        if self.subtract_mean is False:
            return data, targets
        if self.channel_count == 3:
            data_view = data.reshape((3, -1))
            data_view[0] -= self.blue_mean
            data_view[1] -= self.green_mean
            data_view[2] -= self.red_mean
        else:
            data[:] = data - self.gray_mean
        return data, targets


class ImageIngestParams(MediaParams):
    _fields_ = [('resize_at_ingest', ct.c_bool),
                ('lossy_encoding', ct.c_bool),
                ('short_side_min', ct.c_int),
                ('short_side_max', ct.c_int)]
    _defaults_ = {'resize_at_ingest': False,
                  'lossy_encoding': True,
                  'short_side_min': 0,
                  'short_side_max': 0}

    def __init__(self, **kwargs):
        for key in kwargs:
            if not hasattr(self, (key)):
                raise ValueError('Unknown argument %s' % key)
        for key, value in self._defaults_.items():
            setattr(self, key, value)
        super(ImageIngestParams, self).__init__(mtype=MediaType.image, **kwargs)


class VideoParams(MediaParams):
    """
    Used to provide video specific parameters while loading data.

    Arguments:
        frame_prams (ImageParams):
            Properties of video frames.
        frames_per_clip (int):
            The number of frames within each input video clip.
    """
    _fields_ = [('frame_params', ImageParams),
                ('frames_per_clip', ct.c_int)]

    def __init__(self, **kwargs):
        for key in kwargs:
            if not hasattr(self, (key)):
                raise ValueError('Unknown argument %s' % key)
        super(VideoParams, self).__init__(mtype=MediaType.video, **kwargs)

    def get_shape(self):
        return (self.frame_params.channel_count, self.frames_per_clip,
                self.frame_params.height, self.frame_params.width)


class AudioParams(MediaParams):
    """
    Used to provide audio specific parameters while loading data.

    Arguments:
        sampling_freq (int):
            The sampling frequency (in Hz) of input audio.
        clip_duration (int):
            Maximum duration of audio clips in milliseconds.
        frame_duration (int):
            Frame duration in milliseconds.  This defines the window over which
            FFT is computed.
        overlap_percent (int):
            Overlap percent to be used for FFT windows.
        window_func (str):
            Type of windowing.  The options are "none", "hann", "blackman",
            "hamming" and "bartlett".  Defaults to "hann".
        random_scale_percent (float):
            Randomly stretch/shrink the time dimension by this percent.
        add_noise (boolean):
            Superimpose gaussian noise.
        ctc_cost (boolean):
            Whether the CTC cost function is used.
    """

    _fields_ = [('sampling_freq', ct.c_int),
                ('clip_duration', ct.c_int),
                ('frame_duration', ct.c_int),
                ('overlap_percent', ct.c_int),
                ('window_func', ct.c_char * 16),
                ('random_scale_percent', ct.c_float),
                ('add_noise', ct.c_bool),
                ('ctc_cost', ct.c_bool),
                ('window_size', ct.c_int),
                ('overlap', ct.c_int),
                ('stride', ct.c_int),
                ('width', ct.c_int),
                ('height', ct.c_int),
                ('window_type', ct.c_int)]
    _defaults_ = {'frame_duration': 10,
                  'overlap_percent': 30,
                  'window_func': b'hann',
                  'random_scale_percent': 0.0,
                  'add_noise': False,
                  'ctc_cost': False,
                  'window_size': -1,
                  'overlap': -1,
                  'stride': -1,
                  'width': -1,
                  'height': -1,
                  'window_type': -1}
    _windows_ = {b'none': 0,
                 b'hann': 1,
                 b'blackman': 2,
                 b'hamming': 3,
                 b'bartlett': 4}

    def __init__(self, **kwargs):
        for key in kwargs:
            if not hasattr(self, (key)):
                raise ValueError('Unknown argument %s' % key)
        for key, value in iteritems(self._defaults_):
            setattr(self, key, value)
        super(AudioParams, self).__init__(mtype=MediaType.audio, **kwargs)
        for key in ['window_size', 'overlap', 'stride', 'width',
                    'height', 'window_type']:
            if getattr(self, key) != self._defaults_[key]:
                raise ValueError('Argument %s must not be specified' % key)
        if getattr(self, 'window_func') not in self._windows_:
            raise ValueError('Unknown window function: %s' %
                             getattr(self, 'window_func'))
        self.set_shape()

    def set_shape(self):
        self.channel_count = 1
        samples_per_frame = self.sampling_freq * self.frame_duration // 1000
        # Get the closest power of 2.
        log = int(math.log(samples_per_frame) / math.log(2))
        min_pow = 2 ** log
        max_pow = 2 ** (log + 1)

        if (max_pow - samples_per_frame) < (samples_per_frame - min_pow):
            self.window_size = max_pow
        else:
            self.window_size = min_pow

        real_frame_duration = 1000 * self.window_size // self.sampling_freq
        logger.info('Effective frame duration: %dms', real_frame_duration)
        assert self.overlap_percent < 100
        self.overlap = self.window_size * self.overlap_percent // 100
        self.stride = self.window_size - self.overlap
        self.width = (
            (self.clip_duration * self.sampling_freq // 1000 -
             self.window_size) // self.stride) + 1
        self.height = (self.window_size // 2) + 1
        self.window_type = self._windows_[self.window_func]

    def get_shape(self):
        return (self.channel_count, self.height, self.width)

    def alloc(self, loader):
        if self.ctc_cost is True:
            shape = (np.prod(loader.targets[0].shape), 1)
            self.packed_targets = loader.be.empty(shape, dtype=loader.target_dtype)

    def process(self, loader, data, targets, meta):
        if self.ctc_cost is True:
            # FIXME: Do the packing of targets within the context of a
            # background thread inside the loader library.
            start = 0
            target_lens = meta.get()[1]
            for i in range(loader.bsz):
                end = start + target_lens[i]
                self.packed_targets[start:end, 0] = targets[:target_lens[i], i]
                start = end
            return data, (self.packed_targets, meta[1], meta[0])
        return data, targets
