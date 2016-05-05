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

    def process(self, data):
        pass


class ImageParams(MediaParams):
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
                ('gray_mean', ct.c_int)]
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
                  'gray_mean': 127}

    def __init__(self, **kwargs):
        for key in kwargs:
            if not hasattr(self, (key)):
                raise ValueError('Unknown argument %s' % key)
        for key, value in self._defaults_.items():
            setattr(self, key, value)
        super(ImageParams, self).__init__(mtype=MediaType.image, **kwargs)

    def get_shape(self):
        return (self.channel_count, self.height, self.width)

    def process(self, data):
        if self.subtract_mean is False:
            return
        if self.channel_count == 3:
            data_view = data.reshape((3, -1))
            data_view[0] -= self.blue_mean
            data_view[1] -= self.green_mean
            data_view[2] -= self.red_mean
        else:
            data[:] = data - self.gray_mean


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
    _fields_ = [('sampling_freq', ct.c_int),
                # Whether input must be resampled
                ('resample', ct.c_bool),
                # Maximum duration in milliseconds
                ('clip_duration', ct.c_int),
                # Frame duration in milliseconds
                ('frame_duration', ct.c_int),
                ('overlap_percent', ct.c_int),
                # Type of windowing
                ('window_func', ct.c_char * 16),
                # Used to scale the X dimension of the spectrogram
                ('time_scale_factor', ct.c_float),
                # Used to scale the Y dimension of the spectrogram
                ('freq_scale_factor', ct.c_float),
                # The rest are automatically computed
                ('window_size', ct.c_int),
                ('overlap', ct.c_int),
                ('stride', ct.c_int),
                ('time_steps', ct.c_int),
                ('num_freqs', ct.c_int),
                ('window_type', ct.c_int)]
    _defaults_ = {'resample': False,
                  'frame_duration': 10,
                  'overlap_percent': 30,
                  'window_func': 'hann',
                  'time_scale_factor': 1.0,
                  'freq_scale_factor': 1.0,
                  'window_size': -1,
                  'overlap': -1,
                  'stride': -1,
                  'time_steps': -1,
                  'num_freqs': -1,
                  'window_type': -1}
    _windows_ = {'none': 0,
                 'hann': 1,
                 'blackman': 2,
                 'hamming': 3,
                 'bartlett': 4}

    def __init__(self, **kwargs):
        for key in kwargs:
            if not hasattr(self, (key)):
                raise ValueError('Unknown argument %s' % key)
        for key, value in self._defaults_.iteritems():
            setattr(self, key, value)
        super(AudioParams, self).__init__(mtype=MediaType.audio, **kwargs)
        for key in ['window_size', 'overlap', 'stride', 'time_steps',
                    'num_freqs', 'window_type']:
            if getattr(self, key) != self._defaults_[key]:
                raise ValueError('Argument %s must not be specified' % key)
        if getattr(self, 'window_func') not in self._windows_.keys():
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
        self.time_steps = (
            (self.clip_duration * self.sampling_freq // 1000 -
             self.window_size) // self.stride) + 1
        self.num_freqs = (self.window_size // 2) + 1
        self.width = int(self.time_steps * self.time_scale_factor)
        self.height = int(((self.window_size // 2) + 1) * self.freq_scale_factor)
        self.window_type = self._windows_[self.window_func]

    def get_shape(self):
        return (self.channel_count, self.height, self.width)
