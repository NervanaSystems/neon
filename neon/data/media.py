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
import numpy as np
import ctypes as ct


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
    _fields_ = [('channel_count', ct.c_int),
                ('height', ct.c_int),
                ('width', ct.c_int)]
    _defaults_ = {}

    def __init__(self, **kwargs):
        for key in kwargs:
            if not hasattr(self, (key)):
                raise ValueError('Unknown argument %s' % key)
        super(AudioParams, self).__init__(mtype=MediaType.audio, **kwargs)

    def get_shape(self):
        return (self.channel_count, self.height, self.width)
