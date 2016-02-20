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

import ctypes as ct


class MediaType:
    unknown = -1
    image = 0
    video = 1
    audio = 2
    text = 3


class MediaParams(ct.Structure):
    _fields_ = [('mtype', ct.c_int)]


class ImageParams(MediaParams):
    _fields_ = [('channel_count', ct.c_int),
                ('height', ct.c_int),
                ('width', ct.c_int),
                ('augment', ct.c_bool),
                ('center', ct.c_bool),
                ('flip', ct.c_bool),
                ('scale_min', ct.c_int),
                ('scale_max', ct.c_int),
                ('contrast_min', ct.c_int),
                ('contrast_max', ct.c_int),
                ('rotate_min', ct.c_int),
                ('rotate_max', ct.c_int),
                ('aspect_ratio', ct.c_int)]
    _defaults_ = {'augment': False,
                  'center': True,
                  'flip': False,
                  'scale_min': 0,
                  'scale_max': 0,
                  'contrast_min': 100,
                  'contrast_max': 100,
                  'rotate_min': 0,
                  'rotate_max': 0,
                  'aspect_ratio': 0}

    def __init__(self, **kwargs):
        for key in kwargs:
            if not hasattr(self, (key)):
                raise ValueError('Unknown argument %s' % key)
        for key, value in self._defaults_.iteritems():
            setattr(self, key, value)
        super(ImageParams, self).__init__(mtype=MediaType.image, **kwargs)
        if self.augment is False:
            self.check()

    def check(self):
        for key in self._defaults_:
            if getattr(self, key) == self._defaults_[key]:
                continue
            raise ValueError('Invalid combination: augment=False, %s=%s' % (
                key, getattr(self, key)))

    def get_shape(self):
        return (self.channel_count, self.height, self.width)


class ImageIngestParams(MediaParams):
    _fields_ = [('resize_at_ingest', ct.c_bool),
                ('short_side_min', ct.c_int),
                ('short_side_max', ct.c_int)]
    _defaults_ = {'resize_at_ingest': False,
                  'short_side_min': 0,
                  'short_side_max': 0}

    def __init__(self, **kwargs):
        for key in kwargs:
            if not hasattr(self, (key)):
                raise ValueError('Unknown argument %s' % key)
        for key, value in self._defaults_.iteritems():
            setattr(self, key, value)
        super(ImageIngestParams, self).__init__(mtype=MediaType.image, **kwargs)


class VideoParams(MediaParams):
    _fields_ = [('dummy', ct.c_int)]

    def __init__(self, **kwargs):
        super(VideoParams, self).__init__(mtype=MediaType.video, **kwargs)


class AudioParams(MediaParams):
    _fields_ = [('dummy', ct.c_int)]

    def __init__(self, **kwargs):
        super(AudioParams, self).__init__(mtype=MediaType.audio, **kwargs)
