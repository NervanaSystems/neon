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

import ctypes as ct
import logging
import numpy as np
import os
import atexit

from neon import NervanaObject

logger = logging.getLogger(__name__)


class MediaType:
    image = 0
    video = 1
    audio = 2
    text = 3


class MediaParams(ct.Structure):
    _fields_ = [('mtype', ct.c_int)]


class ImageParams(MediaParams):
    def __init__(self, **kwargs):
        super(ImageParams, self).__init__(mtype=MediaType.image,**kwargs)

    _fields_ = [('inner_size', ct.c_int),
                ('center', ct.c_bool),
                ('flip', ct.c_bool),
                ('rgb', ct.c_bool),
                ('scale_min', ct.c_int),
                ('scale_max', ct.c_int),
                ('contrast_min', ct.c_int),
                ('contrast_max', ct.c_int),
                ('rotate_min', ct.c_int),
                ('rotate_max', ct.c_int)]


class VideoParams(MediaParams):
    def __init__(self, **kwargs):
        super(VideoParams, self).__init__(mtype=MediaType.video, **kwargs)

    _fields_ = [('dummy', ct.c_int)]


class AudioParams(MediaParams):
    def __init__(self, **kwargs):
        super(AudioParams, self).__init__(mtype=MediaType.audio, **kwargs)

    _fields_ = [('dummy', ct.c_int)]


DataBufferPair = (ct.c_void_p) * 2
TargetsBufferPair = (ct.c_void_p) * 2
class DeviceParams(ct.Structure):
    _fields_ = [('type', ct.c_int),
                ('id', ct.c_int),
                ('data', DataBufferPair),
                ('targets', TargetsBufferPair)]


class DataLoader(NervanaObject):
    """
    Encapsulates the data loader library and exposes an API to iterate over
    minibatches of generic data.
    """

    def __init__(self, repo_dir, shuffle, media_params,
                 datum_size, target_size,
                 datum_dtype=np.float32, target_dtype=np.float32):
        self.bsz = self.be.bsz
        self.buffer_id = 0
        self.start_idx = 0
        self.repo_dir = repo_dir
        self.shuffle = shuffle
        self.media_params = media_params
        self.datum_size = datum_size
        self.target_size = target_size
        self.datum_dtype = datum_dtype
        self.target_dtype = target_dtype
        self.load_library()
        self.alloc()
        self.start()
        atexit.register(self.stop)

    def load_library(self):
        path = os.path.dirname(os.path.realpath(__file__))
        libpath = os.path.join(path, 'loader', 'loader.so')
        self.loaderlib = ct.cdll.LoadLibrary(libpath)
        self.loaderlib.start.restype = ct.c_void_p
        self.loaderlib.next.argtypes = [ct.c_void_p]
        self.loaderlib.stop.argtypes = [ct.c_void_p]
        self.loaderlib.reset.argtypes = [ct.c_void_p]

    def alloc(self):
        self.data = [self.be.iobuf(self.datum_size, dtype=self.datum_dtype) for i in range(2)]
        self.targets = [self.be.iobuf(self.target_size, dtype=self.target_dtype) for i in range(2)]
        if self.be.device_type == 0:
            data_buffers = [self.data[i].get().ctypes.data_as(ct.c_void_p) for i in range(2)]
            targets_buffers = [self.targets[i].get().ctypes.data_as(ct.c_void_p) for i in range(2)]
        else:
            data_buffers = [ct.cast(int(self.data[i].gpudata), ct.c_void_p) for i in range(2)]
            targets_buffers = [ct.cast(int(self.targets[i].gpudata), ct.c_void_p) for i in range(2)]
        self.device_params = DeviceParams(self.be.device_type, self.be.device_id,
                                          DataBufferPair(data_buffers[0], data_buffers[1]),
                                          TargetsBufferPair(targets_buffers[0], targets_buffers[1]))

    @property
    def nbatches(self):
        return -((self.start_idx - self.ndata) // self.bsz)

    def start(self):
        """
        Launch background threads for loading the data.
        """
        # Limited to a single integer label for now
        assert np.dtype(self.target_dtype).itemsize == 4 
        datum_nbytes = self.datum_size * np.dtype(self.datum_dtype).itemsize 
        target_nbytes = self.target_size * np.dtype(self.target_dtype).itemsize 
        self.loader = self.loaderlib.start(
            self.bsz, self.repo_dir, self.shuffle,
            datum_nbytes, target_nbytes,
            ct.POINTER(MediaParams)(self.media_params),
            ct.POINTER(DeviceParams)(self.device_params))
        if self.loader is None:
            raise RuntimeError('Failed to start data loader.')

    def stop(self):
        """
        Clean up and exit background threads.
        """
        self.loaderlib.stop(self.loader)

    def reset(self):
        """
        Restart data from index 0
        """
        self.buffer_id = 0
        self.start_idx = 0
        self.loaderlib.reset(self.loader)

    def __iter__(self):
        for start in range(self.start_idx, self.ndata, self.bsz):
            end = min(start + self.bsz, self.ndata)
            if end == self.ndata:
                self.start_idx = self.bsz - (self.ndata - start)
            self.loaderlib.next(self.loader)
            self.buffer_id = 1 if self.buffer_id == 0 else 0
            yield self.data[self.buffer_id], self.targets[self.buffer_id]
