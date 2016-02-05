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
from media import MediaParams
from indexer import Indexer

logger = logging.getLogger(__name__)


BufferPair = (ct.c_void_p) * 2


class DeviceParams(ct.Structure):
    _fields_ = [('type', ct.c_int),
                ('id', ct.c_int),
                ('data', BufferPair),
                ('targets', BufferPair)]


class DataLoader(NervanaObject):
    """
    Encapsulates the data loader library and exposes an API to iterate over
    minibatches of generic data.
    """

    def __init__(self, repo_dir, shuffle, media_params,
                 datum_size, target_size,
                 datum_dtype=np.float32, target_dtype=np.float32,
                 onehot=False, nclasses=None):
        if not os.path.exists(repo_dir):
            raise IOError('Directory not found: %s' % repo_dir)
        if onehot is True and nclasses is None:
            raise ValueError('nclasses must be specified for one-hot labels')
        self.repo_dir = repo_dir
        self.item_count = ct.c_int(0)
        self.bsz = self.be.bsz
        self.buffer_id = 0
        self.start_idx = 0
        self.shuffle = shuffle
        self.media_params = media_params
        self.shape = media_params.get_shape()
        self.datum_size = datum_size
        self.target_size = target_size
        self.datum_dtype = datum_dtype
        self.target_dtype = target_dtype
        self.onehot = onehot
        self.nclasses = nclasses
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

        def double_buf(dim0, dtype):
            return [self.be.iobuf(dim0=dim0, dtype=dtype) for _ in range(2)]

        def host_convert(buffers, idx):
            return buffers[idx].get().ctypes.data_as(ct.c_void_p)

        def device_convert(buffers, idx):
            return ct.cast(int(buffers[idx].gpudata), ct.c_void_p)

        def ct_convert(buffers):
            if self.be.device_type == 0:
                return BufferPair(host_convert(buffers, 0),
                                  host_convert(buffers, 1))
            return BufferPair(device_convert(buffers, 0),
                              device_convert(buffers, 1))

        self.data = double_buf(self.datum_size, self.datum_dtype)
        self.targets = double_buf(self.target_size, self.target_dtype)
        self.device_params = DeviceParams(self.be.device_type,
                                          self.be.device_id,
                                          ct_convert(self.data),
                                          ct_convert(self.targets))
        if self.onehot:
            self.onehot_labels = self.be.iobuf(self.nclasses,
                                               dtype=self.be.default_dtype)
        if self.datum_dtype == self.be.default_dtype:
            self.backend_data = None
        else:
            self.backend_data = self.be.iobuf(self.datum_size,
                                              dtype=self.be.default_dtype)

    @property
    def nbatches(self):
        return -((self.start_idx - self.ndata) // self.bsz)

    def start(self):
        """
        Launch background threads for loading the data.
        """
        # Limited to a single integer label for now.
        assert self.target_size == 1
        assert np.dtype(self.target_dtype).itemsize == 4
        indexer = Indexer(self.repo_dir)
        indexer.run()
        datum_nbytes = self.datum_size * np.dtype(self.datum_dtype).itemsize
        target_nbytes = self.target_size * np.dtype(self.target_dtype).itemsize
        self.loader = self.loaderlib.start(
            ct.byref(self.item_count), self.bsz,
            ct.c_char_p(self.repo_dir), self.shuffle,
            datum_nbytes, target_nbytes,
            ct.POINTER(MediaParams)(self.media_params),
            ct.POINTER(DeviceParams)(self.device_params))
        self.ndata = self.item_count.value
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

            if self.backend_data is None:
                data = self.data[self.buffer_id]
            else:
                # Convert data to the required precision.
                self.backend_data[:] = self.data[self.buffer_id]
                data = self.backend_data

            if self.onehot:
                # Convert labels to one-hot encoding.
                self.onehot_labels[:] = self.be.onehot(
                    self.targets[self.buffer_id], axis=0)
                targets = self.onehot_labels
            else:
                targets = self.targets[self.buffer_id]

            self.buffer_id = 1 if self.buffer_id == 0 else 0
            yield data, targets
