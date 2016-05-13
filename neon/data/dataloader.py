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
from builtins import range
import ctypes as ct
import logging
import numpy as np
import os
import atexit

from .media import MediaParams
from .indexer import Indexer
from .dataiterator import NervanaDataIterator

logger = logging.getLogger(__name__)


BufferPair = (ct.c_void_p) * 2


class DeviceParams(ct.Structure):
    _fields_ = [('type', ct.c_int),
                ('id', ct.c_int),
                ('data', BufferPair),
                ('targets', BufferPair)]


class DataLoader(NervanaDataIterator):
    """
    Encapsulates the data loader library and exposes an API to iterate over
    minibatches of generic data.
    """

    def __init__(self, set_name, repo_dir,
                 media_params, target_size,
                 index_file=None,
                 shuffle=False, reshuffle=False,
                 datum_dtype=np.uint8, target_dtype=np.int32,
                 onehot=True, nclasses=None, subset_percent=100,
                 ingest_params=None):
        if onehot is True and nclasses is None:
            raise ValueError('nclasses must be specified for one-hot labels')
        self.set_name = set_name
        repo_dir = os.path.expandvars(os.path.expanduser(repo_dir))
        if not os.path.exists(repo_dir):
            raise IOError('Directory not found: %s' % repo_dir)
        self.macro_start = 0
        self.repo_dir = repo_dir
        parent_dir = os.path.split(repo_dir)[0]
        self.archive_prefix = 'archive-'
        self.archive_dir = os.path.join(parent_dir, set_name + '-ingested')
        self.item_count = ct.c_int(0)
        self.bsz = self.be.bsz
        self.buffer_id = 0
        self.start_idx = 0
        self.media_params = media_params
        self.shape = media_params.get_shape()
        self.datum_size = media_params.datum_size()
        self.target_size = target_size
        if index_file is None:
            self.index_file = set_name + '-index.csv'
        else:
            self.index_file = index_file
        if not os.path.isabs(self.index_file):
            self.index_file = os.path.join(parent_dir, self.index_file)
        self.meta_file = os.path.join(parent_dir, set_name + '-metadata.csv')
        self.shuffle = shuffle
        self.reshuffle = reshuffle
        self.datum_dtype = datum_dtype
        self.target_dtype = target_dtype
        self.onehot = onehot
        self.nclasses = nclasses
        self.subset_percent = int(subset_percent)
        self.ingest_params = ingest_params
        self.load_library()
        self.alloc()
        self.start()
        atexit.register(self.stop)

    def load_library(self):
        path = os.path.dirname(os.path.realpath(__file__))
        libpath = os.path.join(path, os.pardir, os.pardir,
                               'loader', 'bin', 'loader.so')
        self.loaderlib = ct.cdll.LoadLibrary(libpath)
        self.loaderlib.start.restype = ct.c_void_p
        self.loaderlib.next.argtypes = [ct.c_void_p]
        self.loaderlib.stop.argtypes = [ct.c_void_p]
        self.loaderlib.reset.argtypes = [ct.c_void_p]

    def alloc(self):

        def alloc_bufs(dim0, dtype):
            return [self.be.iobuf(dim0=dim0, dtype=dtype) for _ in range(2)]

        def ct_cast(buffers, idx):
            return ct.cast(int(buffers[idx].raw()), ct.c_void_p)

        def cast_bufs(buffers):
            return BufferPair(ct_cast(buffers, 0), ct_cast(buffers, 1))

        self.data = alloc_bufs(self.datum_size, self.datum_dtype)
        self.targets = alloc_bufs(self.target_size, self.target_dtype)
        self.device_params = DeviceParams(self.be.device_type,
                                          self.be.device_id,
                                          cast_bufs(self.data),
                                          cast_bufs(self.targets))
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
        if not os.path.exists(self.archive_dir):
            logger.warning('%s not found. Triggering data ingest...' % self.archive_dir)
            os.makedirs(self.archive_dir)
        if self.item_count.value == 0:
            indexer = Indexer(self.repo_dir, self.index_file)
            indexer.run()
        datum_nbytes = self.datum_size * np.dtype(self.datum_dtype).itemsize
        target_nbytes = self.target_size * np.dtype(self.target_dtype).itemsize
        if self.ingest_params is None:
            ingest_params = ct.POINTER(MediaParams)()
        else:
            ingest_params = ct.POINTER(MediaParams)(self.ingest_params)
        self.loader = self.loaderlib.start(
            ct.byref(self.item_count), self.bsz,
            ct.c_char_p(self.repo_dir.encode()),
            ct.c_char_p(self.archive_dir.encode()),
            ct.c_char_p(self.index_file.encode()),
            ct.c_char_p(self.meta_file.encode()),
            ct.c_char_p(self.archive_prefix.encode()),
            self.shuffle, self.reshuffle,
            self.macro_start,
            ct.c_int(datum_nbytes), ct.c_int(target_nbytes),
            self.subset_percent,
            ct.POINTER(MediaParams)(self.media_params),
            ct.POINTER(DeviceParams)(self.device_params),
            ingest_params)
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

    def next(self, start):
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
        self.media_params.process(data)

        if self.onehot:
            # Convert labels to one-hot encoding.
            self.onehot_labels[:] = self.be.onehot(
                self.targets[self.buffer_id], axis=0)
            targets = self.onehot_labels
        else:
            targets = self.targets[self.buffer_id]

        self.buffer_id = 1 if self.buffer_id == 0 else 0
        return data, targets

    def __iter__(self):
        for start in range(self.start_idx, self.ndata, self.bsz):
            yield self.next(start)
