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
from multiprocessing import Process, Semaphore
from multiprocessing.sharedctypes import Array
import logging
import numpy as np
import os
from PIL import Image as PILImage

from neon import NervanaObject
from neon.util.persist import load_obj

logger = logging.getLogger(__name__)


class Image(object):
    def __init__(self):
        raise NotImplementedError()


def save_pbuf(pbuf, imshape, jpgname):
    """
    Takes a row-wise pixel buffer, reshapes it into the correct image size,
    re-orders the rgb channels and saves out to jpg
    This is purely for debugging
    """
    im = PILImage.fromarray(pbuf.reshape(imshape).transpose(1, 2, 0)[:, :, [2, 1, 0]])
    im.save(jpgname)


class Msg(object):
    """
    Data structure encapsulating a message.
    """
    def __init__(self, size):
        self.s_e = Semaphore(1)
        self.s_f = Semaphore(0)
        self.s_buf = Array(ct.c_ubyte, size)

    def send(self, func):
        self.s_e.acquire()
        self.s_buf.acquire()
        send_result = func(self.s_buf._obj)
        self.s_buf.release()
        self.s_f.release()
        return send_result

    def recv(self, func):
        self.s_f.acquire()
        self.s_buf.acquire()
        recv_result = func(self.s_buf._obj)
        self.s_buf.release()
        self.s_e.release()
        return recv_result


class ImgEndpoint(NervanaObject):
    """
    Parent class that sets up all common dataset config options that the client
    and server will share
    """

    SERVER_KILL = 255
    SERVER_RESET = 254

    def __init__(self, repo_dir, inner_size,
                 do_transforms=True, rgb=True, multiview=False,
                 set_name='train', subset_pct=100):

        assert(subset_pct > 0 and subset_pct <= 100), "subset_pct must be between 0 and 100"
        assert(set_name in ['train', 'validation'])
        self.set_name = set_name if set_name == 'train' else 'val'

        self.repo_dir = repo_dir
        self.inner_size = inner_size
        self.minibatch_size = self.be.bsz

        # Load from repo dataset_cache:
        try:
            cache_filepath = os.path.join(repo_dir, 'dataset_cache.pkl')
            dataset_cache = load_obj(cache_filepath)
        except IOError:
            raise IOError("Cannot find '%s/dataset_cache.pkl'. Run batch_writer to "
                          "preprocess the data and create batch files for imageset"
                          % (repo_dir))

        # Should have following defined:
        req_attributes = ['global_mean', 'nclass', 'val_start', 'ntrain', 'label_names',
                          'train_nrec', 'img_size', 'nval', 'train_start', 'val_nrec',
                          'label_dict', 'batch_prefix']

        for r in req_attributes:
            if r not in dataset_cache:
                raise ValueError("Dataset cache missing required attribute %s" % (r))

        global_mean = dataset_cache['global_mean']
        if global_mean is not None and global_mean.shape != (3, 1):
            raise ValueError('Dataset cache global mean is not in the proper format. Run '
                             'neon/util/update_dataset_cache.py utility on %s.' % cache_filepath)

        self.__dict__.update(dataset_cache)
        self.filename = os.path.join(repo_dir, self.batch_prefix)

        self.center = False if do_transforms else True
        self.flip = True if do_transforms else False
        self.rgb = rgb
        self.multiview = multiview
        self.label = 'l_id'
        if isinstance(self.nclass, dict):
            self.nclass = self.nclass[self.label]

        # Rough percentage
        self.recs_available = getattr(self, self.set_name + '_nrec')
        self.macro_start = getattr(self, self.set_name + '_start')
        self.macros_available = getattr(self, 'n' + self.set_name)
        self.ndata = int(self.recs_available * subset_pct / 100.)

        self.start = 0

    @property
    def nbatches(self):
        return -((self.start - self.ndata) // self.be.bsz)  # ceildiv

    def reset(self):
        pass


class ImgMaster(ImgEndpoint):
    """
    This is just a client that starts its own server process
    """
    def __init__(self, repo_dir, inner_size, do_transforms=True, rgb=True,
                 multiview=False, set_name='train', subset_pct=100, dtype=np.float32):
        super(ImgMaster, self).__init__(repo_dir, inner_size, do_transforms,
                                        rgb, multiview, set_name, subset_pct)

        # Create the communication buffers
        # We have two response buffers b/c we are double buffering

        npix = self.inner_size * self.inner_size * 3
        ishape = (3, self.inner_size, self.inner_size)
        mbsz = self.be.bsz
        self.shape = ishape
        self.response = [Msg(npix * mbsz + 4*mbsz) for i in range(2)]
        self.request = Msg(1)
        self.active_idx = 0
        self.jpg_idx = 0

        self.server_args = [repo_dir, inner_size, do_transforms, rgb,
                            multiview, set_name, subset_pct]
        self.server_args.append((self.request, self.response))

        # For debugging, we can just make a local copy
        self.local_img = np.empty((mbsz, npix), dtype=np.uint8)
        self.local_lbl = np.empty((mbsz,), dtype=np.int32)

        self.dev_X = self.be.iobuf(npix, dtype=dtype)
        self.dev_X_ms = self.dev_X.reshape((ishape[0], -1))  # view for mean subtract
        self.dev_X.lshape = ishape

        self.dev_XT = self.be.empty(self.dev_X.shape[::-1], dtype=np.uint8)
        self.dev_lbls = self.be.iobuf(1, dtype=np.int32)
        self.dev_Y = self.be.iobuf(self.nclass, dtype=dtype)

        # Crop the mean according to the inner_size
        if self.global_mean is not None:
            # switch to BGR order
            self.dev_mean = self.be.array(self.global_mean, dtype=dtype)
        else:
            self.dev_mean = 127.  # Just center uint8 values if missing global mean

        def local_copy(bufobj):
            self.local_img[:] = np.frombuffer(bufobj, dtype=np.uint8,
                                              count=npix*mbsz).reshape(mbsz, npix)
            self.local_lbl[:] = np.frombuffer(bufobj, dtype=np.int32, count=mbsz,
                                              offset=npix*mbsz)

        def device_copy(bufobj):
            self.dev_XT.set(np.frombuffer(bufobj, dtype=np.uint8,
                            count=npix*mbsz).reshape(mbsz, npix))
            self.dev_lbls.set(np.frombuffer(bufobj, dtype=np.int32, count=mbsz,
                              offset=npix*mbsz).reshape(1, mbsz))

        def jpgview():
            outname = 'tmpdir/outv2_' + str(self.jpg_idx) + '_' + str(self.local_lbl[0]) + '.jpg'
            save_pbuf(self.local_img[0], ishape, outname)

        self.local_copy = local_copy
        self.device_copy = device_copy
        self.dump_jpg = jpgview

    def send_request(self, code):
        def set_code(bufobj):
            np.frombuffer(bufobj, dtype=np.uint8, count=1)[:] = code

        self.request.send(set_code)

    def recv_response(self, callback):
        """
        callback is a function that will be executed while we have access
        to the shared block of memory
        we are switching between the response buffers modulo self.active_idx
        """
        self.response[self.active_idx].recv(callback)

    def init_batch_provider(self):
        """
        Launches the server as a separate process and sends an initial request
        """
        def server_start_cmd():
            d = ImgServer(*self.server_args)
            d.run_server()
        p = Process(target=server_start_cmd)
        p.start()
        self.active_idx = 0
        self.send_request(self.active_idx)

    def exit_batch_provider(self):
        """
        Sends kill signal to server
        """
        self.send_request(self.SERVER_KILL)

    def reset(self):
        """
        sends request to restart data from index 0
        """
        if self.start == 0:
            return
        # clear the old request
        self.recv_response(self.device_copy)
        # Reset server state
        self.send_request(self.SERVER_RESET)
        # Reset local state
        self.start = 0
        self.active_idx = 0
        self.send_request(self.active_idx)

    def next(self):
        self.recv_response(self.local_copy)
        self.active_idx = 1 if self.active_idx == 0 else 0
        self.send_request(self.active_idx)
        self.dump_jpg()

    def __iter__(self):
        for start in range(self.start, self.ndata, self.be.bsz):
            end = min(start + self.be.bsz, self.ndata)
            if end == self.ndata:
                self.start = self.be.bsz - (self.ndata - start)
            self.idx = start
            self.recv_response(self.device_copy)
            self.active_idx = 1 if self.active_idx == 0 else 0
            self.send_request(self.active_idx)

            # Separating these steps to avoid possible casting error
            self.dev_X[:] = self.dev_XT.transpose()
            self.dev_X_ms[:] = self.dev_X_ms - self.dev_mean

            # Expanding out the labels on device
            self.dev_Y[:] = self.be.onehot(self.dev_lbls, axis=0)

            yield self.dev_X, self.dev_Y


class ImgServer(ImgEndpoint):
    """
    This class interfaces with the clibrary that does the actual decoding
    """

    def __init__(self, repo_dir, inner_size, do_transforms=True, rgb=True,
                 multiview=False, set_name='train', subset_pct=100, shared_objs=None):
        super(ImgServer, self).__init__(repo_dir, inner_size, do_transforms,
                                        rgb, multiview, set_name, subset_pct)
        assert(shared_objs is not None)
        libpath = os.path.dirname(os.path.realpath(__file__))
        try:
            self._i1klib = ct.cdll.LoadLibrary(os.path.join(libpath,
                                                            'imageset_decoder.so'))
        except:
            logger.error("Unable to load imageset_decoder.so.  Ensure that "
                         "this file has been compiled")
        (self.request, self.response) = shared_objs

        self.worker = self._i1klib.create_data_worker(ct.c_int(self.img_size),
                                                      ct.c_int(self.inner_size),
                                                      ct.c_bool(self.center),
                                                      ct.c_bool(self.flip),
                                                      ct.c_bool(self.rgb),
                                                      ct.c_bool(self.multiview),
                                                      ct.c_int(self.minibatch_size),
                                                      ct.c_char_p(self.filename),
                                                      ct.c_int(self.macro_start),
                                                      ct.c_uint(self.ndata))

        def decode_minibatch(bufobj):
            self._i1klib.process_next_minibatch(self.worker, ct.POINTER(ct.c_ubyte)(bufobj))
        self.decode_minibatch = decode_minibatch

    def recv_request(self):
        def read_code(bufobj):
            return np.frombuffer(bufobj, dtype=np.uint8, count=1)[0]
        return self.request.recv(read_code)

    def send_response(self, active_idx):
        self.response[active_idx].send(self.decode_minibatch)

    def run_server(self):
        while(True):
            active_idx = self.recv_request()
            if active_idx in (0, 1):
                self.send_response(active_idx)
            elif active_idx == self.SERVER_RESET:
                self._i1klib.reset(self.worker)
            else:
                print("Server Exiting")
                break

if __name__ == "__main__":
    from timeit import default_timer
    from neon.backends import gen_backend
    from neon.util.argparser import NeonArgparser
    parser = NeonArgparser(__doc__)
    args = parser.parse_args()

    be = gen_backend(backend='gpu', rng_seed=100)
    NervanaObject.be.bsz = 128

    master = ImgMaster(repo_dir=args.data_dir, set_name='train', inner_size=224, subset_pct=10)
    master.init_batch_provider()
    t0 = default_timer()
    total_time = 0

    for epoch in range(3):
        for x, t in master:
            print "****", epoch, master.start, master.idx, master.ndata
            print t.get().argmax(axis=0)[:17]
    master.send_request(master.SERVER_KILL)
