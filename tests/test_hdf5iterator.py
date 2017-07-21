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

import os
import math
import h5py
import pytest
import logging
import tempfile
import numpy as np

from neon import NervanaObject
from neon.data import HDF5Iterator, HDF5IteratorOneHot, HDF5IteratorAutoencoder
from utils import allclose_with_out

logging.basicConfig(level=20)
logger = logging.getLogger()


def gen_hdf(fn, lshape, N, mean=None, autoencode=False, multidimout=False):
    """
    generates the hdf file with the data for testing
    Arguments:
        fn (str): filename
        lshape (tuple/list): shape of the input data
        N (int): number of data points (input/output pairs)
        mean (optional, tuple/list or ndarray): mean values
    """
    (C, H, W) = lshape

    dat = np.arange(N*C*H*W).reshape((N, C*H*W))

    h5f = h5py.File(fn, 'w')
    inp = h5f.create_dataset('input', dat.shape)
    inp[:] = dat.copy()
    inp.attrs['lshape'] = (C, H, W)
    if mean is not None:
        mean_ = h5f.create_dataset('mean', mean.shape)
        mean_[:] = mean

    if not autoencode:
        if multidimout:
            out = h5f.create_dataset('output', dat.shape)
            out[:] = dat[:, ::-1].copy()
        else:
            out = h5f.create_dataset('output', (N, 1), dtype='i8')
            out[:, 0] = np.arange(N)
            out.attrs['nclass'] = N

    h5f.close()
    return


# fixture to generate an HDF5 file for test of mean subtraction
@pytest.fixture(scope="module", params=['chan_mean', 'full_mean'])
def meansubhdf(request):
    bsz = 128  # assumes default backend is using 128 batch size
    N = 2*bsz
    # generate an empty temp file
    (fid, fn_) = tempfile.mkstemp(suffix='.h5', prefix='tempdata')
    os.close(fid)
    lshape = (3, 2, 2)
    if request.param == 'chan_mean':
        mean = (np.arange(lshape[0]) + 1.1)*2
    else:
        mean = np.zeros(lshape).reshape((lshape[0], -1))
        for ind in range(lshape[0]):
            mean[ind, :] = (ind+1.1)*2
    gen_hdf(fn_, lshape, N, mean=mean)

    def cleanup():
        os.remove(fn_)
    request.addfinalizer(cleanup)
    return (fn_, request.param)


# fixture to generate an HDF5 file imageclassification type dataset
@pytest.fixture(scope="module", params=[0, 1, 10])
def hdf5datafile(request):
    bsz = 128  # assumes default backend is using 128 batch size
    N = 2*bsz + request.param
    # generate an empty temp file
    (fid, fn_) = tempfile.mkstemp(suffix='.h5', prefix='tempdata')
    os.close(fid)

    lshape = (3, 2, 2)
    gen_hdf(fn_, lshape, N)

    def cleanup():
        os.remove(fn_)
    request.addfinalizer(cleanup)
    return fn_


# fixture for output data that is not single dimension
@pytest.fixture(scope="module", params=[0, 1, 10])
def multidimout(request):
    bsz = 128  # assumes default backend is using 128 batch size
    N = 2*bsz + request.param
    # generate an empty temp file
    (fid, fn_) = tempfile.mkstemp(suffix='.h5', prefix='tempdata')
    os.close(fid)

    lshape = (3, 2, 2)
    gen_hdf(fn_, lshape, N, multidimout=True)

    def cleanup():
        os.remove(fn_)
    request.addfinalizer(cleanup)
    return fn_


# fixture to generate an HDF5 file for test of autoencoder type dataset
@pytest.fixture(scope="module")
def hdf5datafile_ae(request):
    bsz = 128  # assumes default backend is using 128 batch size
    N = 2*bsz + 1
    # generate an empty temp file
    (fid, fn_) = tempfile.mkstemp(suffix='.h5', prefix='tempdata')
    os.close(fid)

    lshape = (3, 2, 2)
    gen_hdf(fn_, lshape, N, autoencode=True)

    def cleanup():
        os.remove(fn_)
    request.addfinalizer(cleanup)
    return fn_


@pytest.mark.parametrize("onehot", [False, True])
def test_h5iterator(backend_default, hdf5datafile, onehot):
    NervanaObject.be.bsz = 128
    bsz = 128

    fn = hdf5datafile
    if onehot:
        datit = HDF5IteratorOneHot(fn)
    else:
        datit = HDF5Iterator(fn)
    cnt_image = 0
    cnt_target = 0
    max_len = datit.ndata
    mb_cnt = 0
    MAX_CNT = max_len*datit.inp.shape[1]
    for (x, t) in datit:
        x_ = x.get()
        t_ = t.get()
        assert x_.shape[1] == t_.shape[1]
        assert not np.all(x_ == 0)
        assert not np.all(t_ == 0)

        x_ = x_.T.flatten()
        x_exp = (np.arange(len(x_)) + cnt_image) % MAX_CNT
        assert np.all(x_ == x_exp)
        cnt_image += len(x_)

        if onehot:
            t_ = np.argmax(t_, axis=0).flatten()
        else:
            t_ = t_.flatten()
        t_exp = (np.arange(len(t_)) + cnt_target) % max_len
        assert np.all(t_ == t_exp)
        cnt_target += len(t_)

        mb_cnt += 1
    assert mb_cnt == int(math.ceil(datit.inp.shape[0] / float(bsz)))
    datit.cleanup()


def test_multidimout(backend_default, multidimout):
    NervanaObject.be.bsz = 128

    fn = multidimout
    datit = HDF5Iterator(fn)

    max_len = datit.ndata
    MAX_CNT = max_len*datit.inp.shape[1]
    for (x, t) in datit:
        x_ = x.get()
        t_ = t.get()
        assert x_.shape == t_.shape

        x_ = x_.T % MAX_CNT
        t_ = t_.T % MAX_CNT
        assert np.all(x_ == t_[:, ::-1])

    datit.cleanup()


def test_autoencoder(backend_default, hdf5datafile_ae):
    NervanaObject.be.bsz = 128
    datit = HDF5IteratorAutoencoder(hdf5datafile_ae)
    for x, t in datit:
        assert np.all(x.get() == t.get())
    datit.cleanup()


def test_hdf5meansubtract(backend_default, meansubhdf):
    NervanaObject.be.bsz = 128
    bsz = 128

    datit = HDF5Iterator(meansubhdf[0])
    datit.allocate()
    typ = meansubhdf[1]
    mn = datit.mean.get()
    assert typ in ['chan_mean', 'full_mean']

    cnt_image = 0
    max_len = datit.ndata
    MAX_CNT = max_len*datit.inp.shape[1]
    for x, t in datit:
        x_ = x.get().flatten()
        x_exp = (np.arange(len(x_)) + cnt_image) % MAX_CNT
        x_exp = x_exp.reshape((-1, np.prod(datit.lshape))).T
        if typ == 'chan_mean':
            x_exp = x_exp.reshape((datit.lshape[0], -1)) - mn
        elif typ == 'full_mean':
            x_exp = x_exp.reshape((-1, bsz)) - mn
        x_exp = x_exp.flatten()
        assert allclose_with_out(x_, x_exp, atol=0.0, rtol=1.0e-7)
        cnt_image += len(x_)

    datit.cleanup()


def test_reset(backend_default, hdf5datafile):
    NervanaObject.be.bsz = 128

    fn = hdf5datafile
    datit = HDF5Iterator(fn)

    for (x, t) in datit:
        break
    x_1 = x.get()
    t_1 = t.get()

    for cnt_end in range(2):
        cnt = 0
        for (x, t) in datit:
            cnt += 1
            if cnt > cnt_end:
                break
        datit.reset()
        for (x, t) in datit:
            break
        assert np.all(x.get() == x_1)
        assert np.all(t.get() == t_1)
    datit.cleanup()
