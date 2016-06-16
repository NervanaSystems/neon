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
"""
Pooling layer tests
"""
import itertools as itt
import numpy as np
from neon import NervanaObject
from neon.layers.layer import Pooling
from utils import allclose_with_out


def pytest_generate_tests(metafunc):
    np.random.seed(1)
    if metafunc.config.option.all:
        bsz_rng = [32, 64]
    else:
        bsz_rng = [128]

    if 'poolargs' in metafunc.fixturenames:
        # to check padding, do not need large input
        fargs = []
        if metafunc.config.option.all:
            fs_rng = [2, 3, 5]
            pad_rng = [0, 1]
            nifm_rng = [16, 32]
            in_sz_rng = [8, 16]
        else:
            fs_rng = [2, 4]
            pad_rng = [0, 1]
            nifm_rng = [8]
            in_sz_rng = [8]
        fargs_ = []
        for fs in fs_rng:
            stride_rng = set([1, fs // 2, fs])
            fargs_.append(itt.product(fs_rng, nifm_rng, pad_rng, stride_rng, in_sz_rng, bsz_rng))
        fargs = itt.chain(*fargs_)
        metafunc.parametrize('poolargs', fargs)


def ref_pooling(inp, inp_shape, fshape, padding, strides, be, ncheck=None):
    # given input tensor return the expected polling output for
    # certain batches
    inp_lshape = list(inp_shape)
    bsz = inp.shape[-1]
    if ncheck is None:
        check_inds = np.arange(bsz)
    elif type(ncheck) is int:
        check_inds = np.random.permutation(bsz)
        check_inds = check_inds[0:ncheck]
    else:
        check_inds = ncheck
    check_inds = np.sort(check_inds)

    inp_lshape.append(bsz)
    inpa = inp.get().reshape(inp_lshape)
    outshape = (inp_lshape[0],
                be.output_dim(inp_lshape[1], fshape[0], padding, strides[0], pooling=True),
                be.output_dim(inp_lshape[2], fshape[1], padding, strides[1], pooling=True),
                len(check_inds))

    if padding > 0:
        padded_shape = (inp_lshape[0],
                        inp_lshape[1] + 2 * padding,
                        inp_lshape[2] + 2 * padding,
                        inp_lshape[-1])
        inp_pad = np.zeros(padded_shape)
        inp_pad[:, padding:-padding, padding:-padding, :] = inpa[:, 0:, 0:, :]
    else:
        inp_pad = inpa

    out_exp = np.zeros(outshape)
    for indC in range(outshape[0]):
        for indh in range(outshape[1]):
            hrng = (indh * strides[0], indh * strides[0] + fshape[0])
            for indw in range(outshape[2]):
                wrng = (indw * strides[1], indw * strides[1] + fshape[1])
                for cnt, indb in enumerate(check_inds):
                    inp_check = inp_pad[indC, hrng[0]:hrng[1], wrng[0]:wrng[1], indb]
                    out_exp[indC, indh, indw, cnt] = np.max(inp_check)
    return (out_exp, check_inds)


def test_padding(backend_default, poolargs, deltas_buffer):
    fshape, nifm, padding, stride, in_sz, batch_size = poolargs

    NervanaObject.be.bsz = batch_size

    # basic sanity check with random inputs
    inshape = (nifm, in_sz, in_sz)
    insize = np.prod(inshape)
    neon_layer = Pooling(fshape=fshape, strides=stride, padding=padding)

    inp = neon_layer.be.array(np.random.random((insize, batch_size)))
    inp.lshape = inshape
    neon_layer.configure(inshape)
    neon_layer.prev_layer = True
    neon_layer.allocate()

    neon_layer.allocate_deltas(deltas_buffer)
    neon_layer.set_deltas(deltas_buffer)
    neon_layer.argmax = neon_layer.be.iobuf(neon_layer.outputs.shape[0], dtype=np.uint8)

    out = neon_layer.fprop(inp).get()

    ncheck = [0, batch_size // 2, batch_size - 1]

    (out_exp, check_inds) = ref_pooling(inp, inp.lshape,
                                        (fshape, fshape),
                                        padding,
                                        (stride, stride),
                                        neon_layer.be,
                                        ncheck=ncheck)

    out_shape = list(out_exp.shape[0:3])
    out_shape.append(batch_size)
    outa = out.reshape(out_shape)

    assert allclose_with_out(out_exp, outa[:, :, :, check_inds], atol=0.0, rtol=0.0)
