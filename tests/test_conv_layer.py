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
Convolution layer tests
"""
import itertools as itt
import numpy as np
import pytest
from neon import NervanaObject
from neon.layers.layer import Convolution
from neon.initializers.initializer import Uniform
from utils import allclose_with_out
try:
    from neon.backends.nervanagpu import NervanaGPU
except:
    # stub out the class
    class NervanaGPU(object):
        pass


def pytest_generate_tests(metafunc):
    np.random.seed(1)
    if metafunc.config.option.all:
        bsz_rng = [32, 64]
    else:
        bsz_rng = [128]

    if 'zeros_convargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            fs_rng = [2, 3, 5, 7]
            nofm_rng = [16, 32]
        else:
            fs_rng = [2, 5]
            nofm_rng = [16]
        fargs = itt.product(fs_rng, nofm_rng, bsz_rng)
        metafunc.parametrize('zeros_convargs', fargs)

    if 'ones_convargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            bsz_rng = [64]
            indim_rng = [16, 32]
            nifm_rng = [3, 4, 32]
            fs_rng = [2, 3]
            stride_rng = [1, 2]
            nofm_rng = [16, 32, 64]
            pad_rng = [0, 1, 2]
            fargs1 = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng,
                                 bsz_rng, stride_rng, pad_rng)
            fs_rng = [5]
            stride_rng = [1, 5]
            fargs2 = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng,
                                 bsz_rng, stride_rng, pad_rng)
            fargs = itt.chain(fargs1, fargs2)
        else:
            bsz_rng = [64]
            indim_rng = [32]
            nifm_rng = [4]
            fs_rng = [2, 5]
            nofm_rng = [16]
            stride_rng = [1, 2]
            pad_rng = [0, 1]
            fargs = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng,
                                bsz_rng, stride_rng, pad_rng)
        metafunc.parametrize('ones_convargs', fargs)

    if 'rand_convargs' in metafunc.fixturenames:
        fargs = []
        eps = np.finfo(np.float32).eps
        if metafunc.config.option.all:
            indim_rng = [16, 32]
            nifm_rng = [3, 4]
            fs_rng = [2, 3]
            nofm_rng = [16]
            rng_max_rng = [eps, eps * 10, 1.0, 100]
            wrng = [[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]
            stride_rng = [1, 2, 3]
            pad_rng = [0, 1, 2]
            fargs1 = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng, bsz_rng,
                                 stride_rng, rng_max_rng, wrng, pad_rng)
            fs_rng = [5]
            stride_rng = [1, 5]
            fargs2 = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng, bsz_rng,
                                 stride_rng, rng_max_rng, wrng, pad_rng)
            fargs = itt.chain(fargs1, fargs2)
        else:
            indim_rng = [16]
            nifm_rng = [3, 4]
            fs_rng = [2, 5]
            nofm_rng = [16]
            rng_max_rng = [2.0]
            stride_rng = [1, 2]
            wrng = [[-1.0, 1.0]]
            pad_rng = [0, 1]
            fargs = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng, bsz_rng,
                                stride_rng, rng_max_rng, wrng, pad_rng)
        metafunc.parametrize('rand_convargs', fargs)


def test_conv_zeros(backend_default, zeros_convargs, deltas_buffer):
    fshape, nofm, batch_size = zeros_convargs

    NervanaObject.be.bsz = batch_size

    # basic sanity check with 0 weights random inputs
    init_unif = Uniform(low=0.0, high=0.0)
    inshape = (32, 32, 32)
    insize = np.prod(inshape)
    neon_layer = Convolution(fshape=(fshape, fshape, nofm),
                             strides=1, padding=0, init=init_unif)
    inp = neon_layer.be.array(np.random.random((insize, batch_size)))
    inp.lshape = inshape
    neon_layer.configure(inshape)
    neon_layer.prev_layer = True
    neon_layer.allocate()

    neon_layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    neon_layer.set_deltas(deltas_buffer)

    out = neon_layer.fprop(inp).get()
    assert np.min(out) == 0.0 and np.max(out) == 0.0

    err = np.zeros(out.shape)
    deltas = neon_layer.bprop(neon_layer.be.array(err)).get()
    assert np.min(deltas) == 0.0 and np.max(deltas) == 0.0

    dw = neon_layer.dW.get()
    assert np.min(dw) == 0.0 and np.max(dw) == 0.0
    return


def test_conv_ones(backend_default, ones_convargs, deltas_buffer):
    dtypeu = np.float32
    indim, nifm, fshape, nofm, batch_size, stride, pad = ones_convargs
    if isinstance(NervanaObject.be, NervanaGPU) and NervanaObject.be.compute_capability < (5, 0):
        if nifm % 4 != 0:
            pytest.skip(msg="C dim must be a multiple of 4 for Kepler bprop kernel")

    NervanaObject.be.bsz = batch_size

    # weights set to one
    init_unif = Uniform(low=1.0, high=1.0)

    inshape = (nifm, indim, indim)
    insize = np.prod(inshape)

    neon_layer = Convolution(fshape=(fshape, fshape, nofm),
                             strides=stride, padding=pad, init=init_unif)
    inp = neon_layer.be.array(np.ones((insize, batch_size)))
    inp.lshape = inshape
    neon_layer.configure(inshape)
    neon_layer.prev_layer = True
    neon_layer.allocate()

    neon_layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    neon_layer.set_deltas(deltas_buffer)

    # run fprop
    out = neon_layer.fprop(inp).get()

    # generate the reference layer
    ref_layer = ConvLayerRef(1,
                             batch_size,
                             identity,
                             inshape[0],
                             inshape[1:3],
                             (fshape, fshape),
                             nofm,
                             stride,
                             dtypeu,
                             padding=pad)
    # init weights to ones
    ref_layer.weights = np.ones(neon_layer.W.shape).T.astype(dtypeu)
    ref_layer.fprop(inp.get().T)
    out_exp = ref_layer.y.copy()
    assert allclose_with_out(out_exp.T, out, atol=0.0, rtol=0.0)

    # generate err array
    err = np.ones(out.shape).astype(np.float32)

    # run bprop
    neon_layer.bprop(neon_layer.be.array(err))
    dw = neon_layer.dW.get()

    # run bprop
    ref_layer.bprop(err.T.astype(dtypeu), 1.0)

    # expected output for updates is uniform matrix with
    # all elements == ofmsize*batch_size
    updates_exp = ref_layer.updates.T

    # check dw from neon layer
    assert allclose_with_out(dw, updates_exp, atol=0.0, rtol=0.0)

    # the deltas are more complicated since the matricies are not
    # uniform, going to use the reference code directly here
    # no tolerance here should be exact
    dd = np.abs(ref_layer.berror_nopad.T - neon_layer.deltas.get())
    assert np.max(dd) == 0.0

    return


def test_conv_rand(backend_default, rand_convargs, deltas_buffer):

    indim, nifm, fshape, nofm, batch_size, stride, rng_max, w_rng, pad = rand_convargs
    if isinstance(NervanaObject.be, NervanaGPU) and NervanaObject.be.compute_capability < (5, 0):
        if nifm % 4 != 0:
            pytest.skip(msg="C dim must be a multiple of 4 for Kepler bprop kernel")

    NervanaObject.be.bsz = batch_size
    inp_rng = [0.0, rng_max]
    dtypeu = np.float32
    init_unif = Uniform(low=w_rng[0], high=w_rng[1])

    inshape = (nifm, indim, indim)
    insize = np.prod(inshape)

    # generate neon conv layer
    neon_layer = Convolution(fshape=(fshape, fshape, nofm),
                             strides=stride, padding=pad, init=init_unif)

    # generate the reference layer
    ref_layer = ConvLayerRef(1,
                             batch_size,
                             identity,
                             inshape[0],
                             inshape[1:3],
                             (fshape, fshape),
                             nofm,
                             stride,
                             dtypeu,
                             padding=pad)

    # setup input in range inp_rng
    inpa = np.random.random((insize, batch_size))
    inpa *= inp_rng[1] - inp_rng[0]
    inpa += inp_rng[0]
    inpa = inpa.astype(dtypeu)
    inp = neon_layer.be.array(inpa)
    inp.lshape = inshape

    # run fprop on neon
    neon_layer.configure(inshape)
    neon_layer.prev_layer = True
    neon_layer.allocate()

    neon_layer.allocate_deltas(deltas_buffer)
    deltas_buffer.allocate_buffers()
    neon_layer.set_deltas(deltas_buffer)

    neon_out = neon_layer.fprop(inp).get()

    # pull neon weights into ref layer weights
    ref_layer.weights = neon_layer.W.get().T
    ref_layer.fprop(inpa.T)
    ref_out = np.copy(ref_layer.y)

    # estimate the numerical precision by
    # permuting order of ops in ref layer
    # fprop calculation
    ref_layer.fprop(inpa.T, permute=True)
    ref_out_perm = ref_layer.y
    atol = 4 * np.max(np.abs(ref_out - ref_out_perm))

    # compare ref and neon layer fprop outputs
    # using the empirically determined atol
    assert allclose_with_out(ref_out.T, neon_out, atol=atol, rtol=1.e-4)

    # generate random deltas array
    erra = np.random.random(neon_out.shape)
    erra *= (inp_rng[1] - inp_rng[0])
    erra += inp_rng[0]

    erra = erra.astype(dtypeu)
    err = neon_layer.be.array(erra)

    # run neon bprop
    neon_deltas = neon_layer.bprop(err).get()
    neon_dW = neon_layer.dW.get()

    # run ref code bprop
    ref_layer.bprop(erra.T, 1.0)
    ref_deltas = np.copy(ref_layer.berror_nopad.T)
    ref_dW = np.copy(ref_layer.updates)

    # estimate precision using permutation
    # of operation order on ref layer code
    ref_layer.bprop(erra.T, 1.0, permute=True)
    ref_deltas_perm = ref_layer.berror_nopad.T
    ref_dW_perm = ref_layer.updates

    atol = 4 * np.max(np.abs(ref_deltas - ref_deltas_perm))
    assert allclose_with_out(ref_deltas, neon_deltas, atol=atol, rtol=1.e-4)

    atol = 4 * np.max(np.abs(ref_dW - ref_dW_perm))
    assert allclose_with_out(ref_dW.T, neon_dW, atol=atol, rtol=1.e-4)
    return

"""
Conv check code adapted from ref-des
cnn8
"""


def identity(x):
    return x


def identity_prime(x):
    return np.ones(x.shape)


def get_prime(func):
    if func == identity:
        return identity_prime


class ConvLayerRef(object):

    def __init__(self, pos, mbs, g, nifm, ifmshape_nopad, fshape,
                 nofm, strides, dtypeu, padding=0):
        assert g == identity
        self.ifmheight, self.ifmwidth = ifmshape_nopad
        self.ifmshape_nopad = ifmshape_nopad
        self.padding = padding
        self.ifmshape = (self.ifmheight + 2 * padding, self.ifmwidth + 2 * padding)
        self.fshape = fshape

        self.stride = strides
        self.fheight, self.fwidth = fshape
        self.ofmheight = (self.ifmshape[0] - self.fheight) // strides + 1
        self.ofmwidth = (self.ifmshape[1] - self.fwidth) // strides + 1
        ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmshape[0] * self.ifmshape[1]
        self.ifmsize_nopad = self.ifmshape_nopad[0] * self.ifmshape_nopad[1]
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.nout = self.ofmsize * nofm

        self.nifm = nifm
        self.nofm = nofm
        self.fsize = nifm * self.fheight * self.fwidth
        self.weights = np.zeros((nofm, self.fsize), dtype=dtypeu)
        self.g = g
        self.gprime = get_prime(g)
        self.z = np.zeros((mbs, self.nout), dtype=dtypeu)
        self.y = np.zeros((mbs, self.nout), dtype=dtypeu)
        ofmstarts = np.array(list(range(0, (self.ofmsize * nofm), self.ofmsize)))
        self.ofmlocs = np.zeros((self.ofmsize, nofm), dtype=np.int32)
        for dst in range(self.ofmsize):
            self.ofmlocs[dst, :] = ofmstarts + dst
        # Figure out the connections with the previous layer.
        # This is a list of lists.
        self.links = []
        # sfsize = self.fheight * self.fwidth  # not used
        self.makelinks(nifm, self.ifmsize, self.ifmshape, ofmshape, fshape, strides)
        self.updates = np.zeros(self.weights.shape, dtype=dtypeu)
        self.updateshards = np.zeros((self.fheight * self.fwidth,
                                      nofm, self.fsize), dtype=dtypeu)
        self.updatebuf = np.zeros((nofm, self.fsize), dtype=dtypeu).copy()
        self.pos = pos
        if self.pos > 0:
            self.bpropbuf = np.zeros((mbs, self.fsize), dtype=dtypeu)
            self.berror = np.zeros((mbs, self.ifmsize * nifm), dtype=dtypeu)
            self.berrorshards = np.zeros((self.fheight * self.fwidth, mbs,
                                          self.ifmsize * nifm), dtype=dtypeu)

    def makelinks(self, nifm, ifmsize, ifmshape, ofmshape, fshape, strides):
        # Figure out local connections to the previous layer.
        # This function works for any number of dimensions.
        ndims = len(ifmshape)
        dimsizes = np.empty(ndims, dtype='int32')
        for dim in range(ndims):
            dimsizes[dim] = np.prod(ifmshape[dim:])
        links = []
        for ofmdim in np.ndindex(ofmshape):
            # This variable tracks the top left corner of
            # the receptive field.
            src = ofmdim[-1]
            for dim in range(-1, -ndims, -1):
                src += dimsizes[dim] * ofmdim[dim - 1]
            src *= strides
            indlist = list(range(src, src + fshape[-1]))
            for dim in range(-1, -ndims, -1):
                indarray = np.array(indlist)
                for dimind in range(1, fshape[dim - 1]):
                    indlist.extend(list(indarray + dimind * dimsizes[dim]))
            indarray = np.array(indlist)
            for ifm in range(1, nifm):
                indlist.extend(list(indarray + ifm * ifmsize))
            links.append(indlist)
        self.links = np.array(links, dtype='int32')

    def fprop(self, inputs_nopad, permute=False):
        # add padding
        if self.padding == 0:
            inputs = inputs_nopad.astype(np.float32).copy()
        else:
            shp = inputs_nopad.shape
            shp = [shp[0], self.nifm]
            shp.extend(self.ifmshape_nopad)
            in_rs = inputs_nopad.reshape(shp)
            pad = self.padding
            inputs = np.zeros((shp[0], self.nifm, self.ifmshape[0], self.ifmshape[1]))
            inputs[:, :, pad:-pad, pad:-pad] = in_rs
            inputs = inputs.reshape((shp[0], -1)).astype(np.float32).copy()
        self.inputs = inputs

        for dst in range(self.ofmsize):
            # Compute the weighted average of the receptive field
            # and store the result within the destination feature map.
            # Do this for all filters in one shot.
            rflinks = self.links[dst]
            A = inputs[:, rflinks]
            B = self.weights.T
            if permute:
                inds = np.random.permutation(A.shape[1])
                self.y[:, self.ofmlocs[dst]] = np.dot(A[:, inds], B[inds, :])
            else:
                self.y[:, self.ofmlocs[dst]] = np.dot(A, B)

    def bprop_naive(self, error, permute=False):
        for dst in range(self.ofmsize):
            rflinks = self.links[dst]
            A = error[:, self.ofmlocs[dst]]
            B = self.weights
            if permute:
                inds = np.random.permutation(A.shape[1])
                np.dot(A[:, inds], B[inds, :], self.bpropbuf)
            else:
                np.dot(A, B, self.bpropbuf)
            self.berror[:, rflinks] += self.bpropbuf

    def bprop(self, error, epsilon, permute=False):
        inputs = self.inputs
        if self.pos > 0:
            # Propagate the errors backwards.
            self.berror.fill(0.0)
            self.bprop_naive(error, permute=permute)
        bshp = [self.berror.shape[0], self.nifm, self.ifmshape[0], self.ifmshape[1]]
        pad = self.padding

        # clip the padding out for neon comparison
        if pad > 0:
            self.berror_nopad = self.berror.reshape(bshp)[:, :, pad:-pad, pad:-pad]
            self.berror_nopad = self.berror_nopad.reshape((bshp[0], -1)).copy()
        else:
            self.berror_nopad = self.berror.copy()

        self.updates.fill(0.0)
        for dst in range(self.ofmsize):
            # Accumulate weight updates, going over all
            # corresponding cells in the output feature maps.
            rflinks = self.links[dst]
            deltaslice = error[:, self.ofmlocs[dst]]

            A = deltaslice.T
            B = inputs[:, rflinks]
            if permute:
                inds = np.random.permutation(A.shape[1])
                np.dot(A[:, inds], B[inds, :], out=self.updatebuf)
            else:
                np.dot(A, B, out=self.updatebuf)
            self.updates += self.updatebuf

        # Update the weights.
        np.multiply(self.updates, epsilon, out=self.updates)
        # skip updating weights, just return the dW and deltas
        # np.subtract(self.weights, self.updates, out=self.weights)
