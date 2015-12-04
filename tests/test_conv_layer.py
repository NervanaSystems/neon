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
"""
Convolution layer tests
"""
import itertools as itt
import numpy as np
from neon import NervanaObject
from neon.layers.layer import Convolution
from neon.initializers.initializer import Uniform


def pytest_generate_tests(metafunc):
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
            indim_rng = [16, 32]
            nifm_rng = [1, 2, 3]
            fs_rng = [2, 3]
            nofm_rng = [16, 32, 64]
        else:
            indim_rng = [32]
            nifm_rng = [3]
            fs_rng = [2, 5]
            nofm_rng = [16]
        fargs = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng, bsz_rng)
        metafunc.parametrize('ones_convargs', fargs)

    if 'rand_convargs' in metafunc.fixturenames:
        fargs = []
        eps = np.finfo(np.float32).eps
        if metafunc.config.option.all:
            indim_rng = [16, 32]
            nifm_rng = [1, 3]
            fs_rng = [2, 5]
            nofm_rng = [16]
            rng_max_rng = [eps, eps * 10, 1.0, 1e6, 1e10]
            wrng = [[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]
        else:
            indim_rng = [16]
            nifm_rng = [1, 3]
            fs_rng = [2, 5]
            nofm_rng = [16]
            rng_max_rng = [1.0, 10.0]
            wrng = [[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]
        fargs = itt.product(indim_rng, nifm_rng, fs_rng, nofm_rng, bsz_rng,
                            rng_max_rng, wrng)
        metafunc.parametrize('rand_convargs', fargs)


def test_conv_zeros(backend_default, zeros_convargs):
    fshape, nofm, batch_size = zeros_convargs

    NervanaObject.be.bsz = batch_size

    # basic sanity check with 0 weights random inputs
    init_unif = Uniform(low=0.0, high=0.0)
    inshape = (3, 32, 32)
    insize = np.prod(inshape)
    neon_layer = Convolution(fshape=(fshape, fshape, nofm),
                             strides=1, padding=0, init=init_unif)
    inp = neon_layer.be.array(np.random.random((insize, batch_size)))
    inp.lshape = inshape
    neon_layer.configure(inshape)
    neon_layer.prev_layer = True
    neon_layer.allocate()
    neon_layer.set_deltas([neon_layer.be.iobuf(inshape)])
    out = neon_layer.fprop(inp).get()
    assert np.min(out) == 0.0 and np.max(out) == 0.0

    err = np.zeros(out.shape)
    deltas = neon_layer.bprop(neon_layer.be.array(err)).get()
    assert np.min(deltas) == 0.0 and np.max(deltas) == 0.0

    dw = neon_layer.dW.get()
    assert np.min(dw) == 0.0 and np.max(dw) == 0.0
    return


def test_conv_ones(backend_default, ones_convargs):
    dtypeu = np.float32
    indim, nifm, fshape, nofm, batch_size = ones_convargs
    NervanaObject.be.bsz = batch_size

    # weights set to one
    init_unif = Uniform(low=1.0, high=1.0)

    inshape = (nifm, indim, indim)
    insize = np.prod(inshape)

    neon_layer = Convolution(fshape=(fshape, fshape, nofm),
                             strides=1, padding=0, init=init_unif)
    inp = neon_layer.be.array(np.ones((insize, batch_size)))
    inp.lshape = inshape
    neon_layer.configure(inshape)
    neon_layer.prev_layer = True
    neon_layer.allocate()
    neon_layer.set_deltas([neon_layer.be.iobuf(inshape)])
    # run fprop
    out = neon_layer.fprop(inp).get()
    out_exp = fshape * fshape * nifm
    assert np.min(out) == out_exp and np.max(out) == out_exp

    # generate err array
    err = np.ones(out.shape)

    # run bprop
    neon_layer.bprop(neon_layer.be.array(err)).get()
    dw = neon_layer.dW.get()

    # generate the reference layer
    ref_layer = ConvLayerRef(1,
                             batch_size,
                             identity,
                             inshape[0],
                             inshape[1:3],
                             (fshape, fshape),
                             nofm,
                             1,
                             dtypeu)

    # init weights to ones
    ref_layer.weights = np.ones(neon_layer.W.shape).T.astype(dtypeu)

    # run bprop
    ref_layer.bprop(err.T.astype(dtypeu),
                    inp.get().T.astype(dtypeu),
                    1.0)

    # expected output for updates is uniform matrix with
    # all elements == ofmsize*batch_size
    updates_exp = ref_layer.ofmsize * batch_size

    # check dw from neon layer
    assert np.max(dw) == updates_exp and np.min(dw) == updates_exp

    # the deltas are more complicated since the matricies are not
    # uniform, going to use the reference code directly here
    # no tolerence here should be exact
    dd = np.abs(ref_layer.berror.T - neon_layer.deltas.get())
    assert np.max(dd) == 0.0

    return


def test_conv_rand(backend_default, rand_convargs):
    indim, nifm, fshape, nofm, batch_size, rng_max, w_rng = rand_convargs
    NervanaObject.be.bsz = batch_size
    inp_rng = [0.0, rng_max]
    dtypeu = np.float32
    init_unif = Uniform(low=w_rng[0], high=w_rng[1])

    inshape = (nifm, indim, indim)
    insize = np.prod(inshape)

    # generate neon conv layer
    neon_layer = Convolution(fshape=(fshape, fshape, nofm),
                             strides=1, padding=0, init=init_unif)

    # generate the reference layer
    ref_layer = ConvLayerRef(1,
                             batch_size,
                             identity,
                             inshape[0],
                             inshape[1:3],
                             (fshape, fshape),
                             nofm,
                             1,
                             dtypeu)

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
    neon_layer.set_deltas([neon_layer.be.iobuf(inshape)])
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
    atol = np.max(np.abs(ref_out - ref_out_perm))
    atol += 10  # fudge factor

    # compare ref and neon layer fprop outputs
    # using the empirically determined atol
    assert (np.allclose(ref_out.T, neon_out, atol=atol, rtol=0.0),
            '%e %e' % (np.max(np.abs(ref_out.T - neon_out)), atol))

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
    ref_layer.bprop(erra.T, inpa.T, 1.0)
    ref_deltas = np.copy(ref_layer.berror.T)
    ref_dW = np.copy(ref_layer.updates)

    # estimate precision using permutation
    # of operation order on ref layer code
    ref_layer.bprop(erra.T, inpa.T, 1.0, permute=True)
    ref_deltas_perm = ref_layer.berror.T
    ref_dW_perm = ref_layer.updates

    atol = np.max(np.abs(ref_deltas - ref_deltas_perm))
    atol *= 10.0  # fudge factor
    assert (np.allclose(ref_deltas, neon_deltas, atol=atol, rtol=0.0),
            '%e %e' % (np.max(np.abs(ref_deltas - neon_deltas)), atol))

    atol = np.max(np.abs(ref_dW - ref_dW_perm))
    atol *= 10.0
    print 'atol on bprop dW = %e' % atol
    assert (np.allclose(ref_dW.T, neon_dW, atol=atol, rtol=0.0),
            '%e %e' % (np.max(np.abs(ref_dW.T - neon_dW)), atol))
    return

"""
Conv check code adapted from ref-des
cnn8 currently only using strides = 1
"""


def identity(x):
    return x


def identity_prime(x):
    return np.ones(x.shape)


def get_prime(func):
    if func == identity:
        return identity_prime


class ConvLayerRef(object):

    def __init__(self, pos, mbs, g, nifm, ifmshape, fshape, nofm, strides, dtypeu):
        assert g == identity
        self.ifmheight, self.ifmwidth = ifmshape
        self.fheight, self.fwidth = fshape
        self.ofmheight = (self.ifmheight - self.fheight) / strides + 1
        self.ofmwidth = (self.ifmwidth - self.fwidth) / strides + 1
        ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
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
        ofmstarts = np.array(range(0, (self.ofmsize * nofm), self.ofmsize))
        self.ofmlocs = np.zeros((self.ofmsize, nofm), dtype='i32')
        for dst in range(self.ofmsize):
            self.ofmlocs[dst, :] = ofmstarts + dst
        # Figure out the connections with the previous layer.
        # This is a list of lists.
        self.links = []
        # sfsize = self.fheight * self.fwidth  # not used
        self.makelinks(nifm, self.ifmsize, ifmshape, ofmshape, fshape, strides)
        self.updates = np.zeros(self.weights.shape, dtype=dtypeu)
        self.updateshards = np.zeros((self.fheight * self.fwidth,
                                      nofm, self.fsize), dtype=dtypeu)
        self.updatebuf = np.zeros((nofm, self.fsize), dtype=dtypeu)
        self.pos = pos
        if self.pos > 0:
            self.bpropbuf = np.zeros((mbs, self.fsize), dtype=dtypeu)
            self.berror = np.zeros((mbs, self.ifmsize * nifm),
                                   dtype=dtypeu)
            self.berrorshards = np.zeros((self.fheight * self.fwidth, mbs,
                                          self.ifmsize * nifm),
                                         dtype=dtypeu)

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

    def fprop(self, inputs, permute=False):
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

    def bprop(self, error, inputs, epsilon, permute=False):
        if self.pos > 0:
            # Propagate the errors backwards.
            self.berror.fill(0.0)
            self.bprop_naive(error, permute=permute)

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
