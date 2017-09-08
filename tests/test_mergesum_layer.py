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
from builtins import zip
import numpy as np
from neon import NervanaObject
from neon.layers import Sequential, Conv, MergeSum, SkipNode, Activation
from neon.initializers.initializer import Gaussian, IdentityInit
from neon.transforms import Rectlin
from utils import allclose_with_out
try:
    from neon.backends.nervanamkl import NervanaMKL
except:
    # stub out the class
    class NervanaMKL(object):
        pass

init1 = Gaussian(scale=0.01)
relu = Rectlin()
batch_size = 64


def conv_params(fsize, nfm, stride=1, relu=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=init1,
                batch_norm=True)


def id_params(nfm):
    return dict(fshape=(1, 1, nfm), strides=2, padding=0, activation=None, init=IdentityInit())


def identity_skip(nfm, stride=1):
    mainpath = [Conv(**conv_params(3, nfm, stride=stride)),
                Conv(**conv_params(3, nfm, relu=False))]
    sidepath = [SkipNode() if stride == 1 else Conv(**id_params(nfm))]
    module = [MergeSum([mainpath, sidepath]),
              Activation(Rectlin())]
    return module


def projection_skip(nfm, stride=1):
    mainpath = [Conv(**conv_params(3, nfm, stride=stride)),
                Conv(**conv_params(3, nfm, relu=False))]
    sidepath = [SkipNode() if stride == 1 else Conv(**conv_params(1, nfm, stride, relu=False))]
    module = [MergeSum([mainpath, sidepath]),
              Activation(Rectlin())]
    return module


def module_factory_copy(ref_module, modfunc, nfm, stride=1, name="i"):
    mm = modfunc(nfm, stride)

    for branch_copy, branch_ref in zip(mm[0].layers, ref_module[0].layers):
        for ll, lr in zip(branch_copy.layers, branch_ref.layers):
            if ll.has_params:
                ll.set_params(lr.get_params_serialize())

    return (mm[0].layers[0].layers, mm[0].layers[1].layers)


def test_skip_noupsample(backend_default):
    be = NervanaObject.be
    be.bsz = 64
    mergesum_test_config(be, modfunc=identity_skip, use_stride=1)


def test_skip_upsample(backend_default):
    be = NervanaObject.be
    be.bsz = 64
    mergesum_test_config(be, modfunc=identity_skip, use_stride=2)


def test_proj_upsample(backend_default):
    be = NervanaObject.be
    be.bsz = 64
    mergesum_test_config(be, modfunc=projection_skip, use_stride=2)


def mergesum_test_config(be, modfunc, use_stride=1):
    l1 = Conv(**conv_params(3, 16))
    neon_layer = modfunc(16, use_stride)
    inshape = (16, 32, 32)
    insize = np.prod(inshape)
    inpa = np.random.random((insize, batch_size))

    neon_seq = Sequential([l1] + neon_layer)
    neon_seq.configure(inshape)
    inp = be.array(inpa)

    neon_seq.allocate()
    # neon_layer.layers[0].prev_layer = True

    neon_seq.allocate_deltas()

    neon_out = neon_seq.fprop(inp).get()

    # Now make the reference pathways:
    p1, p2 = module_factory_copy(neon_layer, modfunc, 16, use_stride)
    l11 = Conv(**conv_params(3, 16))
    l12 = Conv(**conv_params(3, 16))

    for ll in (l11, l12):
        for lcopy, lref in zip(ll, l1):
            if lcopy.has_params:
                lcopy.set_params(lref.get_params_serialize())

    path1 = Sequential([l11] + p1)
    path2 = Sequential([l12] + p2)
    for ll in (path1, path2):
        ll.configure(inshape)
        ll.allocate()
        ll.allocate_deltas()

    o1 = path1.fprop(inp)
    o2 = path2.fprop(inp)
    # convert mkl buffer to cpu for following cpu execution
    be.convert_data(o1, False)
    be.convert_data(o2, False)
    neon_out_ref = be.empty_like(o1)
    neon_out_ref[:] = be.maximum(o1 + o2, 0)

    # need to have bsum false for this test to be valid
    assert allclose_with_out(neon_out_ref.get(), neon_out, rtol=0)
    erra = np.random.random(neon_out.shape)
    err = be.array(erra)

    ebr = neon_seq.layers[-1].bprop(err)
    ebr = neon_seq.layers[-2].bprop(ebr)
    trunk_neon = ebr.get()

    err = be.array(erra)
    err[:] = be.greater(neon_out_ref, 0) * err

    pstart = len(l1)
    eb1 = err
    for l in reversed(path1.layers[pstart:]):
        eb1 = l.bprop(eb1)

    eb2 = err
    for l in reversed(path2.layers[pstart:]):
        eb2 = l.bprop(eb2)

    be.convert_data(eb1, False)
    be.convert_data(eb2, False)
    err_ref = be.empty_like(eb1)
    err_ref[:] = eb1 + eb2

    assert allclose_with_out(err_ref.get(), trunk_neon, rtol=0)


if __name__ == '__main__':
    test_skip_noupsample()
