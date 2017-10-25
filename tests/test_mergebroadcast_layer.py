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
import pytest
from neon import NervanaObject, logger as neon_logger
from neon.layers import Sequential, Conv, Pooling, MergeBroadcast, Affine
from neon.initializers.initializer import Gaussian, Constant
from neon.transforms import Rectlin, Softmax
from neon.layers.container import DeltasTree

from utils import allclose_with_out

init1 = Gaussian(scale=0.01)
relu = Rectlin()
bias = Constant(0)
common = dict(activation=relu, init=init1, bias=bias)
commonp1 = dict(activation=relu, init=init1, bias=bias, padding=1)
commonp3s2 = dict(activation=relu, init=init1, bias=bias, padding=3, strides=2)
pool3s1p1 = dict(fshape=3, padding=1, strides=1)
batch_size = 64


def fshape(rs, k):
    return (rs, rs, k)


def inception(kvals, name="i"):
    (p1, p2, p3) = kvals

    branch1 = [Sequential([Conv(fshape(1, p1[0]), **common)])] if p1[0] else []
    branch2 = [Sequential([Conv(fshape(1, p2[0]), **common),
                           Conv(fshape(3, p2[1]), **commonp1)])]
    branch3 = [Sequential([Pooling(op=p3[0], **pool3s1p1)] + (
                          [Conv(fshape(1, p3[1]), **common)] if p3[1] else []))]
    partitions = branch1 + branch2 + branch3
    return [MergeBroadcast(layers=partitions, merge="depth")]


def inception_bare(ref_module, kvals, name="i"):
    (p1, p2, p3) = kvals
    branch1 = [Conv(fshape(1, p1[0]), **common)] if p1[0] else []
    branch2 = [Conv(fshape(1, p2[0]), **common), Conv(fshape(3, p2[1]), **commonp1)]
    branch3 = [Pooling(op=p3[0], **pool3s1p1)] + (
        [Conv(fshape(1, p3[1]), **common)] if p3[1] else [])

    branch1 = Sequential(branch1)
    branch2 = Sequential(branch2)
    branch3 = Sequential(branch3)

    (branch1_ref, branch2_ref, branch3_ref) = ref_module[0].layers

    if p1[0]:
        for ll, lr in zip(branch1.layers, branch1_ref.layers):
            if ll.has_params:
                ll.set_params({'params': {'W': lr.W.get(), 'weight_bias': lr.weight_bias.get()}})

    for ll, lr in zip(branch2.layers, branch2_ref.layers):
        if ll.has_params:
            ll.set_params({'params': {'W': lr.W.get(), 'weight_bias': lr.weight_bias.get()}})

    if p3[1]:
        for ll, lr in zip(branch3.layers, branch3_ref.layers):
            if ll.has_params:
                ll.set_params({'params': {'W': lr.W.get(), 'weight_bias': lr.weight_bias.get()}})

    return (branch1.layers, branch2.layers, branch3.layers)


def main_branch():
    return [Conv(fshape(7, 64), **commonp3s2),
            Pooling(fshape=3, strides=2, padding=1, op="max"),
            Conv(fshape(3, 192), **commonp1),
            Pooling(fshape=3, strides=2, padding=1, op="max")]


def top_branch():
    return [Pooling(fshape=7, strides=1, op="avg"),
            Affine(nout=100, init=init1, activation=Softmax(), bias=bias)]


@pytest.mark.hasgpu
def test_branch_model(backend_gpu):
    np.random.seed(0)
    be = NervanaObject.be
    be.bsz = 64
    main1 = main_branch()
    i1 = inception([(32,), (32, 32), ('max', 16)])
    top = top_branch()
    neon_layer = Sequential(main1 + i1 + top)

    inshape = (4, 224, 224)
    insize = np.prod(inshape)
    inpa = np.random.random((insize, batch_size))
    neon_layer.configure(inshape)
    inp = neon_layer.be.array(inpa)
    neon_layer.allocate()
    neon_logger.display(neon_layer.nested_str())
    neon_layer.layers[0].prev_layer = True

    neon_layer.allocate_deltas()

    neon_out = neon_layer.fprop(inp).get()

    # Now make the reference pathways:
    main_trunk2 = Sequential(main_branch())
    main_trunk2.configure(inshape)
    main2 = main_trunk2.layers
    main2[0].prev_layer = True
    main2[0].deltas = be.iobuf(inshape)
    (b1, b2, b3) = inception_bare(i1, [(32,), (32, 32), ('max', 16)])

    for bb in (b1, b2, b3):
        oshape = inshape
        for ll in main2 + bb:
            oshape = ll.configure(oshape)

    main1_trunk = neon_layer.layers[:6]
    for ll, lo in zip(main2, main1_trunk):
        if ll.has_params:
            ll.set_params({'params': {'W': lo.W.get(), 'weight_bias': lo.weight_bias.get()}})
        ll.allocate()

        temp_buff = DeltasTree()
        ll.allocate_deltas(temp_buff)
        temp_buff.allocate_buffers()
        ll.set_deltas(temp_buff)

    for bb in (b1, b2, b3):
        for ll in bb:
            ll.allocate()
            temp_buff = DeltasTree()
            ll.allocate_deltas(temp_buff)
            temp_buff.allocate_buffers()
            ll.set_deltas(temp_buff)

    # Create the combined output buffer
    merge_output = be.empty_like(neon_layer.layers[6].outputs)

    x = inp
    for ll in main2:
        x = ll.fprop(x)

    start = 0
    for bb in (b1, b2, b3):
        xb = x
        for ll in bb:
            xb = ll.fprop(xb)
        end = start + xb.shape[0]
        merge_output[start:end] = xb
        start = end

    x = merge_output

    top_trunk = Sequential(top).layers
    for ll in top_trunk:
        x = ll.fprop(x)

    neon_out_ref = x.get()
    assert allclose_with_out(neon_out, neon_out_ref, rtol=0)

    neon_logger.display("Beginning Back prop")
    erra = np.random.random(neon_out.shape)
    err = be.array(erra)
    for ll in reversed(neon_layer.layers[6:]):
        err = ll.bprop(err)

    neon_deltas = err.get()
    for bb, errb in zip((b1, b2, b3), neon_layer.layers[6].error_views):
        for ll in reversed(bb):
            errb = ll.bprop(errb)

    # Now sum up the deltas at the root of the branch layer and compare
    ref_deltas = be.zeros_like(b1[0].deltas)
    ref_deltas[:] = b3[0].deltas + b2[0].deltas + b1[0].deltas

    neon_ref_deltas = ref_deltas.get()

    assert allclose_with_out(neon_deltas, neon_ref_deltas, rtol=0)


@pytest.mark.hasgpu
def test_branch_model_fork(backend_gpu):
    from neon.layers import BranchNode, Tree
    np.random.seed(0)
    be = NervanaObject.be
    if be.gpu_memory_size < 6.1 * 1024 * 1024 * 1024:
        pytest.skip(msg='Test requires more than 6.1GB')
    be.bsz = 64
    bnode = BranchNode()
    i1 = inception([(32,), (32, 32), ('max', 16)])
    top1 = top_branch()
    top2 = top_branch()
    p1 = Sequential(main_branch() + [bnode, i1] + top1)
    p2 = [bnode] + top2

    alpha2 = 0.3
    neon_layer = Tree([p1, p2], alphas=[1.0, alpha2])

    inshape = (4, 224, 224)
    insize = np.prod(inshape)
    inpa = np.random.random((insize, batch_size))
    neon_layer.configure(inshape)
    inp = neon_layer.be.array(inpa)

    neon_layer.allocate()

    neon_layer.layers[0].layers[0].prev_layer = True
    neon_layer.allocate_deltas()

    neon_out_dev = neon_layer.fprop(inp)
    neon_out = [d.get() for d in neon_out_dev]

    # Now make the reference pathways:
    main_trunk2 = Sequential(main_branch())
    main_trunk2.configure(inshape)
    main2 = main_trunk2.layers
    main2[0].prev_layer = True
    main2[0].deltas = be.iobuf(inshape)

    branch2 = Sequential(top_branch())
    lbranch2 = branch2.layers
    (b1, b2, b3) = inception_bare(i1, [(32,), (32, 32), ('max', 16)])

    for bb in (b1, b2, b3, lbranch2):
        oshape = inshape
        for ll in main2 + bb:
            oshape = ll.configure(oshape)

    main1_trunk = neon_layer.layers[0].layers[:6]
    for ll, lo in zip(main2, main1_trunk):
        if ll.has_params:
            ll.set_params({'params': {'W': lo.W.get(), 'weight_bias': lo.weight_bias.get()}})
        ll.allocate()
        temp_deltas = DeltasTree()
        temp_deltas.proc_layer(ll)
        temp_deltas.allocate_buffers()
        ll.set_deltas(temp_deltas)

    for ll, lo in zip(lbranch2, neon_layer.layers[1].layers[1:]):
        if ll.has_params:
            ll.set_params({'params': {'W': lo.W.get()}})

    for bb in (b1, b2, b3, lbranch2):
        for ll in bb:
            ll.allocate()
            temp_deltas = DeltasTree()
            temp_deltas.proc_layer(ll)
            temp_deltas.allocate_buffers()
            ll.set_deltas(temp_deltas)

    # Create the combined output buffer
    merge_output = be.empty_like(neon_layer.layers[0].layers[7].outputs)

    x = inp
    for ll in main2:
        x = ll.fprop(x)
    main2_out = x

    start = 0
    for bb in (b1, b2, b3):
        xb = main2_out
        for ll in bb:
            xb = ll.fprop(xb)
        end = start + xb.shape[0]
        merge_output[start:end] = xb
        start = end

    x = merge_output

    top_trunk = Sequential(top1).layers
    for ll in top_trunk:
        x = ll.fprop(x)

    neon_out_ref = x.get()
    assert allclose_with_out(neon_out_ref, neon_out[0], rtol=0)

    # Now do second branch
    neon_out_ref2 = branch2.fprop(main2_out).get()
    assert allclose_with_out(neon_out_ref2, neon_out[1])

    neon_logger.display("Beginning Back prop")
    erra = [np.random.random(d.shape) for d in neon_out]
    err = [be.array(d) for d in erra]
    neon_layer.layers[0].layers[0].deltas = be.iobuf(inshape)
    neon_layer.bprop(err)

    bottom_neon_deltas = neon_layer.layers[0].layers[1].deltas.get()
    middle_neon_deltas = neon_layer.layers[1].layers[1].deltas.get()

    err0 = err[0]
    for ll in reversed(top_trunk):
        err0 = ll.bprop(err0)

    err1 = err[1]
    for ll in reversed(lbranch2):
        err1 = ll.bprop(err1)

    for bb, errb in zip((b1, b2, b3), neon_layer.layers[0].layers[-5].error_views):
        for ll in reversed(bb):
            errb = ll.bprop(errb)

    # Now sum up the deltas at the root of the branch layer and compare
    ref_deltas = be.zeros_like(b1[0].deltas)
    ref_deltas[:] = alpha2 * lbranch2[0].deltas
    ref_deltas[:] = ref_deltas + b3[0].deltas + b2[0].deltas + b1[0].deltas
    neon_ref_deltas = ref_deltas.get()
    assert allclose_with_out(middle_neon_deltas, neon_ref_deltas, rtol=0)

    x = ref_deltas
    main2[0].deltas = be.iobuf(inshape)

    for ll in reversed(main2):
        x = ll.bprop(x)

    bottom_neon_ref_deltas = main2[1].deltas.get()
    assert allclose_with_out(bottom_neon_deltas, bottom_neon_ref_deltas, rtol=0)


@pytest.mark.unsupported
@pytest.mark.skip(reason="Not supported for CPU")
def test_branch_model_mkl(backend_mkl):
    np.random.seed(0)
    be = NervanaObject.be
    be.bsz = 32
    main1 = main_branch()
    i1 = inception([(32,), (32, 32), ('max', 16)])
    top = top_branch()
    neon_layer = Sequential(main1 + i1 + top)

    inshape = (4, 224, 224)
    insize = np.prod(inshape)
    inpa = np.random.random((insize, batch_size))
    neon_layer.configure(inshape)
    inp = neon_layer.be.array(inpa)
    neon_layer.allocate()
    neon_logger.display(neon_layer.nested_str())
    neon_layer.layers[0].prev_layer = True

    neon_layer.allocate_deltas()

    neon_out = neon_layer.fprop(inp).get()

    # Now make the reference pathways:
    main_trunk2 = Sequential(main_branch())
    main_trunk2.configure(inshape)
    main2 = main_trunk2.layers
    main2[0].prev_layer = True
    main2[0].deltas = be.iobuf(inshape)
    (b1, b2, b3) = inception_bare(i1, [(32,), (32, 32), ('max', 16)])

    for bb in (b1, b2, b3):
        oshape = inshape
        for ll in main2 + bb:
            oshape = ll.configure(oshape)

    main1_trunk = neon_layer.layers[:8]
    for ll, lo in zip(main2, main1_trunk):
        if ll.has_params:
            ll.set_params({'params': {'W': lo.W.get()}})
        ll.allocate()

        temp_buff = DeltasTree()
        ll.allocate_deltas(temp_buff)
        temp_buff.allocate_buffers()
        ll.set_deltas(temp_buff)

    for bb in (b1, b2, b3):
        for ll in bb:
            ll.allocate()
            temp_buff = DeltasTree()
            ll.allocate_deltas(temp_buff)
            temp_buff.allocate_buffers()
            ll.set_deltas(temp_buff)

    # Create the combined output buffer
    merge_output = be.empty_like(neon_layer.layers[8].outputs)

    x = inp
    for ll in main2:
        x = ll.fprop(x)

    start = 0
    for bb in (b1, b2, b3):
        xb = x
        for ll in bb:
            xb = ll.fprop(xb)
        end = start + xb.shape[0]
        merge_output[start:end] = xb
        start = end

    x = merge_output

    top_trunk = Sequential(top).layers
    for ll in top_trunk:
        x = ll.fprop(x)

    neon_out_ref = x.get()
    assert allclose_with_out(neon_out, neon_out_ref, rtol=0)

    neon_logger.display("Beginning Back prop")
    erra = np.random.random(neon_out.shape)
    err = be.array(erra)
    for ll in reversed(neon_layer.layers[8:]):
        err = ll.bprop(err)

    neon_deltas = err.get()
    for bb, errb in zip((b1, b2, b3), neon_layer.layers[8].error_views):
        for ll in reversed(bb):
            errb = ll.bprop(errb)

    # Now sum up the deltas at the root of the branch layer and compare
    ref_deltas = be.zeros_like(b1[0].deltas)
    ref_deltas[:] = b3[0].deltas + b2[0].deltas + b1[0].deltas

    neon_ref_deltas = ref_deltas.get()

    assert allclose_with_out(neon_deltas, neon_ref_deltas, rtol=0)


@pytest.mark.unsupported
@pytest.mark.skip(reason="Not supported for CPU")
def test_branch_model_fork_mkl(backend_mkl):
    from neon.layers import BranchNode, Tree
    np.random.seed(0)
    be = NervanaObject.be
    be.bsz = 32
    bnode = BranchNode()
    i1 = inception([(32,), (32, 32), ('max', 16)])
    top1 = top_branch()
    top2 = top_branch()
    p1 = Sequential(main_branch() + [bnode, i1] + top1)
    p2 = [bnode] + top2

    alpha2 = 0.3
    neon_layer = Tree([p1, p2], alphas=[1.0, alpha2])

    inshape = (4, 224, 224)
    insize = np.prod(inshape)
    inpa = np.random.random((insize, batch_size))
    neon_layer.configure(inshape)
    inp = neon_layer.be.array(inpa)

    neon_layer.allocate()

    neon_layer.layers[0].layers[0].prev_layer = True
    neon_layer.allocate_deltas()

    neon_out_dev = neon_layer.fprop(inp)
    neon_out = [d.get() for d in neon_out_dev]

    # Now make the reference pathways:
    main_trunk2 = Sequential(main_branch())
    main_trunk2.configure(inshape)
    main2 = main_trunk2.layers
    main2[0].prev_layer = True
    main2[0].deltas = be.iobuf(inshape)

    branch2 = Sequential(top_branch())
    lbranch2 = branch2.layers
    (b1, b2, b3) = inception_bare(i1, [(32,), (32, 32), ('max', 16)])

    for bb in (b1, b2, b3, lbranch2):
        oshape = inshape
        for ll in main2 + bb:
            oshape = ll.configure(oshape)

    main1_trunk = neon_layer.layers[0].layers[:8]
    for ll, lo in zip(main2, main1_trunk):
        if ll.has_params:
            ll.set_params({'params': {'W': lo.W.get()}})
        ll.allocate()
        temp_deltas = DeltasTree()
        temp_deltas.proc_layer(ll)
        temp_deltas.allocate_buffers()
        ll.set_deltas(temp_deltas)

    for ll, lo in zip(lbranch2, neon_layer.layers[1].layers[1:]):
        if ll.has_params:
            ll.set_params({'params': {'W': lo.W.get()}})

    for bb in (b1, b2, b3, lbranch2):
        for ll in bb:
            ll.allocate()
            temp_deltas = DeltasTree()
            temp_deltas.proc_layer(ll)
            temp_deltas.allocate_buffers()
            ll.set_deltas(temp_deltas)

    # Create the combined output buffer
    merge_output = be.empty_like(neon_layer.layers[0].layers[9].outputs)

    x = inp
    for ll in main2:
        x = ll.fprop(x)
    main2_out = x

    start = 0
    for bb in (b1, b2, b3):
        xb = main2_out
        for ll in bb:
            xb = ll.fprop(xb)
        end = start + xb.shape[0]
        merge_output[start:end] = xb
        start = end

    x = merge_output

    top_trunk = Sequential(top1).layers
    for ll in top_trunk:
        x = ll.fprop(x)

    neon_out_ref = x.get()
    assert allclose_with_out(neon_out_ref, neon_out[0], rtol=0)

    # Now do second branch
    neon_out_ref2 = branch2.fprop(main2_out).get()
    assert allclose_with_out(neon_out_ref2, neon_out[1])

    neon_logger.display("Beginning Back prop")
    erra = [np.random.random(d.shape) for d in neon_out]
    err = [be.array(d) for d in erra]
    neon_layer.layers[0].layers[0].deltas = be.iobuf(inshape)
    neon_layer.bprop(err)

    bottom_neon_deltas = neon_layer.layers[0].layers[1].deltas.get()
    middle_neon_deltas = neon_layer.layers[1].layers[1].deltas.get()

    err0 = err[0]
    for ll in reversed(top_trunk):
        err0 = ll.bprop(err0)

    err1 = err[1]
    for ll in reversed(lbranch2):
        err1 = ll.bprop(err1)

    for bb, errb in zip((b1, b2, b3), neon_layer.layers[0].layers[-5].error_views):
        for ll in reversed(bb):
            errb = ll.bprop(errb)

    # Now sum up the deltas at the root of the branch layer and compare
    ref_deltas = be.zeros_like(b1[0].deltas)
    ref_deltas[:] = alpha2 * lbranch2[0].deltas
    ref_deltas[:] = ref_deltas + b3[0].deltas + b2[0].deltas + b1[0].deltas
    neon_ref_deltas = ref_deltas.get()
    assert allclose_with_out(middle_neon_deltas, neon_ref_deltas, rtol=0)

    x = ref_deltas
    main2[0].deltas = be.iobuf(inshape)

    for ll in reversed(main2):
        x = ll.bprop(x)

    bottom_neon_ref_deltas = main2[1].deltas.get()
    assert allclose_with_out(bottom_neon_deltas, bottom_neon_ref_deltas, rtol=0)


@pytest.mark.unsupported
@pytest.mark.skip(reason="Not supported for CPU")
def test_branch_model_cpu(backend_cpu64):
    np.random.seed(0)
    be = NervanaObject.be
    be.bsz = 32
    main1 = main_branch()
    i1 = inception([(32,), (32, 32), ('max', 16)])
    top = top_branch()
    neon_layer = Sequential(main1 + i1 + top)

    inshape = (4, 224, 224)
    insize = np.prod(inshape)
    inpa = np.random.random((insize, batch_size))
    neon_layer.configure(inshape)
    inp = neon_layer.be.array(inpa)
    neon_layer.allocate()
    neon_logger.display(neon_layer.nested_str())
    neon_layer.layers[0].prev_layer = True

    neon_layer.allocate_deltas()

    neon_out = neon_layer.fprop(inp).get()

    # Now make the reference pathways:
    main_trunk2 = Sequential(main_branch())
    main_trunk2.configure(inshape)
    main2 = main_trunk2.layers
    main2[0].prev_layer = True
    main2[0].deltas = be.iobuf(inshape)
    (b1, b2, b3) = inception_bare(i1, [(32,), (32, 32), ('max', 16)])

    for bb in (b1, b2, b3):
        oshape = inshape
        for ll in main2 + bb:
            oshape = ll.configure(oshape)

    main1_trunk = neon_layer.layers[:8]
    for ll, lo in zip(main2, main1_trunk):
        if ll.has_params:
            ll.set_params({'params': {'W': lo.W.get()}})
        ll.allocate()

        temp_buff = DeltasTree()
        ll.allocate_deltas(temp_buff)
        temp_buff.allocate_buffers()
        ll.set_deltas(temp_buff)

    for bb in (b1, b2, b3):
        for ll in bb:
            ll.allocate()
            temp_buff = DeltasTree()
            ll.allocate_deltas(temp_buff)
            temp_buff.allocate_buffers()
            ll.set_deltas(temp_buff)

    # Create the combined output buffer
    merge_output = be.empty_like(neon_layer.layers[8].outputs)

    x = inp
    for ll in main2:
        x = ll.fprop(x)

    start = 0
    for bb in (b1, b2, b3):
        xb = x
        for ll in bb:
            xb = ll.fprop(xb)
        end = start + xb.shape[0]
        merge_output[start:end] = xb
        start = end

    x = merge_output

    top_trunk = Sequential(top).layers
    for ll in top_trunk:
        x = ll.fprop(x)

    neon_out_ref = x.get()
    assert allclose_with_out(neon_out, neon_out_ref, rtol=0)

    neon_logger.display("Beginning Back prop")
    erra = np.random.random(neon_out.shape)
    err = be.array(erra)
    for ll in reversed(neon_layer.layers[8:]):
        err = ll.bprop(err)

    neon_deltas = err.get()
    for bb, errb in zip((b1, b2, b3), neon_layer.layers[8].error_views):
        for ll in reversed(bb):
            errb = ll.bprop(errb)

    # Now sum up the deltas at the root of the branch layer and compare
    ref_deltas = be.zeros_like(b1[0].deltas)
    ref_deltas[:] = b3[0].deltas + b2[0].deltas + b1[0].deltas

    neon_ref_deltas = ref_deltas.get()

    assert allclose_with_out(neon_deltas, neon_ref_deltas, rtol=0)


@pytest.mark.unsupported
@pytest.mark.skip(reason="Not supported for CPU")
def test_branch_model_fork_cpu(backend_cpu64):
    from neon.layers import BranchNode, Tree
    np.random.seed(0)
    be = NervanaObject.be
    be.bsz = 32
    bnode = BranchNode()
    i1 = inception([(32,), (32, 32), ('max', 16)])
    top1 = top_branch()
    top2 = top_branch()
    p1 = Sequential(main_branch() + [bnode, i1] + top1)
    p2 = [bnode] + top2

    alpha2 = 0.3
    neon_layer = Tree([p1, p2], alphas=[1.0, alpha2])

    inshape = (4, 224, 224)
    insize = np.prod(inshape)
    inpa = np.random.random((insize, batch_size))
    neon_layer.configure(inshape)
    inp = neon_layer.be.array(inpa)

    neon_layer.allocate()

    neon_layer.layers[0].layers[0].prev_layer = True
    neon_layer.allocate_deltas()

    neon_out_dev = neon_layer.fprop(inp)
    neon_out = [d.get() for d in neon_out_dev]

    # Now make the reference pathways:
    main_trunk2 = Sequential(main_branch())
    main_trunk2.configure(inshape)
    main2 = main_trunk2.layers
    main2[0].prev_layer = True
    main2[0].deltas = be.iobuf(inshape)

    branch2 = Sequential(top_branch())
    lbranch2 = branch2.layers
    (b1, b2, b3) = inception_bare(i1, [(32,), (32, 32), ('max', 16)])

    for bb in (b1, b2, b3, lbranch2):
        oshape = inshape
        for ll in main2 + bb:
            oshape = ll.configure(oshape)

    main1_trunk = neon_layer.layers[0].layers[:8]
    for ll, lo in zip(main2, main1_trunk):
        if ll.has_params:
            ll.set_params({'params': {'W': lo.W.get()}})
        ll.allocate()
        temp_deltas = DeltasTree()
        temp_deltas.proc_layer(ll)
        temp_deltas.allocate_buffers()
        ll.set_deltas(temp_deltas)

    for ll, lo in zip(lbranch2, neon_layer.layers[1].layers[1:]):
        if ll.has_params:
            ll.set_params({'params': {'W': lo.W.get()}})

    for bb in (b1, b2, b3, lbranch2):
        for ll in bb:
            ll.allocate()
            temp_deltas = DeltasTree()
            temp_deltas.proc_layer(ll)
            temp_deltas.allocate_buffers()
            ll.set_deltas(temp_deltas)

    # Create the combined output buffer
    merge_output = be.empty_like(neon_layer.layers[0].layers[9].outputs)

    x = inp
    for ll in main2:
        x = ll.fprop(x)
    main2_out = x

    start = 0
    for bb in (b1, b2, b3):
        xb = main2_out
        for ll in bb:
            xb = ll.fprop(xb)
        end = start + xb.shape[0]
        merge_output[start:end] = xb
        start = end

    x = merge_output

    top_trunk = Sequential(top1).layers
    for ll in top_trunk:
        x = ll.fprop(x)

    neon_out_ref = x.get()
    assert allclose_with_out(neon_out_ref, neon_out[0], rtol=0)

    # Now do second branch
    neon_out_ref2 = branch2.fprop(main2_out).get()
    assert allclose_with_out(neon_out_ref2, neon_out[1])

    neon_logger.display("Beginning Back prop")
    erra = [np.random.random(d.shape) for d in neon_out]
    err = [be.array(d) for d in erra]
    neon_layer.layers[0].layers[0].deltas = be.iobuf(inshape)
    neon_layer.bprop(err)

    bottom_neon_deltas = neon_layer.layers[0].layers[1].deltas.get()
    middle_neon_deltas = neon_layer.layers[1].layers[1].deltas.get()

    err0 = err[0]
    for ll in reversed(top_trunk):
        err0 = ll.bprop(err0)

    err1 = err[1]
    for ll in reversed(lbranch2):
        err1 = ll.bprop(err1)

    for bb, errb in zip((b1, b2, b3), neon_layer.layers[0].layers[-5].error_views):
        for ll in reversed(bb):
            errb = ll.bprop(errb)

    # Now sum up the deltas at the root of the branch layer and compare
    ref_deltas = be.zeros_like(b1[0].deltas)
    ref_deltas[:] = alpha2 * lbranch2[0].deltas
    ref_deltas[:] = ref_deltas + b3[0].deltas + b2[0].deltas + b1[0].deltas
    neon_ref_deltas = ref_deltas.get()
    assert allclose_with_out(middle_neon_deltas, neon_ref_deltas, rtol=0)

    x = ref_deltas
    main2[0].deltas = be.iobuf(inshape)

    for ll in reversed(main2):
        x = ll.bprop(x)

    bottom_neon_ref_deltas = main2[1].deltas.get()
    assert allclose_with_out(bottom_neon_deltas, bottom_neon_ref_deltas, rtol=0)


if __name__ == '__main__':
    test_branch_model_fork()
