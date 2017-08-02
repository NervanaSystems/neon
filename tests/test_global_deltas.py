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

import numpy as np
import pytest

from builtins import zip
from neon import NervanaObject
from neon.initializers import Gaussian, Constant, Kaiming
from neon.layers import GeneralizedCost, Affine, Activation, Dropout
from neon.layers import Conv, Pooling, Sequential, MergeMultistream
from neon.layers import MergeSum, SkipNode, BatchNorm, MergeBroadcast, GeneralizedCostMask
from neon.layers import BranchNode, Multicost, SingleOutputTree, LSTM
from neon.layers.container import Broadcast
from neon.models import Model
from neon.transforms import Rectlin, CrossEntropyBinary, Softmax, Tanh
from neon.transforms import Logistic, CrossEntropyMulti


class SequentialModel(object):
    """
    Container for the simple sequential test model
    """
    def __init__(self):
        self.in_shape = (3, 32, 32)
        relu = Rectlin()
        init_use = Constant(0)
        conv = dict(init=init_use,
                    batch_norm=False,
                    activation=relu)
        convp1 = dict(init=init_use,
                      batch_norm=False,
                      bias=init_use,
                      activation=relu,
                      padding=1)
        convp1s2 = dict(init=init_use,
                        batch_norm=False,
                        bias=init_use,
                        padding=1, strides=2)

        layers = [Dropout(keep=.8),
                  Conv((3, 3, 96), **convp1),
                  Conv((3, 3, 96), **convp1),
                  Conv((3, 3, 96), **convp1s2),
                  Dropout(keep=.5),
                  Conv((3, 3, 192), **convp1),
                  Conv((3, 3, 192), **convp1),
                  Conv((3, 3, 192), **convp1s2),
                  Dropout(keep=.5),
                  Conv((3, 3, 192), **convp1),
                  Conv((1, 1, 192), **conv),
                  Conv((1, 1, 16), **conv),
                  Pooling(8, op="avg"),
                  Activation(Softmax())]
        self.layers = layers
        model = Model(layers=layers)
        cost = GeneralizedCost(costfunc=CrossEntropyMulti())
        model.initialize(self.in_shape, cost=cost)
        self.model = model


class ResnetModel(object):
    """
    Container for the resnet test model
    """
    def __init__(self, depth=9):
        self.depth = depth

        depth = 9
        train = (3, 32, 32)

        nfms = [2**(stage + 4) for stage in sorted(list(range(3)) * depth)]
        strides = [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

        # Now construct the network
        layers = [Conv(**self.conv_params(3, 16))]
        layers.append(self.module_s1(nfms[0], True))

        for nfm, stride in zip(nfms[1:], strides):
            res_module = self.module_s1(nfm) if stride == 1 else self.module_s2(nfm)
            layers.append(res_module)
        layers.append(BatchNorm())
        layers.append(Activation(Rectlin()))
        layers.append(Pooling('all', op='avg'))
        layers.append(Affine(10, init=Kaiming(local=False),
                             batch_norm=True, activation=Softmax()))
        self.layers = layers
        model = Model(layers=layers)
        cost = GeneralizedCost(costfunc=CrossEntropyMulti())
        model.initialize(train, cost=cost)
        self.model = model

    @staticmethod
    def conv_params(fsize, nfm, stride=1, relu=True, batch_norm=True):
        return dict(fshape=(fsize, fsize, nfm),
                    strides=stride,
                    padding=(1 if fsize > 1 else 0),
                    activation=(Rectlin() if relu else None),
                    init=Kaiming(local=True),
                    batch_norm=batch_norm)

    def module_s1(self, nfm, first=False):
        '''
        non-strided
        '''
        sidepath = Conv(**self.conv_params(1, nfm * 4, 1, False, False)) if first else SkipNode()
        mainpath = [] if first else [BatchNorm(), Activation(Rectlin())]
        mainpath.append(Conv(**self.conv_params(1, nfm)))
        mainpath.append(Conv(**self.conv_params(3, nfm)))
        mainpath.append(Conv(**self.conv_params(1, nfm * 4, relu=False, batch_norm=False)))

        return MergeSum([sidepath, mainpath])

    def module_s2(self, nfm):
        '''
        strided
        '''
        module = [BatchNorm(), Activation(Rectlin())]
        mainpath = [Conv(**self.conv_params(1, nfm, stride=2)),
                    Conv(**self.conv_params(3, nfm)),
                    Conv(**self.conv_params(1, nfm * 4, relu=False, batch_norm=False))]
        sidepath = [Conv(**self.conv_params(1, nfm * 4, stride=2, relu=False, batch_norm=False))]
        module.append(MergeSum([sidepath, mainpath]))
        return module


class InceptionModel(object):
    """
    Container for the inception-like test model
    """
    def __init__(self):
        self.in_shape = (3, 299, 299)
        self.nout = 100
        self.pool3s1p1 = dict(fshape=3, padding=1, strides=1, op='avg')
        self.pool3s2p0 = dict(fshape=3, strides=2, op='max')

        layers = self.main_branch(nout=self.nout)
        self.cost = GeneralizedCost(costfunc=CrossEntropyMulti())
        self.model = Model(layers=layers)
        self.model.initialize(self.in_shape, cost=self.cost)
        self.layers = layers

    @staticmethod
    def conv_params(fsize, nfm, padding='SAME', strides=1,
                    activation=Rectlin(), batch_norm=True):
        fsize = fsize if isinstance(fsize, tuple) else (fsize, fsize)
        fshape = fsize + (nfm,)
        padding = {'pad_h': (fsize[0] // 2 if padding == 'SAME' else 0),
                   'pad_w': (fsize[1] // 2 if padding == 'SAME' else 0),
                   'pad_d': 0}
        strides = {'str_h': strides, 'str_w': strides, 'str_d': 1}
        return dict(fshape=fshape,
                    strides=strides,
                    activation=activation,
                    padding=padding,
                    batch_norm=batch_norm,
                    init=Kaiming(local=True))

    def inception(self, kvals, b2fsz=5):
        (p1, p2, p3, p4) = kvals
        branch1 = [Conv(**self.conv_params(1, p1[0], padding=0, strides=1))]
        branch2 = [Conv(**self.conv_params(1, p2[0], padding=0, strides=1)),
                   Conv(**self.conv_params(b2fsz, p2[1], padding='SAME', strides=1))],
        branch3 = [Conv(**self.conv_params(1, p3[0], padding=0, strides=1)),
                   Conv(**self.conv_params(3, p3[1], padding='SAME', strides=1)),
                   Conv(**self.conv_params(3, p3[1], padding='SAME', strides=1))]
        branch4 = [Pooling(**self.pool3s1p1),
                   Conv(**self.conv_params(1, p4[0], padding=0, strides=1))]
        return MergeBroadcast(layers=[branch1, branch2, branch3, branch4], merge="depth")

    def inception_inception(self, kvals):
        (p1, p2, p3, p4) = kvals
        branch1 = [Conv(**self.conv_params(1, p1[0], padding=0, strides=1))]

        i2_branch1 = [Conv(**self.conv_params((1, 3), p2[1], padding='SAME', strides=1))]
        i2_branch2 = [Conv(**self.conv_params((3, 1), p2[1], padding='SAME', strides=1))]
        branch2 = [Conv(**self.conv_params(1, p2[0], padding=0, strides=1)),
                   MergeBroadcast(layers=[i2_branch1, i2_branch2], merge="depth")]

        i3_branch1 = [Conv(**self.conv_params((1, 3), p3[2], padding='SAME', strides=1))]
        i3_branch2 = [Conv(**self.conv_params((3, 1), p3[2], padding='SAME', strides=1))]
        branch3 = [Conv(**self.conv_params(1, p3[0], padding=0, strides=1)),
                   Conv(**self.conv_params(3, p3[1], padding='SAME', strides=1)),
                   MergeBroadcast(layers=[i3_branch1, i3_branch2], merge="depth")]

        branch4 = [Pooling(**self.pool3s1p1),
                   Conv(**self.conv_params(1, p4[0], padding=0, strides=1))]

        return MergeBroadcast(layers=[branch1, branch2, branch3, branch4], merge="depth")

    def main_branch(self, nout=100):
        return [Conv(**self.conv_params(3, 32, strides=2, padding=0)),
                Pooling(**self.pool3s2p0),
                Conv(**self.conv_params(1, 80, strides=1, padding=0)),
                Pooling(**self.pool3s2p0),
                self.inception([(64, ), (48, 64), (64, 96), (32, )], b2fsz=5),
                self.inception([(64, ), (48, 64), (64, 96), (64, )], b2fsz=5),
                self.inception_inception([(64, ), (64, 64), (64, 64, 64), (192, )]),
                self.inception_inception([(64, ), (64, 64), (64, 64, 64), (192, )]),
                Pooling(fshape='all', strides=1, op="avg"),
                Dropout(keep=0.8),
                Conv(**self.conv_params(1, nout, activation=Softmax()))]


class TreeModel(object):
    """
    Container for Tree style test model"
    """
    def __init__(self):
        self.in_shape = (1, 32, 32)

        init_norm = Gaussian(loc=0.0, scale=0.01)

        normrelu = dict(init=init_norm, activation=Rectlin())
        normsigm = dict(init=init_norm, activation=Logistic(shortcut=True))
        normsoft = dict(init=init_norm, activation=Softmax())

        # setup model layers
        b1 = BranchNode(name="b1")
        b2 = BranchNode(name="b2")

        p1 = [Affine(nout=100, name="main1", **normrelu),
              b1,
              Affine(nout=32, name="main2", **normrelu),
              Affine(nout=160, name="main3", **normrelu),
              b2,
              Affine(nout=32, name="main2", **normrelu),
              # make next layer big to check sizing
              Affine(nout=320, name="main2", **normrelu),
              Affine(nout=10, name="main4", **normsoft)]

        p2 = [b1,
              Affine(nout=16, name="branch1_1", **normrelu),
              Affine(nout=10, name="branch1_2", **normsigm)]

        p3 = [b2,
              Affine(nout=16, name="branch2_1", **normrelu),
              Affine(nout=10, name="branch2_2", **normsigm)]

        self.cost = Multicost(costs=[GeneralizedCost(costfunc=CrossEntropyMulti()),
                              GeneralizedCost(costfunc=CrossEntropyBinary()),
                              GeneralizedCost(costfunc=CrossEntropyBinary())],
                              weights=[1, 0., 0.])

        self.layers = SingleOutputTree([p1, p2, p3], alphas=[1, .2, .2])
        self.model = Model(layers=self.layers)
        self.model.initialize(self.in_shape, cost=self.cost)


class MultistreamModel(object):
    """
    Container for a multistream test model
    """
    def __init__(self):
        self.in_shape = [1024, (2538, 38)]

        init = Constant(0)
        image_path = Sequential([Affine(20, init, bias=init),
                                 Affine(10, init, bias=init)])
        sent_path = Sequential([Affine(30, init, bias=init),
                                Affine(10, init)])

        layers = [MergeMultistream(layers=[image_path, sent_path], merge="recurrent"),
                  Dropout(keep=0.5),
                  LSTM(4, init, activation=Logistic(), gate_activation=Tanh(), reset_cells=True),
                  Affine(20, init, bias=init, activation=Softmax())]
        self.layers = layers
        self.cost = GeneralizedCostMask(CrossEntropyMulti())

        self.model = Model(layers=layers)
        self.model.initialize(self.in_shape, cost=self.cost)


def check_broadcast(ms_layer):
    for branch in ms_layer.layers:
        assert type(branch) is Sequential
        # first layer in each branch should share
        # the container deltas
        assert branch.layers[0].deltas.ptr == ms_layer.deltas.ptr
        ptr = None
        for layer in branch.layers[1:]:
            if ptr is None:
                assert layer.deltas is not None
                ptr = layer.deltas.ptr
            else:
                assert layer.deltas.ptr != ptr
                ptr = layer.deltas.ptr


def check_deltas_swap(root_layer):
    # make sure that the deltas_buffers are
    # swapping back and forth and not using
    # the same allocated memory between layers
    last_buffer = None
    last_layer = None
    for cnt, ll in enumerate(root_layer):
        if cnt == 0:
            # first layer has no deltas
            if type(ll) is not BranchNode:
                assert ll.deltas is None
            last_layer = ll
            continue

        if ll.deltas is not None:
            assert ll.deltas.size >= np.prod(ll.in_shape)

        if last_buffer is None:
            # set the last_buffer var if not set yet
            if ll.owns_delta:
                assert ll.deltas is not None
                last_buffer = ll.deltas.ptr
            last_layer = ll
            continue

        if ll.deltas is not None:
            if type(last_layer) is not BranchNode:
                assert ll.deltas.ptr != last_buffer
            else:
                assert ll.deltas.ptr == last_buffer
            last_buffer = ll.deltas.ptr

        last_layer = ll
        if issubclass(ll.__class__, Broadcast):
            check_broadcast(ll)
            continue


def check_tree_model(root_layer):
    for branch in root_layer.layers:
        check_deltas_swap(branch.layers)


def check_ms_model(root_layer):
    for branch in root_layer.layers[0].layers:
        check_deltas_swap(branch.layers)
    check_deltas_swap(root_layer.layers[1:])


@pytest.mark.hasgpu
def test_inception_gpu(backend_gpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = InceptionModel()
    model = network.model
    check_deltas_swap(model.layers.layers)


@pytest.mark.skip(reason="Not implemented")
def test_inception_mkl(backend_mkl):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = InceptionModel()
    model = network.model
    check_deltas_swap(model.layers.layers)


@pytest.mark.skip(reason="Not implemented")
def test_inception_cpu(backend_cpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = InceptionModel()
    model = network.model
    check_deltas_swap(model.layers.layers)


@pytest.mark.hasgpu
def test_tree_gpu(backend_gpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = TreeModel()
    model = network.model
    check_tree_model(model.layers)

    print_deltas(model)


@pytest.mark.skip(reason="Not implemented")
def test_tree_mkl(backend_mkl):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = TreeModel()
    model = network.model
    check_tree_model(model.layers)

    print_deltas(model)


@pytest.mark.skip(reason="Not implemented")
def test_tree_cpu(backend_cpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = TreeModel()
    model = network.model
    check_tree_model(model.layers)

    print_deltas(model)


@pytest.mark.hasgpu
def test_multistream_gpu(backend_gpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = MultistreamModel()
    model = network.model
    check_ms_model(model.layers)


@pytest.mark.skip(reason="Not implemented")
def test_multistream_mkl(backend_mkl):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = MultistreamModel()
    model = network.model
    check_ms_model(model.layers)


@pytest.mark.skip(reason="Not implemented")
def test_multistream_cpu(backend_cpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = MultistreamModel()
    model = network.model
    check_ms_model(model.layers)


def print_deltas(model):
    layer_start = model.layers.layers[0]
    last_buffer = None
    deltas_ = [None]
    for ll in model.layers.layers_fprop():
        if ll.deltas is not None:
            print(ll.in_shape)
        gd = getattr(ll.deltas, 'ptr', None)
        if type(ll) is Sequential or ll == layer_start:
            assert gd is None
            continue

        if gd not in deltas_:
            deltas_.append(gd)
        if last_buffer is None:
            last_buffer = deltas_.index(gd)
        print('%s - %d' % (ll.name, deltas_.index(gd)))


@pytest.mark.hasgpu
def test_resnet_gpu(backend_gpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = ResnetModel()
    model = network.model
    check_deltas_swap(model.layers.layers)


@pytest.mark.skip(reason="Not implemented")
def test_resnet_mkl(backend_mkl):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = ResnetModel()
    model = network.model
    check_deltas_swap(model.layers.layers)


@pytest.mark.skip(reason="Not implemented")
def test_resnet_cpu(backend_cpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = ResnetModel()
    model = network.model
    check_deltas_swap(model.layers.layers)


@pytest.mark.hasgpu
def test_sequential_gpu(backend_gpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = SequentialModel()
    model = network.model
    check_deltas_swap(model.layers.layers)


@pytest.mark.skip(reason="Not implemented")
def test_sequential_mkl(backend_mkl):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = SequentialModel()
    model = network.model
    check_deltas_swap(model.layers.layers)


@pytest.mark.skip(reason="Not implemented")
def test_sequential_cpu(backend_cpu):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = 32
    network = SequentialModel()
    model = network.model
    check_deltas_swap(model.layers.layers)
