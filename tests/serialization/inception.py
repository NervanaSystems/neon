#!/usr/bin/env python
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
googlenet model for serialzation testing.  script is adapted to only run
a small number of mini batches.
"""
import numpy as np

from neon.layers import Conv, Pooling, MergeBroadcast, BranchNode, Affine, Tree, Dropout
from neon.layers import GeneralizedCost, Multicost
from neon.initializers import Constant, Xavier
from neon.optimizers import GradientDescentMomentum, PolySchedule, MultiOptimizer
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.util.persist import load_obj, save_obj

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--resume', action='store_true',
                    help='first run without this and then run with so resume from saved file')
args = parser.parse_args()

if args.backend == 'mgpu':
    batch_size = 32*8  # for some reason bs 128 does not work with bias layers
else:
    batch_size = 32

# subset pct is set to make sure that every epoch has the same mb count
img_set_options = dict(repo_dir=args.data_dir,
                       inner_size=224,
                       dtype=np.float32,
                       subset_pct=0.09990891117239205)
train = ImageLoader(set_name='train', scale_range=(256, 256), shuffle=False,
                    do_transforms=False, **img_set_options)

init1 = Xavier(local=False)
initx = Xavier(local=True)
bias = Constant(val=0.20)
relu = Rectlin()

common = dict(activation=relu, init=initx, bias=bias)
commonp1 = dict(activation=relu, init=initx, bias=bias, padding=1)
commonp2 = dict(activation=relu, init=initx, bias=bias, padding=2)
pool3s1p1 = dict(fshape=3, padding=1, strides=1)
pool3s2p1 = dict(fshape=3, padding=1, strides=2, op='max')


def inception(kvals, name):
    (p1, p2, p3, p4) = kvals

    branch1 = [Conv((1, 1, p1[0]), name=name+'1x1', **common)]
    branch2 = [Conv((1, 1, p2[0]), name=name+'3x3_reduce', **common),
               Conv((3, 3, p2[1]), name=name+'3x3', **commonp1)]
    branch3 = [Conv((1, 1, p3[0]), name=name+'5x5_reduce', **common),
               Conv((5, 5, p3[1]), name=name+'5x5', **commonp2)]
    branch4 = [Pooling(op="max", name=name+'pool', **pool3s1p1),
               Conv((1, 1, p4[0]), name=name+'pool_proj', **common)]
    return MergeBroadcast(layers=[branch1, branch2, branch3, branch4], merge="depth")


def main_branch(branch_nodes):
    return [Conv((7, 7, 64), padding=3, strides=2, name='conv1/7x7_s2', **common),
            Pooling(name="pool1/3x3_s2", **pool3s2p1),
            Conv((1, 1, 64), name='conv2/3x3_reduce', **common),
            Conv((3, 3, 192), name="conv2/3x3",  **commonp1),
            Pooling(name="pool2/3x3_s2", **pool3s2p1),
            inception([(64, ), (96, 128), (16, 32), (32, )], name='inception_3a/'),
            inception([(128,), (128, 192), (32, 96), (64, )], name='inception_3b/'),
            Pooling(name='pool3/3x3_s2', **pool3s2p1),
            inception([(192,), (96, 208), (16, 48), (64, )], name='inception_4a/'),
            branch_nodes[0],
            inception([(160,), (112, 224), (24, 64), (64, )], name='inception_4b/'),
            inception([(128,), (128, 256), (24, 64), (64, )], name='inception_4c/'),
            inception([(112,), (144, 288), (32, 64), (64, )], name='inception_4d/'),
            branch_nodes[1],
            inception([(256,), (160, 320), (32, 128), (128,)], name='inception_4e/'),
            Pooling(name='pool4/3x3_s2', **pool3s2p1),
            inception([(256,), (160, 320), (32, 128), (128,)], name='inception_5a/'),
            inception([(384,), (192, 384), (48, 128), (128,)], name="inception_5b/"),
            Pooling(fshape=7, strides=1, op="avg", name='pool5/7x7_s1'),
            Affine(nout=1000, init=init1, activation=Softmax(),
                   bias=Constant(0), name='loss3/classifier')]


def aux_branch(bnode, ind):
    # TODO put dropout back in
    nm = 'loss%d/' % ind
    return [bnode,
            Pooling(fshape=5, strides=3, op="avg", name=nm+'ave_pool'),
            Conv((1, 1, 128), name=nm+'conv', **common),
            Affine(nout=1024, init=init1, activation=relu, bias=bias, name=nm+'fc'),
            Dropout(keep=1.0, name=nm+'drop_fc'),
            Affine(nout=1000, init=init1, activation=Softmax(),
                   bias=Constant(0), name=nm+'classifier')]


# setup cost function as CrossEntropy
cost = Multicost(costs=[GeneralizedCost(costfunc=CrossEntropyMulti()),
                        GeneralizedCost(costfunc=CrossEntropyMulti()),
                        GeneralizedCost(costfunc=CrossEntropyMulti())],
                 weights=[1, 0., 0.])  # We only want to consider the CE of the main path

if not args.resume:
    # build the model from scratch and run it

    # Now construct the model
    branch_nodes = [BranchNode(name='branch' + str(i)) for i in range(2)]
    main1 = main_branch(branch_nodes)
    aux1 = aux_branch(branch_nodes[0], ind=1)
    aux2 = aux_branch(branch_nodes[1], ind=2)

    model = Model(layers=Tree([main1, aux1, aux2], alphas=[1.0, 0.3, 0.3]))

else:
    # load up the save model
    model = Model('serialize_test_2.pkl')
    model.initialize(train, cost=cost)

# configure callbacks
callbacks = Callbacks(model, progress_bar=True, output_file='temp1.h5',
                      serialize=1, history=3, save_path='serialize_test.pkl')

lr_sched = PolySchedule(total_epochs=10, power=0.5)
opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0002, schedule=lr_sched)
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=lr_sched)

opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})
if not args.resume:
    # fit the model for 3 epochs
    model.fit(train, optimizer=opt, num_epochs=3, cost=cost, callbacks=callbacks)

train.reset()
# get 1 image
for im, l in train:
    break
train.exit_batch_provider()
save_obj((im.get(), l.get()), 'im1.pkl')
im_save = im.get().copy()
if args.resume:
    (im2, l2) = load_obj('im1.pkl')
    im.set(im2)
    l.set(l2)

# run fprop and bprop on this minibatch save the results
out_fprop = model.fprop(im)

out_fprop_save = [x.get() for x in out_fprop]
im.set(im_save)
out_fprop = model.fprop(im)
out_fprop_save2 = [x.get() for x in out_fprop]
for x, y in zip(out_fprop_save, out_fprop_save2):
    assert np.max(np.abs(x-y)) == 0.0, '2 fprop iterations do not match'

# run fit fot 1 minibatch
# have to do this by hand
delta = model.cost.get_errors(im, l)
model.bprop(delta)
if args.resume:
    model.optimizer = opt
model.optimizer.optimize(model.layers_to_optimize, epoch=model.epoch_index)

# run fprop again as a measure of the model state
out_fprop = model.fprop(im)
out_fprop_save2 = [x.get() for x in out_fprop]

if not args.resume:
    save_obj([out_fprop_save, out_fprop_save2], 'serial_test_out1.pkl')
else:
    # load up the saved file and compare
    run1 = load_obj('serial_test_out1.pkl')

    # compare the initial fprops
    for x, y in zip(run1[0], out_fprop_save):
        assert np.max(np.abs(x-y)) == 0.0, 'Deserialized model not matching serialized model'

    # and post extra training fprops
    for x, y in zip(run1[1], out_fprop_save2):
        if np.max(np.abs(x-y)) != 0.0:
            print np.max(np.abs(x-y))
            raise ValueError('Deserialized training not matching serialized training')

    # see if the single epoch of optimization had any real effect
    for x, y in zip(out_fprop_save, out_fprop_save2):
        assert np.max(np.abs(x-y)) > 0.0, 'Training had no effect on model'
    print 'passed'
