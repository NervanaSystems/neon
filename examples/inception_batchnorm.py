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
Googlenet V1 implementation
"""

import sys
from neon.util.argparser import NeonArgparser
from neon.layers import Conv, Pooling, MergeBroadcast, BranchNode, Affine, Tree
from neon.layers import GeneralizedCost, Multicost
from neon.initializers import Constant, Xavier
from neon.backends import gen_backend
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImgMaster
from neon.callbacks.callbacks import Callbacks, Callback

parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
batch_size = 64

# setup backend
be = gen_backend(backend=args.backend, rng_seed=args.rng_seed, device_id=args.device_id,
                 batch_size=batch_size, default_dtype=args.datatype)

try:
    train = ImgMaster(repo_dir=args.data_dir, inner_size=227, set_name='train')
    test = ImgMaster(repo_dir=args.data_dir, inner_size=227, set_name='validation',
                     do_transforms=False)
except (OSError, IOError, ValueError) as err:
    print err
    sys.exit(0)

train.init_batch_provider()
test.init_batch_provider()

init1 = Xavier()
init2 = Xavier(local=False)
bias = Constant(val=0.)
relu = Rectlin()

# drop LR by 4% every 2 epochs
weight_sched = Schedule(range(args.epochs)[2::2], 0.96)
opt_gdm = GradientDescentMomentum(learning_rate=0.015, momentum_coef=0.9, wdecay=0.00004,
                                  schedule=weight_sched)

common = dict(activation=relu, init=init1, batch_norm=True)
commonp1 = dict(activation=relu, init=init1, batch_norm=True, padding=1)
commonp2 = dict(activation=relu, init=init1, batch_norm=True, padding=2)
pool3s1p1 = dict(fshape=3, padding=1, strides=1)
pool3s2p1 = dict(fshape=3, padding=1, strides=2, op='max')
pool3s2 = dict(fshape=3, strides=2, op='max')


def fshape(rs, k):
    return (rs, rs, k)


def inception(kvals):
    (p1, p2, p3, p4) = kvals
    branch1 = [Conv(fshape(1, p1[0]), **common)]
    branch2 = [Conv(fshape(1, p2[0]), **common),
               Conv(fshape(3, p2[1]), **commonp1)]
    branch3 = [Conv(fshape(1, p3[0]), **common),
               Conv(fshape(3, p3[1]), **commonp1),
               Conv(fshape(3, p3[1]), **commonp1)]
    branch4 = [Pooling(op=p4[1], **pool3s1p1),
               Conv(fshape(1, p4[0]), **common)]
    return MergeBroadcast(layers=[branch1, branch2, branch3, branch4], merge="depth")


def inception2(kvals):
    (p1, p2) = kvals
    branch1 = [Conv(fshape(1, p1[0]), **common),
               Conv(fshape(3, p1[1]), strides=2, **common)]
    branch2 = [Conv(fshape(1, p2[0]), **common),
               Conv(fshape(3, p2[1]), **commonp1),
               Conv(fshape(3, p2[1]), strides=2, **common)]
    branch3 = [Pooling(**pool3s2)]
    return MergeBroadcast(layers=[branch1, branch2, branch3], merge="depth")


def main_branch(bnode):
    return [Conv(fshape(7, 64), strides=2, **commonp1),
            Pooling(**pool3s2),
            Conv(fshape(3, 192), **commonp1),
            Pooling(**pool3s2),
            inception([(64,), (64, 64), (64, 96), (32, 'avg')]),
            inception([(64,), (64, 96), (64, 96), (64, 'avg')]),
            inception2([(128, 160), (64, 96)]),
            inception([(224,), (64, 96), (96, 128), (128, 'avg')]),
            inception([(192,), (96, 128), (96, 128), (128, 'avg')]),
            inception([(160,), (128, 160), (128, 160), (128, 'avg')]),
            inception([(96,), (128, 192), (160, 192), (128, 'avg')]),
            bnode,
            inception2([(128, 192), (192, 256)]),
            inception([(352,), (192, 320), (160, 224), (128, 'avg')]),
            inception([(352,), (192, 320), (192, 224), (128, 'max')]),
            Pooling(fshape=7, strides=1, op="avg"),
            Affine(nout=1000, init=init2, activation=Softmax(), bias=bias)]


def aux_branch(bnode):
    return [bnode,
            Pooling(fshape=5, strides=3, op="avg"),
            Conv(fshape(1, 128), **common),
            Affine(nout=1024, init=init2, activation=relu, batch_norm=True),
            Affine(nout=1000, init=init2, activation=Softmax(), bias=bias)]


# Now construct the model
branch_node = BranchNode(name='branchnode')
main1 = main_branch(branch_node)
aux1 = aux_branch(branch_node)

mlp = Model(layers=Tree([main1, aux1], alphas=[1.0, 0.3]))

if args.model_file:
    import os
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    mlp.load_weights(args.model_file)

# configure callbacks
callbacks = Callbacks(mlp, train, output_file=args.output_file)

if args.validation_freq:
    class TopKMetrics(Callback):
        def __init__(self, valid_set, epoch_freq=args.validation_freq):
            super(TopKMetrics, self).__init__(epoch_freq=epoch_freq)
            self.valid_set = valid_set

        def on_epoch_end(self, epoch):
            self.valid_set.reset()
            allmetrics = TopKMisclassification(k=5)
            stats = mlp.eval(self.valid_set, metric=allmetrics)
            print ", ".join(allmetrics.metric_names) + ": " + ", ".join(map(str, stats.flatten()))

    callbacks.add_callback(TopKMetrics(test))

if args.save_path:
    callbacks.add_serialize_callback(args.serialize, args.save_path, history=2)

# setup cost function as CrossEntropy
cost = Multicost(costs=[GeneralizedCost(costfunc=CrossEntropyMulti()),
                        GeneralizedCost(costfunc=CrossEntropyMulti())],
                 weights=[1, 0.])  # We only want to consider the CE of the main path

mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

test.exit_batch_provider()
train.exit_batch_provider()
