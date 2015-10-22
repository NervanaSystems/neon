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

from neon.util.argparser import NeonArgparser
from neon.layers import Conv, Pooling, MergeBroadcast, BranchNode, Affine, Tree, Dropout
from neon.layers import GeneralizedCost, Multicost
from neon.initializers import Constant, Xavier
from neon.backends import gen_backend
from neon.optimizers import GradientDescentMomentum, PolySchedule, MultiOptimizer
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImgMaster
from neon.callbacks.callbacks import Callbacks, Callback

parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
batch_size = 128

# setup backend
be = gen_backend(backend=args.backend, rng_seed=args.rng_seed, device_id=args.device_id,
                 batch_size=batch_size, default_dtype=args.datatype)

train = ImgMaster(repo_dir=args.data_dir, inner_size=224, set_name='train')
test = ImgMaster(repo_dir=args.data_dir, inner_size=224, set_name='validation',
                 do_transforms=False)

train.init_batch_provider()
test.init_batch_provider()

init1 = Xavier(local=False)
initx = Xavier(local=True)
bias = Constant(val=0.20)
relu = Rectlin()

lr_sched = PolySchedule(total_epochs=args.epochs, power=0.5)

# scale original learning rates by sqrt(4) since we are doing batch size of 128 instead of 32
opt_gdm = GradientDescentMomentum(0.02, 0.9, wdecay=0.0004, schedule=lr_sched)
opt_biases = GradientDescentMomentum(0.04, 0.9, schedule=lr_sched)
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

common = dict(activation=relu, init=initx, bias=bias)
commonp1 = dict(activation=relu, init=initx, bias=bias, padding=1)
commonp2 = dict(activation=relu, init=initx, bias=bias, padding=2)
pool3s1p1 = dict(fshape=3, padding=1, strides=1)
pool3s2p1 = dict(fshape=3, padding=1, strides=2, op='max')


def fshape(rs, k):
    return (rs, rs, k)


def inception(kvals):
    (p1, p2, p3, p4) = kvals

    branch1 = [Conv(fshape(1, p1[0]), **common)]
    branch2 = [Conv(fshape(1, p2[0]), **common),
               Conv(fshape(3, p2[1]), **commonp1)]
    branch3 = [Conv(fshape(1, p3[0]), **common),
               Conv(fshape(5, p3[1]), **commonp2)]
    branch4 = [Pooling(op="max", **pool3s1p1),
               Conv(fshape(1, p4[0]), **common)]
    return MergeBroadcast(layers=[branch1, branch2, branch3, branch4], merge="depth")


def main_branch(branch_nodes):
    return [Conv(fshape(7, 64), padding=3, strides=2, **common),
            Pooling(**pool3s2p1),
            Conv(fshape(1, 64), **common),
            Conv(fshape(3, 192), **commonp1),
            Pooling(**pool3s2p1),
            inception([(64, ), (96, 128), (16, 32), (32, )]),
            inception([(128,), (128, 192), (32, 96), (64, )]),
            Pooling(**pool3s2p1),
            inception([(192,), (96, 208), (16, 48), (64, )]),
            branch_nodes[0],
            inception([(160,), (112, 224), (24, 64), (64, )]),
            inception([(128,), (128, 256), (24, 64), (64, )]),
            inception([(112,), (144, 288), (32, 64), (64, )]),
            branch_nodes[1],
            inception([(256,), (160, 320), (32, 128), (128,)]),
            Pooling(**pool3s2p1),
            inception([(256,), (160, 320), (32, 128), (128,)]),
            inception([(384,), (192, 384), (48, 128), (128,)]),
            Pooling(fshape=7, strides=1, op="avg"),
            Affine(nout=1000, init=init1, activation=Softmax(), bias=Constant(0))]


def aux_branch(bnode):
    return [bnode,
            Pooling(fshape=5, strides=3, op="avg"),
            Conv(fshape(1, 128), **common),
            Affine(nout=1024, init=init1, activation=relu, bias=bias),
            Dropout(keep=0.3),
            Affine(nout=1000, init=init1, activation=Softmax(), bias=Constant(0))]


# Now construct the model
branch_nodes = [BranchNode(name='branch' + str(i)) for i in range(2)]
main1 = main_branch(branch_nodes)
aux1 = aux_branch(branch_nodes[0])
aux2 = aux_branch(branch_nodes[1])

mlp = Model(layers=Tree([main1, aux1, aux2], alphas=[1.0, 0.3, 0.3]))

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
                        GeneralizedCost(costfunc=CrossEntropyMulti()),
                        GeneralizedCost(costfunc=CrossEntropyMulti())],
                 weights=[1, 0., 0.])  # We only want to consider the CE of the main path

mlp.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

test.exit_batch_provider()
train.exit_batch_provider()
