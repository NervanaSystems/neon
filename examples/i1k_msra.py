#!/usr/bin/env python
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
Runs one epoch of Alexnet on imagenet data.
For running complete alexnet
alexnet.py -e 90 -eval 1 -s <save-path> -w <path-to-saved-batches>
"""

from builtins import zip
from neon import logger as neon_logger
from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, GeneralizedCost, Activation
from neon.layers import MergeSum, SkipNode
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks
import itertools as itt
import sys

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=0,
                    help='network configuration')
parser.add_argument('--bottleneck', action="store_true",
                    help="use bottleneck modules compared to double 3x3 modules")
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

if args.depth in (0, 18):
    stages = (2, 2, 2, 2)
elif args.depth in (1, 34, 50):
    stages = (3, 4, 6, 3)
elif args.depth in (2, 68, 101):
    stages = (3, 4, 23, 3)
elif args.depth in (3, 102, 152):
    stages = (3, 8, 36, 3)
elif args.depth in (4, 98, 138):
    stages = (3, 7, 35, 3)
else:
    neon_logger.display("Bad depth parameter")
    sys.exit()

# setup data provider
img_set_options = dict(repo_dir=args.data_dir,
                       inner_size=224,
                       subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', scale_range=(
    256, 480), shuffle=True, **img_set_options)
test = ImageLoader(set_name='validation', scale_range=0,
                   do_transforms=False, **img_set_options)


def conv_params(fsize, nfm, strides=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm),
                strides=strides,
                activation=(Rectlin() if relu else None),
                padding=(1 if fsize > 1 else 0),
                batch_norm=batch_norm,
                init=Kaiming(local=True))


def module_factory(nfm, stride=1):
    nfm_out = nfm * 4 if args.bottleneck else nfm
    use_skip = True if stride == 1 else False
    stride = abs(stride)
    sidepath = [SkipNode() if use_skip else Conv(
        **conv_params(1, nfm_out, stride, False))]

    if args.bottleneck:
        mainpath = [Conv(**conv_params(1, nfm)),
                    Conv(**conv_params(3, nfm, stride)),
                    Conv(**conv_params(1, nfm_out, relu=False))]
    else:
        mainpath = [Conv(**conv_params(3, nfm, stride)),
                    Conv(**conv_params(3, nfm, relu=False))]
    return [MergeSum([mainpath, sidepath]),
            Activation(Rectlin())]


layers = [Conv(**conv_params(7, 64, strides=2)),
          Pooling(3, strides=2)]


# Structure of the deep residual part of the network:
# args.depth modules of 2 convolutional layers each at feature map depths
# of 64, 128, 256, 512
nfms = list(itt.chain.from_iterable(
    [itt.repeat(2**(x + 6), r) for x, r in enumerate(stages)]))
strides = [-1] + [1 if cur == prev else 2 for cur,
                  prev in zip(nfms[1:], nfms[:-1])]

for nfm, stride in zip(nfms, strides):
    layers.append(module_factory(nfm, stride))

layers.append(Pooling('all', op='avg'))
layers.append(
    Conv(**conv_params(1, train.nclass, relu=False, batch_norm=False)))
layers.append(Activation(Softmax()))
model = Model(layers=layers)

weight_sched = Schedule([30, 60], 0.1)
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001, schedule=weight_sched)

# configure callbacks
valmetric = TopKMisclassification(k=5)
callbacks = Callbacks(model, eval_set=test,
                      metric=valmetric, **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
model.fit(train, optimizer=opt, num_epochs=args.epochs,
          cost=cost, callbacks=callbacks)
