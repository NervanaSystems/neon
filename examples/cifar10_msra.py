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

# This will allow you to run a deep residual network on cifar10 data as detailed in:
# He et. al., "Deep Residual Learning for Image Recognition", http://arxiv.org/abs/1512.03385
#
# Prior to running, you need to write out padded cifar10 batches for ImageLoader to consume
#
# batch_writer.py --set_type cifar10 \
#       --data_dir <path-to-save-batches> \
#       --macro_size 10000 \
#       --target_size 40
#
# Then run the example:
#
# cifar10_msra.py -r 0 -vv \
#      --log <logfile> \
#      --no_progress_bar \
#      --epochs 150 \
#      --save_path <save-path> \
#      --eval_freq 1 \
#      --backend gpu \
#      --data_dir <path-to-saved-batches> \
#      --network resnet
#
# This setting should get to ~6.7% top-1 error. (Could be as low as 6.5)
#
# NB:  It is good practice to set your data_dir where your batches are stored
# to be local to your machine (to avoid accessing the macrobatches over network if,
# for example, your data_dir is in an NFS mounted location)

from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming, IdentityInit
from neon.layers import Conv, Pooling, GeneralizedCost, Activation, Affine
from neon.layers import MergeSum, SkipNode
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--network', default='plain', choices=['plain', 'resnet'],
                    help='type of network to create (plain or resnet)')
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 6n+2)')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

# setup data provider
imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                      repo_dir=args.data_dir, subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', shuffle=True, do_transforms=True, **imgset_options)
test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, **imgset_options)


def conv_params(fsize, nfm, stride=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=batch_norm)


def id_params(nfm):
    return dict(fshape=(1, 1, nfm), strides=2, padding=0, activation=None, init=IdentityInit())


def module_factory(nfm, stride=1):
    mainpath = [Conv(**conv_params(3, nfm, stride=stride)),
                Conv(**conv_params(3, nfm, relu=False))]
    sidepath = [SkipNode() if stride == 1 else Conv(**id_params(nfm))]

    module = [MergeSum([mainpath, sidepath]),
              Activation(Rectlin())]
    return module

# Structure of the deep residual part of the network:
# args.depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
nfms = [2**(stage + 4) for stage in sorted(range(3) * args.depth)]
strides = [1] + [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

# Now construct the network
layers = [Conv(**conv_params(3, 16))]
for nfm, stride in zip(nfms, strides):
    layers.append(module_factory(nfm, stride))
layers.append(Pooling('all', op='avg'))
layers.append(Affine(10, init=Kaiming(local=False), batch_norm=True, activation=Softmax()))

model = Model(layers=layers)
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001, schedule=Schedule([90, 135], 0.1))

# configure callbacks
valmetric = Misclassification()
callbacks = Callbacks(model, eval_set=test, metric=valmetric, **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
