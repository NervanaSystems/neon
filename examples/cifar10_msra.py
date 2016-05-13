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

# This will allow you to run a deep residual network on cifar10 data as detailed in:
# He et. al., "Deep Residual Learning for Image Recognition", http://arxiv.org/abs/1512.03385
#
# Edit:  This example has now been updated to use the "preactivation" structure described in:
# He et. al., "Identity Mappings in Deep Residual Networks", http://arxiv.org/abs/1603.05027
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
#      --epochs 165 \
#      --depth 111 \
#      --save_path <save-path> \
#      --eval_freq 1 \
#      --backend gpu \
#      --batch_size 64 \
#      --data_dir <path-to-saved-batches>
#
# This setting should get to ~4.84% top-1 error. (Could be as low as 4.7)
#
# NB:  It is good practice to set your data_dir where your batches are stored
# to be local to your machine (to avoid accessing the macrobatches over network if,
# for example, your data_dir is in an NFS mounted location)

from builtins import range, zip
from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, GeneralizedCost, Activation, Affine
from neon.layers import MergeSum, SkipNode, BatchNorm
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks, BatchNormTuneCallback

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 9n+2)')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

# setup data provider
imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                      repo_dir=args.data_dir, subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', shuffle=True, do_transforms=True, **imgset_options)
test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, **imgset_options)
tune_set = ImageLoader(set_name='train', shuffle=False, do_transforms=False, inner_size=32,
                       scale_range=40, repo_dir=args.data_dir, subset_pct=20)


def conv_params(fsize, nfm, stride=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=batch_norm)


def module_s1(nfm, first=False):
    '''
    non-strided
    '''
    sidepath = Conv(**conv_params(1, nfm * 4, 1, False, False)) if first else SkipNode()
    mainpath = [] if first else [BatchNorm(), Activation(Rectlin())]
    mainpath.append(Conv(**conv_params(1, nfm)))
    mainpath.append(Conv(**conv_params(3, nfm)))
    mainpath.append(Conv(**conv_params(1, nfm * 4, relu=False, batch_norm=False)))

    return MergeSum([sidepath, mainpath])


def module_s2(nfm):
    '''
    strided
    '''
    module = [BatchNorm(), Activation(Rectlin())]
    mainpath = [Conv(**conv_params(1, nfm, stride=2)),
                Conv(**conv_params(3, nfm)),
                Conv(**conv_params(1, nfm * 4, relu=False, batch_norm=False))]
    sidepath = [Conv(**conv_params(1, nfm * 4, stride=2, relu=False, batch_norm=False))]
    module.append(MergeSum([sidepath, mainpath]))
    return module

# Structure of the deep residual part of the network:
# args.depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
nfms = [2**(stage + 4) for stage in sorted(list(range(3)) * args.depth)]
strides = [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

# Now construct the network
layers = [Conv(**conv_params(3, 16))]
layers.append(module_s1(nfms[0], True))

for nfm, stride in zip(nfms[1:], strides):
    res_module = module_s1(nfm) if stride == 1 else module_s2(nfm)
    layers.append(res_module)
layers.append(BatchNorm())
layers.append(Activation(Rectlin()))
layers.append(Pooling('all', op='avg'))
layers.append(Affine(10, init=Kaiming(local=False), batch_norm=True, activation=Softmax()))

model = Model(layers=layers)
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001, schedule=Schedule([82, 124], 0.1))

# configure callbacks
valmetric = Misclassification()
callbacks = Callbacks(model, eval_set=test, metric=valmetric, **args.callback_args)
callbacks.add_callback(BatchNormTuneCallback(tune_set), insert_pos=0)

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
