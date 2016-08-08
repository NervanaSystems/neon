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
Deep Residual Network on CIFAR10 data.

Reference:

    Deep Residual Learning for Image Recognition `[He2015]`_
..  _[He2015]: http://arxiv.org/abs/1512.03385

This example has also been updated to use the "preactivation" structure
described in: He et. al., "Identity Mappings in Deep Residual Networks",
http://arxiv.org/abs/1603.05027

Usage:

    python examples/cifar10_msra.py -r 0 -vv \
        --log <logfile> \
        --no_progress_bar \
        --epochs 165 \
        --depth 111 \
        --save_path <save-path> \
        --eval_freq 1 \
        --backend gpu \
        --batch_size 64 \

    This setting should get to ~4.84% top-1 error. (Could be as low as 4.7)

    NB:  It is good practice to set your data_dir where your batches are stored
    to be local to your machine (to avoid accessing the macrobatches over network if,
    for example, your data_dir is in an NFS mounted location)

"""

import os
from builtins import zip
from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, GeneralizedCost, Activation, Affine
from neon.layers import MergeSum, SkipNode, BatchNorm
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.callbacks.callbacks import Callbacks, BatchNormTuneCallback
from neon.data.dataloader_transformers import OneHot, TypeCast, ImageMeanSubtract
from neon.util.persist import get_data_cache_dir
from neon import NervanaObject
import numpy as np
from aeon import DataLoader
from ingesters import ingest_cifar10

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 9n+2)')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')


def make_aeon_config(manifest_filename, minibatch_size, do_randomize=False, subset_pct=100):
    image_decode_cfg = dict(
        height=32, width=32,
        scale=[0.8, 0.8],            # cropboxes of size 0.8 * 40 = 32
        flip=do_randomize,           # whether to do random flips
        center=(not do_randomize))   # whether to do random crops

    return dict(
        manifest_filename=manifest_filename,
        minibatch_size=minibatch_size,
        macrobatch_size=5000,
        cache_directory=get_data_cache_dir('/usr/local/data', subdir='cifar10_cache'),
        subset_fraction=float(subset_pct/100.0),
        shuffle_manifest=do_randomize,
        shuffle_every_epoch=do_randomize,
        type='image,label',
        label={'binary': False},
        image=image_decode_cfg)


def transformers(dl):
    dl = OneHot(dl, nclasses=10, index=1)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = ImageMeanSubtract(dl, index=0, pixel_mean=[104.41227722, 119.21331787, 126.80609131])
    return dl


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


args = parser.parse_args()

image_dir = get_data_cache_dir(args.data_dir, subdir='cifar10_extracted')
cache_dir = get_data_cache_dir(args.data_dir, subdir='cifar10_cache')

# perform ingest if it hasn't already been done and return manifest files
train_manifest, val_manifest = ingest_cifar10(out_dir=image_dir, padded_size=40, overwrite=False)

# setup data provider
train_config = make_aeon_config(train_manifest, args.batch_size,
                                do_randomize=True, subset_pct=args.subset_pct)

val_config = make_aeon_config(val_manifest, args.batch_size)

tune_config = make_aeon_config(train_manifest, args.batch_size, subset_pct=20)

train = transformers(DataLoader(train_config, NervanaObject.be))
test = transformers(DataLoader(val_config, NervanaObject.be))
tune_set = transformers(DataLoader(tune_config, NervanaObject.be))


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
