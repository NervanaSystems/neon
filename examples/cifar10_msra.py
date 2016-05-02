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
#
# To run the example:
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

import os
from builtins import zip
from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, GeneralizedCost, Activation, Affine
from neon.layers import MergeSum, SkipNode, BatchNorm
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import DataLoader, ImageParams
from neon.callbacks.callbacks import Callbacks, BatchNormTuneCallback

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 9n+2)')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')


def extract_images(out_dir, padded_size):
    '''
    Save CIFAR-10 dataset as PNG files
    '''
    import numpy as np
    from neon.data import load_cifar10
    from PIL import Image
    dataset = dict()
    dataset['train'], dataset['val'], _ = load_cifar10(out_dir, normalize=False)
    pad_size = (padded_size - 32) / 2 if padded_size > 32 else 0
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))

    for setn in ('train', 'val'):
        data, labels = dataset[setn]

        img_dir = os.path.join(out_dir, setn)
        ulabels = np.unique(labels)
        for ulabel in ulabels:
            subdir = os.path.join(img_dir, str(ulabel))
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        for idx in range(data.shape[0]):
            im = np.pad(data[idx].reshape((3, 32, 32)), pad_width, mode='mean')
            im = np.uint8(np.transpose(im, axes=[1, 2, 0]).copy())
            im = Image.fromarray(im)
            path = os.path.join(img_dir, str(labels[idx][0]), str(idx) + '.png')
            im.save(path, format='PNG')


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

train_dir = os.path.join(args.data_dir, 'train')
test_dir = os.path.join(args.data_dir, 'val')
if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
    extract_images(args.data_dir, 40)

# setup data provider
shape = dict(channel_count=3, height=32, width=32)
train_params = ImageParams(center=False, aspect_ratio=110, **shape)
test_params = ImageParams(**shape)
common = dict(target_size=1, nclasses=10)

train = DataLoader(set_name='train', repo_dir=train_dir, media_params=train_params,
                   shuffle=True, subset_percent=args.subset_pct, **common)
test = DataLoader(set_name='val', repo_dir=test_dir, media_params=test_params, **common)
tune_set = DataLoader(set_name='train', repo_dir=train_dir, media_params=train_params,
                      subset_percent=20, **common)


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
