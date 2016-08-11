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
Deep Residual Network on ImageNet data.

Reference:
    Deep Residual Learning for Image Recognition `[He2015]`_
..  _[He2015]: http://arxiv.org/abs/1512.03385

Usage:

    Before training, prepare ImageNet macrobatches as described at
    http://neon.nervanasys.com/docs/latest/datasets.html#imagenet

    python examples/i1k_msra.py -w </path/to/ImageNet/macrobatches>

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
from neon.callbacks.callbacks import Callbacks, BatchNormTuneCallback
from neon.data.dataloader_transformers import OneHot, TypeCast, ImageMeanSubtract
from neon.util.persist import get_data_cache_dir
from aeon import DataLoader
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
manifest_dir = get_data_cache_dir('/usr/local/data', subdir='i1k_test')
cache_dir = get_data_cache_dir('/usr/local/data', subdir='i1k_cache')


def make_aeon_config(manifest_filename, cache_directory, minibatch_size, do_randomize=False, subset_pct=100):
    image_decode_cfg = dict(height=224, width=224, scale=[0.875, 0.875])
    if do_randomize:
        image_decode_cfg['scale'] = [0.08, 1.0]  # 8% of area to 100% of area for cropbox
        image_decode_cfg['do_area_scale'] = True
        image_decode_cfg['aspect_ratio'] = [0.75, 1.33]
        image_decode_cfg['photometric'] = [-0.1, 0.1]
        image_decode_cfg['lighting'] = [0.0, 0.01]
        image_decode_cfg['flip_enable'] = True
        image_decode_cfg['lighting'] = False

    return dict(
        manifest_filename=manifest_filename,
        minibatch_size=minibatch_size,
        macrobatch_size=1024,
        cache_directory=cache_directory,
        subset_fraction=float(subset_pct/100.0),
        shuffle_manifest=do_randomize,
        shuffle_every_epoch=do_randomize,
        type='image,label',
        label={'binary': False},
        image=image_decode_cfg)

def transformers(dl):
    dl = OneHot(dl, nclasses=1000, index=1)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = ImageMeanSubtract(dl, index=0, pixel_mean=[104.41227722, 119.21331787, 126.80609131])
    return dl

train_config = make_aeon_config(os.path.join(manifest_dir, 'train_file.csv'), cache_dir, args.batch_size,
                                do_randomize=True, subset_pct=args.subset_percent)

valid_config = make_aeon_config(os.path.join(manifest_dir, 'val_file.csv'), cache_dir, args.batch_size)
tune_config = make_aeon_config(os.path.join(manifest_dir, 'train_file.csv'), cache_dir, args.batch_size,
                               subset_pct=20)

train = transformers(DataLoader(train_config, model.be))
valid = transformers(DataLoader(valid_config, model.be))

img_set_options = dict(repo_dir=args.data_dir,
                       inner_size=224,
                       subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', scale_range={'min_area_pct': 8, 'max_area_pct': 100},
                    aspect_ratio=133, contrast_range=(60, 140), shuffle=True, **img_set_options)
test = ImageLoader(set_name='validation', scale_range=256, do_transforms=False, **img_set_options)
tune = ImageLoader(set_name='train', scale_range=256, do_transforms=False, repo_dir=args.data_dir,
                   inner_size=224, subset_pct=10)


def conv_params(fsize, nfm, strides=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm),
                strides=strides,
                activation=(Rectlin() if relu else None),
                padding=(fsize // 2),
                batch_norm=batch_norm,
                init=Kaiming(local=True))


def module_factory(nfm, stride=1):
    nfm_out = nfm * 4 if args.bottleneck else nfm
    use_skip = True if stride == 1 else False
    stride = abs(stride)
    sidepath = [SkipNode() if use_skip else Conv(
        **conv_params(1, nfm_out, stride, False))]

    if args.bottleneck:
        mainpath = [Conv(**conv_params(1, nfm, stride)),
                    Conv(**conv_params(3, nfm)),
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
layers.append(Conv(**conv_params(1, train.nclass, relu=False)))
layers.append(Activation(Softmax()))
model = Model(layers=layers)

weight_sched = Schedule([30, 60], 0.1)
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001, schedule=weight_sched)

# configure callbacks
valmetric = TopKMisclassification(k=5)
callbacks = Callbacks(model, eval_set=test, metric=valmetric, **args.callback_args)
callbacks.add_callback(BatchNormTuneCallback(tune), insert_pos=0)

cost = GeneralizedCost(costfunc=CrossEntropyMulti())
model.fit(train, optimizer=opt, num_epochs=args.epochs,
          cost=cost, callbacks=callbacks)
