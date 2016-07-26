#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
Classify music clips according to genre.

Download the example dataset from http://marsyasweb.appspot.com/download/data_sets/
After unpacking the dataset, point the script to the unpacked directory.

Usage:

    python examples/music_genres.py -e 8 --tar_file </path/to/genres.tar.gz> -w </destination/path> -r 0

"""

import os
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.initializers import Gaussian, GlorotUniform
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, DeepBiRNN, RecurrentMean
from neon.optimizers import Adagrad
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.data.dataloader_transformers import OneHot, TypeCast
from aeon import DataLoader
from ingesters import ingest_genre_data

def make_aeon_config(manifest_filename, minibatch_size, do_randomize=False):
    audio_decode_cfg = dict(
        sample_freq_hz=22050,
        max_duration="31 seconds",
        frame_length="20 milliseconds",
        frame_stride="14 milliseconds",
        time_scale_fraction=[0.95, 1.05] if do_randomize else [1.0, 1.0])

    return dict(
        manifest_filename=manifest_filename,
        minibatch_size=minibatch_size,
        macrobatch_size=100,
        cache_dir=get_data_cache_dir('/usr/local/data', subdir='music_genres_cache'),
        shuffle_manifest=do_randomize,
        shuffle_every_epoch=do_randomize,
        type='audio,label',
        label={'binary': False},
        audio=audio_decode_cfg)


def transformers(dl):
    dl = OneHot(dl, nclasses=10, index=1)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


parser = NeonArgparser(__doc__)
parser.add_argument('--tar_file', type=string, required=True, help='Input tar filename')
args = parser.parse_args()

train_idx, val_idx = ingest_genre_data(args.tar_file, args.data_dir)

# setup data provider
train_config = make_aeon_config(train_idx, args.batch_size, do_randomize=True)
val_config = make_aeon_config(val_idx, args.batch_size)

train = transformers(DataLoader(train_config, NervanaObject.be))
val = transformers(DataLoader(val_config, NervanaObject.be))


init = Gaussian(scale=0.01)
layers = [Conv((7, 7, 32), init=init, activation=Rectlin(),
               strides=dict(str_h=2, str_w=4)),
          Pooling(2, strides=2),
          Conv((5, 5, 64), init=init, batch_norm=True, activation=Rectlin(),
               strides=dict(str_h=1, str_w=2)),
          DeepBiRNN(128, init=GlorotUniform(), batch_norm=True, activation=Rectlin(),
                    reset_cells=True, depth=3),
          RecurrentMean(),
          Affine(nout=common['nclasses'], init=init, activation=Softmax())]

model = Model(layers=layers)
opt = Adagrad(learning_rate=0.01)
metric = Misclassification()
callbacks = Callbacks(model, eval_set=val, metric=metric, **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
print('Misclassification error = %.1f%%' % (model.eval(val, metric=metric)*100))
