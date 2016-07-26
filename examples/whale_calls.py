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
Detect presence of whale calls in sound clips.

Download the dataset from https://www.kaggle.com/c/whale-detection-challenge
After unpacking the dataset, point the script to the unpacked directory.
unzip -q -d dstdir whale_data/whale_data.zip && find dstdir -name '*.aiff'  | parallel --progress 'sox {} {.}.wav && rm {}
Usage:

    python examples/whale_calls.py -e 16 --zip_file </path/to/whale_data.zip> -w </destination/path> -r 0

"""

import numpy as np
from neon.util.argparser import NeonArgparser
from neon.initializers import Gaussian
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, DeepBiRNN, RecurrentLast
from neon.optimizers import Adadelta
from neon.transforms import Rectlin, Softmax, CrossEntropyBinary, Misclassification
from neon.models import Model
from neon.data import DataLoader, AudioParams
from neon.data.dataloader_transformers import OneHot, TypeCast
from neon.callbacks.callbacks import Callbacks
from ingesters import ingest_whale_data


def make_aeon_config(manifest_filename, minibatch_size, do_randomize=False):
    audio_decode_cfg = dict(
        sample_freq_hz=2000,
        max_duration="2 seconds",
        frame_length="80 milliseconds",
        frame_stride="40 milliseconds")

    return dict(
        manifest_filename=manifest_filename,
        minibatch_size=minibatch_size,
        macrobatch_size=1024,
        cache_dir=get_data_cache_dir('/usr/local/data', subdir='whale_calls_cache'),
        shuffle_manifest=do_randomize,
        shuffle_every_epoch=do_randomize,
        type='audio,label',
        label={'binary': False},
        audio=audio_decode_cfg)


def make_aeon_config_inference(manifest_filename, minibatch_size):
    audio_decode_cfg = dict(
        sample_freq_hz=2000,
        max_duration="2 seconds",
        frame_length="80 milliseconds",
        frame_stride="40 milliseconds")

    return dict(
        manifest_filename=manifest_filename,
        minibatch_size=minibatch_size,
        macrobatch_size=1024,
        cache_dir=get_data_cache_dir('/usr/local/data', subdir='whale_calls_cache'),
        type='audio,inference',
        audio=audio_decode_cfg)


def transform_train(dl):
    dl = OneHot(dl, nclasses=2, index=1)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl

def transform_inference(dl):
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


def run(train, test):
    init = Gaussian(scale=0.01)
    layers = [Conv((3, 3, 128), init=init, activation=Rectlin(),
                   strides=dict(str_h=1, str_w=2)),
              Conv((3, 3, 256), init=init, batch_norm=True, activation=Rectlin()),
              Pooling(2, strides=2),
              Conv((2, 2, 512), init=init, batch_norm=True, activation=Rectlin()),
              DeepBiRNN(256, init=init, activation=Rectlin(), reset_cells=True, depth=3),
              RecurrentLast(),
              Affine(32, init=init, batch_norm=True, activation=Rectlin()),
              Affine(nout=common['nclasses'], init=init, activation=Softmax())]

    model = Model(layers=layers)
    opt = Adadelta()
    metric = Misclassification()
    callbacks = Callbacks(model, eval_set=test, metric=metric, **args.callback_args)
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())

    model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
    return model


parser = NeonArgparser(__doc__)
parser.add_argument('--zip_file', type=string, required=True, help='Input zip filename')
args = parser.parse_args()

train_idx, val_idx, test_idx, all_idx, noise_idx = ingest_whale_data(args.zip_file, args.data_dir)

train_config = make_aeon_config(train_idx, args.batch_size, do_randomize=True)
val_config = make_aeon_config(val_idx, args.batch_size)

train = transformer_train(DataLoader(train_config, NervanaObject.be))
val = transformer_train(DataLoader(val_config, NervanaObject.be))

# Validate...
model = run(train, val)
print('Misclassification error = %.1f%%' % (model.eval(val, metric=Misclassification())*100))

# Test...
all_config = make_aeon_config(all_idx, args.batch_size)
alltrain = transformer_train(DataLoader(all_config, NervanaObject.be))

test_config = make_aeon_config_inference(test_idx, args.batch_size)
test = transformer_inference(DataLoader(test_config, NervanaObject.be))

model = run(alltrain, test)
preds = model.get_outputs(test)
np.savetxt('subm.txt', preds[:, 1], fmt='%.5f')
