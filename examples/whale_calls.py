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
Detect presence of whale calls in sound clips
Dataset can be downloaded from https://www.kaggle.com/c/whale-detection-challenge
After unpacking the dataset, point the script to the unpacked directory:

./whale_calls.py -e16 -w /path/to/data -r0
"""

import os
import glob
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.initializers import Gaussian
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, DeepBiRNN, RecurrentLast
from neon.optimizers import Adadelta
from neon.transforms import Rectlin, Softmax, CrossEntropyBinary, Misclassification
from neon.models import Model
from neon.data import DataLoader, AudioParams
from neon.callbacks.callbacks import Callbacks


def from_csv(filename):
    return np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=str)


def to_csv(filename, index):
    np.savetxt(filename, index, fmt='%s', delimiter=',', header='filename,label1')


def create_index_files(data_dir, train_percent=80):
    def get_path(filename):
        return os.path.join(data_dir, filename)

    assert os.path.exists(data_dir)
    train_idx = get_path('train-index.csv')
    val_idx = get_path('val-index.csv')
    all_idx = get_path('all-index.csv')
    test_idx = get_path('test-index.csv')
    if os.path.exists(test_idx):
        return train_idx, val_idx, all_idx, test_idx

    train = from_csv(os.path.join(data_dir, 'train.csv'))

    # Split into training and validation subsets
    np.random.seed(0)
    np.random.shuffle(train)
    train_count = train.shape[0] * train_percent / 100
    to_csv(all_idx, train)
    to_csv(train_idx, train[:train_count])
    to_csv(val_idx, train[train_count:])
    # Test set
    test_files = glob.glob(os.path.join(data_dir, 'test', '*.aiff'))
    test = np.zeros((len(test_files), 2), dtype=object)
    test[:, 0] = map(os.path.basename, test_files)
    to_csv(test_idx, test)
    return train_idx, val_idx, all_idx, test_idx


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
args = parser.parse_args()
train_idx, val_idx, all_idx, test_idx = create_index_files(args.data_dir)

common_params = dict(sampling_freq=2000, clip_duration=2000, frame_duration=80, overlap_percent=50)
train_params = AudioParams(add_noise=True, randomize_time_scale_by=5, **common_params)
test_params = AudioParams(**common_params)
common = dict(target_size=1, nclasses=2)

# Validate...
train_dir = os.path.join(args.data_dir, 'train')
train = DataLoader(set_name='train', repo_dir=train_dir, media_params=train_params,
                   index_file=train_idx, **common)
test = DataLoader(set_name='val', repo_dir=train_dir, media_params=test_params,
                  index_file=val_idx, **common)
model = run(train, test)
print('Misclassification error = %.1f%%' % (model.eval(test, metric=Misclassification())*100))

# Test...
test_dir = os.path.join(args.data_dir, 'test')
train = DataLoader(set_name='all', repo_dir=train_dir, media_params=train_params,
                   index_file=train_idx, **common)
test = DataLoader(set_name='test', repo_dir=test_dir, media_params=test_params,
                  index_file=test_idx, **common)
model = run(train, test)
preds = model.get_outputs(test)
np.savetxt('subm.txt', preds[:, 1], fmt='%.5f')
