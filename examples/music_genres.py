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
Classify music clips according to genre
Example dataset can be downloaded from http://marsyasweb.appspot.com/download/data_sets/
After unpacking the dataset, point the script to the unpacked directory:

./music_genres.py -e8 -w /path/to/data -r0
"""

import os
import glob
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.initializers import Gaussian
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Dropout
from neon.optimizers import Adadelta
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import DataLoader, AudioParams
from neon.callbacks.callbacks import Callbacks


def create_index_files(source_path, train_percent=80, pattern='*'):
    assert os.path.exists(source_path)
    train_idx = os.path.join(source_path, 'train-index.csv')
    val_idx = os.path.join(source_path, 'val-index.csv')
    if os.path.exists(train_idx) and os.path.exists(val_idx):
        return train_idx, val_idx
    subdirs = glob.iglob(os.path.join(source_path, '*'))
    subdirs = filter(lambda x: os.path.isdir(x), subdirs)
    classes = sorted(map(lambda x: os.path.basename(x), subdirs))
    class_map = {key: val for key, val in zip(classes, range(len(classes)))}

    # Split into training and validation subsets.
    np.random.seed(0)
    with open(train_idx, 'w') as train_fd, open(val_idx, 'w') as val_fd:
        train_fd.write('filename,label1\n')
        val_fd.write('filename,label1\n')
        for subdir in subdirs:
            label = class_map[os.path.basename(subdir)]
            files = glob.glob(os.path.join(subdir, pattern))
            np.random.shuffle(files)
            train_count = len(files) * train_percent / 100
            for idx, filename in enumerate(files):
                fd = train_fd if idx < train_count else val_fd
                rel_path = os.path.join(os.path.basename(subdir),
                                        os.path.basename(filename))
                fd.write(rel_path + ',' + str(label) + '\n')
    return train_idx, val_idx

parser = NeonArgparser(__doc__)
args = parser.parse_args()
train_idx, val_idx = create_index_files(args.data_dir)

common_params = dict(sampling_freq=22050, clip_duration=31000, frame_duration=20)
train_params = AudioParams(add_noise=True, **common_params)
val_params = AudioParams(**common_params)
common = dict(target_size=1, nclasses=10, repo_dir=args.data_dir)
train = DataLoader(set_name='genres-train', media_params=train_params,
                   index_file=train_idx, shuffle=True, **common)
val = DataLoader(set_name='genres-val', media_params=val_params,
                 index_file=val_idx, shuffle=False, **common)
init = Gaussian(scale=0.01)
layers = [Conv((5, 5, 64), init=init, activation=Rectlin(),
               strides=dict(str_h=2, str_w=4)),
          Pooling(2, strides=2),
          Conv((5, 5, 128), init=init, batch_norm=True, activation=Rectlin(),
               strides=dict(str_h=1, str_w=4)),
          Pooling(2, strides=2),
          Conv((3, 3, 256), init=init, batch_norm=True, activation=Rectlin()),
          Pooling(2, strides=2),
          Conv((3, 3, 512), init=init, batch_norm=True, activation=Rectlin()),
          Pooling(2, strides=2),
          Conv((2, 2, 1024), init=init, batch_norm=True, activation=Rectlin()),
          Pooling(2, strides=2),
          Conv((2, 2, 2048), init=init, batch_norm=True, activation=Rectlin()),
          Dropout(),
          Affine(256, init=init, batch_norm=True, activation=Rectlin()),
          Affine(nout=common['nclasses'], init=init, activation=Softmax())]

model = Model(layers=layers)
opt = Adadelta()
metric = Misclassification()
callbacks = Callbacks(model, eval_set=val, metric=metric, **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
print('Misclassification error = %.1f%%' % (model.eval(val, metric=metric)*100))
