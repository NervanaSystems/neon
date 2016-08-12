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

import os
import numpy as np
from neon.data.dataloader_transformers import OneHot, TypeCast
from neon.util.persist import ensure_dirs_exist
from aeon import DataLoader


def get_ingest_file(filename):
    '''
    prepends the environment variable data path after checking that it has been set
    '''
    if os.environ.get('V3D_DATA_PATH') is None:
        raise RuntimeError("Missing required env variable V3D_DATA_PATH")

    return os.path.join(os.environ['V3D_DATA_PATH'], 'ucf-extracted', filename)


def common_config(manifest_file, batch_size):
    root_dir = os.environ['V3D_DATA_PATH']
    cache_root = ensure_dirs_exist(os.path.join(root_dir, 'ucf-cache/'))
    return {
               'manifest_filename': manifest_file,
               'minibatch_size': batch_size,
               'macrobatch_size': batch_size * 12,
               'type': 'video,label',
               'cache_directory': cache_root,
               'video': {'max_frame_count': 16,
                         'frame': {'height': 112,
                                   'width': 112,
                                   'scale': [0.875, 0.875]}},
               'label': {'binary': False}
            }


def make_test_loader(backend_obj):
    manifest_file = get_ingest_file('test-index.csv')
    aeon_config = common_config(manifest_file, backend_obj.bsz)
    dl = DataLoader(aeon_config, backend_obj)
    dl = OneHot(dl, index=1, nclasses=101)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


def make_train_loader(backend_obj, random_seed=0):
    manifest_file = get_ingest_file('train-index.csv')
    aeon_config = common_config(manifest_file, backend_obj.bsz)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    aeon_config['random_seed'] = random_seed

    aeon_config['video']['frame']['center'] = False
    aeon_config['video']['frame']['flip_enable'] = True

    dl = DataLoader(aeon_config, backend_obj)
    dl = OneHot(dl, index=1, nclasses=101)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


def make_inference_loader(manifest_file, backend_obj):
    aeon_config = common_config(manifest_file, backend_obj.bsz)
    aeon_config['type'] = 'video'  # No labels provided
    aeon_config.pop('label', None)
    dl = DataLoader(aeon_config, backend_obj)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


def make_category_map():
    category_file = get_ingest_file('category-index.csv')
    categories = np.genfromtxt(category_file, dtype=None, delimiter=',')
    return {t[0]: t[1] for t in categories}


def accumulate_video_pred(clip_preds):
    #  Index file will look like:
    #  video_clip_file,label_file
    #  video_clip_file will be video_path/v_WritingOnBoard_g05,
    #  where WritingOnBoard_g05 is the video name
    video_pred = {}
    manifest_file = get_ingest_file('test-index.csv')
    clip_files = np.genfromtxt(manifest_file, dtype=None, delimiter=',', usecols=(0))
    for clip_file, pred in zip(clip_files, clip_preds):
        video_name = '_'.join(os.path.basename(clip_file).split('_')[1:-2])
        category = os.path.split(os.path.dirname(clip_file))[-1]
        video_pred.setdefault(video_name, (category, [])).__getitem__(1).append(pred)

    return video_pred
