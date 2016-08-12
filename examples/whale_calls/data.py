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
from neon.util.persist import ensure_dirs_exist
from neon.data.dataloader_transformers import OneHot, TypeCast
from aeon import DataLoader


def get_ingest_file(filename):
    '''
    prepends the environment variable data path after checking that it has been set
    '''
    if os.environ.get('WHALE_DATA_PATH') is None:
        raise RuntimeError("Missing required env variable WHALE_DATA_PATH")

    return os.path.join(os.environ['WHALE_DATA_PATH'], 'whale-extracted', filename)


def wrap_dataloader(dl):
    dl = OneHot(dl, index=1, nclasses=2)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


def common_config(set_name, batch_size):
    manifest_file = get_ingest_file(set_name + '-index.csv')
    cache_root = ensure_dirs_exist(os.path.join(os.environ['WHALE_DATA_PATH'], 'whale-cache/'))

    return {
               'manifest_filename': manifest_file,
               'minibatch_size': batch_size,
               'macrobatch_size': batch_size * 12,
               'type': 'audio,label',
               'cache_directory': cache_root,
               'audio': {'sample_freq_hz': 2000,
                         'max_duration': '2 seconds',
                         'frame_length': '80 milliseconds',
                         'frame_stride': '40 milliseconds'},
               'label': {'binary': False}
            }


def make_val_loader(set_name, backend_obj):
    aeon_config = common_config(set_name, backend_obj.bsz)
    return wrap_dataloader(DataLoader(aeon_config, backend_obj))


def make_train_loader(set_name, backend_obj, random_seed=0):
    aeon_config = common_config(set_name, backend_obj.bsz)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    aeon_config['random_seed'] = random_seed

    aeon_config['audio']['noise_index_file'] = get_ingest_file('noise-index.csv')
    aeon_config['audio']['add_noise_probability'] = 0.5
    # aeon_config['audio']['noise_level'] = [0.0, 0.5]

    return wrap_dataloader(DataLoader(aeon_config, backend_obj))


def make_test_loader(set_name, backend_obj):
    aeon_config = common_config(set_name, backend_obj.bsz)
    aeon_config['type'] = 'audio'  # No labels provided
    aeon_config.pop('label', None)
    dl = DataLoader(aeon_config, backend_obj)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl
