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
import numpy as np
from neon.util.persist import get_data_cache_or_nothing
from neon.data.dataloader_transformers import OneHot, TypeCast, BGRMeanSubtract
from neon.data.aeon_shim import AeonDataLoader


def common_config(manifest_file, manifest_root, batch_size, subset_pct):
    cache_root = get_data_cache_or_nothing('i1k-cache/')

    return {
               'manifest_filename': manifest_file,
               'manifest_root': manifest_root,
               'minibatch_size': batch_size,
               'subset_fraction': float(subset_pct/100.0),
               'macrobatch_size': 5000,
               'type': 'image,label',
               'cache_directory': cache_root,
               'image': {'height': 224,
                         'width': 224,
                         'scale': [0.875, 0.875]},
               'label': {'binary': False}
            }


def wrap_dataloader(dl):
    dl = OneHot(dl, index=1, nclasses=1000)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = BGRMeanSubtract(dl, index=0)
    return dl


def make_alexnet_train_loader(manifest_file, manifest_root, backend_obj,
                              subset_pct=100, random_seed=0):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    aeon_config['random_seed'] = random_seed
    aeon_config['image']['center'] = False
    aeon_config['image']['flip_enable'] = True

    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


def make_msra_train_loader(manifest_file, manifest_root, backend_obj,
                           subset_pct=100, random_seed=0):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    aeon_config['random_seed'] = random_seed
    aeon_config['image']['center'] = False
    aeon_config['image']['flip_enable'] = True
    aeon_config['image']['scale'] = [0.08, 1.0]
    aeon_config['image']['do_area_scale'] = True
    aeon_config['image']['horizontal_distortion'] = [0.75, 1.33]
    aeon_config['image']['lighting'] = [0.0, 0.01]
    aeon_config['image']['contrast'] = [0.9, 1.1]
    aeon_config['image']['brightness'] = [0.9, 1.1]
    aeon_config['image']['saturation'] = [0.9, 1.1]

    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


def make_validation_loader(manifest_file, manifest_root, backend_obj, subset_pct=100):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct)

    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


def make_tuning_loader(manifest_file, manifest_root, backend_obj):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct=10)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))
