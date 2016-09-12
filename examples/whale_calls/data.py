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
from zipfile import ZipFile
from glob import glob
from neon.util.persist import ensure_dirs_exist
from neon.data.aeon_shim import AeonDataLoader
from neon.data.dataloader_transformers import OneHot, TypeCast


def get_ingest_file(filename):
    '''
    prepends the environment variable data path after checking that it has been set
    '''
    if os.environ.get('WHALE_DATA_PATH') is None:
        raise RuntimeError("Missing required env variable WHALE_DATA_PATH")

    return os.path.join(os.environ['WHALE_DATA_PATH'], 'whale-extracted', filename)


def ingest_whales(zfilename, train_frac=0.8):
    '''
    save_ingested_whale_files
    '''
    set_names = ['all', 'val', 'train', 'noise', 'test']

    manifests = {s: get_ingest_file(s + '-index.csv') for s in set_names}

    if all([os.path.exists(manifests[sn]) for sn in set_names]):
       return

    out_dir = os.path.join(os.environ['WHALE_DATA_PATH'], 'whale-extracted')

    with ZipFile(zfilename, 'r') as zf:
        zf.extractall(out_dir)

        # create label files
        lbl_files = [os.path.join(out_dir, 'data', lbl + '.txt') for lbl in ('neg', 'pos')]

        np.savetxt(lbl_files[0], [0], fmt='%d')
        np.savetxt(lbl_files[1], [1], fmt='%d')

        input_csv = os.path.join(out_dir, 'data', 'train.csv')
        train_records = np.genfromtxt(input_csv, delimiter=',', skip_header=1, dtype=None)

        pos_list, neg_list = [], []

        for aiff, lbl in train_records:
            record = (os.path.join(out_dir, 'data', 'train', aiff), lbl_files[lbl])
            if lbl == 1:
                pos_list.append(record)
            else:
                neg_list.append(record)

        neg_part, pos_part = int(len(neg_list) * train_frac), int(len(pos_list) * train_frac)

        set_lists = dict()
        set_lists['all'] = neg_list + pos_list
        set_lists['train'] = neg_list[:neg_part] + pos_list[:pos_part]
        set_lists['val'] = neg_list[neg_part:] + pos_list[pos_part:]

        # Use just the files from the non-whale calls to use as a set of noise samples
        set_lists['noise'] = [(a) for a, l in neg_list[:neg_part]]

        # Write out the test files
        set_lists['test'] = glob(os.path.join(out_dir, 'data', 'test', '*.aiff'))

        np.random.seed(0)

        for sn in set_names:
            if sn != 'test':
                np.random.shuffle(set_lists[sn])

            format_str = '%s' if sn in ('noise', 'test') else '%s,%s'
            np.savetxt(manifests[sn], set_lists[sn], fmt=format_str)


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
    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


def make_train_loader(set_name, backend_obj, random_seed=0):
    aeon_config = common_config(set_name, backend_obj.bsz)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    aeon_config['random_seed'] = random_seed

    aeon_config['audio']['noise_index_file'] = get_ingest_file('noise-index.csv')
    aeon_config['audio']['add_noise_probability'] = 0.5
    # aeon_config['audio']['noise_level'] = [0.0, 0.5]

    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


def make_test_loader(set_name, backend_obj):
    aeon_config = common_config(set_name, backend_obj.bsz)
    aeon_config['type'] = 'audio'  # No labels provided
    aeon_config.pop('label', None)
    dl = AeonDataLoader(aeon_config, backend_obj)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


if __name__ == '__main__':
    from configargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--zipfile', required=True, help='path to whale_data.zip')
    args = parser.parse_args()

    ingest_whales(args.zipfile)

