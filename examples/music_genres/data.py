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
import tarfile
import numpy as np
from neon.util.persist import ensure_dirs_exist, get_data_cache_or_nothing
from neon.data.aeon_shim import AeonDataLoader
from neon.data.dataloader_transformers import OneHot, TypeCast
from tqdm import tqdm


def ingest_genre_data(input_dir, out_dir, train_percent=80):
    in_tar = os.path.join(input_dir, 'genres.tar.gz')
    ingest_dir = os.path.join(out_dir, 'music-extracted')

    train_idx = os.path.join(ingest_dir, 'train-index.csv')
    val_idx = os.path.join(ingest_dir, 'val-index.csv')

    cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
    log_file = os.path.join(out_dir, 'train.log')
    manifest_list_cfg = ', '.join(['train:' + train_idx, 'val:' + val_idx])

    with open(cfg_file, 'w') as f:
        f.write('manifest = [{}]\n'.format(manifest_list_cfg))
        f.write('log = {}\n'.format(log_file))
        f.write('epochs = 16\nrng_seed = 0\nverbose = True\neval_freq = 1\n')

    if os.path.exists(train_idx) and os.path.exists(val_idx):
        return train_idx, val_idx

    assert os.path.exists(in_tar)

    # convert files as we extract
    snd_files = dict()
    with tarfile.open(in_tar, 'r') as tf_archive:
        infiles = [elem for elem in tf_archive.getmembers()]
        for tf_elem in tqdm(infiles):
            dirpath = tf_elem.name.split('/')
            outpath = os.path.join(ingest_dir, *dirpath[1:])
            if tf_elem.isdir() and not os.path.exists(outpath):
                os.makedirs(outpath)
            elif tf_elem.isfile():
                snd_files.setdefault(dirpath[1], []).append(outpath)
                with open(outpath, 'wb') as of:
                    of.write(tf_archive.extractfile(tf_elem).read())

    # make target files
    lbl_files = dict()
    for lbl, cls_name in enumerate(sorted(snd_files.keys())):
        lbl_files[cls_name] = ensure_dirs_exist(
            os.path.join(ingest_dir, 'labels', str(cls_name) + '.txt'))
        np.savetxt(lbl_files[cls_name], [lbl], fmt='%d')

    np.random.seed(0)
    train_records, val_records = [], []

    for cls_name in snd_files.keys():
        files, label = snd_files[cls_name], lbl_files[cls_name]
        np.random.shuffle(files)
        train_count = (len(files) * train_percent) // 100
        for filename in files[:train_count]:
            train_records.append((filename, label))
        for filename in files[train_count:]:
            val_records.append((filename, label))

    np.savetxt(train_idx, train_records, fmt='%s,%s')
    np.savetxt(val_idx, val_records, fmt='%s,%s')

    return train_idx, val_idx


def wrap_dataloader(dl):
    dl = OneHot(dl, index=1, nclasses=10)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


def common_config(manifest_file, batch_size):
    cache_root = get_data_cache_or_nothing('music-cache/')

    return {
               'manifest_filename': manifest_file,
               'minibatch_size': batch_size,
               'macrobatch_size': batch_size * 12,
               'type': 'audio,label',
               'cache_directory': cache_root,
               'audio': {'sample_freq_hz': 22050,
                         'max_duration': '31 seconds',
                         'frame_length': '16 milliseconds',
                         'frame_stride': '247 samples'},
               'label': {'binary': False}
            }


def make_val_loader(manifest_file, backend_obj):
    aeon_config = common_config(manifest_file, backend_obj.bsz)
    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


def make_train_loader(manifest_file, backend_obj, random_seed=0):
    aeon_config = common_config(manifest_file, backend_obj.bsz)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    aeon_config['random_seed'] = random_seed

    # aeon_config['audio']['time_scale_fraction'] = [0.95, 1.05]

    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


if __name__ == '__main__':
    from configargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='path to genres.tar.gz')
    parser.add_argument('--out_dir', required=True, help='path to extract files into')
    args = parser.parse_args()

    generated_files = ingest_genre_data(args.input_dir, args.out_dir)

    print("Manifest files written to:\n" + "\n".join(generated_files))
