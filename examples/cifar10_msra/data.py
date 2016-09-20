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
import os
from tqdm import tqdm
from neon.data import CIFAR10
from neon.data.aeon_shim import AeonDataLoader
from neon.data.dataloader_transformers import OneHot, TypeCast, BGRMeanSubtract
from neon.util.persist import get_data_cache_or_nothing, ensure_dirs_exist
from PIL import Image


def ingest_cifar10(out_dir, padded_size, overwrite=False):
    '''
    Save CIFAR-10 dataset as PNG files
    '''
    dataset = dict()
    cifar10 = CIFAR10(path=out_dir, normalize=False)
    dataset['train'], dataset['val'], _ = cifar10.load_data()
    pad_size = (padded_size - 32) // 2 if padded_size > 32 else 0
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))

    set_names = ('train', 'val')
    manifest_files = [os.path.join(out_dir, setn + '-index.csv') for setn in set_names]

    cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
    log_file = os.path.join(out_dir, 'train.log')
    manifest_list_cfg = ', '.join([k+':'+v for k, v in zip(set_names, manifest_files)])

    with open(cfg_file, 'w') as f:
        f.write('manifest = [{}]\n'.format(manifest_list_cfg))
        f.write('log = {}\n'.format(log_file))
        f.write('epochs = 165\nrng_seed = 0\nverbose = True\neval_freq = 1\n')
        f.write('backend = gpu\nbatch_size = 64\n')

    if (all([os.path.exists(manifest) for manifest in manifest_files]) and not overwrite):
        return manifest_files

    # Write out label files and setup directory structure
    lbl_paths, img_paths = dict(), dict(train=dict(), val=dict())
    for lbl in range(10):
        lbl_paths[lbl] = ensure_dirs_exist(os.path.join(out_dir, 'labels', str(lbl) + '.txt'))
        np.savetxt(lbl_paths[lbl], [lbl], fmt='%d')
        for setn in ('train', 'val'):
            img_paths[setn][lbl] = ensure_dirs_exist(os.path.join(out_dir, setn, str(lbl) + '/'))

    # Now write out image files and manifests
    for setn, manifest in zip(set_names, manifest_files):
        records = []
        for idx, (img, lbl) in enumerate(tqdm(zip(*dataset[setn]))):
            img_path = os.path.join(img_paths[setn][lbl[0]], str(idx) + '.png')
            im = np.pad(img.reshape((3, 32, 32)), pad_width, mode='mean')
            im = Image.fromarray(np.uint8(np.transpose(im, axes=[1, 2, 0]).copy()))
            im.save(img_path, format='PNG')
            records.append((img_path, lbl_paths[lbl[0]]))
        np.savetxt(manifest, records, fmt='%s,%s')

    return manifest_files


def common_config(manifest_file, batch_size, subset_pct):
    cache_root = get_data_cache_or_nothing('cifar-cache/')

    return {
               'manifest_filename': manifest_file,
               'minibatch_size': batch_size,
               'subset_fraction': float(subset_pct/100.0),
               'macrobatch_size': 5000,
               'type': 'image,label',
               'cache_directory': cache_root,
               'image': {'height': 32,
                         'width': 32,
                         'scale': [0.8, 0.8]},
               'label': {'binary': False}
            }


def wrap_dataloader(dl):
    dl = OneHot(dl, index=1, nclasses=10)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = BGRMeanSubtract(dl, index=0)
    return dl


def make_train_loader(manifest_file, backend_obj, subset_pct=100, random_seed=0):
    aeon_config = common_config(manifest_file, backend_obj.bsz, subset_pct)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    aeon_config['random_seed'] = random_seed
    aeon_config['image']['center'] = False
    aeon_config['image']['flip_enable'] = True

    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


def make_validation_loader(manifest_file, backend_obj, subset_pct=100):
    aeon_config = common_config(manifest_file, backend_obj.bsz, subset_pct)
    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


def make_tuning_loader(manifest_file, backend_obj):
    aeon_config = common_config(manifest_file, backend_obj.bsz, subset_pct=20)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))


if __name__ == '__main__':
    from configargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--out_dir', required=True, help='path to extract files into')
    parser.add_argument('--input_dir', default=None, help='unused argument')
    parser.add_argument('--padded_size', type=int, default=40,
                        help='Size of image after padding (each side)')
    args = parser.parse_args()

    generated_files = ingest_cifar10(args.out_dir, args.padded_size)

    print("Manifest files written to:\n" + "\n".join(generated_files))
