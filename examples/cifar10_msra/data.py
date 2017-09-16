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
from neon.util.persist import get_data_cache_or_nothing
from PIL import Image


def ingest_cifar10(out_dir, padded_size, overwrite=False):
    """
    Save CIFAR-10 dataset as PNG files
    """
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
        f.write('manifest_root = {}\n'.format(out_dir))
        f.write('log = {}\n'.format(log_file))
        f.write('epochs = 165\nrng_seed = 0\nverbose = True\neval_freq = 1\n')
        f.write('backend = gpu\nbatch_size = 64\n')

    if (all([os.path.exists(manifest) for manifest in manifest_files]) and not overwrite):
        print("Found existing manfiest files, skipping ingest, use --overwrite to rerun ingest.")
        return manifest_files

    # Now write out image files and manifests
    for setn, manifest in zip(set_names, manifest_files):
        img_path = os.path.join(out_dir, setn)
        if not os.path.isdir(img_path):
            os.makedirs(img_path)

        records = [('@FILE', 'STRING')]

        for idx, (img, lbl) in enumerate(tqdm(zip(*dataset[setn]))):
            fname = os.path.join(img_path, '{}_{:05d}.png'.format(lbl[0], idx))
            im = np.pad(img.reshape((3, 32, 32)), pad_width, mode='mean')
            im = Image.fromarray(np.uint8(np.transpose(im, axes=[1, 2, 0]).copy()))
            im.save(fname, format='PNG')
            records.append((os.path.relpath(fname, out_dir), lbl[0]))

        np.savetxt(manifest, records, fmt='%s\t%s')

    print("Manifest files written to:\n" + "\n".join(manifest_files))


def common_config(manifest_file, manifest_root, batch_size, subset_pct):
    cache_root = get_data_cache_or_nothing('cifar-cache/')

    image_config = {"type": "image",
                    "height": 32,
                    "width": 32}
    label_config = {"type": "label",
                    "binary": False}
    augmentation = {"type": "image",
                    "scale": [0.8, 0.8],
                    "crop_enable": True}

    return {'manifest_filename': manifest_file,
            'manifest_root': manifest_root,
            'batch_size': batch_size,
            'block_size': 5000,
            'subset_fraction': float(subset_pct/100.0),
            'cache_directory': cache_root,
            'etl': [image_config, label_config],
            'augmentation': [augmentation]}


def wrap_dataloader(dl):
    dl = OneHot(dl, index=1, nclasses=10)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = BGRMeanSubtract(dl, index=0)
    return dl


def make_train_loader(manifest_file, manifest_root, backend_obj, subset_pct=100, random_seed=0):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct)

    aeon_config['iteration_mode'] = "ONCE"
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_enable'] = True
    aeon_config['random_seed'] = random_seed
    aeon_config['augmentation'][0]["center"] = False
    aeon_config['augmentation'][0]["flip_enable"] = True

    return wrap_dataloader(AeonDataLoader(aeon_config))


def make_validation_loader(manifest_file, manifest_root, backend_obj, subset_pct=100):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct)
    return wrap_dataloader(AeonDataLoader(aeon_config))


def make_tuning_loader(manifest_file, manifest_root, backend_obj):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct=20)
    aeon_config['shuffle_manifest'] = True
    return wrap_dataloader(AeonDataLoader(aeon_config))


if __name__ == '__main__':
    from configargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--out_dir', required=True, help='Directory to write ingested files')
    parser.add_argument('--padded_size', type=int, default=40, help='Size of image after padding')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite files')
    args = parser.parse_args()

    ingest_cifar10(args.out_dir, args.padded_size, overwrite=args.overwrite)
