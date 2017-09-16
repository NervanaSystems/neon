# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
import shutil
import subprocess
import json
import zipfile
import logging

from tqdm import tqdm
from PIL import Image
from neon.data.aeon_shim import AeonDataLoader
from neon.data.dataloader_transformers import OneHot, TypeCast, ValueNormalize
from neon.util.persist import get_data_cache_or_nothing

from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below
from future.moves.urllib.request import urlopen  # noqa
logger = logging.getLogger(__name__)
try:
    import lmdb
except ImportError as e:
    logger.exception("Dependency not installed")
    raise(e)

LSUN_URL = "http://lsun.cs.princeton.edu/htbin/"
MAP_SIZE = 1099511627776
MAX_NUM_INGEST_PROC = 100


def lsun_categories(tag):
    """
    Query LSUN_URL and return a list of LSUN categories

    Argument:
        tag (str): version tag, use "latest" for most recent
    """
    f = urlopen(LSUN_URL + 'list.cgi?tag=' + tag)
    return json.loads(f.read())


def download_lsun(lsun_dir, category, dset, tag, overwrite=False):
    """
    Download LSUN data and unpack

    Arguments:
        lsun_dir (str): LSUN data directory
        category (str): LSUN category
        dset (str): dataset, "train", "val", or "test"
        tag (str): version tag, use "latest" for most recent
        overwrite (bool): whether to overwrite existing data
    """
    dfile = 'test_lmdb' if dset == 'test' else '{0}_{1}_lmdb'.format(category, dset)
    dfile = os.path.join(lsun_dir, dfile)
    if not os.path.exists(dfile) or overwrite:
        dfile += '.zip'
        if os.path.exists(dfile):
            os.remove(dfile)
        url = LSUN_URL + 'download.cgi?tag={0}&category={1}&set={2}'.format(tag, category, dset)
        print('Data download might take a long time.')
        print('Downloading {0} {1} set...'.format(category, dset))
        subprocess.call(['curl', url, '-o', dfile])
        print('Extracting {0} {1} set...'.format(category, dset))
        zf = zipfile.ZipFile(dfile, 'r')
        zf.extractall(lsun_dir)
        zf.close()
        print('Deleting {}...'.format(dfile))
        os.remove(dfile)
    else:
        pass  # data already downloaded
    print("LSUN {0} {1} dataset downloaded and unpacked.".format(category, dset))


def ingest_lsun(lsun_dir, category, dset, lbl_map, overwrite=False, png_conv=False):
    """
    Save LSUN dataset as WEBP or PNG files and generate config and log files

    Arguments:
        lsun_dir (str): LSUN data directory
        category (str): LSUN category
        dset (str): dataset, "train", "val", or "test"
        lbl_map (dict(str:int)): maps a category to an integer
        overwrite (bool): whether to overwrite existing data
        png_conv (bool): whether to convert to PNG images
    """
    cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
    log_file = os.path.join(lsun_dir, 'train.log')
    dpath = 'test' if dset == 'test' else '{0}_{1}'.format(category, dset)
    dpath = os.path.join(lsun_dir, dpath)
    manifest_file = '{}_index.csv'.format(dpath)

    with open(cfg_file, 'w') as f:
        f.write('manifest = [{}:{}]\n'.format(dset, manifest_file))
        f.write('manifest_root = {}\n'.format(lsun_dir))
        f.write('log = {}\n'.format(log_file))
        f.write('epochs = 25\nrng_seed = 0\nverbose = True\neval_freq = 0\n')
        f.write('backend = gpu\nbatch_size = 64\n')
    if os.path.exists(manifest_file) and not overwrite:
        print("LSUN {0} {1} dataset ingested.".format(category, dset))
        print("Manifest file is: " + manifest_file)
        return manifest_file
    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    if os.path.exists(manifest_file):
        os.remove(manifest_file)
    os.makedirs(dpath)

    print('Exporting images...')
    env = lmdb.open(dpath+'_lmdb', map_size=MAP_SIZE,
                    max_readers=MAX_NUM_INGEST_PROC, readonly=True)
    count, records = 0, [('@FILE', 'STRING')]
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in tqdm(cursor):
            image_out_path = os.path.join(dpath, key + '.webp')
            with open(image_out_path, 'w') as fp:
                fp.write(val)
            count += 1
            if png_conv:  # in case WEBP is not supported, extra step of conversion to PNG
                image_out_path_ = image_out_path
                image_out_path = os.path.join(dpath, key + '.png')
                im = Image.open(image_out_path_).convert('RGB')
                im.save(image_out_path, 'png')
                os.remove(image_out_path_)
            records.append((os.path.relpath(image_out_path, lsun_dir), lbl_map[category]))
        np.savetxt(manifest_file, records, fmt='%s\t%s')
    print("LSUN {0} {1} dataset ingested.".format(category, dset))
    print("Manifest file is: " + manifest_file)
    return manifest_file


def common_config(manifest_file, manifest_root, batch_size, subset_pct):
    cache_root = get_data_cache_or_nothing('lsun_cache/')

    image_config = {"type": "image",
                    "height": 64,
                    "width": 64}
    label_config = {"type": "label",
                    "binary": False}
    augmentation = {"type": "image",
                    "scale": [1., 1.]}

    return {'manifest_filename': manifest_file,
            'manifest_root': manifest_root,
            'batch_size': batch_size,
            'subset_fraction': float(subset_pct/100.0),
            'block_size': 5000,
            'cache_directory': cache_root,
            'etl': [image_config, label_config],
            'augmentation': [augmentation]}


def wrap_dataloader(dl):
    dl = OneHot(dl, index=1, nclasses=10)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = ValueNormalize(dl, index=0, source_range=[0., 255.], target_range=[-1., 1.])
    return dl


def make_loader(manifest_file, manifest_root, backend_obj, subset_pct=100, random_seed=0):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_enable'] = True
    aeon_config['random_seed'] = random_seed
    aeon_config['augmentation'][0]['center'] = True
    aeon_config['augmentation'][0]['flip_enable'] = False

    return wrap_dataloader(AeonDataLoader(aeon_config))


if __name__ == '__main__':
    from configargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='data path')
    parser.add_argument('-c', '--category', type=str, default="bedroom", help='data category')
    parser.add_argument('-s', '--dset', type=str, default="train", help='train, val or test')
    parser.add_argument('-t', '--tag', type=str, default="latest", help='version tag')
    parser.add_argument('-w', '--overwrite', action='store_true', help='overwrite existing data')
    parser.add_argument('-p', '--png', action='store_true', help='conversion to PNG images')
    args = parser.parse_args()

    assert os.path.exists(args.out_dir), "Output directory does not exist"
    categories = lsun_categories(args.tag)
    assert args.category in categories, "Unrecognized LSUN category: {}".format(args.category)
    # download and unpack LSUN data if not yet done so
    download_lsun(lsun_dir=args.out_dir, category=args.category,
                  dset=args.dset, tag=args.tag, overwrite=args.overwrite)
    # ingest LSUN data for AEON loader if not yet done so
    manifest_file = ingest_lsun(lsun_dir=args.out_dir, category=args.category, dset=args.dset,
                                lbl_map=dict(zip(categories, range(len(categories)))),
                                overwrite=args.overwrite, png_conv=args.png)
