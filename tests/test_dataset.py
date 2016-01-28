# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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

import logging
import numpy as np
import os
import pickle

from neon import NervanaObject
from neon.data import ArrayIterator, load_mnist
from neon.data.text import Text
from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.data import PASCALVOC
from neon.data.datasets import Dataset

logging.basicConfig(level=20)
logger = logging.getLogger()


def test_dataset(backend_default, data):
    (X_train, y_train), (X_test, y_test), nclass = load_mnist(path=data)

    train_set = ArrayIterator(X_train, y_train, nclass=nclass)
    train_set.be = NervanaObject.be

    for i in range(2):
        for X_batch, y_batch in train_set:
            print X_batch.shape, y_batch.shape
        train_set.index = 0


def test_text(backend_default):
    text_data = (
        'Lorem ipsum dolor sit amet, consectetur adipisicing elit, '
        'sed do eiusmod tempor incididunt ut labore et dolore magna '
        'aliqua. Ut enim ad minim veniam, quis nostrud exercitation '
        'ullamco laboris nisi ut aliquip ex ea commodo consequat. '
        'Duis aute irure dolor in reprehenderit in voluptate velit '
        'esse cillum dolore eu fugiat nulla pariatur. Excepteur sint '
        'occaecat cupidatat non proident, sunt in culpa qui officia '
        'deserunt mollit anim id est laborum.'
    )
    data_path = 'tmp_test_text_data'
    with open(data_path, 'w') as f:
        f.write(text_data)

    NervanaObject.be.bsz = 4
    time_steps = 6
    valid_split = 0.2

    # load data and parse on character-level
    train_path, valid_path = Text.create_valid_file(
        data_path, valid_split=valid_split)
    train_set = Text(time_steps, train_path)
    valid_set = Text(time_steps, valid_path, vocab=train_set.vocab)

    train_set.be = NervanaObject.be
    bsz = train_set.be.bsz

    for i, (X_batch, y_batch) in enumerate(train_set):
        if i > 2:
            break
        chars = [train_set.index_to_token[x]
                 for x in np.argmax(X_batch.get(), axis=0).tolist()]
        # First sent of first batch will be contiguous with first sent of next
        # batch
        for batch in range(bsz):
            sent = ''.join(chars[batch::bsz])
            start = i*time_steps + batch * time_steps * train_set.nbatches
            sent_ref = text_data[start:start+time_steps]
            assert sent == sent_ref

    valid_start = int(len(text_data) * (1 - valid_split))
    for i, (X_batch, y_batch) in enumerate(valid_set):
        if i > 2:
            break
        chars = [train_set.index_to_token[x]
                 for x in np.argmax(X_batch.get(), axis=0).tolist()]
        for batch in range(bsz):
            sent = ''.join(chars[batch::bsz])
            start = i*time_steps + batch * time_steps * \
                valid_set.nbatches + valid_start
            sent_ref = text_data[start:start+time_steps]
            assert sent == sent_ref

    os.remove(data_path)
    os.remove(train_path)
    os.remove(valid_path)


def test_pascalvoc(backend_default, data):
    url = 'https://s3-us-west-1.amazonaws.com/nervana-pascal-voc-data'
    filename = 'pascal_data_ref.pkl'
    size = 423982870

    workdir, filepath = Dataset._valid_path_append(data, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    with open(filepath, 'rb') as handle:
        neon_data_ref = pickle.loads(handle.read())

    n_mb = neon_data_ref['n_mb']
    img_per_batch = neon_data_ref['img_per_batch']
    rois_per_img = neon_data_ref['rois_per_img']
    dataset = neon_data_ref['dataset']
    year = neon_data_ref['year']
    output_type = neon_data_ref['output_type']
    rois_random_sample = neon_data_ref['rois_random_sample']
    shuffle = neon_data_ref['shuffle']

    train_set = PASCALVOC(dataset, year, path=data, output_type=output_type, n_mb=n_mb,
                          img_per_batch=img_per_batch, rois_per_img=rois_per_img,
                          rois_random_sample=rois_random_sample, shuffle=shuffle)

    # X_batch[0]: image - (3e6, 2)
    # X_batch[1]: ROIs - (128, 5)
    # Y_batch[0]: labels - (21, 128)
    # Y_batch[1][0]: bbtarget - (84, 128)
    # Y_batch[1][1]: bb mask - (84, 128)
    for mb_i, (X_batch, y_batch) in enumerate(train_set):

        image_neon = X_batch[0].get()
        image_ref = neon_data_ref['X_batch_img'][mb_i]

        rois_neon = X_batch[1].get()
        rois_ref = neon_data_ref['X_batch_rois'][mb_i]

        label_neon = y_batch[0].get()
        label_ref = neon_data_ref['y_batch_label'][mb_i]

        bbtarget_neon = y_batch[1][0].get()
        bbtarget_ref = neon_data_ref['y_batch_bbtarget'][mb_i]

        mask_neon = y_batch[1][1].get()
        mask_ref = neon_data_ref['y_batch_mask'][mb_i]

        assert np.allclose(image_neon, image_ref, atol=1e-5, rtol=0)
        assert np.allclose(rois_neon, rois_ref, atol=1e-5, rtol=0)
        assert np.allclose(label_neon, label_ref, atol=1e-5, rtol=0)
        assert np.allclose(bbtarget_neon, bbtarget_ref, atol=1e-5, rtol=0)
        assert np.allclose(mask_neon, mask_ref, atol=1e-5, rtol=0)


if __name__ == '__main__':

    # setup backend
    parser = NeonArgparser(__doc__)
    args = parser.parse_args(gen_be=False)
    args.batch_size = 128

    # setup backend
    be = gen_backend(**extract_valid_args(args, gen_backend))
    test_pascalvoc(be, args.data_dir)
