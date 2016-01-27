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

from neon import NervanaObject
from neon.data import ArrayIterator, load_mnist
from neon.data.text import Text
from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.data import PASCALVOC

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


if __name__ == '__main__':

    # setup backend
    parser = NeonArgparser(__doc__)
    args = parser.parse_args(gen_be=False)
    args.batch_size = 128

    # setup backend
    be = gen_backend(**extract_valid_args(args, gen_backend))

    # setup dataset
    n_mb = None
    img_per_batch = 2
    rois_per_img = 64
    train_set = PASCALVOC('trainval', '2007', path=args.data_dir, output_type=0,
                          n_mb=n_mb, img_per_batch=img_per_batch,
                          rois_per_img=rois_per_img, rois_random_sample=False)

    # load reference data
    import pickle
    with open('/home/users/yinyin/nervana/data/frcn_pascal_data_30.pkl', 'rb') as handle:
        blobs_caffe = pickle.loads(handle.read())

    # X_batch[0]: image - (3e6, 2)
    # X_batch[1]: ROIs - (128, 5)
    # Y_batch[0]: labels - (21, 128)
    # Y_batch[1][0]: bbtarget - (84, 128)
    # Y_batch[1][1]: bb mask - (84, 128)
    for mb_i, (X_batch, y_batch) in enumerate(train_set):
        if mb_i > 30-1:
            break
        else:
            print mb_i

        # image won't match
        image_caffe = blobs_caffe[mb_i]['data'].transpose(1, 2, 3, 0)
        image_neon = X_batch[0].get().reshape(3, 1000, 1000, 2)
        image_neon = image_neon[:, :image_caffe.shape[1], :image_caffe.shape[2], :]

        # labels: caffe uses indices, neon uses one-hot
        label_caffe = blobs_caffe[mb_i]['labels']
        label_neon = y_batch[0].get().argmax(0)

        print np.abs((image_neon-image_caffe)).max()
        print np.abs((image_neon-image_caffe)).mean()
        assert np.allclose(label_neon, label_caffe, atol=1e-5)
        assert np.allclose(
            X_batch[1].get(), blobs_caffe[mb_i]['rois'], atol=1e-5)
        assert np.allclose(
            y_batch[1][0].T.get(), blobs_caffe[mb_i]['bbox_targets'], atol=1e-5)
        assert np.allclose(
            y_batch[1][1].T.get(), blobs_caffe[mb_i]['bbox_loss_weights'], atol=1e-5)
