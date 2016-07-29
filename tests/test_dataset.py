# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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

from neon import NervanaObject
from neon import logger as neon_logger
from neon.data import MNIST
from neon.data.text import Text


def test_dataset(backend_default, data):
    dataset = MNIST(path=data)
    dataset.gen_iterators()
    train_set = dataset.data_dict['train']
    train_set.be = NervanaObject.be

    for i in range(2):
        for X_batch, y_batch in train_set:
            neon_logger.display("Xshape: {}, yshape: {}".format(X_batch.shape, y_batch.shape))
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
            start = i * time_steps + batch * time_steps * train_set.nbatches
            sent_ref = text_data[start:start + time_steps]
            assert sent == sent_ref

    valid_start = int(len(text_data) * (1 - valid_split))
    for i, (X_batch, y_batch) in enumerate(valid_set):
        if i > 2:
            break
        chars = [train_set.index_to_token[x]
                 for x in np.argmax(X_batch.get(), axis=0).tolist()]
        for batch in range(bsz):
            sent = ''.join(chars[batch::bsz])
            start = i * time_steps + batch * time_steps * \
                valid_set.nbatches + valid_start
            sent_ref = text_data[start:start + time_steps]
            assert sent == sent_ref

    os.remove(data_path)
    os.remove(train_path)
    os.remove(valid_path)
