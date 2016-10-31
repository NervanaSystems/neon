#!/usr/bin/env python
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

from __future__ import division

import numpy as np
from neon.backends import gen_backend
import itertools as itt

from anchor_target_layer_ref import AnchorTargetLayer
from objectlocalization import ObjectLocalization, PASCALVOC
from neon.data.dataloader_transformers import TypeCast
from aeon import DataLoader
import os

MIN_SIZE = 600
MAX_SIZE = 1000


def pytest_generate_tests(metafunc):
    if 'fargs' in metafunc.fixturenames:
        height = [1000]
        width = [1000]
        fargs = itt.product(height, width)
        metafunc.parametrize('fargs', fargs)


def test_anchor_target_layer(backend_default, fargs):
    (height, width) = fargs

    manifest_path = os.environ['PASCAL_MANIFEST_PATH']
    assert manifest_path is not None, "Please set the PASCAL_MANIFEST_PATH variable."

    manifest_root = os.environ['PASCAL_MANIFEST_ROOT']
    assert manifest_root is not None, "Please set the PASCAL_MANIFEST_ROOT variable."

    config = PASCALVOC(manifest_path, manifest_root, cache_dir='',
                       height=height, width=width, inference=False)
    config['subset_fraction'] = 0.1

    dl = DataLoader(config, backend_default)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    train_set = ObjectLocalization(dl, frcn_rois_per_img=128)

    for idx, (X, Y) in enumerate(train_set):
        reference_test(train_set, X, Y)


def reference_test(dataloader, X, Y):
        bbtargets_mask = Y[1][1]

        target = AnchorTargetLayer()
        im_shape = dataloader.im_shape.get()
        im_scale = dataloader.im_scale.get()[0][0]
        num_gt_boxes = dataloader.num_gt_boxes.get()[0][0]

        # prepare inputs
        bottom = [0, 1, 2]

        bottom[0] = np.zeros((dataloader.conv_height, dataloader.conv_width))
        bottom[1] = dataloader.gt_boxes.get()[:num_gt_boxes]
        bottom[2] = [im_shape[0], im_shape[1], im_scale]

        # obtain forward pass output
        top = [0, 1, 2, 3]
        target.setup(bottom, top)
        target.forward(bottom, top)
        py_labels, py_bbtargets, py_iw, py_ow = top

        label = bbtargets_mask.get().reshape((4, -1))[0, :]

        # positive labels should match
        if np.sum(label == 1) < 128:

            # assert positive labels match since positives (usually) dont get under sampled
            assert np.allclose(np.where(label == 1)[0],
                               np.where(py_labels.flatten() == 1)[0])

            # our bboxes are in 4 * K, whereas reference is in K * 4 order, so reshape
            bb = Y[1][0].get() * Y[1][1].get()

            pybb = py_bbtargets * py_iw
            pybb = pybb.reshape((1, 9, 4, dataloader.conv_height, dataloader.conv_width)) \
                .transpose(0, 2, 1, 3, 4)
            pybb = pybb.reshape(1, 36, dataloader.conv_height, dataloader.conv_width) \
                .flatten()
            # bounding box target locations and values must match
            assert np.allclose(np.where(bb != 0)[0], np.where(pybb != 0)[0], atol=0.001)
            assert np.allclose(bb[np.where(bb != 0)], pybb[np.where(pybb != 0)], atol=0.001)


if __name__ == "__main__":
    be = gen_backend()
    fargs = (1000, 1000)  # height, width
    test_anchor_target_layer(be, fargs)
