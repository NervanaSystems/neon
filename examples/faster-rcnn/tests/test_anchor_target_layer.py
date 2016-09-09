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
from builtins import next

import numpy as np
import itertools as itt
from neon import NervanaObject
from neon.backends import gen_backend

from anchor_target_layer_ref import AnchorTargetLayer
from objectlocalization import PASCAL

MIN_SIZE = 600
MAX_SIZE = 1000


def pytest_generate_tests(metafunc):
    if 'fargs' in metafunc.fixturenames:
        if metafunc.config.option.all:
            _conv_size = [62, 63]
            rpn_rois_per_img = [128, 256, 512]
        else:
            _conv_size = [62]
            rpn_rois_per_img = [256]
        fargs = itt.product(_conv_size, rpn_rois_per_img)
        metafunc.parametrize('fargs', fargs)


def calculate_scale_shape(size):
    im_shape = np.array(size, np.int32)
    im_size_min = np.min(im_shape)
    im_size_max = np.max(im_shape)
    im_scale = float(MIN_SIZE) / float(im_size_min)
    # Prevent the biggest axis from being more than FRCN_MAX_SIZE
    if np.round(im_scale * im_size_max) > MAX_SIZE:
        im_scale = float(MAX_SIZE) / float(im_size_max)
    im_shape = np.round((im_shape * im_scale)).astype(int)
    return im_scale, im_shape


def test_anchor_target_layer(backend_default, fargs):
    _conv_size, rpn_rois_per_img = fargs

    gt_bb = np.array([[262, 210, 323, 338],
                      [164, 263, 252, 371],
                      [240, 193, 294, 298]])

    mock_db = {'gt_bb': gt_bb,
               'gt_classes': np.array([[9], [9], [9]]),
               'img_id': '000005',
               'flipped': False,
               'img_path': '../nervana/data/VOCdevkit/VOC2007/JPEGImages/000005.jpg',
               'img_shape': (500, 375)}

    im_scale, im_shape = calculate_scale_shape(mock_db['img_shape'])

    # setup backend
    NervanaObject.be.bsz = 1
    NervanaObject.be.enable_winograd = 4

    train_set = PASCAL('trainval', '2007', path='../nervana/data', n_mb=1,
                       rpn_rois_per_img=rpn_rois_per_img, frcn_rois_per_img=128,
                       add_flipped=True, shuffle=False, rebuild_cache=False,
                       mock_db=mock_db, conv_size=_conv_size)

    # get the first training point
    X, Y = next(train_set.__iter__())

    label = train_set.roi_db[0]['labels']

    target = AnchorTargetLayer()

    # prepare inputs
    bottom = [0, 1, 2]

    bottom[0] = np.zeros((_conv_size, _conv_size))
    bottom[1] = train_set.roi_db[0]['gt_bb'] * im_scale
    bottom[2] = [im_shape[0], im_shape[1], im_scale]

    # obtain forward pass output
    top = [0, 1, 2, 3]
    target.setup(bottom, top)
    target.RPN_BATCHSIZE = rpn_rois_per_img
    target.forward(bottom, top)
    py_labels, py_bbtargets, py_iw, py_ow = top

    # positive labels should match
    # assert positive labels match since positives (usually) dont get under sampled
    assert np.allclose(np.where(label == 1)[0],
                       np.where(py_labels.flatten() == 1)[0])

    # our bboxes are in 4 * K, whereas reference is in K * 4 order, so reshape
    bb = Y[1][0].get() * Y[1][1].get()

    pybb = py_bbtargets * py_iw
    pybb = pybb.reshape((1, 9, 4, _conv_size, _conv_size)) \
        .transpose(0, 2, 1, 3, 4)
    pybb = pybb.reshape(1, 36, _conv_size, _conv_size) \
        .flatten()
    # bounding box target locations and values must match
    assert np.allclose(np.where(bb != 0)[0], np.where(pybb != 0)[0])
    assert np.allclose(bb[np.where(bb != 0)], pybb[np.where(pybb != 0)])


if __name__ == "__main__":
    be = gen_backend()
    fargs = (62, 256)
    test_anchor_target_layer(be, fargs)
