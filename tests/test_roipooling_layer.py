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
'''
Test of the ROI pooling layer
'''
from builtins import round
import itertools as itt
import numpy as np
from timeit import default_timer as timeit

from neon import NervanaObject, logger as neon_logger
from neon.backends import gen_backend
from utils import allclose_with_out

spatial_scale = 1.0 / 16


def _fprop_slice_np(h, stride, H, roi_offset):
    """
    slicing in this 1 dimension
    h: is the index on the pooled map (output index)
    stride:
    H: the max of the input map
    roi_offset: how far hstart is from 0
    """
    hstart = int(np.floor(float(h) * stride))
    hend = int(np.ceil(float(h + 1) * stride))

    hstart = min(max(hstart + roi_offset, 0), H)
    hend = min(max(hend + roi_offset, 0), H)

    return slice(hstart, hend), hend - hstart


def pytest_generate_tests(metafunc):

    if 'fargs' in metafunc.fixturenames:
        fargs = []
        bsz_rng = [2]
        roi_num_rng = [2]
        img_fm_c_rng = [2]
        img_fm_h_rng = [62]
        img_fm_w_rng = [62]
        roi_size_rng = [6]
        fargs = itt.product(roi_num_rng, img_fm_c_rng, img_fm_h_rng,
                            img_fm_w_rng, roi_size_rng, bsz_rng)
        metafunc.parametrize('fargs', fargs)


def bprop_roipooling_ref(fm, rois, error, fm_channel, fm_height, fm_width,
                         bsz, rois_per_image, H, W):
    """
    Function to perform a bprop of ROIPooling. It uses a different way from the
    that in CPU backend
    """
    feature_maps = fm.reshape(fm_channel, fm_height, fm_width, bsz)
    rois_per_batch = rois_per_image * bsz
    error_in = error.reshape(fm_channel, H, W, rois_per_batch)

    delta = np.zeros(feature_maps.shape).reshape(fm_channel, fm_height, fm_width, bsz)

    # combine the feature map with ROIs
    for b_id in range(rois_per_batch):
        [idx, xmin, ymin, xmax, ymax] = rois[b_id]
        xmin = int(round(xmin * spatial_scale))
        xmax = int(round(xmax * spatial_scale))
        ymin = int(round(ymin * spatial_scale))
        ymax = int(round(ymax * spatial_scale))
        roi_width = max(xmax - xmin + 1, 1)
        roi_height = max(ymax - ymin + 1, 1)

        stride_h = float(roi_height) / float(H)
        stride_w = float(roi_width) / float(W)

        for h_out in range(H):
            sliceh, lenh = _fprop_slice_np(h_out, stride_h, fm_height, ymin)
            if sliceh.stop <= sliceh.start:
                continue
            for w_out in range(W):
                slicew, lenw = _fprop_slice_np(w_out, stride_w, fm_width, xmin)
                if slicew.stop <= slicew.start:
                    continue
                else:
                    array_I = feature_maps[:, sliceh, slicew, int(idx)].reshape(
                        fm_channel, -1)
                    max_idx = np.argmax(array_I, axis=1)

                    delta_view = delta[:, sliceh, slicew, int(idx)].reshape(
                        fm_channel, -1)
                    delta_view[
                        list(range(fm_channel)), max_idx] += error_in[:, h_out, w_out, b_id]
                    delta[:, sliceh, slicew, int(idx)] = delta_view.reshape(fm_channel,
                                                                            lenh,
                                                                            lenw)

    return delta


def fprop_roipooling_ref(fm, rois, fm_channel, fm_height, fm_width, bsz, rois_per_image, H, W):

    feature_maps = fm.reshape(fm_channel, fm_height, fm_width, bsz)
    rois_per_batch = rois_per_image * bsz
    outputs = np.zeros((fm_channel, H, W, rois_per_batch))

    # combine the feature map with ROIs
    for b_id in range(rois_per_batch):
        [idx, xmin, ymin, xmax, ymax] = rois[b_id]
        xmin = int(round(xmin * spatial_scale))
        xmax = int(round(xmax * spatial_scale))
        ymin = int(round(ymin * spatial_scale))
        ymax = int(round(ymax * spatial_scale))
        roi_width = max(xmax - xmin + 1, 1)
        roi_height = max(ymax - ymin + 1, 1)

        stride_h = float(roi_height) / H
        stride_w = float(roi_width) / W

        for h_out in range(H):
            sliceh, _ = _fprop_slice_np(h_out, stride_h, fm_height, ymin)
            if sliceh.stop <= sliceh.start:
                continue
            for w_out in range(W):
                slicew, _ = _fprop_slice_np(w_out, stride_w, fm_width, xmin)
                if slicew.stop <= slicew.start:
                    continue
                else:
                    array_I = feature_maps[:, sliceh, slicew, int(idx)].reshape(
                        fm_channel, -1)
                    outputs[:, h_out, w_out, b_id] = np.max(array_I, axis=1)

    return outputs.reshape(-1, rois_per_batch)


def test_roipooling_fprop_random(backend_default, fargs):

    rois_per_image, img_fm_c, img_fm_h, img_fm_w, roi_size, bsz = fargs

    # generate a random feature map and some random ROIs
    feature_maps = np.random.random(
        (img_fm_c, img_fm_h, img_fm_w, bsz)).reshape(-1, bsz)
    rois_per_batch = rois_per_image * bsz

    rois_idx = np.vstack([i * np.ones((rois_per_image, 1)) for i in range(bsz)])
    rois = np.random.random((rois_per_batch, 4)) * min(img_fm_h, img_fm_w)

    rois = np.zeros((rois_per_batch, 4))
    rois[:, 0] = np.random.random((rois_per_batch,)) * 10 / spatial_scale
    rois[:, 1] = np.random.random((rois_per_batch,)) * 25 / spatial_scale
    rois[:, 2] = (
        np.random.random((rois_per_batch,)) * 27 + (img_fm_w - 27)) / spatial_scale
    rois[:, 3] = (
        np.random.random((rois_per_batch,)) * 12 + (img_fm_h - 12)) / spatial_scale

    rois = np.hstack((rois_idx, rois))

    # run the numpy roi fprop (function inside this test script)
    outputs_np = fprop_roipooling_ref(feature_maps, rois,
                                      img_fm_c, img_fm_h, img_fm_w,
                                      bsz, rois_per_image, roi_size, roi_size)

    # call backend roipooling kernel
    NervanaObject.be.bsz = bsz
    be = NervanaObject.be
    input_dev = be.array(feature_maps)
    rois_dev = be.array(rois)
    output_shape = (img_fm_c, roi_size, roi_size, rois_per_batch)
    outputs_dev = be.zeros(output_shape)
    # make sure the type being int
    argmax_dev = be.zeros(output_shape, np.int32)

    start_time = timeit()
    be.roipooling_fprop(input_dev, rois_dev, outputs_dev, argmax_dev, rois_per_batch,
                        img_fm_c, img_fm_h, img_fm_w, roi_size, roi_size, spatial_scale)
    neon_logger.display("Nervana backend roipooling fprop (sec): {}".format(timeit() - start_time))

    outputs_be = outputs_dev.get().reshape(-1, rois_per_batch)
    assert allclose_with_out(outputs_np, outputs_be, atol=1e-6, rtol=0)


def test_roipooling_fprop_ref(backend_default, rois=None, inputs=None, outputs_ref=None):

    if rois is None and inputs is None and outputs_ref is None:
        return

    (bsz, img_fm_c, img_fm_h, img_fm_w) = inputs.shape
    (rois_per_batch, _, roi_size, _) = outputs_ref.shape
    outputs_ref_in = outputs_ref.reshape(rois_per_batch, -1).T
    rois_per_image = rois_per_batch // bsz
    feature_maps = inputs.reshape(bsz, -1).T.astype(np.float, order='C')

    # run the numpy roi fprop (function inside this test script)
    outputs_np = fprop_roipooling_ref(feature_maps, rois,
                                      img_fm_c, img_fm_h, img_fm_w,
                                      bsz, rois_per_image, roi_size, roi_size)

    assert allclose_with_out(outputs_ref_in, outputs_np, atol=1e-6, rtol=0)

    # call NervanaGPU roipooling kernel
    NervanaObject.be.bsz = bsz
    be = NervanaObject.be
    input_dev = be.array(feature_maps)
    rois_dev = be.array(rois)
    output_shape = (img_fm_c, roi_size, roi_size, rois_per_batch)
    outputs_dev = be.zeros(output_shape, dtype=np.float32)
    # make sure the type being int
    argmax_dev = be.zeros(output_shape, dtype=np.int32)

    start_time = timeit()
    be.roipooling_fprop(input_dev, rois_dev, outputs_dev, argmax_dev, rois_per_batch,
                        img_fm_c, img_fm_h, img_fm_w, roi_size, roi_size, spatial_scale)

    outputs_backend = outputs_dev.get().reshape(-1, rois_per_batch)

    neon_logger.display("Nervana backend roipooling fprop (sec): {}".format(timeit() - start_time))

    assert allclose_with_out(outputs_ref_in, outputs_backend, atol=1e-6, rtol=0)


def test_roipooling_bprop_random(backend_default, fargs):

    rois_per_image, img_fm_c, img_fm_h, img_fm_w, roi_size, bsz = fargs
    rois_per_batch = rois_per_image * bsz
    # generate a random feature map and some random ROIs
    feature_map_size = img_fm_c * img_fm_h * img_fm_w * bsz

    feature_maps = np.array(list(range(feature_map_size))).reshape(
        (img_fm_c, img_fm_h, img_fm_w, bsz))
    input_errors = np.zeros(
        (img_fm_c, roi_size, roi_size, rois_per_batch))

    range_num = roi_size * roi_size
    input_errors[0, :, :, rois_per_batch - 1] = np.array(
        list(range(range_num))).reshape(input_errors[0, :, :, rois_per_batch - 1].shape)

    rois_idx = np.vstack([i * np.ones((rois_per_image, 1)) for i in range(bsz)])
    rois = np.random.random((rois_per_batch, 4)) * min(img_fm_h, img_fm_w)

    # use full frame as ROI
    rois = np.zeros((rois_per_batch, 4))
    rois[:, 0] = np.ones((rois_per_batch,))
    rois[:, 1] = np.ones((rois_per_batch,))
    rois[:, 2] = np.ones((rois_per_batch,)) * img_fm_w / spatial_scale
    rois[:, 3] = np.ones((rois_per_batch,)) * img_fm_w / spatial_scale

    rois = np.hstack((rois_idx, rois))

    # run the numpy roi fprop (function inside this test script)
    outputs_np = bprop_roipooling_ref(feature_maps, rois, input_errors,
                                      img_fm_c, img_fm_h, img_fm_w,
                                      bsz, rois_per_image, roi_size, roi_size)

    # call backend roipooling kernel
    NervanaObject.be.bsz = bsz
    be = NervanaObject.be
    input_dev = be.array(feature_maps)
    rois_dev = be.array(rois)
    output_shape = (img_fm_c, roi_size, roi_size, rois_per_batch)
    outputs_dev = be.zeros(output_shape, dtype=np.float32)
    # make sure the type being int
    argmax_dev = be.zeros(output_shape, dtype=np.int32)
    input_error_dev = be.array(input_errors)
    output_error_dev = be.zeros(feature_maps.shape)

    be.roipooling_fprop(input_dev, rois_dev, outputs_dev, argmax_dev, rois_per_batch,
                        img_fm_c, img_fm_h, img_fm_w, roi_size, roi_size, spatial_scale)
    start_time = timeit()
    be.roipooling_bprop(input_error_dev, rois_dev, output_error_dev, argmax_dev,
                        rois_per_batch, img_fm_c, img_fm_h, img_fm_w, roi_size,
                        roi_size, spatial_scale)
    neon_logger.display("Nervana backend roipooling bprop (sec): {}".format(timeit() - start_time))

    assert output_error_dev.get().reshape(
        img_fm_c, img_fm_h, img_fm_w, bsz)[:, :, :, 0].sum() == 0
    assert output_error_dev.get().reshape(
        img_fm_c, img_fm_h, img_fm_w, bsz)[:, :, :, -1].sum() != 0

    assert output_error_dev.get().sum() == input_errors.sum()

    outputs_be = output_error_dev.get()
    assert allclose_with_out(outputs_np, outputs_be, atol=1e-6, rtol=0)


def test_roipooling_bprop_ref(backend_default, rois=None, inputs=None, outputs_fprop_ref=None,
                              input_errors=None):

    if rois is None and inputs is None and outputs_fprop_ref is None and input_errors is None:
        return

    (bsz, img_fm_c, img_fm_h, img_fm_w) = inputs.shape
    (rois_per_batch, _, roi_size, _) = input_errors.shape

    outputs_fprop_ref_in = outputs_fprop_ref.reshape(rois_per_batch, -1).T
    feature_maps = inputs.reshape(bsz, -1).T.astype(np.float, order='C')
    input_errors_in = input_errors.reshape(
        rois_per_batch, -1).T.astype(np.float, order='C')

    # compare with GPU kernel, need to call fprop first, then bprop
    NervanaObject.be.bsz = bsz
    be = NervanaObject.be
    input_dev = be.array(feature_maps)
    rois_dev = be.array(rois)
    output_shape = (img_fm_c, roi_size, roi_size, rois_per_batch)
    outputs_dev = be.zeros(output_shape, dtype=np.float32)
    # make sure the type being int
    argmax_dev = be.zeros(output_shape, dtype=np.int32)
    input_error_dev = be.array(input_errors_in)
    output_error_dev = be.zeros(outputs_fprop_ref_in.shape)

    be.roipooling_fprop(input_dev, rois_dev, outputs_dev, argmax_dev, rois_per_batch,
                        img_fm_c, img_fm_h, img_fm_w, roi_size, roi_size, spatial_scale)

    outputs_fprop_be = outputs_dev.get().reshape(-1, rois_per_batch)

    assert allclose_with_out(
        outputs_fprop_ref_in, outputs_fprop_be, atol=1e-6, rtol=0)

    start_time = timeit()
    be.roipooling_bprop(input_error_dev, rois_dev, output_error_dev, argmax_dev,
                        rois_per_batch, img_fm_c, img_fm_h, img_fm_w, roi_size,
                        roi_size, spatial_scale)
    neon_logger.display("NervanaGPU roipooling bprop (sec): {}".format(timeit() - start_time))
    outputs_backend = output_error_dev.get()

    assert allclose_with_out(outputs_fprop_ref_in, outputs_backend, atol=1e-6, rtol=0)


if __name__ == '__main__':

    bsz = 2
    be = gen_backend(backend='gpu', batch_size=bsz, compat_mode='caffe')

    # compare using random data
    fargs = (2, 2, 62, 62, 6, bsz)
    test_roipooling_fprop_random(be, fargs)
    test_roipooling_bprop_random(be, fargs)
