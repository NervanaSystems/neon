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
"""
Utility functions for Faster-RCNN example and demo.

Reference:
    "Faster R-CNN"
    https://arxiv.org/pdf/1506.01497
    https://github.com/rbgirshick/py-faster-rcnn

"""
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
import numpy as np
import os
import pickle

from neon.initializers import Constant, Xavier, Gaussian
from neon.transforms import Rectlin, Identity, Softmax, PixelwiseSoftmax
from neon.layers import Conv, Pooling, Affine, BranchNode, Tree, Dropout
from neon.models import Model
from neon.util.persist import load_obj
from neon.data.datasets import Dataset
from voc_eval import voc_eval

from roi_pooling import RoiPooling
from proposal_layer import ProposalLayer


def add_vgg_layers():

    # setup layers
    init1_vgg = Xavier(local=True)
    relu = Rectlin()

    conv_params = {'strides': 1,
                   'padding': 1,
                   'init': init1_vgg,
                   'bias': Constant(0),
                   'activation': relu}

    # Set up the model layers
    vgg_layers = []

    # set up 3x3 conv stacks with different feature map sizes
    vgg_layers.append(Conv((3, 3, 64), name="skip", **conv_params))
    vgg_layers.append(Conv((3, 3, 64), name="skip", **conv_params))
    vgg_layers.append(Pooling(2, strides=2))
    vgg_layers.append(Conv((3, 3, 128), name="skip", **conv_params))
    vgg_layers.append(Conv((3, 3, 128), name="skip", **conv_params))
    vgg_layers.append(Pooling(2, strides=2))
    vgg_layers.append(Conv((3, 3, 256), **conv_params))
    vgg_layers.append(Conv((3, 3, 256), **conv_params))
    vgg_layers.append(Conv((3, 3, 256), **conv_params))
    vgg_layers.append(Pooling(2, strides=2))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Pooling(2, strides=2))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    vgg_layers.append(Conv((3, 3, 512), **conv_params))
    # not used after this layer
    # vgg_layers.append(Pooling(2, strides=2))
    # vgg_layers.append(Affine(nout=4096, init=initfc, bias=Constant(0), activation=relu))
    # vgg_layers.append(Dropout(keep=0.5))
    # vgg_layers.append(Affine(nout=4096, init=initfc, bias=Constant(0), activation=relu))
    # vgg_layers.append(Dropout(keep=0.5))
    # vgg_layers.append(Affine(nout=1000, init=initfc, bias=Constant(0), activation=Softmax()))

    return vgg_layers


def scale_bbreg_weights(model, means, stds, num_classes):
    means = np.array(num_classes * means)
    stds = np.array(num_classes * stds)

    means_be = model.be.array(means)
    stds_be = model.be.array(stds)
    model.layers.layers[2].layers[1].layers[1].layers[-3].W[:] = \
        model.layers.layers[2].layers[1].layers[1].layers[-3].W * stds_be
    model.layers.layers[2].layers[1].layers[1].layers[-2].W[:] = \
        model.layers.layers[2].layers[1].layers[1].layers[-2].W * stds_be + means_be
    return model


def load_vgg_all_weights(model, path):
    # load a pre-trained VGG16 from Neon model zoo to the local
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/'
    filename = 'VGG_D.p'
    size = 554227541

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    print('De-serializing the pre-trained VGG16 model...')
    pdict = load_obj(filepath)

    param_layers = [l for l in model.layers.layers[0].layers]
    param_dict_list = pdict['model']['config']['layers']

    i = 0
    for layer, ps in zip(param_layers, param_dict_list):
        i += 1
        if i == 43:
            break
        layer.load_weights(ps, load_states=True)
        print(layer.name + " <-- " + ps['config']['name'])

    # to load the fc6 and fc7 from caffe into neon fc layers after ROI pooling
    neon_fc_layers = model.layers.layers[2].layers[1].layers[0].layers[2:5] +\
        model.layers.layers[2].layers[1].layers[0].layers[6:9]
    vgg_fc_layers = param_dict_list[44:47] + param_dict_list[48:51]

    for layer, ps in zip(neon_fc_layers, vgg_fc_layers):
        layer.load_weights(ps, load_states=True)
        print(layer.name + " <-- " + ps['config']['name'])


def load_vgg_weights(model, path):
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/'
    filename = 'VGG_D_Conv.p'
    size = 169645138

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    print('De-serializing the pre-trained VGG16 model...')
    pdict = load_obj(filepath)

    param_layers = [l for l in model.layers.layers[0].layers]
    param_dict_list = pdict['model']['config']['layers']

    for layer, ps in zip(param_layers, param_dict_list):
        layer.load_weights(ps, load_states=True)
        print(layer.name + " <-- " + ps['config']['name'])


def build_model(dataset, frcn_rois_per_img, inference=False):
    """
    Returns the Faster-RCNN model. For inference, also returns a reference to the
    proposal layer.

    Faster-RCNN contains three modules: VGG, the Region Proposal Network (RPN),
    and the Classification Network (ROI-pooling + Fully Connected layers), organized
    as a tree. Tree has 4 branches:

    VGG -> b1 -> Conv (3x3) -> b2 -> Conv (1x1) -> CrossEntropyMulti (objectness label)
                               b2 -> Conv (1x1) -> SmoothL1Loss (bounding box targets)
           b1 -> PropLayer -> ROI -> Affine -> Affine -> b3 -> Affine -> CrossEntropyMulti
                                                         b3 -> Affine -> SmoothL1Loss

    When the model is constructed for inference, several elements are different:
    - The number of regions to keep before and after non-max suppression is (6000, 300) for
      training and (12000, 2000) for inference.
    - The out_shape of the proposalLayer of the network is equal to post_nms_N (number of rois
      to keep after performaing nms). This is configured by passing the inference flag to the
      proposalLayer constructor.

    Arguments:
        dataset (objectlocalization): Dataset object.
        frcn_rois_per_img (int): Number of ROIs per image considered by the classification network.
        inference (bool): Construct the model for inference. Default is False.

    Returns:
        model (Model): Faster-RCNN model.
        proposalLayer (proposalLayer): Reference to proposalLayer in the model.
                                       Returned only for inference=True.
    """
    num_classes = dataset.num_classes

    # define the branch points
    b1 = BranchNode(name="conv_branch")
    b2 = BranchNode(name="rpn_branch")
    b3 = BranchNode(name="roi_branch")

    # define VGG
    VGG = add_vgg_layers()

    # define RPN
    rpn_init = dict(strides=1, init=Gaussian(scale=0.01), bias=Constant(0))
    # these references are passed to the ProposalLayer.
    RPN_3x3 = Conv((3, 3, 512), activation=Rectlin(), padding=1, **rpn_init)
    RPN_1x1_obj = Conv((1, 1, 18), activation=PixelwiseSoftmax(c=2), padding=0, **rpn_init)
    RPN_1x1_bbox = Conv((1, 1, 36), activation=Identity(), padding=0, **rpn_init)

    # inference uses different network settings
    if not inference:
        pre_nms_N = 12000
        post_nms_N = 2000
    else:
        pre_nms_N = 6000
        post_nms_N = 300

    proposalLayer = ProposalLayer([RPN_1x1_obj, RPN_1x1_bbox],
                                  dataset.get_global_buffers(), pre_nms_N=pre_nms_N,
                                  post_nms_N=post_nms_N, num_rois=frcn_rois_per_img,
                                  inference=inference)

    # define ROI classification network
    ROI = [proposalLayer,
           RoiPooling(HW=(7, 7)),
           Affine(nout=4096, init=Gaussian(scale=0.005),
                  bias=Constant(.1), activation=Rectlin()),
           Dropout(keep=0.5),
           Affine(nout=4096, init=Gaussian(scale=0.005),
                  bias=Constant(.1), activation=Rectlin()),
           Dropout(keep=0.5)]

    ROI_category = Affine(nout=num_classes, init=Gaussian(scale=0.01),
                          bias=Constant(0), activation=Softmax())
    ROI_bbox = Affine(nout=4 * num_classes, init=Gaussian(scale=0.001),
                      bias=Constant(0), activation=Identity())

    # build the model
    # the four branches of the tree mirror the branches listed above
    frcn_tree = Tree([ROI + [b3, ROI_category],
                     [b3, ROI_bbox]
                      ])

    model = Model(layers=Tree([VGG + [b1, RPN_3x3, b2, RPN_1x1_obj],
                               [b2, RPN_1x1_bbox],
                               [b1] + [frcn_tree],
                               ]))

    if inference:
        return (model, proposalLayer)
    else:
        return model


def get_bboxes(outputs, proposals, num_proposals, num_classes,
               im_shape, im_scale, max_per_image=100, thresh=0.001, nms_thresh=0.4):
    """
    Returns bounding boxes for detected objects, organized by class.

    Transforms the proposals from the region proposal network to bounding box predictions
    using the bounding box regressions from the classification network:
    (1) Applying bounding box regressions to the region proposals.
    (2) For each class, take proposed boxes where the corresponding objectness score is greater
        then THRESH.
    (3) Apply non-maximum suppression across classes using NMS_THRESH
    (4) Limit the maximum number of detections over all classes to MAX_PER_IMAGE

    Arguments:
        outputs (list of tensors): Faster-RCNN model outputs
        proposals (Tensor): Proposed boxes from the model's proposalLayer
        num_proposals (int): Number of proposals
        num_classes (int): Number of classes
        im_shape (tuple): Shape of image
        im_scale (float): Scaling factor of image
        max_per_image (int): Maximum number of allowed detections per image. Default is 100.
                             None indicates no enforced maximum.
        thresh (float): Threshold for objectness score. Default is 0.001.
        nms_thresh (float): Threshold for non-maximum suppression. Default is 0.4.

    Returns:
        detections (list): List of bounding box detections, organized by class. Each element
                           contains a numpy array of bounding boxes for detected objects
                           of that class.
    """
    detections = [[] for _ in range(num_classes)]

    proposals = proposals.get()[:num_proposals, :]  # remove padded proposals
    boxes = proposals[:, 1:5] / im_scale  # scale back to real image space

    # obtain bounding box corrections from the frcn layers
    scores = outputs[2][0].get()[:, :num_proposals].T
    bbox_deltas = outputs[2][1].get()[:, :num_proposals].T

    # apply bounding box corrections to the region proposals
    pred_boxes = bbox_transform_inv(boxes, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape)

    # Skip the background class, start processing from class 1
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > thresh)[0]

        # obtain class-specific boxes and scores
        cls_scores = scores[inds, j]
        cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

        # apply non-max suppression
        keep = nms(cls_dets, nms_thresh)
        cls_dets = cls_dets[keep, :]

        # store results
        detections[j] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image is not None:

        # obtain flattened list of all image scores
        image_scores = np.hstack([detections[j][:, -1]
                                  for j in range(1, num_classes)])

        if len(image_scores) > max_per_image:
            # compute threshold needed to keep the top max_per_image
            image_thresh = np.sort(image_scores)[-max_per_image]

            # apply threshold
            for j in range(1, num_classes):
                keep = np.where(detections[j][:, -1] >= image_thresh)[0]
                detections[j] = detections[j][keep, :]

    return detections


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def run_voc_eval(annopath, imagesetfile, year, image_set, classes, output_dir):
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = 'voc_{}_{}_{}.txt'.format(
            year, image_set, cls)
        filepath = os.path.join(output_dir, filename)
        rec, prec, ap = voc_eval(filepath, annopath, imagesetfile, cls,
                                 output_dir, ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
