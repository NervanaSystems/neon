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
from neon.initializers import Constant, Gaussian
from neon.transforms import Rectlin, Identity, Softmax, PixelwiseSoftmax
from neon.layers import Conv, Affine, BranchNode, Tree, Dropout, RoiPooling
from neon.models import Model
from neon.data.dataloader_transformers import BGRMeanSubtract, TypeCast
from neon.data.aeon_shim import AeonDataLoader
from proposal_layer import ProposalLayer
from objectlocalization import ObjectLocalization
import util
import numpy as np


def build_model(dataset, frcn_rois_per_img, train_pre_nms_N=12000,
                train_post_nms_N=2000, test_pre_nms_N=6000, test_post_nms_N=300, inference=False):
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
    VGG = util.add_vgg_layers()

    # define RPN
    rpn_init = dict(strides=1, init=Gaussian(scale=0.01), bias=Constant(0))
    # these references are passed to the ProposalLayer.
    RPN_3x3 = Conv((3, 3, 512), activation=Rectlin(), padding=1, **rpn_init)
    RPN_1x1_obj = Conv((1, 1, 18), activation=PixelwiseSoftmax(c=2), padding=0, **rpn_init)
    RPN_1x1_bbox = Conv((1, 1, 36), activation=Identity(), padding=0, **rpn_init)

    # inference uses different network settings
    if not inference:
        pre_nms_N = train_pre_nms_N  # default 12000
        post_nms_N = train_post_nms_N   # default 2000
    else:
        pre_nms_N = test_pre_nms_N   # default 6000
        post_nms_N = test_post_nms_N   # default 300

    proposalLayer = ProposalLayer([RPN_1x1_obj, RPN_1x1_bbox],
                                  dataset, pre_nms_N=pre_nms_N,
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


def build_dataloader(config, frcn_rois_per_img):
    """
    Builds the dataloader for the Faster-RCNN network using our aeon loader.
    Besides, the base loader, we add several operations:
    1. Cast the image data into float32 format
    2. Subtract the BGRMean from the image. We used pre-defined means from training
       the VGG network.
    3. Repack the data for Faster-RCNN model. This model has several nested branches, so
       The buffers have to repacked into nested tuples to match the branch leafs. Additionally,
       buffers for training the RCNN portion of the model are also allocated and provisioned
       to the model.

    Arguments:
        config (dict): dataloader configuration
        be (backend): compute backend
        frcn_rois_per_img (int): Number of ROIs to use for training the RCNN portion of the
            model. This is used to create the target buffers for RCNN.

    Returns:
        dataloader object.
    """
    dl = AeonDataLoader(config)
    dl = TypeCast(dl, index=0, dtype=np.float32)  # cast image to float
    dl = BGRMeanSubtract(dl, index=0, pixel_mean=util.FRCN_PIXEL_MEANS)  # subtract means
    dl = ObjectLocalization(dl, frcn_rois_per_img=frcn_rois_per_img)  # repack faster-rcnn
    return dl


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
        detections (array): Array of bounding box detections in a N x 6 array. Each bounding box
                            has the following attributes:[xmin, ymin, xmas, ymax, score, class]

    """

    proposals = proposals.get()[:num_proposals, :]  # remove padded proposals
    boxes = proposals[:, 1:5] / im_scale  # scale back to real image space

    # obtain bounding box corrections from the frcn layers
    scores = outputs[2][0].get()[:, :num_proposals].T
    bbox_deltas = outputs[2][1].get()[:, :num_proposals].T

    # apply bounding box corrections to the region proposals
    pred_boxes = util.bbox_transform_inv(boxes, bbox_deltas)
    pred_boxes = util.clip_boxes(pred_boxes, im_shape)

    detections = []
    # Skip the background class, start processing from class 1
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > thresh)[0]

        # obtain class-specific boxes and scores
        cls_labels = j * np.ones((len(inds), 1))
        cls_scores = scores[inds, j]
        cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis],
                              cls_labels)).astype(np.float32, copy=False)

        # apply non-max suppression
        keep = util.nms(cls_dets, nms_thresh)
        cls_dets = cls_dets[keep, :]

        # store results
        if cls_dets.size != 0:
            detections.append(cls_dets)  # detections[j] = cls_dets

    # guard against no detections
    if len(detections) != 0:
        detections = np.vstack(detections)

    # Limit to max_per_image detections *over all classes*
    if max_per_image is not None:
        if len(detections) > max_per_image:
            # compute threshold needed to keep the top max_per_image
            image_thresh = np.sort(detections[:, -2])[-max_per_image]

            keep = np.where(detections[:, -2] >= image_thresh)[0]
            detections = detections[keep, :]

    # For each bounding box:
    # [xmin, ymin, xmax, ymax, score, class]

    return detections
