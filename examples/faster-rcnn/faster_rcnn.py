from neon.initializers import Gaussian, Constant
from neon.transforms import Rectlin, Identity, Softmax, PixelwiseSoftmax
from neon.layers import Conv, Affine, BranchNode, Tree, Dropout
from neon.models import Model
from roi_pooling import RoiPooling
from proposal_layer import ProposalLayer
from bbox_transform import bbox_transform_inv, clip_boxes
import util
import numpy as np


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
    VGG = util.add_vgg_layers()

    # define RPN
    rpn_init = dict(strides=1, init=Gaussian(scale=0.01), bias=Constant(0))
    # these references are passed to the ProposalLayer.
    RPN_3x3 = Conv((3, 3, 512), activation=Rectlin(), padding=1, **rpn_init)
    RPN_1x1_obj = Conv((1, 1, 18), activation=PixelwiseSoftmax(c=2), padding=0, **rpn_init)
    RPN_1x1_bbox = Conv((1, 1, 36), activation=Identity(), padding=0, **rpn_init)

    # inference uses different network settings
    if inference:
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
    detections = [[] for _ in xrange(num_classes)]

    proposals = proposals.get()[:num_proposals, :]  # remove padded proposals
    boxes = proposals[:, 1:5] / im_scale  # scale back to real image space

    # obtain bounding box corrections from the frcn layers
    scores = outputs[2][0].get()[:, :num_proposals].T
    bbox_deltas = outputs[2][1].get()[:, :num_proposals].T

    # apply bounding box corrections to the region proposals
    pred_boxes = bbox_transform_inv(boxes, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape)

    # Skip the background class, start processing from class 1
    for j in xrange(1, num_classes):
        inds = np.where(scores[:, j] > thresh)[0]

        # obtain class-specific boxes and scores
        cls_scores = scores[inds, j]
        cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

        # apply non-max suppression
        keep = util.nms(cls_dets, nms_thresh)
        cls_dets = cls_dets[keep, :]

        # store results
        detections[j] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image is not None:

        # obtain flattened list of all image scores
        image_scores = np.hstack([detections[j][:, -1]
                                  for j in xrange(1, num_classes)])

        if len(image_scores) > max_per_image:
            # compute threshold needed to keep the top max_per_image
            image_thresh = np.sort(image_scores)[-max_per_image]

            # apply threshold
            for j in xrange(1, num_classes):
                keep = np.where(detections[j][:, -1] >= image_thresh)[0]
                detections[j] = detections[j][keep, :]

    return detections
