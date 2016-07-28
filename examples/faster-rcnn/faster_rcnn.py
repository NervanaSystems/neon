from neon.initializers import Gaussian, Constant
from neon.transforms import Rectlin, Identity, Softmax, PixelwiseSoftmax
from neon.layers import Conv, Affine, BranchNode, Tree, Dropout
from neon.models import Model
from roi_pooling import RoiPooling
from proposal_layer import ProposalLayer
import util


def build_model(dataset, frcn_rois_per_img, inference=False):
    num_classes = dataset.num_classes

    # Faster-RCNN contains three models: VGG, the Region Proposal Network (RPN),
    # and the Classification Network (ROI-pooling + Fully Connected layers), organized
    # as a tree. Tree has 4 branches:
    #
    # VGG -> b1 -> Conv (3x3) -> b2 -> Conv (1x1) -> CrossEntropyMulti (objectness label)
    #                            b2 -> Conv (1x1) -> SmoothL1Loss (bounding box targets)
    #        b1 -> PropLayer -> ROI -> Affine -> Affine -> b3 -> Affine -> CrossEntropyMulti
    #                                                      b3 -> Affine -> SmoothL1Loss
    #

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
