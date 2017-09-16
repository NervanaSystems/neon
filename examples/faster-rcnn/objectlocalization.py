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
Defines PASCAL_VOC datatset handling.
"""
import numpy as np
from neon.data.dataloader_transformers import DataLoaderTransformer
from neon.util.persist import get_data_cache_or_nothing


class ObjectLocalization(DataLoaderTransformer):
    """
    Transforms the dataloader inputs for provisioning to the Faster-RCNN model.
    Provided dataloader must be of the (image, localization_rcnn) type.

    To support the Faster-RCNN model, this transformer:
    1. allocates buffers to start the targets for the classification layers. These
       buffers are populated during fprop by part of the network.
    2. stores metadata for each image that are populated with each minibatch.
    3. provides functions for the Faster-RCNN network to access the above target
       and metdata buffers.
    4. Repacks the buffers provided by the dataloader for the tree structure of the
       Faster-RCNN model.

    Args:
        dataloader (DataLoader): dataloader to transform
        frcn_rois_per_img (int): how many rois to sample to train frcnn
    """

    def __init__(self, dataloader, frcn_rois_per_img, *args, **kwargs):
        super(ObjectLocalization, self).__init__(dataloader, None, *args, **kwargs)

        # obtain settings from the dataloader config
        config = dataloader.config
        self.CLASSES = config['etl'][1]['class_names']
        self.conv_scale = config['etl'][1]['scaling_factor']
        self.conv_height = int(np.floor(config['etl'][0]['height'] *
                                        self.conv_scale))
        self.conv_width = int(np.floor(config['etl'][0]['width'] *
                                       self.conv_scale))
        self.rpn_rois_per_img = config['etl'][1]['rois_per_image']

        self.num_classes = len(self.CLASSES)
        self.frcn_rois_per_img = frcn_rois_per_img

        # create placeholders for data needed by ProposalLayer
        self.im_shape = None  # image shape
        self.gt_boxes = None  # ground truth boxes
        self.num_gt_boxes = None  # number of gt boxes
        self.gt_classes = None  # class label for each gt box
        self.im_scale = None  # image scale
        self.difficult = None  # whether the gt boxes are marked difficult

        # allocate buffers
        self.allocate()

    def allocate(self):

        # Data needed for training the RPN are provided by the DataLoader

        # For training, Fast-RCNN needs:
        # 1. class labels
        # 2. bbox targets
        #
        # The above are computed during fprop by the ProposalLayer,
        # so here we create the buffers to pass that to layer.
        self.dev_y_frcn_labels = self.be.zeros(
            (self.num_classes, self.frcn_rois_per_img), dtype=np.int32)
        self.dev_y_frcn_labels_mask = self.be.zeros(
            (self.num_classes, self.frcn_rois_per_img), dtype=np.int32)
        self.dev_y_frcn_bbtargets = self.be.zeros(
            (self.num_classes * 4, self.frcn_rois_per_img), dtype=np.float32)
        self.dev_y_frcn_bbmask = self.be.zeros(
            (self.num_classes * 4, self.frcn_rois_per_img), dtype=np.float32)

    def get_target_buffers(self):

        return ((self.dev_y_frcn_labels, self.dev_y_frcn_labels_mask),
                (self.dev_y_frcn_bbtargets, self.dev_y_frcn_bbmask))

    def get_metadata_buffers(self):

        return (self.im_shape, self.im_scale, self.gt_boxes, self.gt_classes,
                self.num_gt_boxes, self.difficult)

    def transform(self, t):
        # unpack buffers
        (img, bbtargets, bbtargets_mask, labels, labels_mask, self.im_shape, gt_boxes,
         self.num_gt_boxes, self.gt_classes, self.im_scale, self.difficult) = t

        self.gt_boxes = gt_boxes.reshape((-1, 4))

        # repack for faster-rcnn model
        X = img
        Y = ((labels, labels_mask),
             (bbtargets, bbtargets_mask),
             ((self.dev_y_frcn_labels, self.dev_y_frcn_labels_mask),
              (self.dev_y_frcn_bbtargets, self.dev_y_frcn_bbmask)))

        return (X, Y)


def PASCALVOC(manifest_file, manifest_root, rois_per_img=256,
              height=1000, width=1000, inference=False):
    """
    Returns the aeon dataloader configuration for PASCAL VOC dataset.
    """

    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    # if inference, we do not shuffle or flip
    do_transforms = not inference

    image_config = {"type": "image",
                    "height": height,
                    "width": width}

    localization_config = {"type": "localization_rcnn",
                           "height": height,
                           "width": width,
                           "rois_per_image": rois_per_img,
                           "class_names": CLASSES,
                           "scaling_factor": 1. / 16}

    augmentation = {"type": "image",
                    "fixed_aspect_ratio": True,
                    "flip_enable": do_transforms,
                    "crop_enable": False}

    return {'manifest_filename': manifest_file,
            'manifest_root': manifest_root,
            'etl': [image_config, localization_config],
            'cache_directory': get_data_cache_or_nothing(subdir='pascalvoc_cache'),
            'shuffle_enable': do_transforms,
            'shuffle_manifest': do_transforms,
            'batch_size': 1,
            'block_size': 100,
            'augmentation': [augmentation]}


def KITTI(manifest_file, manifest_root, rois_per_img=256,
          height=375, width=1242, inference=False):
    """
    Returns the aeon dataloader configuration for KITTI dataset.
    """

    CLASSES = ('__background__',
               'Car', 'Van', 'Truck',
               'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
               'Misc', 'DontCare')

    # if inference, we do not shuffle or flip
    do_transforms = not inference

    image_config = {"type": "image",
                    "height": height,
                    "width": width}

    localization_config = {"type": "localization_rcnn",
                           "rois_per_image": rois_per_img,
                           "height": height,
                           "width": width,
                           "class_names": CLASSES,
                           "scaling_factor": 1. / 16}

    augmentation = {"type": "image",
                    "fixed_aspect_ratio": True,
                    "flip_enable": do_transforms,
                    "crop_enable": False}

    return {'manifest_filename': manifest_file,
            'manifest_root': manifest_root,
            'etl': [image_config, localization_config],
            'cache_directory': get_data_cache_or_nothing(subdir='kitti_cache'),
            'shuffle_enable': do_transforms,
            'shuffle_manifest': do_transforms,
            'batch_size': 1,
            'block_size': 100,
            'augmentation': [augmentation]}
