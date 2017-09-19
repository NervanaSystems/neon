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
import numpy as np
from neon.data.dataloader_transformers import DataLoaderTransformer
from neon.data.dataloader_transformers import TypeCast, BGRMeanSubtract
from aeon import DataLoader
from neon.data.dataloaderadapter import DataLoaderAdapter


class ObjectLocalization(DataLoaderTransformer):
    def __init__(self, dataloader, *args, **kwargs):
        # load_to_host - if True will load the target data directly to the host
        super(ObjectLocalization, self).__init__(dataloader, None, *args, **kwargs)

        # According to SSD documentation, img_shape is on the first place in shapes.
        self.img_shape = dataloader.shapes()[0]
        self.dataloader = dataloader

    def transform(self, t):
        # unpack buffers
        (im_shape, gt_boxes, num_gt_boxes, gt_classes, difficult, img) = t

        self.img_shape = im_shape  # all images in mb are same shape

        gt_boxes = gt_boxes.get().reshape((-1, 4, self.be.bsz))

        gt_boxes = gt_boxes.reshape((-1, self.be.bsz))

        return (img, (gt_boxes, gt_classes.get(), num_gt_boxes.get(),
                      difficult.get(), im_shape.get()))

    def get_img_shape(self):
        return self.img_shape

    def set_classes(self, classes):
        self.CLASSES = classes
        self.num_classes = len(self.CLASSES)


def build_dataloader(config, manifest_root, batch_size, subset_pct=100,
                     PIXEL_MEANS=np.array([104, 117, 123])):
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

    Returns:
        dataloader object.
    """
    # assert config['minibatch_size'] == be.bsz,
    # 'Dataloader config\'s minibatch size not matching backend bsz'
    config["manifest_root"] = manifest_root
    config["batch_size"] = batch_size
    config["subset_fraction"] = float(subset_pct/100.0)

    dl = DataLoaderAdapter(DataLoader(config))
    dl = TypeCast(dl, index=5, dtype=np.float32)  # cast image to float

    dl = BGRMeanSubtract(dl, index=5, pixel_mean=PIXEL_MEANS)  # subtract means
    dl = ObjectLocalization(dl)
    dl.set_classes(config['etl'][0]['class_names'])
    dl.shape = dl.shapes()[5]
    return dl
