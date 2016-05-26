from neon.layers.layer import Layer
import numpy as np


class RoiPooling(Layer):
    """
    RoiPooling uses max pooling to convert the features inside any ROI into a small
    feature map with a fixed spatial extend of H x W, where H and W are layer
    parameters independent of any particular ROI.
    Each ROI is defined as a 4-tuple as (xmin, ymin, xmax, ymax)

    ROIPooling is applied independently to each feature map channel, as in standard
    max pooling.

    ROIPooling takes as input a tuple (img_fm, rois) where:
    (1) img_fm: output from the convolutional layers (e.g. for VGG-16, 62x62)
    (2) rois: proposed ROIs, in the form (rois_per_img, 5). The first index is the
        image_id within the minibatch. Since faster-rcnn uses batch size 1, this is always 0.

    The output shape (out_shape) is a tuple - (batch_size, rois_per_img), then
    the following layers will allocate buffers accordingly.
    """

    def __init__(self, HW=(7, 7), spatial_scale=0.0625, name=None):
        super(RoiPooling, self).__init__(name)

        self.HW = HW
        self.roi_H, self.roi_W = self.HW
        self.spatial_scale = spatial_scale  # 0.0625 is 1/16

        # it has its own output buffer besides being a container
        self.owns_output = True
        self.owns_delta = True

        self.img = None
        self.rois = None
        self.rois_per_img = None
        self.fm_channel = None
        self.fm_height = None
        self.fm_width = None

        # self.rois_per_batch = self.be.bsz * 64

    def configure(self, in_obj):
        """
        Must receive a list of shapes for configurations
        Need both the layer container and roi dataset to configure shapes
        'in_obj' will include be [image_shape, roi_shape] (e.g [(3, 600, 1000), 5])

        Arguments:
            in_obj:

        Returns:

        """
        # configure to get the shape of feature map
        assert len(in_obj) == 2, "Input to ROIpooling must be a 2-tuple"
        self.prev_layer = in_obj
        img_fm, rois = in_obj

        # configure number of rois
        assert rois.out_shape[0] == 5, "Input ROIs must be a 5-tuple"
        self.rois_per_img = rois.out_shape[1]

        # configure input image feature map shapes
        self.in_shape = img_fm.out_shape
        (self.fm_channel, self.fm_height, self.fm_width) = self.in_shape
        self.error_in_reshape = (self.fm_channel, -1)
        self.fm_reshape_shape = (
            self.fm_channel, self.fm_height * self.fm_width, self.be.bsz)

        # make the out_shape as a tuple, as if the roi_per_image a
        # time_step dimension
        self.out_shape = (self.fm_channel * self.roi_H * self.roi_W, self.rois_per_img)
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(RoiPooling, self).allocate(shared_outputs)
        self.owns_output = True
        self.error = self.be.iobuf(self.in_shape)
        self.outputs = self.be.iobuf(self.out_shape, shared=shared_outputs)
        self.max_idx = self.be.iobuf(self.out_shape, dtype=np.int32)

    # def set_deltas(self, delta_buffers):
    #     """
    #     Use pre-allocated (by layer containers) list of buffers for backpropagated error.
    #     Only set deltas for layers that own their own deltas
    #     Only allocate space if layer owns its own deltas

    #     Arguments:
    #         delta_buffers (list): list of pre-allocated tensors (provided by layer container)
    #     """
    #     self.allocate_deltas()

    def init_buffers(self, inputs):
        """
        Initialize buffers for images and ROIs

        Arguments:
            inputs:

        Returns:

        """
        assert len(inputs) == 2, "inputs must contain both images and ROIs"
        self.img = inputs[0]
        self.rois = inputs[1].transpose()
        assert self.rois.shape[1] == 5, "ROI entry must be 5-value tuple"

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        self.init_buffers(inputs)

        self.outputs.fill(0)
        self.max_idx.fill(0)

        # fprop through the roipooling layer
        self.be.roipooling_fprop(self.img, self.rois, self.outputs, self.max_idx,
                                 self.rois_per_img, self.fm_channel, self.fm_height,
                                 self.fm_width, self.roi_H, self.roi_W, self.spatial_scale)

        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """

        self.error.fill(0)

        if self.bprop_enabled:
            # # bprop through the roipooling layer
            self.be.roipooling_bprop(error, self.rois, self.error, self.max_idx,
                                     self.rois_per_batch, self.fm_channel, self.fm_height,
                                     self.fm_width, self.roi_H, self.roi_W, self.spatial_scale)

        # # bprop back through the imagenet layer container
        # self.deltas = super(RoiPooling, self).bprop(self.error, alpha, beta)
