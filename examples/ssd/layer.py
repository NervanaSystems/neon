from neon.layers.layer import Layer, ParameterLayer
import numpy as np
from neon.transforms import Softmax
from neon.initializers.initializer import Constant
import math
from collections import OrderedDict


class Normalize(ParameterLayer):
    def __init__(self, init=Constant(20.0), name=None):
        super(Normalize, self).__init__(name=name, init=init)
        self.bottom_data = None
        self.norm_data = None
        self.owns_outputs = True

    def allocate(self, shared_outputs=None):
        super(Normalize, self).allocate()
        self.outputs_view = self.outputs.reshape(self.channels, -1)

    def configure(self, in_obj):
        self.prev_layer = in_obj
        self.in_shape = in_obj.out_shape
        self.out_shape = in_obj.out_shape

        assert len(self.in_shape) == 3, "Normalize layer must have (C, H, W) input"
        self.channels = self.in_shape[0]
        self.weight_shape = (self.channels, 1)
        return self

    def fprop(self, inputs, inference=True):
        self.bottom_data = inputs.reshape(self.channels, -1)
        self.norm_data = self.be.sqrt(self.be.sum(self.be.square(self.bottom_data), axis=0))
        self.outputs_view[:] = self.W * self.bottom_data / self.norm_data
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        error_rs = error.reshape(self.channels, -1)
        self.dW[:] = self.be.sum(self.outputs_view*error_rs, axis=1)/self.W

        self.deltas_view = self.deltas.reshape(self.channels, -1)

        # we may be able to join these back together into 1 assing call
        self.deltas_view[:] = -self.outputs_view * self.be.sum(self.bottom_data * error_rs, axis=0)
        self.deltas_view[:] = self.deltas_view / self.be.square(self.norm_data)
        # this is separate
        self.deltas_view[:] += self.W * error_rs / self.norm_data
        return self.deltas


class ConcatTranspose(Layer):
    """
        Takes a list of inputs, each with a shape CHWN, and transposes
        to HWCN, then concatenates along the HWC axis.
    """

    def __init__(self, name=None, parallelism='Disabled'):
        super(ConcatTranspose, self).__init__(name, parallelism=parallelism)

    def configure(self, in_obj):
        # we expect a list of layers
        assert isinstance(in_obj, list)

        self.in_shapes = [l.out_shape for l in in_obj]

        self.num_elements = np.sum(np.prod(l.out_shape) for l in in_obj)
        self.out_shape = (self.num_elements)

        # store the number of channels from the layer shapes
        self.channels = [l.out_shape[0] for l in in_obj]

    def allocate(self, shared_outputs=None):
        self.outputs = self.be.iobuf((self.num_elements))

        # create destination delta buffers
        self.deltas = [self.be.iobuf(in_shape) for in_shape in self.in_shapes]

    def fprop(self, inputs):

        start = 0
        for (layer, num_channels) in zip(inputs, self.channels):

            # reshape (C, HW, N)
            rlayer = layer.reshape((num_channels, -1, self.be.bsz))

            # transpose to (HW, C, N) and store in buffer
            C, HW, N = rlayer.shape
            end = start + C * HW
            output_view = self.outputs[start:end, :].reshape((HW, C, N))

            self.be.copy_transpose(rlayer, output_view, axes=(1, 0, 2))
            start = end

        return self.outputs

    def bprop(self, error):
        # error is in (HWC, N)
        # need to transpose to (CHW, N) and unstack
        start = 0
        for (delta, num_channels) in zip(self.deltas, self.channels):

            # reshape (C, HW, N)
            rdelta = delta.reshape((num_channels, -1, self.be.bsz))

            C, HW, N = rdelta.shape
            end = start + C * HW

            error_view = error[start:end, :].reshape((HW, C, N))

            self.be.copy_transpose(error_view, rdelta, axes=(1, 0, 2))
            start = end

        return self.deltas


class DetectionOutput(Layer):

    def __init__(self, num_classes, nms_threshold=0.45,
                 nms_topk=400, topk=200, threshold=0.01, name=None):
            super(DetectionOutput, self).__init__(name)
            self.num_classes = num_classes
            self.nms_threshold = nms_threshold
            self.nms_topk = nms_topk
            self.topk = topk
            self.threshold = 0.01
            self.softmax = Softmax(axis=1)

    def configure(self, in_obj):
        self.out_shape = (self.topk, 5)

        # we expect a list of layers from the SSD model
        (leafs, prior_boxes) = in_obj

        # store total number of boxes
        self.num_boxes = np.sum([prior_box.num_boxes for prior_box in prior_boxes])

    def allocate(self, shared_outputs=None):
        self.conf = self.be.iobuf((self.num_boxes * self.num_classes))
        self.loc = self.be.iobuf((self.num_boxes * 4))

        # intermediate buffer for compute
        # these are needed to keep compute on the GPU
        # 1. proposals for each class and image
        # 2. store detections after sort/threshold
        # 3. store softmax
        self.proposals = self.be.empty((self.num_boxes, 4))
        self.detections = self.be.empty((self.nms_topk, 5))
        self.scores = self.be.empty((self.num_boxes, self.num_classes))

    def fprop(self, inputs, inference=True):
        # assumes the inputs are a tuple of (outputs, prior_boxes),
        # where outputs is a vector of outputs from the model.

        # flatten the nested vector generated by tree-in-tree
        # also reorder the list in: 4_3, fc7, conv6, conv7, conv8, conv9
        # x = self.reorder(inputs[0])
        self.loc = inputs[0][0]
        self.conf = inputs[0][1]
        prior_boxes = inputs[1]

        # reshape loc from (HWC, N) to (HWK, 4, N)
        # reshape conf from (HWC, N) to (HWK, 21, N)
        conf_view = self.conf.reshape((-1, self.num_classes, self.be.bsz))
        loc_view = self.loc.reshape((-1, 4, self.be.bsz))

        # convert the prior boxes to bbox predictions by applying
        # the loc regression targets
        # process each image individually

        batch_all_detections = [None] * self.be.bsz
        for k in range(self.be.bsz):

            self.bbox_transform_inv(prior_boxes, loc_view[:, :, k], self.proposals)

            all_detections = np.zeros((0, 6))  # detections for this image

            conf = conf_view[:, :, k]
            self.scores[:] = self.softmax(conf)

            for c in range(self.num_classes):
                if (c == 0):  # skip processing of background classes
                    continue

                # apply softmax
                scores = self.scores[:, c]

                # 1. apply threshold, sort, and get the top nms_k
                top_N_ind = self.get_top_N_index(scores, self.nms_topk, self.threshold)

                # fill the detections
                if len(top_N_ind) > 0:
                    self.detections.fill(0)
                    self.detections[:len(top_N_ind), :4] = self.proposals[top_N_ind, :]
                    self.detections[:len(top_N_ind), 4] = scores[top_N_ind]

                    # 2. apply NMS
                    keep = self.be.nms(self.detections, self.nms_threshold, normalized=True)
                    keep = keep[:self.nms_topk]

                    # 3. store the detections per class
                    # add an additional dimension for the category label
                    dets = np.append(self.detections[keep, :].get(),
                                     c * np.ones((len(keep), 1)), axis=1)
                    all_detections = np.vstack([all_detections, dets])

            if all_detections.shape[0] > self.topk:
                top_N_ind = self.get_top_N_index(all_detections[:, 4], self.topk, None)
                all_detections = all_detections[top_N_ind, :]

            batch_all_detections[k] = all_detections

        return batch_all_detections

    def bprop(self, error, alpha=1.0, beta=0.0):
        raise NotImplementedError

    def get_top_N_index(self, scores, N, threshold):
        # this function handles scores still being device tensors

        # move scores to host if needed
        if isinstance(scores, np.ndarray):
            np_scores = scores.ravel()
        else:
            np_scores = scores.get().ravel()

        # apply threshold if needed
        if threshold is None:
            count = len(np_scores)
        else:
            count = len(np.where(np_scores > threshold)[0])

        order = np_scores.argsort()[::-1].tolist()
        order = order[:count]
        if N > 0:
            order = order[:N]

        return order

    def bbox_transform_inv(self, boxes, deltas, output, variance=[0.1, 0.1, 0.2, 0.2]):

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        pred_ctr_x = variance[0] * dx * widths + ctr_x
        pred_ctr_y = variance[1] * dy * heights + ctr_y
        pred_w = self.be.exp(variance[2] * dw) * widths
        pred_h = self.be.exp(variance[3] * dh) * heights

        # pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        output[:, 0] = pred_ctr_x - 0.5 * pred_w
        # y1
        output[:, 1] = pred_ctr_y - 0.5 * pred_h
        # x2
        output[:, 2] = pred_ctr_x + 0.5 * pred_w
        # y2
        output[:, 3] = pred_ctr_y + 0.5 * pred_h

        return output


class PriorBox(Layer):

    def __init__(self, min_sizes, max_sizes, step=None, aspect_ratios=[2, 3], img_shape=(300, 300),
                 flip=True, clip=False, variance=[0.1, 0.1, 0.2, 0.2], offset=0.5, name=None):
        super(PriorBox, self).__init__(name)
        self.offset = offset
        self.variance = variance
        self.flip = flip
        self.clip = clip
        if type(step) in (dict, OrderedDict):
            assert set(step.keys()) == set(('step_w', 'step_h'))
            self.step_w = step['step_w']
            self.step_h = step['step_h']
        else:
            assert step is not None
            self.step_w = step
            self.step_h = step
        self.prior_boxes = None
        self.img_w = img_shape[0]
        self.img_h = img_shape[1]

        assert isinstance(min_sizes, tuple)
        assert isinstance(max_sizes, tuple)

        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        assert len(self.min_sizes) == len(self.max_sizes)

        # compute the number of prior boxes
        # with flip, order the aspect ratios in the same was as caffe
        self.aspect_ratios = []
        for ar in aspect_ratios:
            self.aspect_ratios.extend([ar])

            if(self.flip):
                self.aspect_ratios.extend([1.0 / float(ar)])

        # number of prior baxes per feature map pixel
        # there is 1 box with AR=1 for each min and max sizes
        self.num_priors_per_pixel = len(self.min_sizes) * 2
        # and one box for each aspect ratio at the min size
        self.num_priors_per_pixel += len(self.aspect_ratios) * len(self.min_sizes)

    def configure(self, in_objs):
        conv_layer = in_objs[1]

        self.in_shape = conv_layer.out_shape
        (_, self.layer_height, self.layer_width) = self.in_shape

        if self.step_w is None or self.step_h is None:
            self.step_w = math.ceil(float(self.img_w) / self.layer_width)
            self.step_h = math.ceil(float(self.img_h) / self.layer_height)

        self.num_boxes = self.layer_height * self.layer_width * self.num_priors_per_pixel
        self.out_shape = (4*self.num_priors_per_pixel, self.layer_height, self.layer_width)

    def allocate(self, shared_outputs=None):
        self.outputs = self.be.empty((self.num_boxes, 4))

    def fprop(self, inputs, inference=True):
        # the priors will be of shape layer_width * layer_height * num_priors_per_pixel * 4
        # with 2 chans per element (one for mean fnd one or vairance)

        # we only need to calculate these once if the image size does not change
        # right now we don't support changing image sizes anyways
        if self.prior_boxes is not None:
            return self.outputs

        img_shape = [self.img_w, self.img_h]

        self.prior_boxes = []

        def gen_box(center, box_size, image_size, variance, clip):
            box_ = [None] * 4
            box_[0] = (center[0] - box_size[0] * 0.5) / image_size[0]  # xmin
            box_[1] = (center[1] - box_size[1] * 0.5) / image_size[1]  # ymin
            box_[2] = (center[0] + box_size[0] * 0.5) / image_size[0]  # xmax
            box_[3] = (center[1] + box_size[1] * 0.5) / image_size[1]  # ymax

            if clip:
                for ind in range(4):
                    box_[ind] = min([max([box_[ind], 0.0]), 1.0])
            return box_

        offset = self.offset

        # the output is 2 chans (the 4 prior coordinates, the 4 prior variances) for
        # each output feature map pixel  so the output array is
        # 2 x layer_height x layer_width x num_priors x 4
        center = [0, 0]
        for h in range(self.layer_height):
            center[1] = (h + offset) * self.step_h
            for w in range(self.layer_width):
                center[0] = (w + offset) * self.step_w
                # do the min and max boxes with aspect ratio 1
                for (min_size, max_size) in zip(self.min_sizes, self.max_sizes):
                    # do the min box
                    box_shape = [min_size, min_size]
                    self.prior_boxes += (gen_box(center, box_shape, img_shape,
                                                 self.variance, self.clip))

                    # do the max size box
                    sz_ = math.sqrt(min_size * max_size)
                    box_shape = [sz_, sz_]
                    self.prior_boxes += (gen_box(center, box_shape, img_shape,
                                                 self.variance, self.clip))

                    # now do the different aspect ratio boxes
                    for ar in self.aspect_ratios:
                        assert np.abs(ar - 1.0) > 1.0e-6
                        box_width = min_size * math.sqrt(ar)
                        box_height = min_size / math.sqrt(ar)
                        box_shape = [box_width, box_height]
                        self.prior_boxes += (gen_box(center, box_shape, img_shape,
                                                     self.variance, self.clip))

        self.outputs.set(np.array(self.prior_boxes).reshape(-1, 4))
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        raise NotImplementedError
