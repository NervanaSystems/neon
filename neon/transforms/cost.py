# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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
from builtins import str
from neon import NervanaObject
import numpy as np


class Cost(NervanaObject):

    """
    Base class for cost functions that are used during training.

    Child classes can either implement the below `__call__` and `bprop` methods, or alternatively
    define `self.func` and self.funcgrad`. The latter is typically used for code
    compactness when the operations can be fit into a lambda function.
    """

    def __call__(self, y, t):
        """
        Applies the cost function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the cost
        """
        return self.func(y, t)

    def bprop(self, y, t):
        """
        Computes the derivative of the cost function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the derivative of the cost function
        """
        return self.funcgrad(y, t)


class CrossEntropyBinary(Cost):

    """
    Binary cross-entropy cost.

    The binary cross-entropy cost is used when the labels have two classes: 0 and 1.
    The cost is computed as :math:`C = \sum -t\log(y)-(1-t)\log(1-y)`, where :math:`t` is
    the target label and :math:`y` is the network output.

    Note:
    The backpropagation assumes that this cost is coupled with an output layer
    that uses the Softmax() activation function. This allows for a shortcut in
    the deriviate that saves computation.
    """

    def __init__(self, scale=1):
        """
        Args:
            scale (float, optional): Amount by which to scale the backpropagated error (default: 1)
        """
        self.scale = scale

    def __call__(self, y, t):
        """
        Returns the binary cross entropy cost.

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the binary cross entropy cost
        """
        assert y.shape == t.shape, "CrossEntropy requires network output shape to match targets"
        return self.be.sum(self.be.safelog(1 - y) * (t - 1) - self.be.safelog(y) * t, axis=0)

    def bprop(self, y, t):
        """
        Returns the derivative of the binary cross entropy cost.

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the (mean) shortcut derivative of the binary entropy
                    cost function ``(y - t) / y.shape[1]``
        """
        return self.scale * (y - t)


class CrossEntropyMulti(Cost):

    """
    Multi-class cross-entropy cost.

    The multi-class cross-entropy cost is used when the labels have multiple classes.
    The cost is computed as :math:`C = \sum -t*\log(y)`, where :math:`t` is
    the target label and :math:`y` is the network output.

    The target labels :math:`t` are expected to be in an one-hot encoding. By default,
    the natural logarithm is used, but a cost that returns bits instead (e.g. log base 2)
    can also be specified with the ``usebits`` argument.

    Note:
        The back-propogation assumes that this cost is coupled with an output layer
        that uses the Softmax() activation function. This allows for a shortcut in
        the deriviate that saves computation.
    """

    def __init__(self, scale=1, usebits=False):
        """
        Args:
            scale (float, optional): scale factor for the backpropagated error (default: 1)
            usebits (boolean, optional): Display costs in bits (default: False)
        """
        super(CrossEntropyMulti, self).__init__()
        self.usebits = usebits
        self.scale = scale
        self.logscale = np.float(1. / np.log(2.0) if usebits else 1.)

    def __call__(self, y, t):
        """
        Returns the multiclass cross entropy cost

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the multiclass cross entropy cost
        """
        assert y.shape == t.shape, "CrossEntropy requires network output shape to match targets"
        return (self.be.sum(-t * self.logscale * self.be.safelog(y), axis=0))

    def bprop(self, y, t):
        """
        Returns the derivative of the multiclass cross entropy cost.

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the (mean) shortcut derivative of the multiclass
            entropy cost function ``(y - t) / y.shape[1]``
        """
        return self.scale * (y - t)


class SumSquared(Cost):

    """
    Total Squared Error cost function. Computes :math:`\sum_i (y_i-t_i)^2`.


    """

    def __init__(self):
        """
        Define the cost function and its gradient as lambda functions.
        """
        self.func = lambda y, t: self.be.sum(self.be.square(y - t), axis=0) / 2.
        self.funcgrad = lambda y, t: (y - t)


class MeanSquared(Cost):

    """
    Average Squared Error cost function. Computes :math:`\\frac{1}{N}\\sum_i (y_i-t_i)^2`.
    """

    def __init__(self):
        """
        Define the cost function and its gradient as lambda functions.
        """
        self.func = lambda y, t: self.be.mean(self.be.square(y - t), axis=0) / 2.
        self.funcgrad = lambda y, t: (y - t) / y.shape[0]


class SmoothL1Loss(Cost):

    """
    Smooth L1 cost function.

    The L1 loss is less sensitive to outliers than the L2 loss.
    See `Girshick 2015 <http://arxiv.org/pdf/1504.08083v2.pdf>`__. This
    cost is used for training object localization models such as Fast-RCNN.
    """

    def smoothL1(self, x):
        """
        Returns the Smooth-L1 cost
        """
        return (0.5 * self.be.square(x) * (self.be.absolute(x) < 1) +
                (self.be.absolute(x) - 0.5) * (self.be.absolute(x) >= 1))

    def smoothL1grad(self, x):
        """
        Returns the gradient of the Smooth-L1 cost.
        """
        return (x * (self.be.absolute(x) < 1) + self.be.sgn(x) *
                (self.be.absolute(x) >= 1))

    def __init__(self):
        """
        Define the cost function and its gradient as lambda functions.
        """
        self.func = lambda y, t: self.be.sum(self.smoothL1(y - t), axis=0)
        self.funcgrad = lambda y, t: self.smoothL1grad(y - t)


class SquareHingeLoss(Cost):

    """
    Applies the square hinge loss cost function
    """

    def squarehinge(self, y, t):
        t = 2 * t - 1
        return self.be.mean(self.be.square(self.be.maximum(self.margin - t * y, 0)), axis=0)

    def squarehingegrad(self, y, t):
        t = 2 * t - 1
        return -2 * t * self.be.maximum(self.margin - t * y, 0)/float(y.shape[0])

    def __init__(self, margin=1):
        """
        Initialize the square hinge loss cost function
        """
        self.margin = margin
        self.func = lambda y, t: self.squarehinge(y, t)
        self.funcgrad = lambda y, t: self.squarehingegrad(y, t)


class Metric(Cost):

    """
    Base class for Metrics. Metrics are quantities not used during training
    for the back-propogration but are useful to compute and display to check
    on progress.

    For example, when training on image classification network,
    we may want to use the Cross-entropy cost to train the weights, but display
    the misclassification rate metric.
    """

    def __call__(self, y, t):
        """
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            float: Returns the metric
        """
        raise NotImplementedError()


class LogLoss(Metric):

    """
    LogLoss metric.

    Computes :math:`\\log\\left(\\sum y*t\\right)`.
    """

    def __init__(self):
        self.correctProbs = self.be.iobuf(1)
        self.metric_names = ['LogLoss']

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            numpy array : Returns the log loss  metric in numpy array,
                         [LogLoss]
        """
        self.correctProbs[:] = self.be.sum(y * t, axis=0)
        self.correctProbs[:] = -self.be.safelog(self.correctProbs)
        return np.array(self.correctProbs.get()[:, calcrange].mean())


class TopKMisclassification(Metric):

    """
    Multiple misclassification metrics.

    Computes the LogLoss metric, the Top-1 Misclassification rate, and the Top-K
    misclassification rate.
    """

    def __init__(self, k):
        """
        Arguments:
            k (integer): Number of guesses to allow.
        """
        self.correctProbs = self.be.iobuf(1)
        self.top1 = self.be.iobuf(1)
        self.topk = self.be.iobuf(1)

        self.k = k
        self.metric_names = ['LogLoss', 'Top1Misclass', 'Top' + str(k) + 'Misclass']

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Returns a numpy array of metrics for: LogLoss, Top-1, and Top-K.

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y
            calcrange (slice, optional): Slice of data used for the metric (default: all)

        Returns:
            numpy array : Returns the metrics in a numpy array:
                          [LogLoss, Top 1 misclass, Top k misclass]
        """
        be = self.be
        self.correctProbs[:] = be.sum(y * t, axis=0)
        nSlots = self.k - be.sum((y > self.correctProbs), axis=0)
        nEq = be.sum(y == self.correctProbs, axis=0)
        self.topk[:] = 1. - (nSlots > 0) * ((nEq <= nSlots) * (1 - nSlots / nEq) + nSlots / nEq)
        self.top1[:] = 1. - (be.max(y, axis=0) == self.correctProbs) / nEq
        self.correctProbs[:] = -be.safelog(self.correctProbs)
        return np.array((self.correctProbs.get()[:, calcrange].mean(),
                         self.top1.get()[:, calcrange].mean(),
                         self.topk.get()[:, calcrange].mean()))


class Misclassification(Metric):

    """
    Misclassification error metric.
    """

    def __init__(self):
        """
        Initialize the metric.
        """
        self.preds = self.be.iobuf(1, persist_values=False)
        self.hyps = self.be.iobuf(1, persist_values=False)
        self.outputs = self.preds  # Contains per record metric
        self.metric_names = ['Top1Misclass']

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Returns the misclassification error metric

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            float: Returns the metric
        """
        # convert back from onehot and compare
        self.preds[:] = self.be.argmax(y, axis=0)
        self.hyps[:] = self.be.argmax(t, axis=0)
        self.outputs[:] = self.be.not_equal(self.preds, self.hyps)

        return self.outputs.get()[:, calcrange].mean()


class Accuracy(Metric):

    """
    Accuracy metric (correct rate).

    """

    def __init__(self):
        self.preds = self.be.iobuf(1)
        self.hyps = self.be.iobuf(1)
        self.outputs = self.preds  # Contains per record metric
        self.metric_names = ['Accuracy']

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Returns the accuracy.

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            float: Returns the metric
        """
        # convert back from onehot and compare
        self.preds[:] = self.be.argmax(y, axis=0)
        self.hyps[:] = self.be.argmax(t, axis=0)
        self.outputs[:] = self.be.equal(self.preds, self.hyps)

        return self.outputs.get()[:, calcrange].mean()


class PrecisionRecall(Metric):
    """
    Precision and Recall metrics.

    Typically used in a conjunction with a multi-classification model.
    """
    def __init__(self, num_classes, binarize=False, epsilon=1e-6):
        """
        Arguments:
            num_classes (int): Number of different output classes.
            binarize (bool, optional): If True will attempt to convert the model
                                       outputs to a one-hot encoding (in place).
                                       Defaults to False.
            epsilon (float, optional): Smoothing to apply to avoid division by zero.
                                       Defaults to 1e-6.
        """
        self.outputs = self.be.empty((num_classes, 2))
        self.token_stats = self.be.empty((num_classes, 3))
        self.metric_names = ['Precision', 'Recall']
        if binarize:
            self.bin_buf = self.be.iobuf(1, dtype=np.int32)
        else:
            self.bin_buf = None
        self.eps = epsilon

    def __call__(self, y, t):
        """
        Returns a numpy array with the precision and recall metrics.

        Args:
            y (Tensor or OpTree): Output of previous layer or model (we assume
                                  already binarized, or you need to ensure
                                  binarize is True during construction).
            t (Tensor or OpTree): True targets corresponding to y (we assume
                                  already binarized)

        Returns:
            ndarray: The class averaged precision (item 0) and recall (item
                     1) values.  Per-class statistics remain in self.outputs.
        """
        if self.bin_buf is not None:
            self.be.argmax(y, axis=0, out=self.bin_buf)
            y[:] = self.be.onehot(self.bin_buf, axis=0)
        # True positives
        self.token_stats[:, 0] = self.be.sum(y * t, axis=1)

        # Prediction
        self.token_stats[:, 1] = self.be.sum(y, axis=1)

        # Targets
        self.token_stats[:, 2] = self.be.sum(t, axis=1)

        # Precision
        self.outputs[:, 0] = self.token_stats[:, 0] / (self.token_stats[:, 1] + self.eps)

        # Recall
        self.outputs[:, 1] = self.token_stats[:, 0] / (self.token_stats[:, 2] + self.eps)

        return self.outputs.get().mean(axis=0)


class ObjectDetection(Metric):

    """
    The object detection metric includes object label accuracy, and
    bounding box regression.
    """

    def __init__(self):
        self.metric_names = ['Accuracy', 'SmoothL1Loss']
        self.label_ind = 0
        self.bbox_ind = 1

    def smoothL1(self, x):
        """
        Returns the Smooth L1 cost.
        """
        return (0.5 * self.be.square(x) * (self.be.absolute(x) < 1) +
                (self.be.absolute(x) - 0.5) * (self.be.absolute(x) >= 1))

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Returns a numpy array with the accuracy and the Smooth-L1 metrics.

        Args:
            y (Tensor or OpTree): Output of a model like Fast-RCNN model with 2 elements:
                                    1. class label: (# classes, # batchsize for ROIs)
                                    2. object bounding box (# classes * 4, # bacthsize for ROIs)
            t (Tensor or OpTree): True targets corresponding to y, with 2 elements:
                                    1. class labels: (# classes, # batchsize for ROIs)
                                        1.1 class labels
                                                    (# classes, # batchsize for ROIs)
                                        1.2 class labels mask
                                                    (# classes, # batchsize for ROIs)
                                    2. object bounding box and mask, where mask will indicate the
                                        real object to detect other than the background objects
                                        2.1 object bounding box
                                                    (# classes * 4, # bacthsize for ROIs)
                                        2.2 object bounding box mask
                                                    (# classes * 4, # bacthsize for ROIs)

        Returns:
            numpy ary : Returns the metrics in numpy array [Label Accuracy, Bounding Box Smooth-L1]
        """
        t_bb = t[self.bbox_ind][0]
        t_bb_mask = t[self.bbox_ind][1]
        y_bb = y[self.bbox_ind]

        self.detectionMetric = self.be.empty((1, t_bb.shape[1]))
        self.detectionMetric[:] = self.be.sum(self.smoothL1(y_bb * t_bb_mask - t_bb), axis=0)

        if isinstance(t[self.label_ind], tuple):
            t_lbl = t[self.label_ind][0] * t[self.label_ind][1]
            y_lbl = y[self.label_ind] * t[self.label_ind][1]
        else:
            t_lbl = t[self.label_ind]
            y_lbl = y[self.label_ind]

        self.preds = self.be.empty((1, y_lbl.shape[1]))
        self.hyps = self.be.empty((1, t_lbl.shape[1]))
        self.labelMetric = self.be.empty((1, y_lbl.shape[1]))

        self.preds[:] = self.be.argmax(y_lbl, axis=0)
        self.hyps[:] = self.be.argmax(t_lbl, axis=0)
        self.labelMetric[:] = self.be.equal(self.preds, self.hyps)

        return np.array((self.labelMetric.get()[:, calcrange].mean(),
                         self.detectionMetric.get()[:, calcrange].mean()))
