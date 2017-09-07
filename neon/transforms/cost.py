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
from collections import Counter
from neon import logger as neon_logger


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
    that uses the Logistic() activation function. This allows for a shortcut in
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
        if y.shape != t.shape:
            raise ValueError((
                "CrossEntropy requires network output shape to match "
                "targets. Network output shape was {} and targets shape "
                "was {}"
            ).format(y.shape, t.shape))
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
        return self.logscale * self.scale * (y - t)


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
        return (0.5 * self.be.square(x) * self._sigma2 * (self.be.absolute(x) < 1/self._sigma2) +
                (self.be.absolute(x) - 0.5/self._sigma2) * (self.be.absolute(x) >= 1/self._sigma2))

    def smoothL1grad(self, x):
        """
        Returns the gradient of the Smooth-L1 cost.
        """
        return (x * self._sigma2 * (self.be.absolute(x) < 1/self._sigma2) +
                self.be.sgn(x) * (self.be.absolute(x) >= 1/self._sigma2))

    def __init__(self, sigma=1.0):
        """
        Define the cost function and its gradient as lambda functions.
        """
        self.sigma = sigma
        self._sigma2 = self.be.square(sigma)
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


class MultiMetric(Metric):
    """
    A wrapper Metric which can be used with Tree models which have more than
    one output.  Tree models have tuples of tensors, one tensor per output.
    Wrapping a Metric with a MultiMetric ensures that the metric sees only one
    of those tensors in the output tuple instead of all of them.
    """

    def __init__(self, metric, index):
        """
        Args:
            metric (Metric): Metric to apply in this multi-output context
            index (integer): The index into the model's output tuple to apply
                             the metric to
        """
        self.metric = metric
        self.index = index

    def __call__(self, y, t, *args, **kwargs):
        """
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            numpy array : Returns the log loss  metric in numpy array,
                         [LogLoss]
        """
        return self.metric(y[self.index], y[self.index], *args, **kwargs)

    def __getattr__(self, key):
        return getattr(self.metric, key)


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

    def __init__(self, steps=1):
        """
        Initialize the metric.
        """
        self.preds = self.be.iobuf((1, steps), persist_values=False)
        self.hyps = self.be.iobuf((1, steps), persist_values=False)
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

    def __call__(self, y, t, calcrange=slice(0, None)):
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


class BLEUScore(Metric):
    """
    Compute BLEU score metric
    """

    def __init__(self, unk='<unk>'):
        self.metric_names = ['BLEU']
        self.end_token = '.'
        self.unk_symbol = unk

    def __call__(self, y, t, N=4, brevity_penalty=False, lower_case=True):
        """
        Args:
            y (list): list of predicted sentences
            t (list): list of reference sentences where each element is a list
                      of multiple references
            N (int, optional): compute all ngram modified precisions up to this N
            brevity_penalty (bool, optional): if True, use brevity penalty
            lower_case (bool, optional): if True, convert all words to lower case
        """

        y_list = list(y)
        t_list = list(t)

        if lower_case:
            for ii, sent in enumerate(y_list):
                y_list[ii] = sent.lower()

        # convert all sentences to lists of words
        for ii, sent in enumerate(y_list):
            y_list[ii] = sent.strip(self.end_token).split()

        for ii, refs in enumerate(t_list):
            tmp = []
            for ref in refs:
                tmp += [ref.split()]
            t_list[ii] = tmp

        def ngram_counts(sentence, counts, N):
            for n in range(1, N+1):
                num = len(sentence) - n + 1     # number of n-grams
                for jj in range(num):
                    ngram = ' '.join(sentence[jj:jj+n])
                    ngram = repr(n) + ' ' + ngram
                    counts[ngram] += 1

        # compute ngram counts
        totals = np.zeros(N)    # ngram counts over all candidates
        correct = np.zeros(N)   # correct ngrams (compared to max over references)
        len_translation, len_reference = (0, 0)
        for ii, sent in enumerate(y_list):
            counts_ref_max = Counter()    # maximum ngram count over all references for an example
            # count ngrams in candidate sentence
            counts_cand = Counter()
            ngram_counts(sent, counts_cand, N)
            # process reference sentences
            closest_diff, closest_len = (float("inf"), float("inf"))
            for ref in t_list[ii]:
                counts_ref = Counter()
                # find closest length of reference sentence for current example
                diff = abs(len(sent) - len(ref))
                if diff < closest_diff:
                    closest_len = len(ref)
                elif diff == closest_diff:
                    closest_len = min(closest_len, len(ref))
                # compute all ngram counts up to specified n=N for this reference
                ngram_counts(ref, counts_ref, N)
                for ngram, count in counts_ref.items():
                    if counts_ref_max[ngram] < count:
                        counts_ref_max[ngram] = count
            len_reference += closest_len
            len_translation += len(sent)
            for ngram, count in counts_cand.items():
                n = int(ngram[0])
                ind = n - 1
                totals[ind] += count
                # only match if there are no UNK
                if ngram.find(self.unk_symbol) == -1:
                    r = counts_ref_max[ngram]
                    c = count if r >= count else r
                    correct[ind] += c

        # calculate bleu scores
        precision = correct/totals + 0.0000001

        if (brevity_penalty and len_translation < len_reference):
            bp = np.exp(1-float(len_reference)/len_translation)
        else:
            bp = 1.0

        logprec = np.log(precision)
        self.bleu_n = [100*bp*np.exp(sum(logprec[:nn+1])/(nn+1)) for nn in range(N)]

        neon_logger.display("Bleu scores: " + " ".join([str(np.round(f, 2)) for f in self.bleu_n]))

        return self.bleu_n[-1]


class GANCost(Cost):

    """
    Discriminator cost for a Generative Adversarial Network
    The Discriminator cost is a packaged cross-entropy where the inputs with label 0
    and the inputs with label 1 are passed in separately. It takes the form
    :math:`C = \log (y_data) + \log (1 - y_noise)` where :math:`y_data` are the fprop
    outputs of the data minibatch, and :math:`y_noise` are the outputs of the generator-
    discriminator stack on a noise batch.
    """
    def __init__(self, scale=1., cost_type="dis", func='modified'):
        """
        Args:
            scale (float, optional): Amount by which to scale the backpropagated error (default: 1)
            cost_type (string): select discriminator cost "dis" or generator cost "gen"
            cost_func (string): cost function: choice from "original", "modified" and "wasserstein"
                                (Goodfellow et al. 2014, Arjovski et al. 2017)
        """
        self.scale = scale
        self.cost_type = cost_type
        self.func = func
        err_str = "Illegal GAN cost type, can only be: gen or dis"
        assert self.cost_type in ['dis', 'gen']
        err_str = "Unsupported GAN cost function, supported: original, modified, wasserstein"
        assert self.func in ['original', 'modified', 'wasserstein'], err_str
        self.one_buf = self.be.iobuf(1)
        self.one_buf.fill(1)

    def __call__(self, y_data, y_noise, cost_type='dis'):
        """
        Returns the discriminator cost. Note sign flip of the discriminator
        cost relative to Goodfellow et al. 2014 so we can minimize the cost
        rather than maximizing discriminiation.
        Args:
            y_data (Tensor or OpTree): Output of the data minibatch
            y_noise (Tensor or OpTree): Output of noise minibatch
            cost_type (str): 'dis' (default), 'dis_data', 'dis_noise' or 'gen'
        Returns:
            OpTree: discriminator or generator cost, controlled by cost_type
        """
        assert y_data.shape == y_noise.shape, "Noise and data output shape mismatch"
        if self.func == 'original':
            cost_dis_data = -self.be.safelog(y_data)
            cost_dis_noise = -self.be.safelog(1-y_noise)
            cost_gen = -self.be.safelog(y_noise)
        elif self.func == 'modified':
            cost_dis_data = -self.be.safelog(y_data)
            cost_dis_noise = -self.be.safelog(1-y_noise)
            cost_gen = self.be.safelog(1-y_noise)
        elif self.func == 'wasserstein':
            cost_dis_data = y_data
            cost_dis_noise = -y_noise
            cost_gen = y_noise

        if cost_type == 'dis':
            return self.be.mean(cost_dis_data + cost_dis_noise, axis=0)
        elif cost_type == 'dis_data':
            return self.be.mean(cost_dis_data, axis=0)
        elif cost_type == 'dis_noise':
            return self.be.mean(cost_dis_noise, axis=0)
        elif cost_type == 'gen':
            return self.be.mean(cost_gen, axis=0)

    def bprop_noise(self, y_noise):
        """
        Derivative of the discriminator cost wrt. y_noise.
        """
        if self.func in ['original', 'modified']:
            return self.scale / (1. - y_noise)
        elif self.func == 'wasserstein':
            return -self.scale * self.one_buf

    def bprop_data(self, y_data):
        """
        Derivative of the discriminator cost wrt. y_data.
        """
        if self.func in ['original', 'modified']:
            return self.scale * (-1. / y_data)
        elif self.func == 'wasserstein':
            return self.scale * self.one_buf

    def bprop_generator(self, y_noise):
        """
        Derivative of the generator cost wrt. y_noise.
        """
        if self.func == 'original':
            return self.scale / (y_noise - 1.)
        elif self.func == 'modified':
            return -self.scale / y_noise
        elif self.func == 'wasserstein':
            return self.scale * self.one_buf
