.. ---------------------------------------------------------------------------
.. Copyright 2016 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------


Costs
=====

Cost refers to the loss function used to train the model. Each cost
inherits from :py:class:`Cost<neon.transforms.cost.Cost>`. Neon currently supports the
following cost functions:

.. csv-table::
   :header: "Name", "Description"
   :widths: 20, 20
   :escape: ~

   :py:class:`neon.transforms.CrossEntropyBinary<neon.transforms.cost.CrossEntropyBinary>`, :math:`-t\log(y)-(1-t)\log(1-y)`
   :py:class:`neon.transforms.CrossEntropyMulti<neon.transforms.cost.CrossEntropyMulti>`, :math:`\sum t log(y)`
   :py:class:`neon.transforms.SumSquared<neon.transforms.cost.SumSquared>`, :math:`\sum_i (y_i-t_i)^2`
   :py:class:`neon.transforms.MeanSquared<neon.transforms.cost.MeanSquared>`, :math:`\frac{1}{N}\sum_i (y_i-t_i)^2`
   :py:class:`neon.transforms.SmoothL1Loss<neon.transforms.cost.SmoothL1Loss>`, Smooth :math:`L_1` loss (see `Girshick 2015 <http://arxiv.org/pdf/1504.08083v2.pdf>`__)


To create new cost functions, subclass from :py:class:`neon.transforms.Cost<neon.transforms.cost.Cost>` and
implement two methods: :py:meth:`~.Cost.__call__` and :py:meth:`~.Cost.bprop`. Both methods take as
input:

* ``y`` (Tensor or OpTree): Output of model
* ``t`` (Tensor or OpTree): True targets corresponding to y

and returns an OpTree with the cost and the derivative for :py:meth:`~.Cost.__call__`
and :py:meth:`~.Cost.bprop` respectively.

Metrics
=======

We define metrics to evaluate the performance of a trained model.
Similar to costs, each metric takes as input the output of the model
``y`` and the true targets ``t``. Metrics may be initialized with
additional parameters. Each metric returns a numpy array of the metric.
Neon supports the following metrics:

.. csv-table::
   :header: "Name", "Description"
   :widths: 20, 20
   :escape: ~

   :py:class:`neon.transforms.LogLoss<neon.transforms.cost.LogLoss>`, :math:`\log\left(\sum y*t\right)`
   :py:class:`neon.transforms.Misclassification<neon.transforms.cost.Misclassification>`, Incorrect rate
   :py:class:`neon.transforms.TopKMisclassification<neon.transforms.cost.TopKMisclassification>`, Incorrect rate from Top :math:`K` guesses
   :py:class:`neon.transforms.Accuracy<neon.transforms.cost.Accuracy>`, Correct Rate
   :py:class:`neon.transforms.PrecisionRecall<neon.transforms.cost.PrecisionRecall>`, Class averaged precision (item 0) and recall (item 1) values.
   :py:class:`neon.transforms.ObjectDetection<neon.transforms.cost.ObjectDetection>`, Correct rate (item 0) and L1 loss on the bounding box (item 1)

To create your own metric, subclass from :py:class:`Metric<neon.transforms.cost.Metric>` and implement the
:py:meth:`~.Cost.__call__` method, which takes as input Tensors ``y`` and ``t`` and
returns a numpy array of the resulting metrics. If you need to allocate
buffer space for the backend to store calculations, or accept additional
parameters, remember to do so in the class constructor.
