.. ---------------------------------------------------------------------------
.. Copyright 2015 Nervana Systems Inc.
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

Overview
========


High Level Overview
-------------------
To build a network in neon, you create a :doc:`model <models>` that has a
:doc:`backend <backends>`, a :doc:`dataset <datasets>` and
:doc:`layers <layers>`. In addition, you need to define the
:doc:`activation functions <activations>`, the
:doc:`learning rules (optimizers) <optimizers>`, and the
:doc:`cost function <costs>`.
Finally, you can choose one or more :doc:`metrics <metrics>` by which to
evaluate your model's performance.


Neon currently supports the following:

* Backends - :py:class:`NervanaGPU<neon.backends.nervanagpu.NervanaGPU>`, :py:class:`NervanaCPU<neon.backends.nervanacpu.NervanaCPU>`
* Datasets - `CIFAR-10 <http://www.cs.toronto.edu/~kriz/cifar.html>`_, `MNIST <http://yann.lecun.com/exdb/mnist/>`_, `Penn Treebank <https://www.cis.upenn.edu/~treebank/>`_, `flickr8k <http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html>`_, `flickr30k <http://shannon.cs.illinois.edu/DenotationGraph/>`_, `COCO <http://mscoco.org/>`_, `Hutter-prize <http://mattmahoney.net/dc/textdata>`_, `William Shakespeare Text <http://cs.stanford.edu/people/karpathy/char-rnn>`_, `Imagenet 1K (Separate license and data download required) <http://image-net.org/>`_
* Initializers - :py:class:`Constant<neon.initializers.initializer.Constant>`, :py:class:`Uniform<neon.initializers.initializer.Uniform>`, :py:class:`Gaussian<neon.initializers.initializer.Gaussian>`, :py:class:`Glorot Uniform<neon.initializers.initializer.GlorotUniform>`
* Learning rules - :py:class:`Gradient Descent with Momentum<neon.optimizers.optimizer.GradientDescentMomentum>`, :py:class:`RMSProp<neon.optimizers.optimizer.RMSProp>`, :py:class:`AdaDelta<neon.optimizers.optimizer.Adadelta>`, :py:class:`Adam<neon.optimizers.optimizer.Adam>`, :py:class:`Adagrad<neon.optimizers.optimizer.Adagrad>`
* Activations - :py:class:`Rectified Linear<neon.transforms.activation.Rectlin>`, :py:class:`Softmax<neon.transforms.activation.Softmax>`, :py:class:`Tanh<neon.transforms.activation.Tanh>`, :py:class:`Logistic<neon.transforms.activation.Logistic>`, :py:class:`Identity<neon.transforms.activation.Identity>`
* Layers - :py:class:`Linear<neon.layers.layer.Linear>`, :py:class:`Convolution<neon.layers.layer.Convolution>`, :py:class:`Pooling<neon.layers.layer.Pooling>`, :py:class:`Deconvolution<neon.layers.layer.Deconv>`, :py:class:`Dropout<neon.layers.layer.Dropout>`, :py:class:`Recurrent<neon.layers.recurrent.Recurrent>`, :py:class:`Long Short-Term Memory<neon.layers.recurrent.LSTM>`, :py:class:`Gated Recurrent Unit<neon.layers.recurrent.GRU>`, :py:class:`BatchNorm<neon.layers.layer.BatchNorm>`
* Costs - :py:class:`Binary Cross Entropy<neon.transforms.cost.CrossEntropyBinary>`, :py:class:`Multiclass Cross Entropy<neon.transforms.cost.CrossEntropyMulti>`, :py:class:`Sum of Squares Error<neon.transforms.cost.SumSquared>`
* Metrics - :py:class:`Misclassification (Top 1 only)<neon.transforms.cost.Misclassification>`, :py:class:`Misclassification (Top1, Topk, LogLoss)<neon.transforms.cost.TopKMisclassification>`


You can choose to specify your model using a YAML or Python file.  To see an
example of how to construct the model in either format see
`How to run a model <user_guide.html#how-to-run-a-model>`__.
