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

Welcome to neon! The typical workflow for a deep learning model is as
follows:


1. **Generate a backend**

   The backend defines where computations are executed in neon. We support CPU, MKL, and GPU (Pascal, Maxwell or Kepler architectures) backends.

   See :doc:`neon backend <backends>`

2. **Load data**

   Neon supports loading of both common and custom datasets. Data should
   be loaded as a python iterator, providing one minibatch of data at a
   time during training.

   See :doc:`Datasets <datasets>`, :doc:`Data loaders <loading_data>`,

3. **Specify model architecture** (layers, activation functions, weight
   initializers)

   Create your model by providing a list of layers. For layers with
   weights, provide a function to initialize the weights prior to
   training.

   See :doc:`Layers <layers>`, :doc:`Layer
   containers <layer_containers>`,
   :doc:`Activations <activations>`, :doc:`Initializers <initializers>`.

4. **Train model**

   To train a model, provide the training data (as an iterator), cost
   function, and an optimization algorithm for updating the model's
   weights. To modify the learning rate over the training
   time, provide a learning schedule.

   See :doc:`Datasets <datasets>`, :doc:`Costs and
   metrics <costs>`, :doc:`Optimizers <optimizers>`,
   and :doc:`Learning schedules <learning_schedules>`

5. **Evaluate**

   Evaluate a trained model based on a validation dataset and a provided
   Metric.

   See :doc:`Models <models>`, :doc:`Costs and
   metrics <costs>`

Neon Features
~~~~~~~~~~~~~

Neon currently supports the following:

-  Backends - :py:class:`NervanaGPU<neon.backends.nervanagpu.NervanaGPU>`, :py:class:`NervanaCPU<neon.backends.nervanacpu.NervanaCPU>`, :py:class:`NervanaCPU<neon.backends.nervanamkl.NervanaMKL>`
-  Datasets

   -  Images: MNIST, CIFAR-10, ImageNet 1K, PASCAL VOC, Mini-Places2
   -  Text: IMDB, Penn Treebank, Shakespeare Text, bAbI, Hutter-prize
   -  Video: UCF101
   -  Others: flickr8k, flickr30k, COCO
   -  Custom datasets

-  Initializers - :py:class:`Constant<neon.initializers.initializer.Constant>`, :py:class:`Uniform<neon.initializers.initializer.Uniform>`, :py:class:`Gaussian<neon.initializers.initializer.Gaussian>`, :py:class:`Glorot Uniform<neon.initializers.initializer.GlorotUniform>`, :py:class:`Xavier<neon.initializers.initializer.Xavier>`, :py:class:`Kaiming<neon.initializers.initializer.Kaiming>`, :py:class:`IdentityInit<neon.initializers.initializer.IdentityInit>`, :py:class:`Orthonormal<neon.initializers.initializer.Orthonormal>`
-  Optimizers - :py:class:`Gradient Descent with Momentum<neon.optimizers.optimizer.GradientDescentMomentum>`, :py:class:`RMSProp<neon.optimizers.optimizer.RMSProp>`, :py:class:`Adadelta<neon.optimizers.optimizer.Adadelta>`, :py:class:`Adam<neon.optimizers.optimizer.Adam>`, :py:class:`Adagrad<neon.optimizers.optimizer.Adagrad>`, :py:class:`MultiOptimizer<neon.optimizers.optimizer.MultiOptimizer>`
- Activations - :py:class:`Rectified Linear<neon.transforms.activation.Rectlin>`, :py:class:`Softmax<neon.transforms.activation.Softmax>`, :py:class:`Tanh<neon.transforms.activation.Tanh>`, :py:class:`Logistic<neon.transforms.activation.Logistic>`, :py:class:`Identity<neon.transforms.activation.Identity>`, :py:class:`ExpLin<neon.transforms.activation.Explin>`
-  Layers - :py:class:`Linear<neon.layers.layer.Linear>`, :py:class:`Convolution<neon.layers.layer.Convolution>`, :py:class:`Pooling<neon.layers.layer.Pooling>`, :py:class:`Deconvolution<neon.layers.layer.Deconv>`, :py:class:`Dropout<neon.layers.layer.Dropout>`, :py:class:`Recurrent<neon.layers.recurrent.Recurrent>`, :py:class:`Long Short-Term Memory<neon.layers.recurrent.LSTM>`, :py:class:`Gated Recurrent Unit<neon.layers.recurrent.GRU>`, :py:class:`BatchNorm<neon.layers.layer.BatchNorm>`, :py:class:`LookupTable<neon.layers.layer.LUT>`, :py:class:`Local Response Normalization<neon.layers.layer.LRN>`, :py:class:`Bidirectional-RNN<neon.layers.recurrent.BiRNN>`, :py:class:`Bidirectional-LSTM<neon.layers.recurrent.BiLSTM>`
- Costs - :py:class:`Binary Cross Entropy<neon.transforms.cost.CrossEntropyBinary>`, :py:class:`Multiclass Cross Entropy<neon.transforms.cost.CrossEntropyMulti>`, :py:class:`Sum of Squares Error<neon.transforms.cost.SumSquared>`
- Metrics - Misclassification (:py:class:`Top1<neon.transforms.cost.Misclassification>`, :py:class:`TopK<neon.transforms.cost.TopKMisclassification>`), :py:class:`LogLoss<neon.transforms.cost.LogLoss>`, :py:class:`Accuracy<neon.transforms.cost.Accuracy>`, :py:class:`PrecisionRecall<neon.transforms.cost.PrecisionRecall>`, :py:class:`ObjectDetection<neon.transforms.cost.ObjectDetection>`
