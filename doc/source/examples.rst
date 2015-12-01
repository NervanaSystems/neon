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
..  ---------------------------------------------------------------------------

Examples
*********

Convolutional Neural Nets (CNN)
===============================

Example 1 - All CNN on CIFAR10
------------------------------
This is an implementation of the All-CNN style convolutional network
(convolution layers with large strides are used in place of alternating max
pooling layers usually employed).  See: http://arxiv.org/pdf/1412.6806.pdf

cifar10_allcnn.py_

.. _cifar10_allcnn.py: https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_allcnn.py

Example 2 - Small CNN on CIFAR10
--------------------------------
cifar10_conv.py_

.. _cifar10_conv.py: https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_conv.py

Convolutional Auto-encoder (CAE)
================================

Example 1 - MNIST
-----------------
This is a convolutional autoencoder for the MNIST data set. It reconstructs an image
using the deconvolution layers. Matplotlib is required in order to generate the plots.

conv_autoencoder.py_

.. _conv_autoencoder.py: https://github.com/NervanaSystems/neon/blob/master/examples/conv_autoencoder.py

Recurrent Neural Nets (RNN)
===========================

Example 1 - Single Unit RNN on Penn Treebank (char level)
---------------------------------------------------------
This example trains a network with one recurrent layer of tanh units on Penn Treebank data, parsed on
the character level.

char_rnn.py_

.. _char_rnn.py: https://github.com/NervanaSystems/neon/blob/master/examples/char_rnn.py

Example 2 - LSTM or GRU layer on Penn Treebank (char level)
-----------------------------------------------------------
This example trains a network using a LSTM layer or GRU layer on Penn Treebank data, parsed on the
character level. Inside the example script, you can switch between using a LSTM or GRU Layer.

char_lstm.py_

.. _char_lstm.py: https://github.com/NervanaSystems/neon/blob/master/examples/char_lstm.py

Example 3 - LSTM or GRU layer on Penn Treebank (word level)
-----------------------------------------------------------
This example trains a network using a LSTM layer or GRU layer on Penn Treebank data, parsed on the word level.
Inside the example script, you can switch between using a LSTM or GRU Layer.

word_lstm.py_

.. _word_lstm.py: https://github.com/NervanaSystems/neon/blob/master/examples/word_lstm.py

Example 4 - LSTM on Shakespeare data (char level)
-------------------------------------------------
This example trains a network using a LSTM on Shakespeare data, parsed on the character level. After training,
the network will generate text from a seed sequence.

text_generation_lstm.py_

.. _text_generation_lstm.py: https://github.com/NervanaSystems/neon/blob/master/examples/text_generation_lstm.py

Example 5 - Time series learning and prediction
-------------------------------------------------
This example trains a network using a LSTM on synthetic multi-dimensional time series data. After training,
the network will generate the sequences. The results can be visualized by the plots generated as PNG files.

timeseries_lstm.py_

.. _timeseries_lstm.py: https://github.com/NervanaSystems/neon/blob/master/examples/timeseries_lstm.py

Q&A model
===================

Example 1 - Baseline GRU/LSTM on bAbI dataset
----------------------------------------------------
This is an implementation of Facebook's baseline GRU/LSTM model on the bAbI dataset. 
This model connects two paths of networks to process the story and question seperately. Each path includes word embedding and GRU/LSTM layers. Refer to the README in the babi directory for how to run the interactive demo.

train.py_
demo.py_

.. _train.py: https://github.com/NervanaSystems/neon/blob/master/examples/babi/train.py
.. _demo.py: https://github.com/NervanaSystems/neon/blob/master/examples/babi/demo.py

Fully-connected Nets (MLP)
==========================

Example 1 - CIFAR10
-------------------

cifar10.py_

.. _cifar10.py: https://github.com/NervanaSystems/neon/blob/master/examples/cifar10.py

Example 2 - MNIST
-----------------
This example can be enabled with different neon features through
command line arguments. Through the command line, you can choose whether or not to save the model
via serialization.

mnist_mlp.py_

.. _mnist_mlp.py: https://github.com/NervanaSystems/neon/blob/master/examples/mnist_mlp.py

Example 3 - MNIST in YAML
-------------------------
This is the same example as MLP on MNIST, but it is implemented using YAML.

mnist_mlp.yaml_

.. _mnist_mlp.yaml: https://github.com/NervanaSystems/neon/blob/master/examples/mnist_mlp.yaml

Image caption model
===================

Example 1 - VGG features
------------------------
This model connects image features with sentences to learn how to caption unseen images.
It concatenates the precomputed VGG features and a sentence and uses that as data to train the RNN.

image_caption.py_

.. _image_caption.py: https://github.com/NervanaSystems/neon/blob/master/examples/image_caption.py

Merge layer example
===================

Example 1 - MNIST
-----------------
This example uses MNIST data to demonstrate how to use a network that merges two paths of input.

mnist_merge.py_

.. _mnist_merge.py: https://github.com/NervanaSystems/neon/blob/master/examples/mnist_merge.py

Multiple optimizer example
==========================

Example 1 - MNIST
-----------------
This example demonstrates the ability to apply different optimizers to different layers, or
different components of the same layer. The multi-optimizer will pair layers and optimizers using
layer names.

multi_optimizer.py_

.. _multi_optimizer.py: https://github.com/NervanaSystems/neon/blob/master/examples/multi_optimizer.py

Early stopping example
======================

Example 1 - MNIST
-----------------
This model trains a MLP using MNIST data and stops the training when a stopping criterion is satisfied
or when the number of training epochs is completed, whichever happens first.

early_stopping.py_

.. _early_stopping.py: https://github.com/NervanaSystems/neon/blob/master/examples/early_stopping.py
