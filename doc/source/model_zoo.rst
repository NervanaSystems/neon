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

Model Zoo
=========

Neon features fast implementations of most state-of-the-art models
reported in the academic literature.

Several examples are packaged with neon in the ``neon/examples`` folder.
Note that these are sometimes shortened models to reduce training time,
and are meant to illustrate different ways to use neon.

Our `Model Zoo`_ contains complete models, with python scripts as well as
pre-trained weights. For the latest updates, we recommend paying a visit to
the `Model Zoo`_. Admission is free!

Multilayer Perceptron (MLP)
---------------------------

These are the simplest models, applying multilayer perceptron (MLP) to
the problem of recognizing handwritten digits (MNIST dataset). One
example is included with the CIFAR-10 dataset (60,000 natural images
from 10 categories).


.. csv-table::
   :header: "Model", "Description"
   :widths: 20, 40
   :escape: ~

   `mnist_mlp.py <https://github.com/NervanaSystems/neon/blob/master/examples/mnist_mlp.py>`__, Simple MLP model
   `mnist_branch.py <https://github.com/NervanaSystems/neon/blob/master/examples/mnist_branch.py>`__, Small MLP with multiple branches
   `mnist_merge.py <https://github.com/NervanaSystems/neon/blob/master/examples/mnist_merge.py>`__, MLP model that demonstrates merging
   `cifar10.py <https://github.com/NervanaSystems/neon/blob/master/examples/cifar10.py>`__, Small MLP applied to natural images

Convolutional Neural Networks
-----------------------------

Convolutional neural networks are the state-of-art architecture for many
image and video processing problems. The main datasets involved are:

1. `ImageNet <http://image-net.org/>`__: a large corpus of 1 million natural images (256x256 pixels), divided into 1000 categories.

2. `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`__ : 60,000 natural images (32 x 32 pixels) from 10 categories.

3. `PASCAL_VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`__: A subset of ImageNet images with object bounding boxes.

4. `UCF101 <http://crcv.ucf.edu/data/UCF101.php>`__: 13,320 videos from 101 action categories.

5. `Mini-Places2 <http://6.869.csail.mit.edu/fa15/project.html>`__: Subset of the Places2 dataset. Includes 100,000 images from 100 scene categories.

Example scripts
~~~~~~~~~~~~~~~

These python scripts are found in the ``neon/examples`` folder. While
these examples load a particular image dataset, in principle they can be
adapted to any dataset.

.. csv-table::
   :header: "Model", "Dataset", "Description"
   :widths: 20, 15, 40
   :escape: ~
   :delim: |

   `cifar10_allcnn.py <https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_allcnn.py>`__| CIFAR-10| All-convolutional neural network
   `cifar10_conv.py <https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_conv.py>`__| CIFAR-10| Small all-convolution network demonstrating use of fp16 data format
   `cifar10_msra <https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_msra>`__| CIFAR-10| Deep residual network detailed in `He, 2015`_
   `alexnet.py <https://github.com/NervanaSystems/neon/blob/master/examples/imagenet/alexnet.py>`__| ImageNet| Implementation of `AlexNet`_
   `imagenet_allcnn.py <https://github.com/NervanaSystems/neon/blob/master/examples/imagenet/allcnn.py>`__| ImageNet| All-convolutional network based on `Springenberg, 2014`_
   `fast_rcnn <https://github.com/NervanaSystems/neon/tree/master/examples/fast-rcnn>`__| PASCAL VOC| Fast region-based CNN (`R-CNN`_) for object localization and detection. Uses a pre-trained VGG16 network trained on ImageI1K to initialize the convolution layers.
   `conv_autoencoder.py <https://github.com/NervanaSystems/neon/blob/master/examples/conv_autoencoder.py>`__| MNIST| Autoencoder convolutional network that reconstructs the image with deconvolutional layers

Model Zoo
~~~~~~~~~

Our model zoo also includes complete models with both the model script
and pre-trained weights upon which to build your networks. The links
below lead to individual pages where you can download the model and
weights.


.. csv-table::
   :header: "Model", "Dataset", "Description"
   :widths: 20, 15, 40
   :escape: ~
   :delim: |

   `Alexnet <https://github.com/NervanaSystems/ModelZoo/tree/master/ImageClassification/ILSVRC2012/Alexnet>`__ | ImageNet| Implementation of Alexnet described in `Krizhevsky, 2012`_
   `VGG <https://github.com/NervanaSystems/ModelZoo/tree/master/ImageClassification/ILSVRC2012/VGG>`__ | ImageNet| Adapted the 16 and 19 layer `VGG <http://arxiv.org/abs/1409.1556>`__ model from Caffe for use with neon.
   `GoogleNet`_| ImageNet| 22-layer CNN with multiple branches. See `Szegedy, 2014`_
   `ALLCNN`_| CIFAR10| All convolutional model inspired by `Springenberg, 2014`_
   `DeepResNet <https://github.com/apark263/cfmz>`__ | CIFAR10| Deep residual network detailed in `He, 2015`_
   `DeepResNet <https://github.com/hunterlang/mpmz/>`__| mini-Places2| `Deep residual network`_ for scene classification
   `FastRCNN`_| Pascal-VOC| `Fast-RCNN model`_ for object localization. The CNN layers are seeded by Alexnet pre-trained in neon with ImageNet.
   `C3D model`_| UCF101| `3D convolutional networks`_ for video action recognition


Recurrent Neural Networks
-------------------------

Neon has implementations for all-to-all recurrent neural networks
(RNNs), as well as Long short-term memory (LSTM) networks, and Gated
Recurrent Units (GRU) networks. Training datasets include:

1. `Penn Treebank (PTB) <https://www.cis.upenn.edu/~treebank/>`__: Text corpus with ~1 million words. Vocabulary is limited to 10,000 words. The task is predicting downstream words/characters.

2. `Shakespeare <http://cs.stanford.edu/people/karpathy/char-rnn/>`__: Complete text from Shakespeare's works.

3. `IMDB reviews <https://s3.amazonaws.com/text-datasets>`__: 25,000 movie reviews, labeled as positive or negative

4. `Facebook bAbI <https://research.facebook.com/researchers/1543934539189348>`__: As set of 20 question & answer tasks, each with 1,000 training examples.

5. `Flickr8k <http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html>`__, `COCO <http://mscoco.org/>`__: Images with associated caption (sentences). Flickr8k consists of 8,092 images captioned by AmazonTurkers with ~40,000 captions. COCO has 328,000 images, each with 5 captions. The COCO images also come with labeled objects using segmentation algorithms.

Example scripts
~~~~~~~~~~~~~~~

These examples scripts, found in ``neon/examples`` demonstrate how to
load and preprocess text data (for some models) and construct the
recurrent networks.

.. csv-table::
   :header: "Model", "Dataset", "Description"
   :widths: 20, 15, 40
   :escape: ~
   :delim: |

   `word_lstm.py <https://github.com/NervanaSystems/neon/blob/master/examples/word_lstm.py>`__| PTB (word) | `LSTM`_/GRU network for prediction
   `char_lstm.py <https://github.com/NervanaSystems/neon/blob/master/examples/char_lstm.py>`__| PTB (char) | `LSTM/GRU`_ network for prediction
   `char_rnn.py <https://github.com/NervanaSystems/neon/blob/master/examples/char_rnn.py>`__| PTB (char)| One-layer RNN with tanh units for prediction
   `text_generation_lstm.py <https://github.com/NervanaSystems/neon/blob/master/examples/text_generation_lstm.py>`__| Shakespeare | Trains an LSTM network then demonstrates how to draw samples from the network
   `timeseries_lstm.py <https://github.com/NervanaSystems/neon/blob/master/examples/timeseries_lstm.py>`__| Time series| Trains a network on a synthetic time series and generates sequences
   `imdb_lstm.py <https://github.com/NervanaSystems/neon/blob/master/examples/imdb_lstm.py>`__| IMDB| Performs sentiment analysis on IMDB (see `Li, 2015`_)
   `image_caption.py <https://github.com/NervanaSystems/neon/blob/master/examples/image_caption.py>`__| Flickr, COCO| This model connects image features with sentences to learn how to caption unseen images. Uses precomputed VGG features and a sentence to train a LSTM. See `Karpathy Neural Talk`_.

Model Zoo
~~~~~~~~~

.. csv-table::
   :header: "Model", "Dataset", "Description"
   :widths: 20, 15, 40
   :escape: ~
   :delim: |

   `Image Captioning`_| Flickr8k | Image captioning model based on `Vinyals, 2015`_ using `precomputed`_ VGG features.
   `Question & Answering`_| bABI| Facebook's baseline `GRU/LSTM model`_
   `Sentiment analysis`_| IMDB| LSTM model for classifying movie reviews as positive/negative (`Li, 2015`_)

Other Examples
--------------
.. csv-table::
   :header: "Model", "Dataset", "Description"
   :widths: 20, 15, 40
   :escape: ~
   :delim: |

   `Deep-Q Network`_ | Atari video games | Deep reinforcement learning model to play video games (based on `Minh, 2015`_)


.. |(TM)| unicode:: U+2122
   :ltrim:
.. _Model Zoo: https://github.com/NervanaSystems/ModelZoo
.. _AlexNet: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
.. _He, 2015: http://arxiv.org/abs/1512.03385
.. _Springenberg, 2014: http://arxiv.org/pdf/1412.6806.pdf
.. _R-CNN: http://arxiv.org/pdf/1504.08083v2.pdf
.. _Krizhevsky, 2012: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
.. _GoogleNet: https://github.com/NervanaSystems/ModelZoo/tree/master/ImageClassification/ILSVRC2012/Googlenet
.. _Szegedy, 2014: http://arxiv.org/pdf/1409.4842.pdf
.. _AllCNN: https://github.com/NervanaSystems/ModelZoo/tree/master/ImageClassification/CIFAR10/All_CNN
.. _Deep residual network: http://arxiv.org/abs/1512.03385
.. _FastRCNN: https://github.com/NervanaSystems/ModelZoo/tree/master/ObjectLocalization/FastRCNN
.. _Fast-RCNN model: http://arxiv.org/pdf/1504.08083v2.pdf
.. _C3D model: https://github.com/NervanaSystems/neon/tree/master/examples/video-c3d
.. _3D convolutional networks: http://arxiv.org/pdf/1412.0767v4.pdf
.. _LSTM: http://arxiv.org/pdf/1308.0850.pdf
.. _LSTM/GRU: https://github.com/karpathy/char-rnn
.. _Li, 2015: http://arxiv.org/pdf/1503.00185v5.pdf
.. _Karpathy Neural Talk: https://github.com/karpathy/neuraltalk
.. _Image Captioning: https://github.com/NervanaSystems/ModelZoo/tree/master/ImageCaptioning/LSTM
.. _Vinyals, 2015: http://arxiv.org/abs/1411.4555
.. _precomputed: http://cs.stanford.edu/people/karpathy/deepimagesent/
.. _Question & Answering: https://github.com/NervanaSystems/ModelZoo/tree/master/NLP/QandA/bAbI
.. _Sentiment analysis: https://github.com/NervanaSystems/ModelZoo/tree/master/NLP/SentimentClassification/IMDB
.. _Deep-Q Network: https://github.com/tambetm/simple_dqn
.. _Minh, 2015: http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
.. _GRU/LSTM model: https://research.facebook.com/researchers/1543934539189348
