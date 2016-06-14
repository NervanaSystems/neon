.. ---------------------------------------------------------------------------
.. Copyright 2015-2016 Nervana Systems Inc.
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
.. neon documentation master file

Previous Versions
=================

neon v1.4.0
-----------

|Docs140|_

neon v1.4.0 released Apr 29 2016 supporting:

* VGG16 based Fast R-CNN model using winograd kernels
* new, backward compatible, generic data loader
* C3D video loader model trained on UCF101 dataset
* Deep Dream example
* make conv layer printout more informative [#222]
* fix some examples to use new arg override capability
* improve performance for relu for small N
* better support for arbitrary batch norm layer placement
* documentation updates [#210, #213, #236]

neon v1.3.0
-----------

|Docs130|_

neon v1.3.0 released Mar 3 2016 supporting:

* Winograd kernels and associated autotuning routines
* benchmarking scripts
* deprecation of deterministic argument for backend constructor
* improve batch norm stability with fp16 backend
* allow strided support for dimshuffle kernel
* speed up zero momentum gradient descent

neon v1.2.2
-----------

|Docs122|_

neon v1.2.2 released Feb 24 2016 supporting:

* Benchmarking enhancements
* fast dimshuffle, transpose, other kernel speedups and refactoring
* batch norm states fix, deterministic updates
* example fixes for fast rcnn and conv_autoencoder
* image decoding rescaling method fix
* deserialization fixes for RNN's, refactoring
* caffe compatibility fixes
* documentation updates

neon v1.2.1
-----------

|Docs121|_

neon v1.2.1 released Feb 15 2016 supporting:

* New MergeSum, Colornoise layers
* support for aspect_ratio scaling augmentation
* updated IMDB sentiment analysis example
* generic CSV batchwriter
* various build and deserialization bugfixes, doc updates

neon v1.2.0
-----------

|Docs120|_

neon v1.2.0 released Jan 31 2016 supporting:

* Kepler GPU kernel support [#80]
* new dataloader format, updated docs [#115, #170]
* new serialization format
* FastRCNN implementation, ROI pooling support [#135]
* deep residual nets implementation and example
* expanded model zoo
* Ticker dataset and copy, repeat copy tasks
* autodiff transpose support [#173]
* numerous bug fixes and documentation updates.

neon v1.1.5
-----------

|Docs115|_

neon v1.1.5 released Jan 15 2016 supporting:

* CUDA kernels for lookuptable layer (up to 4x speedup)
* support for determinstic Conv layer updatesa
* LRN layer support
* custom dataset walkthrough utilizing bAbI data
* reduced number of threads in deep reduction EW kernels [#171]
* additional (de)serialization routines [#106]
* CPU tensor slicing fix
* corrections for PrecisionRecall, MultiLabelStats [#148]
* explicitly specify python2.7 for virtualenv [#155]
* default to SM50 when no working GPU found [#186]
* Add alpha to ELU activation [#164]
* deconv callback fix [#162]
* various documentation updates [#151, #152]


neon v1.1.4
-----------

|Docs114|_

neon v1.1.4 released Jan 4 2016 supporting:

* Add support for bidirectional RNNs and LSTMs
* added ELU, leaky ReLU activations
* significantly faster GPU kernel builds (using ptx instead of cuda-c)
* data shuffling enhancements, removal of old data loader code.
* caffe conv, pool, dropout layer matching and compatibility flags
* add scheduling support for RMSProp
* callback enhancements, additional unit tests
* documentation auditing, added links to introductory video tutorials

neon v1.1.3
-----------

|Docs113|_

neon v1.1.3 released Dec 1 2015 supporting:

* deconvolution and weight histogram visualization examples and documentation
* CPU convolution and pooling layer speedups (~2x faster)
* bAbI question and answer interactive demo, dataset support.
* various ImageLoader enhancements.
* interactive usage improvements (shortcut Callback import, multiple Callbacks init, doc updates, single item batch size support)
* set default verbosity level to warning
* CIFAR10 example normalization updates
* CUDA detection enhancements [#132]
* only parse batch_writer arguments when used as a script, allow undefined global_mean [#137, #140]


neon v1.1.2
-----------

|Docs112|_

neon v1.1.2 released Nov 17 2015 supporting:

* completely re-written C++ multithreaded dataloader
* new weight initialization options for recurrent layers
* Added deconvolution visualization support (guided backprop)
* new bAbI question answering example network
* Improved performance of cifar10_allcnn, word_lstm examples
* new CUDA-C max and avg pooling kernels
* Additional bugfixes and documentation updates


neon v1.1.1
-----------

|Docs111|_

neon v1.1.1 released Nov 6 2015 supporting:

* Callback initialization bug fix [#127]
* IMDB LSTM example bug fix [#130]
* Added cuda-convnet2 style binary dropout variant
* Added benchmark function to model (separate fprop, bprop, update timings)
* Remove h_buffer references in lieu of outputs for recurrent layers
* Multi-cost output buffer bugfix for inference [#131]
* New timeseries prediction and generation example
* Change Callback initialization to re-support named arguments. Separate out these arguments in argparser. [#128]

neon v1.1.0
-----------

|Docs110|_

neon v1.1.0 released Oct 30 2015 supporting:

* Sentiment analysis support (LSTM lookupTable based), new IMDB example
* Support for merge and branch layer stacks via LayerContainers
  * Sequential, Tree, MergeBroadcast, MergeMultiStream
* Support for freezing layer stacks
* Adagrad optimizer support
* new GPU kernels for fast compounding batch norm, conv and pooling engine updates, new kernel build system and flags.
* Modifications for Caffe support

  * conv, pooling, P/Q updates, dropout layer normalization more in-line with Caffe approach. NOTE: this breaks backwards compatibility with some strided conv/pool related models serialized using older versions of neon as the output sizes may now be different. See the FAQ for more info.
  * serialization enhancements to make caffe model import/export easier
  * use per-channel mean subtraction instead of single global. NOTE: this breaks backwards compatibility with ImgMaster saved datasets prior to this revision. To correct, please use the included update_dataset_cache.py script in the util directory.

* Default training cost display during progress bar is now calculated on a rolling window basis rather than from the beginning of each epoch
* Separate Layer configuration and initialization steps
* YAML based alexnet example
* Callback enhancements.

  * now pass args instead of having to spell out callbacks in each example
  * Changed validation callback to loss callback, validation_frequency now evaluation_frequency
  * Generic metric callback.

* Various bug fixes

  * non-contiguous array get for GPUTensors
  * 1D slicing returns 2D matrices
  * bin/neon serialization fixes for RNNs
  * 3D conv fixes for fprop, bprop
  * batch norm inference fix
  * bias layer size fix

* Documentation updates and improvements

neon v1.0.0
-----------

|Docs100|_

neon v1.0.0 released Sep 9 2015, a major top to bottom re-write of
the codebase that features the following enhancements:

* RNN/LSTM

  * Code is cleaner and achieves state of the art results on the Penn Tree Bank dataset using RNN/LSTM/GRU
  * Fast image captioning model (~200x faster than CPU based NeuralTalk) on flickr8k dataset

* Basic automatic differentiation support
* Framework for visualizations (supported via callbacks)
* Top-down refactoring & redesign to enable quicker iteration while keeping the speedups offered by our nervanagpu kernels

  * Datasets are easier to specify
  * Backend now uses OpTrees (similar to nervanagpu) to support autodiff
  * nervanagpu merged in as a neon backend to simplify development and use
  * YAML syntax is simplified (but not backwards compatible)
  * Better documentation and wider test coverage

neon v0.9.0
-----------

|Docs9|_ 

neon v0.9.0 supports:

* Hyperparameter optimization
* Multi GPU 

neon v0.8.2
------------

|Docs8|_

neon v0.8.2 supports:

* Integration with our cudanet_ fork of Alex Krizhevsky's cuda-convnet2 library for Kepler GPU is

We will add support for previous generation GPUs, multi-GPU and hyperparameter optimization in the
upcoming releases. 

neon v0.8.1
------------

Initial public release of neon.

.. |Docs140| replace:: Docs
.. |Docs130| replace:: Docs
.. |Docs122| replace:: Docs
.. |Docs121| replace:: Docs
.. |Docs120| replace:: Docs
.. |Docs115| replace:: Docs
.. |Docs114| replace:: Docs
.. |Docs113| replace:: Docs
.. |Docs112| replace:: Docs
.. |Docs111| replace:: Docs
.. |Docs110| replace:: Docs
.. |Docs100| replace:: Docs
.. |Docs9| replace:: Docs
.. |Docs8| replace:: Docs
.. _cudanet: https://github.com/NervanaSystems/cuda-convnet2
.. _Docs140: http://neon.nervanasys.com/docs/1.4.0
.. _Docs130: http://neon.nervanasys.com/docs/1.3.0
.. _Docs122: http://neon.nervanasys.com/docs/1.2.2
.. _Docs121: http://neon.nervanasys.com/docs/1.2.1
.. _Docs120: http://neon.nervanasys.com/docs/1.2.0
.. _Docs115: http://neon.nervanasys.com/docs/1.1.5
.. _Docs114: http://neon.nervanasys.com/docs/1.1.4
.. _Docs113: http://neon.nervanasys.com/docs/1.1.3
.. _Docs112: http://neon.nervanasys.com/docs/1.1.2
.. _Docs111: http://neon.nervanasys.com/docs/1.1.1
.. _Docs110: http://neon.nervanasys.com/docs/1.1.0
.. _Docs100: http://neon.nervanasys.com/docs/1.0.0
.. _Docs9: http://neon.nervanasys.com/docs/0.9.0
.. _Docs8: http://neon.nervanasys.com/docs/0.8.2
