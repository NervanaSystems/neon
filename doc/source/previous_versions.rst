.. ---------------------------------------------------------------------------
.. Copyright 2015-2017 Nervana Systems Inc.
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
neon v2.2.0
-----------

|Docs220|_

* Update MKLML version 20170908 that fixes a bug related to data conversions)
* Add SSD example for bounding box object detection that works for both GPU and MKL backend
* Add DeepSpeech2 MKL backend optimization that features ~3X improvement
* Update aeon to 1.0.0 including new version of manifest (doc/source/loading_data.rst#aeon-dataloader)
* Add CHWD Support for Batch Normalization in mkl backend
* Modify ResNet-50 model's last layer to match the original ResNet-50 model paper
* Enable Seq2Seq testing and benchmarking

neon v2.1.0
-----------

neon v2.1.0 released August 2, 2017 supporting:

* Set MKL backend (-b mkl) as the default CPU backend on Linux (use -b cpu to specify original CPU backend)
* Update MKLML version 20170720 (AVX512 code paths enabled by default and conversion optimizations)
* Simplify ResNet example
* Makefiles now check for virtualenv and pkg-config (NervanaSystems/neon#383)
* Fix Deep Speech2 model on MKL backend
* Fix MKL installation for "make sysinstall"

neon v2.0.0
-----------

|Docs200|_

neon v2.0.0 released June 27, 2017 supporting:

* Added support for MKL backend (-b mkl) on Linux, which boosts neon CPU performance significantly
* Added WGAN model examples for LSUN and MNIST data
* Enabled WGAN and DCGAN model examples for Python3
* Added fix (using file locking) to prevent race conditions running multiple jobs on the same machine with multiple GPUs
* Added functionality to display some information about hardware, OS and model used
* Updated appdirs to 1.4.3 to be compatibile on Centos 7.3 for appliance

neon v1.9.0
-----------

|Docs190|_

neon v1.9.0 released May 3, 2017 supporting:

* Add support for 3D deconvolution
* Generative Adversarial Networks (GAN) implementation, and MNIST DCGAN example, following GoodFellow 2014 (http://arXiv.org/abs/1406.2661)
* Implement Wasserstein GAN cost function and make associated API changes for GAN models
* Add a new benchmarking script with per-layer timings
* Add weight clipping for GDM, RMSProp, Adagrad, Adadelta and Adam optimizers
* Make multicost an explicit choice in mnist_branch.py example
* Enable NMS kernels to work with normalized boxes and offset
* Fix missing links in api.rst [#366]
* Fix docstring for --datatype option to neon [#367]
* Fix perl shebang in maxas.py and allow for build with numpy 1.12 [#356]
* Replace os.path.join for Windows interoperability [#351]
* Update aeon to 0.2.7 to fix a seg fault on termination

neon v1.8.2
-----------

|Docs182|_

neon v1.8.2 released February 23, 2017 supporting:

* Make the whale calls example stable and shuffle dataset before splitting into subsets
* Reduce default depth in cifar_msra example to 2
* Fix the formatting of the conv layer description
* Fix documentation error in the video-c3d example
* Support greyscale videos

neon v1.8.1
-----------

|Docs181|_

neon v1.8.1 released January 17, 2017 supporting:

* Bug fix: Add dilation to object dict and assign defaults to dil_w = dil_h = 1 [#335, #336]
* Bug fix: Prevent GPU backend from ignoring non-zero slope in Rectlinclip and change default slope to 0
* Bug fix: Nesterov momentum was updating velocities incorrectly

neon v1.8.0
-----------

|Docs180|_

neon v1.8.0 released December 28, 2016 supporting:

* Skip Thought Vectors (http://arxiv.org/abs/1506.06726) example
* Dilated convolution support
* Nesterov Accelerated Gradient option to SGD optimizer
* MultiMetric class to allow wrapping Metric classes
* Support for serializing and deserializing encoder-decoder models
* Allow specifying the number of time steps to evaluate during beam search
* A new community-contributed Docker image
* Improved error messages when a tensor is created with an invalid shape or reshaped to an incompatible size
* Fix bugs in MultiCost support
* Documentation fixes [#331]

neon v1.7.0
-----------

|Docs170|_

neon v1.7.0 released November 11 2016 supporting:

* Update Data Loader to aeon https://github.com/NervanaSystems/aeon
* Add Neural Machine Translation model
* Remove Fast RCNN model (use Faster RCNN model instead)
* Remove music_genres example
* Fix super blocking for small N with 1D conv
* Fix update-direct conv kernel for small N
* Add gradient clipping to Adam optimizer
* Documentation updates and bug fixes

neon v1.6.0
-----------

|Docs160|_

neon v1.6.0 released September 21 2016 supporting:

* Faster RCNN model
* Sequence to Sequence container and char_rae recurrent autoencoder model
* Reshape Layer that reshapes the input [#221]
* Pip requirements in requirements.txt updated to latest versions [#289]
* Remove deprecated data loaders and update docs
* Use NEON_DATA_CACHE_DIR envvar as archive dir to store DataLoader ingested data
* Eliminate type conversion for FP16 for CUDA compute capability >= 5.2
* Use GEMV kernels for batch size 1
* Alter delta buffers for nesting of merge-broadcast layers
* Support for ncloud real-time logging
* Add fast_style Makefile target
* Fix Python 3 builds on Ubuntu 16.04
* Run setup.py for sysinstall to generate version.py [#282]
* Fix broken link in mnist docs
* Fix conv/deconv tests for CPU execution and fix i32 data type
* Fix for average pooling with batch size 1
* Change default scale_min to allow random cropping if omitted
* Fix yaml loading
* Fix bug with image resize during injest
* Update references to the ModelZoo and neon examples to their new locations

neon v1.5.4
-----------

|Docs154|_

neon v1.5.4 released July 15 2016 supporting:

* Implement Binarized Neural Networks from http://arxiv.org/pdf/1602.02830v3.pdf
* Bug fixes [#268]

neon v1.5.3
-----------

|Docs153|_

neon v1.5.3 released July 7 2016 supporting:

* Bug fixes [#267]

neon v1.5.2
-----------

|Docs152|_

neon v1.5.2 released July 6 2016 supporting:

* Bug fixes to audio loader


neon v1.5.1
-----------

|Docs151|_

neon v1.5.1 released June 30 2016 supporting:

* Bug fixes

neon v1.5.0
-----------

|Docs150|_

neon v1.5.0 released June 29 2016 supporting:

* Python2/Python3 compatibility [#191]
* Support for Pascal GPUs
* Persistent RNN kernels [#262]
* Dataloader enhancements (audio loader with examples)
* HDF5 file data iterator
* Convolution kernel improvements
* Winograd kernel for fprop/bprop and 5x5 stride 1 filters
* API documentation improvements [#234, #244, #263]
* Cache directory cleanup
* Reorganization of all unit tests
* Check for compatible shapes before doing a memcpy [#182, #183]
* Bug fixes [#231, #241, #253, #257, #259]

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

.. |Docs220| replace:: Docs
.. |Docs200| replace:: Docs
.. |Docs190| replace:: Docs
.. |Docs182| replace:: Docs
.. |Docs181| replace:: Docs
.. |Docs180| replace:: Docs
.. |Docs170| replace:: Docs
.. |Docs160| replace:: Docs
.. |Docs154| replace:: Docs
.. |Docs153| replace:: Docs
.. |Docs152| replace:: Docs
.. |Docs151| replace:: Docs
.. |Docs150| replace:: Docs
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
.. _Docs220: http://neon.nervanasys.com/docs/2.2.0
.. _Docs200: http://neon.nervanasys.com/docs/2.0.0
.. _Docs190: http://neon.nervanasys.com/docs/1.9.0
.. _Docs182: http://neon.nervanasys.com/docs/1.8.2
.. _Docs181: http://neon.nervanasys.com/docs/1.8.1
.. _Docs180: http://neon.nervanasys.com/docs/1.8.0
.. _Docs170: http://neon.nervanasys.com/docs/1.7.0
.. _Docs160: http://neon.nervanasys.com/docs/1.6.0
.. _Docs154: http://neon.nervanasys.com/docs/1.5.4
.. _Docs153: http://neon.nervanasys.com/docs/1.5.3
.. _Docs152: http://neon.nervanasys.com/docs/1.5.2
.. _Docs151: http://neon.nervanasys.com/docs/1.5.1
.. _Docs150: http://neon.nervanasys.com/docs/1.5.0
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
