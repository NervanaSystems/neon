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

neon
====

:Release: |version|
:Date: |today|

|neo|_ is Intel Nervana_ 's reference deep learning framework committed to best
performance on all hardware. Designed for ease-of-use and extensibility.

Features include:

* Support for commonly used models including convnets, RNNs, LSTMs, and
  autoencoders.  You can find many pre-trained implementations of these in our
  `model zoo`_
* Tight integration with our `state-of-the-art`_ GPU kernel library
* 3s/macrobatch (3072 images) on AlexNet on Titan X (Full run on 1 GPU ~ 32 hrs)
* Basic automatic differentiation support
* Framework for visualization
* Swappable hardware backends: write code once and deploy on CPUs, GPUs, or Nervana hardware

New features in this release:

* Optimized DeepSpeech2 MKL backend performance (~7X improvement over the CPU backend)
* Fused convolution and bias layer which significantly boosted AlexNet and VGG performance on Intel architectures with MKL backend
* Made SSD and Faster-RNN use VGG weight files in new format
* Fixed use of reset_cells hyperparameter
* Fixed MKL backend bug for GAN and Faster-RCNN models
* See more in the `change log`_.

We use neon internally at Intel Nervana to solve our `customers' problems`_
in many domains. Consider joining us. We are hiring across several
roles. Apply here_!


.. |(TM)| unicode:: U+2122
   :ltrim:
.. _nervana: http://nervanasys.com
.. |neo| replace:: neon
.. _neo: https://github.com/nervanasystems/neon
.. _model zoo: https://github.com/NervanaSystems/ModelZoo
.. _state-of-the-art: https://github.com/soumith/convnet-benchmarks
.. _customers' problems: http://www.nervanasys.com/solutions
.. _here: http://www.nervanasys.com/careers
.. _highest performance: https://github.com/soumith/convnet-benchmarks
.. _change log: https://github.com/NervanaSystems/neon/blob/master/ChangeLog




..
.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Introduction

   installation.rst
   overview.rst
   running_models.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Essentials

   tutorials.rst
   model_zoo.rst
   backends.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: neon Fundamentals

   loading_data.rst
   datasets.rst
   layers.rst
   layer_containers.rst
   activations.rst
   costs.rst
   initializers.rst
   optimizers.rst
   learning_schedules.rst
   models.rst
   callbacks.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   faq.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: For Developers

   developer_guide.rst
   design.rst
   ml_operational_layer.rst

.. toctree::
    :hidden:

    resources.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Full API

   api.rst

.. toctree::
   :hidden:
   :caption: neon Versions

   previous_versions.rst
