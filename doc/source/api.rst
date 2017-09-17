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
.. neon API documentation

API
===

This API documentation covers each model within neon. Most modules have a
corresponding user guide section that introduces the main concepts. See this
API for specific function definitions.

.. csv-table::
    :header: "Module API", "Description", "User Guide"
    :widths: 20, 40, 30
    :delim: |

    :py:mod:`neon` | Holds NervanaObject, the base object available to all other classes |
    :py:mod:`neon.backends` | Computational backend (CPU, MKL or GPU) | :doc:`neon backend<backends>`
    :py:mod:`neon.data` | Data loading and handling | :doc:`Data loading<loading_data>`, :doc:`Datasets<datasets>`
    :py:mod:`neon.models` | Model architecture | :doc:`Models<models>`
    :py:mod:`neon.layers` | Layer objects | :doc:`Layers<layers>`, :doc:`Creating new layers<creating_new_layers>`, :doc:`Layer containers<layer_containers>`
    :py:mod:`neon.initializers` | Weight initializer methods | :doc:`Initializers<initializers>`
    :py:mod:`neon.transforms` | Activation functions and Costs/Metrics | :doc:`Activations<activations>`, :doc:`Costs and Metrics<costs>`
    :py:mod:`neon.callbacks` | Callbacks during model training | :doc:`Callbacks<callbacks>`
    :py:mod:`neon.optimizers` | Learning algorithms | :doc:`Optimizers<optimizers>`, :doc:`Learning schedules<learning_schedules>`
    :py:mod:`neon.visualizations` | Visualization of training cost and weight histograms | :doc:`Visualizing results<tools>`
    :py:mod:`neon.util` | Utility module |


``neon``
--------
.. py:module: neon

The base (global) object ``NervanaObject`` contains the attribute ``be``, the reference to the computational
backend.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.NervanaObject


``neon.backends``
-----------------
.. py:module:: neon.backends

This module defines the computational backend of neon, either based on CPU or GPU
hardware. Included are classes that implement neon's auto-differentiation feature.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.backends.gen_backend
   neon.backends.backend.Tensor
   neon.backends.backend.Backend
   neon.backends.backend.OpTreeNode
   neon.backends.backend.Block
   neon.backends.nervanacpu.CPUTensor
   neon.backends.nervanacpu.NervanaCPU
   neon.backends.nervanamkl.MKLTensor
   neon.backends.nervanamkl.NervanaMKL
   neon.backends.nervanagpu.GPUTensor
   neon.backends.nervanagpu.NervanaGPU
   neon.backends.autodiff.Autodiff
   neon.backends.autodiff.GradNode
   neon.backends.autodiff.GradUtil.get_grad_back
   neon.backends.autodiff.GradUtil.is_invalid


``neon.data``
-------------
.. py:module:: neon.data

Data-related classes and methods comprise this module, including methods for loading data
and iterating through minibatches of data during training.

.. autosummary::
  :toctree: generated/
  :nosignatures:

  neon.data.dataiterator.NervanaDataIterator
  neon.data.dataiterator.ArrayIterator
  neon.data.hdf5iterator.HDF5Iterator
  neon.data.hdf5iterator.HDF5IteratorAutoencoder
  neon.data.hdf5iterator.HDF5IteratorOneHot

.. warning:: The :py:class:`.DataLoader` and :py:class:`.ImageLoader` classes were deprecated in favor of the new Aeon-based DataLoader. For documentation of the aeon package, see http://aeon.nervanasys.com.

The new Aeon-based dataloader supports several classes that perform transformations on the data provisioned by aeon:

.. autosummary::
  :toctree: generated/
  :nosignatures:

  neon.data.dataloader_transformers.DataLoaderTransformer
  neon.data.dataloader_transformers.OneHot
  neon.data.dataloader_transformers.PixelWiseOneHot
  neon.data.dataloader_transformers.TypeCast
  neon.data.dataloader_transformers.BGRMeanSubtract
  neon.data.dataloader_transformers.DumpImage

Dataset objects for storing data from common modalities (e.g. Text), as well as specific stock datasets (e.g. MNIST, CIFAR-10, Penn Treebank) are included.

.. autosummary::
  :toctree: generated/
  :nosignatures:

  neon.data.datasets.Dataset
  neon.data.image.MNIST
  neon.data.image.CIFAR10
  neon.data.imagecaption.ImageCaption
  neon.data.imagecaption.Flickr8k
  neon.data.imagecaption.Flickr30k
  neon.data.imagecaption.Coco
  neon.data.text.Text
  neon.data.text.Shakespeare
  neon.data.text.PTB
  neon.data.text.HutterPrize
  neon.data.text.IMDB
  neon.data.questionanswer.QA
  neon.data.questionanswer.BABI
  neon.data.ticker.Ticker
  neon.data.ticker.Task
  neon.data.ticker.CopyTask
  neon.data.ticker.RepeatCopyTask
  neon.data.ticker.PrioritySortTask

``neon.models``
---------------
.. py:module:: neon.models

The Model class stores a list of layers describing the model. Methods are provided
to train the model weights, perform inference, and save/load the model.

.. autosummary::
 :toctree: generated/
 :nosignatures:

 neon.models.model.Model


``neon.layers``
---------------
.. py:module:: neon.layers

This modules contains class definitions for common neural network layers. Base
layers from which other layers are subclassed are

.. autosummary::
    :toctree: generated/
    :nosignatures:

    neon.layers.layer.Layer
    neon.layers.layer.ParameterLayer
    neon.layers.layer.CompoundLayer

Common Layers

.. autosummary::
    :toctree: generated/
    :nosignatures:

    neon.layers.layer.Bias
    neon.layers.layer.Linear
    neon.layers.layer.Affine
    neon.layers.layer.Dropout
    neon.layers.layer.LookupTable
    neon.layers.layer.Activation
    neon.layers.layer.BatchNorm
    neon.layers.layer.BatchNormAutodiff
    neon.layers.layer.Pooling
    neon.layers.layer.LRN
    neon.layers.layer.DataTransform
    neon.layers.layer.BranchNode
    neon.layers.layer.SkipNode

Convolutional Layers

.. autosummary::
    :toctree: generated/
    :nosignatures:

    neon.layers.layer.Convolution
    neon.layers.layer.Conv
    neon.layers.layer.Deconvolution
    neon.layers.layer.Deconv

Recurrent Layers

.. autosummary::
    :toctree: generated/
    :nosignatures:

    neon.layers.recurrent.Recurrent
    neon.layers.recurrent.LSTM
    neon.layers.recurrent.GRU
    neon.layers.recurrent.BiRNN
    neon.layers.recurrent.BiLSTM
    neon.layers.recurrent.DeepBiRNN
    neon.layers.recurrent.DeepBiLSTM
    neon.layers.recurrent.RecurrentOutput
    neon.layers.recurrent.RecurrentSum
    neon.layers.recurrent.RecurrentMean
    neon.layers.recurrent.RecurrentLast

Containers govern the structure of the model. For a linear cascade of layers,
the ``Sequential`` container is sufficient. Models that have branching and merging
should use the other containers.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    neon.layers.container.LayerContainer
    neon.layers.container.Sequential
    neon.layers.container.Tree
    neon.layers.container.SingleOutputTree
    neon.layers.container.Broadcast
    neon.layers.container.MergeSum
    neon.layers.container.MergeBroadcast
    neon.layers.container.MergeMultistream
    neon.layers.layer.RoiPooling

Generic cost layers are implemented in the following classes. Note that these
classes subclass from `NervanaObject`, not any base layer class.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    neon.layers.layer.GeneralizedCost
    neon.layers.layer.GeneralizedCostMask
    neon.layers.container.Multicost


``neon.initializers``
---------------------
.. py:module:: neon.initializers

Layer weights can be initialized with the following approaches

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.initializers.initializer.Initializer
   neon.initializers.initializer.Array
   neon.initializers.initializer.Constant
   neon.initializers.initializer.Gaussian
   neon.initializers.initializer.IdentityInit
   neon.initializers.initializer.Uniform
   neon.initializers.initializer.GlorotUniform
   neon.initializers.initializer.Kaiming
   neon.initializers.initializer.Orthonormal
   neon.initializers.initializer.Xavier

``neon.transforms``
-------------------
.. py:module:: neon.transforms

This modules contain activation functions, costs, and metrics.


Activation functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.transforms.transform.Transform
   neon.transforms.activation.Identity
   neon.transforms.activation.Explin
   neon.transforms.activation.Rectlin
   neon.transforms.activation.Softmax
   neon.transforms.activation.Tanh
   neon.transforms.activation.Logistic
   neon.transforms.activation.Normalizer

Costs
~~~~~

.. autosummary::
  :toctree: generated/
  :nosignatures:

  neon.transforms.cost.Cost
  neon.transforms.cost.CrossEntropyBinary
  neon.transforms.cost.CrossEntropyMulti
  neon.transforms.cost.SumSquared
  neon.transforms.cost.MeanSquared
  neon.transforms.cost.LogLoss

Metrics
~~~~~~~

.. autosummary::
  :toctree: generated/
  :nosignatures:

  neon.transforms.cost.Metric
  neon.transforms.cost.Misclassification
  neon.transforms.cost.TopKMisclassification
  neon.transforms.cost.Accuracy
  neon.transforms.cost.PrecisionRecall
  neon.transforms.cost.ObjectDetection

``neon.optimizers``
-------------------
.. py:module:: neon.optimizers

neon implements the following learning algorithms for updating the weights.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.optimizers.optimizer.Optimizer
   neon.optimizers.optimizer.GradientDescentMomentum
   neon.optimizers.optimizer.RMSProp
   neon.optimizers.optimizer.Adadelta
   neon.optimizers.optimizer.Adagrad
   neon.optimizers.optimizer.Adam
   neon.optimizers.optimizer.MultiOptimizer

For some optimizers, users can adjust the learning rate over the course of training
by providing a schedule.

.. autosummary::
  :toctree: generated/
  :nosignatures:

  neon.optimizers.optimizer.Schedule
  neon.optimizers.optimizer.StepSchedule
  neon.optimizers.optimizer.PowerSchedule
  neon.optimizers.optimizer.ExpSchedule
  neon.optimizers.optimizer.PolySchedule

``neon.callbacks``
------------------
.. py:module:: neon.callbacks

Callbacks are methods that are called at user-defined times during training. They can
be scheduled to occur at the beginning/end of training/minibatch/epoch. Callbacks can
be used to, for example, periodically report training loss or save weight histograms.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.callbacks.callbacks.Callbacks
   neon.callbacks.callbacks.Callback
   neon.callbacks.callbacks.RunTimerCallback
   neon.callbacks.callbacks.TrainCostCallback
   neon.callbacks.callbacks.ProgressBarCallback
   neon.callbacks.callbacks.TrainLoggerCallback
   neon.callbacks.callbacks.SerializeModelCallback
   neon.callbacks.callbacks.LossCallback
   neon.callbacks.callbacks.MetricCallback
   neon.callbacks.callbacks.MultiLabelStatsCallback
   neon.callbacks.callbacks.HistCallback
   neon.callbacks.callbacks.SaveBestStateCallback
   neon.callbacks.callbacks.EarlyStopCallback
   neon.callbacks.callbacks.DeconvCallback
   neon.callbacks.callbacks.BatchNormTuneCallback
   neon.callbacks.callbacks.WatchTickerCallback

``neon.visualizations``
-----------------------
.. py:module:: neon.visualizations

This module generates visualizations using the ``nvis`` command line function.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.visualizations.data
   neon.visualizations.figure


``neon.util``
-------------
.. py:module:: neon.util

Useful utility functions, including parsing the command line and saving/loading
of objects.

.. autosummary::
  :toctree: generated/
  :nosignatures:

  neon.util.argparser.NeonArgparser
  neon.util.argparser.extract_valid_args
  neon.util.compat
  neon.util.persist.load_class
  neon.util.persist.load_obj
  neon.util.persist.save_obj
  neon.util.modeldesc.ModelDescription
  neon.util.yaml_parse
