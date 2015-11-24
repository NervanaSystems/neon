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


``neon`` module
---------------

.. automodule:: neon


``neon.backends``
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.backends.gen_backend
   neon.backends.backend.Tensor
   neon.backends.backend.Backend
   neon.backends.backend.OpTreeNode
   neon.backends.nervanacpu.CPUTensor
   neon.backends.nervanacpu.NervanaCPU
   neon.backends.nervanagpu.GPUTensor
   neon.backends.nervanagpu.NervanaGPU
   neon.backends.autodiff.Autodiff
   neon.backends.autodiff.GradNode
   neon.backends.autodiff.GradUtil.get_grad_back
   neon.backends.autodiff.GradUtil.is_invalid


``neon.callbacks``
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.callbacks.callbacks.Callbacks
   neon.callbacks.callbacks.Callback
   neon.callbacks.callbacks.SerializeModelCallback
   neon.callbacks.callbacks.TrainCostCallback
   neon.callbacks.callbacks.LossCallback
   neon.callbacks.callbacks.MetricCallback
   neon.callbacks.callbacks.HistCallback
   neon.callbacks.callbacks.ProgressBarCallback
   neon.callbacks.callbacks.TrainLoggerCallback
   neon.callbacks.callbacks.SaveBestStateCallback
   neon.callbacks.callbacks.EarlyStopCallback
   neon.callbacks.callbacks.DeconvCallback


``neon.data``
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.data.dataiterator.DataIterator
   neon.data.imageloader.ImageLoader
   neon.data.imagecaption.ImageCaption
   neon.data.datasets.load_cifar10
   neon.data.datasets.load_mnist
   neon.data.datasets.load_text
   neon.data.datasets.load_dataset
   neon.data.text.Text


``neon.initializers``
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.initializers.initializer.GlorotUniform
   neon.initializers.initializer.Constant
   neon.initializers.initializer.Gaussian
   neon.initializers.initializer.Uniform


``neon.layers``
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.layers.layer.Layer
   neon.layers.layer.Pooling
   neon.layers.layer.ParameterLayer
   neon.layers.layer.Convolution
   neon.layers.layer.Deconv
   neon.layers.layer.Linear
   neon.layers.layer.Bias
   neon.layers.layer.Activation
   neon.layers.layer.Affine
   neon.layers.layer.Conv
   neon.layers.layer.Dropout
   neon.layers.layer.LookupTable
   neon.layers.layer.GeneralizedCost
   neon.layers.layer.GeneralizedCostMask
   neon.layers.layer.BatchNorm
   neon.layers.recurrent.Recurrent
   neon.layers.recurrent.LSTM
   neon.layers.recurrent.GRU
   neon.layers.recurrent.RecurrentSum
   neon.layers.recurrent.RecurrentMean
   neon.layers.recurrent.RecurrentLast
   neon.layers.container.LayerContainer
   neon.layers.container.Sequential
   neon.layers.container.Tree
   neon.layers.container.MergeBroadcast
   neon.layers.container.MergeMultistream
   neon.layers.container.Multicost


``neon.models``
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.models.model.Model


``neon.optimizers``
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.optimizers.optimizer.Adadelta
   neon.optimizers.optimizer.Adagrad
   neon.optimizers.optimizer.Adam
   neon.optimizers.optimizer.GradientDescentMomentum
   neon.optimizers.optimizer.RMSProp
   neon.optimizers.optimizer.Schedule
   neon.optimizers.optimizer.ExpSchedule
   neon.optimizers.optimizer.MultiOptimizer

``neon.activations``
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.transforms.activation.Identity
   neon.transforms.activation.Rectlin
   neon.transforms.activation.Softmax
   neon.transforms.activation.Tanh
   neon.transforms.activation.Logistic

``neon.costs``
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.transforms.cost.CrossEntropyBinary
   neon.transforms.cost.CrossEntropyMulti
   neon.transforms.cost.SumSquared
   neon.transforms.cost.MeanSquared

``neon.metrics``
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.transforms.cost.Misclassification
   neon.transforms.cost.TopKMisclassification
   neon.transforms.cost.Accuracy


``neon.util``
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.util.argparser.NeonArgparser
   neon.util.persist.load_obj
   neon.util.persist.save_obj


``neon.visualizations``
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neon.visualizations.data.create_minibatch_x
   neon.visualizations.data.create_epoch_x
   neon.visualizations.data.h5_cost_data
   neon.visualizations.figure.x_label
   neon.visualizations.figure.cost_fig
