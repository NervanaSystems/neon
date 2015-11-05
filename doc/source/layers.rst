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

Layers
======

Neural network models are typically composed of a data source, several weight
layers and a cost layer.  In addition to standard weight layers, there
is a variety of model specific weight, pooling and normalization layers.
Each layer represents a grouping of nodes (artificial neurons).


Pooling Layers
--------------
.. autosummary::
   neon.layers.layer.Pooling

Parameter Layer
---------------
.. autosummary::
   neon.layers.layer.ParameterLayer

Convolutional Layers
--------------------
.. autosummary::
   neon.layers.layer.Convolution
   neon.layers.layer.Deconvolution

Linear Layer
------------
.. autosummary::
   neon.layers.layer.Linear

Bias Layer
----------
.. autosummary::
   neon.layers.layer.Bias

Activation Layer
----------------
.. autosummary::
   neon.layers.layer.Activation

Composite Layers
----------------
.. autosummary::
   neon.layers.layer.Affine
   neon.layers.layer.Conv
   neon.layers.layer.Deconv

Dropout Layers
--------------
.. autosummary::
   neon.layers.layer.Dropout
   neon.layers.layer.DropoutBinary


LookupTable Layer
-----------------
.. autosummary::
   neon.layers.layer.LookupTable

Cost
----
.. autosummary::
   neon.layers.layer.GeneralizedCost
   neon.layers.layer.GeneralizedCostMask
   neon.layers.container.Multicost

Batch Norm Layer
----------------
.. autosummary::
   neon.layers.layer.BatchNorm
   neon.layers.layer.BatchNormAutodiff

Gated Recurrent Unit Layer
--------------------------
.. autosummary::
   neon.layers.recurrent.GRU

Long Short-Term Memory Layer
----------------------------
.. autosummary::
   neon.layers.recurrent.LSTM

Recurrent Layer
----------------
.. autosummary::
   neon.layers.recurrent.Recurrent
   neon.layers.recurrent.RecurrentOutput
   neon.layers.recurrent.RecurrentSum
   neon.layers.recurrent.RecurrentMean
   neon.layers.recurrent.RecurrentLast
