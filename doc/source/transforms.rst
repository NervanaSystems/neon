.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.
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

Transforms
==========

Transforms are functions that can be applied to modify data values.
Often these will represent things like non-linear activation functions, or
cost/loss functions.


Available Transforms
--------------------

.. autosummary::
   :toctree: generated/

   neon.transforms.linear.Linear
   neon.transforms.rectified.RectLin
   neon.transforms.rectified.RectLeaky
   neon.transforms.logistic.Logistic
   neon.transforms.tanh.Tanh
   neon.transforms.softmax.Softmax
   neon.transforms.batch_norm.BatchNorm

   neon.transforms.sum_squared.SumSquaredDiffs
   neon.transforms.cross_entropy.CrossEntropy
   neon.transforms.xcov.XCovariance
