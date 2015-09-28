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

Optimizers
===========
These classes define how the learning proceeds, that is, how updates are
applied to the parameters of the network.

Gradient Descent with Momentum
------------------------------
.. autosummary::
   neon.optimizers.optimizer.GradientDescentMomentum

RMSProp
----------------------------
.. autosummary::
   neon.optimizers.optimizer.RMSProp

Adadelta
--------
.. autosummary::
   neon.optimizers.optimizer.Adadelta

Adam
----
.. autosummary::
   neon.optimizers.optimizer.Adam
