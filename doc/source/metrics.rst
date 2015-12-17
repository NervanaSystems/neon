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

Metrics
=======
These classes represent various performance evaluation functions that can be
employed to determine how accurate the learned model is against a given
dataset.

Misclassification
-----------------
.. autosummary::
   neon.transforms.cost.Misclassification
   neon.transforms.cost.TopKMisclassification

Computes the fraction of data samples that have been misclassified. Multiply by 100 to obtain misclassification percentage.


Accuracy
--------
.. autosummary::
   neon.transforms.cost.Accuracy

Computes the fraction of data samples that have been correctly classified. Multiply by 100 to obtain accuracy percentage.


Precision and Recall
--------------------
.. autosummary::
   neon.transforms.cost.PrecisionRecall

Computes the precision (portion of samples correctly predicted to have a given
class out of all predictions made of that class), and recall (portion of
samples predicted to have a given class out of all samples that are actually of
that class).
