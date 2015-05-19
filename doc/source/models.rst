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

Models
======


Available Models
----------------

.. autosummary::
   :toctree: generated/

   neon.models.mlp.MLP
   neon.models.autoencoder.Autoencoder
   neon.models.rbm.RBM
   neon.models.dbn.DBN
   neon.models.rnn.RNN
   neon.models.balance.Balance

.. _extending_model:

Adding a new type of Model
--------------------------

#. Create a new subclass of :class:`neon.models.model.Model`
#. At a minimum implement :func:`neon.models.model.Model.fit` to learn
   parameters from a training dataset
#. Write :func:`neon.models.model.Model.predict` to apply learned parameters
   to make predictions about another dataset.

Saving models
-------------

Models can be saved by using the `serialized_path` option in the yaml file.
To save snapshots of the model during training, use the `serialize_schedule`
option.  The schedule can be either a list of ints if specific epochs are to
be saved or a single int if saving is meant to occur at a constant interval.
