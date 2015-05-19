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

.. _saving_models:

Saving models
-------------

Models can be saved by including the ``serialized_path`` option in the yaml
file, and passing a string specifying the path and file name of the saved
model.  Models are saved as serialized python dictionaries.

To periodically save snapshots of the model during training (useful to
checkpoint long running jobs), use the ``serialize_schedule`` parameter.  The
schedule can either be a list of ints if specific epochs are to be saved or a
single int if saving is meant to occur at a constant interval.  As an example:

.. code-block:: yaml

    model: !obj:models.MLP {
      num_epochs: 100,
      serialized_path: './my_mlp_model.prm',
      serialize_schedule: [25, 50, 70, 90],

      # other parameters ...

The example above would result in the current model state being save to disk in
the file ``my_mlp_model.prm`` at the end of training the 25th, 50th, 70th, and
90th epochs respectively.  The existing file will be overwritten at each step.
Alternatively, the example:

.. code-block:: yaml

    model: !obj:models.MLP {
      num_epochs: 100,
      serialized_path: './my_mlp_model.prm',
      serialize_schedule: 10,

      # other parameters ...

would result in the model state being snapshot to disk in ``my_mlp_model.prm``
after each of the 10th, 20th, 30th, 40th, ... epochs of training.


Loading saved models
--------------------

As an alternative to training models from scratch each time, it is beneficial
to be able to save and later restore a given model's state so that training can
resume from that given point in time.  See :ref:`saving_models` to learn how to
write saved model state to disk.

Armed with a particular saved model file, we can restore and utilize it in the
training process by setting either the ``deserialized_path`` or
``serialized_path`` model parameters to point at this file.
``serialized_path`` is lower priority and will only be examined if
``deserialized_path`` does not exist.

Restoring a model in this way is typically useful for generating predictions
but if you'd like to train your model further from this loaded state, you'll
also likely need to increase the ``num_epochs`` parameter.  Since this is
saved as part of the model's state, the model will not train further unless
this is overridden.  Here's an example showing how to take a saved model
called ``my_mlp_model.prm`` trained for 10 epochs, load it and train it for a
further 90 epochs from that state:

.. code-block:: yaml

    model: !obj:models.MLP {
      overwrite_list: ['num_epochs'],
      num_epochs: 100,
      deserialized_path: './my_mlp_model.prm',

      # other parameters ...

