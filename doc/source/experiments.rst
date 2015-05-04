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

Experiments
===========

Current Implementations
-----------------------

.. autosummary::
   :toctree: generated/

   neon.experiments.experiment.Experiment
   neon.experiments.fit.FitExperiment
   neon.experiments.fit_predict_err.FitPredictErrorExperiment
   neon.experiments.check_grad.GradientChecker

.. _extending_experiment:

Adding a new type of Experiment
-------------------------------

#. Subclass :class:`neon.experiments.experiment.Experiment`

.. _gen_predictions:

Generating Predictions
----------------------
Model predictions can be saved to disk when running a
:class:`neon.experiments.fit_predict_err.FitPredictErrorExperiment`.  To do so
add the following inside your top-level experiment:

.. code-block:: yaml

    predictions: ['train', 'test'],

This will result in the generation of new python serialized object (.pkl)
files being written to the directory in which your dataset resides.  The
generated file will have the suffix ``-inference.pkl``, and will contain a
numpy ndarray object of predicted model outputs for that dataset (one per row).

In the example above we've requested saved outputs for the training and test
datasets, though 'validation' datasets can also be included if supported.

If you have a trained model you'd like to use just for generating predictions
(i.e. don't bother training from scratch), this can be accomplished as follows:

* Train your model, being sure to save model parameters to disk and use an
  Experiment of class
  :class:`neon.experiments.fit_predict_err.FitPredictErrorExperiment`.
  This can be accomplished by adding the following to the model definition in
  your yaml file (note that you can use any model type not just MLP's):

.. code-block:: yaml

    model: !obj:models.MLP {
        serialized_path: './my_mlp_model.pkl',

        # other model parameters here...
    }

* Simply re-run your experiment file.  Even though the type of Experiment
  includes a fit step, this ends up being skipped as your previously saved
  model will be loaded and there are no further epochs to run.  The reason why
  your model gets loaded is because the yaml contains a ``serialized_path``
  model parameter, though you can also set ``deserialized_path`` too which
  would take priority.  One of the attributes that gets saved when we
  serialize a model is how many epochs have been run.  Since this will match
  the total number of epochs requested, no further fitting is required and we
  go straight to generating predictions.

* This strategy can also be used to checkpoint a model periodically, or
  continue training from a previous point.  Follow the steps above but prior
  to re-running your YAML file, edit the model section as follows (this will
  start training at epoch 101 if you previously saved at 100 epochs and now
  would like 1000 total):

.. code-block:: yaml

    model: !obj:models.MLP {
        num_epochs: 1000, # set it to a larger number than previously run
        overwrite_list: ['num_epochs'],

        serialize_path: './my_mlp_model.pkl',

        # other model parameters here...
    }
