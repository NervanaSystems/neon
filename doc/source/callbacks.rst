.. ---------------------------------------------------------------------------
.. Copyright 2016 Nervana Systems Inc.
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

Callbacks
=========

Neon provides a callback API for operations performed during model fit.

Callbacks are classes that derive from |Callback| and implement one or more of
the provided ``on_[train, minibatch, epoch]_[begin, end]`` functions.

A Callbacks object is created once in the experiment definition and
provided to ``model.fit()``, which calls each of the callback functions
at the appropriate times.

.. code-block:: python

    # creates a Callbacks object with the provided model, validation set, and any
    # callback-related command line arguments.
    callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

    # pass callbacks during training
    model.fit(train_set, optimizer=opt, cost=cost, callbacks=callbacks)

Neon implements the following callbacks. Callbacks with an asterisk (\*)
are enabled by default.


.. csv-table::
   :header: "Name", "Description"
   :widths: 20, 40
   :escape: ~

   \* :py:class:`RunTimerCallback<neon.callbacks.callbacks.RunTimerCallback>`, Tracks total training time
   \* :py:class:`TrainCostCallback<neon.callbacks.callbacks.TrainCostCallback>`, Computes average training cost for each minibatch
   \* :py:class:`ProgressBarCallback<neon.callbacks.callbacks.ProgressBarCallback>`, Live progress bar for training
   \* :py:class:`TrainLoggerCallback<neon.callbacks.callbacks.TrainLoggerCallback>`, Logs training progress every `epoch_freq` epochs and `minibatch_freq` minibatches.
   :py:class:`SerializeModelCallback<neon.callbacks.callbacks.SerializeModelCallback>`, Saves the model every `epoch_freq` epochs. Can be enabled with the `--serialize [epoch_freq]` command line argument.
   :py:class:`LossCallback<neon.callbacks.callbacks.LossCallback>`, Computes loss every epoch. Can be enabled with the `--eval_freq [epoch_freq]` command line argument (validation set must be passed to Callback).
   :py:class:`MetricCallback<neon.callbacks.callbacks.MetricCallback>`, Computes a given `metric` every `epoch_freq` epochs. Can be enabled with the `--eval_freq [epoch_freq]` command line argument (metric must be passed to Callback)
   :py:class:`MultiLabelStatsCallback<neon.callbacks.callbacks.MultiLabelStatsCallback>`, Computes multi-label metrics (e.g. PrecisionRecall) every `epoch_freq` epochs
   :py:class:`HistCallback<neon.callbacks.callbacks.HistCallback>`, Collect histograms of weights of all layers once per minibatch/epoch. Histograms stored to hdf5 output file for visualization with `nvis` tool.
   :py:class:`SaveBestStateCallback<neon.callbacks.callback.SaveBestStateCallback>`, Saves the best model so far (defined as the loss on the validation set) to the file provided in `path`.
   :py:class:`EarlyStopCallback<neon.callbacks.callbacks.EarlyStopCallback>`, Halts training when a threshold is triggered (such as reaching a performance target)
   :py:class:`DeconvCallback<neon.callbacks.callbacks.DeconvCallback>`, Stores projections of the activations back to pixel space using guided backpropagation `(Springenberg~, 2014) <http://arxiv.org/abs/1412.6806>`__. Used for visualization with the `nvis` tool.
   :py:class:`BatchNormTuneCallback<neon.callbacks.callbacks.BatchNormTuneCallback>`, Callback for tuning batch norm parameters with unbiased estimators for global mean and variance.
   :py:class:`WatchTickerCallback<neon.callbacks.callbacks.WatchTickerCallback>`, Callback that examines a single input output pair using a validation set.


Callbacks are added in three different ways:

1. Use the :py:class:`.add_callback` method.

  .. code-block:: python

    callbacks.add_callback(LossCallback(eval_set=valid_set, epoch_freq=1))

2. For some callbacks, use a provided convenience function

   .. code-block:: python

       callbacks.add_hist_callback(plot_per_mini=True)

3. Some callbacks can be enabled from the command line arguments. First,
   create Callbacks via ``callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)``
   This passes command line arguments to Callbacks. Then, use the following
   command line arguments:

   .. code-block:: bash

        # enables LossCallback, provided that an
        # eval_set is specified in the python script
        ./mnist_mlp.py --eval_freq 1

        # enables SerializeModelCallback
        ./mnist_mlp.py --serialize 2 --save_path mlp.o

Example usage
-------------

In the following example, the Callbacks ``__init__`` method takes a
reference to the model and any command line callbacks. The method then
generates the default callbacks (see asterisks above). Here we add a
callback to save the best performing model in the output file
``"best_state.pkl"``

.. code-block:: python

    # configure default callbacks for computing train and validation cost
    # and displaying a progress bar. Here we pass eval_freq=1 to create the
    # LossCallback needed for the SaveBestStateCallback
    callbacks = Callbacks(model, eval_set=valid_set, eval_freq=1)

    # add a callback that saves the best model state
    callbacks.add_save_best_state_callback("./best_state.pkl")

    # pass callbacks to model, which calls the callback functions during fit
    model.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs,
            cost=cost, callbacks=callbacks)

Callback dependencies
---------------------

Some callbacks depend on other callbacks to work. For example, the
:py:class:`.SaveBestStateCallback` depends on :py:class:`.LossCallback` to compute the
loss used to determine when to save the model.

Callbacks provide a data sharing mechanism that allows callbacks to
decouple computation of metrics from further processing or consumption
of those metrics. For example the :py:class:`.LossCallback` evaluates the
training loss/cost function on the provided validation set at a
configurable epoch frequency. Such decoupling prevents unnecessary
re-computation of the validation cost.

Callback shared data can also be saved to a file for archival or
visualization purposes. To save the callback data, provide the optional
``output_file`` argument to the Callback's ``__init__`` function. For
example,

.. code-block:: python

    # save callback data to disk
    callbacks = Callbacks(model, train_set, output_file="./data.h5")

Creating callbacks
------------------

To create a custom callback, subclass from |Callback| and implement
one or more of the following functions

.. code-block:: python

    # Arguments:
    #     callback_data (HDF5 dataset): shared data between callbacks
    #     model (Model): model object
    #     epoch (int): index of current epoch
    #     epochs (int): total number of epochs
    #     minibatch (int): index of minibatch that is ending

    def on_train_begin(self, callback_data, model, epochs):

    def on_train_end(self, callback_data, model):

    def on_epoch_begin(self, callback_data, model, epoch):

    def on_epoch_end(self, callback_data, model, epoch):

    def on_minibatch_begin(self, callback_data, model, epoch, minibatch):

    def on_minibatch_end(self, callback_data, model, epoch, minibatch):

.. |Callback| replace:: :py:class:`Callback<neon.callbacks.callbacks.Callback>`
