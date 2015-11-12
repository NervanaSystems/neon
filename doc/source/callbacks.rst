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

Callbacks
=========
Neon provides a callback api for operations performed during model fit.

Callbacks are classes that derive from
:py:class:`Callback<neon.callbacks.callbacks.Callback>` and implement one or
more of the provided on_[train, minibatch, epoch]_[begin, end] functions.

A :py:class:`Callbacks<neon.callbacks.callbacks.Callbacks>` object is created
once in the experiment definition and provided to model.fit(), which calls
each of the callback functions at the appropriate times.

Callback implementations are provided for computing cost, displaying runtime
state, and saving state or interrupting training based on performance. For a
complete list of provided callback implementations see the
:py:class:`neon.callbacks` directory.

In the following example the Callbacks ``__init__`` method takes a reference
to the model and ``train_set`` object, which are needed by most callbacks.  It
also takes optional arguments that control creation of other utility
callbacks for computing validation cost and displaying a progress bar.

.. code-block:: python

    # configure default callbacks for computing train and validation cost
    # and displaying a progress bar
    callbacks = Callbacks(model, train_set, eval_set=valid_set, **args.callback_args)

    # add a callback that saves the best model state
    callbacks.add_save_best_state_callback("./best_state.pkl")

    # pass callbacks to model, which calls the callback functions during fit
    model.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs,
            cost=cost, callbacks=callbacks)

:py:class:`Callbacks<neon.callbacks.callbacks.Callbacks>` provides a shared
data mechanism that allows callbacks to decouple computation of metrics from
further processing or consumption of those metrics.  For example the
:py:class:`LossCallback<neon.callbacks.callbacks.LossCallback>`
evaluates the training loss/cost function on the provided evaluation set at some configurable
epoch frequency.  This metric is used by the
:py:class:`ProgressBarCallback<neon.callbacks.callbacks.ProgressBarCallback>`
for display purposes, and by the
:py:class:`SaveBestStateCallback<neon.callbacks.callbacks.SaveBestStateCallback>`
to decide when to save state.  Such decoupling prevents having to recompute
the validation cost in several callbacks.

Callback shared data can be optionally saved to a file for archival or
visualization with the ``nvis`` tool. To enable saving the callback data,
provide the optional output_file arg to the Callbacks ``__init__`` function.

.. code-block:: python

    # save callback data to disk
    callbacks = Callbacks(model, train_set, output_file="./data.h5")

Users can create their own callbacks by subclassing
:py:class:`Callback<neon.callbacks.callbacks.Callback>` and providing a
helper function in :py:class:`Callbacks <neon.callbacks.callbacks.Callbacks>`
to register it and provide it with a reference to the shared callback data
object.
