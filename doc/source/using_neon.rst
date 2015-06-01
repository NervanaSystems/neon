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

Using neon
==========

Running
-------
A command line executable named neon is included:

.. code-block:: bash

    neon my_example_cfg.yaml

Some example yaml files are available in the ``examples`` directory of the
package.  To understand options available with the command you can issue
``-h`` or ``--help``:

.. code-block:: bash

    neon --help

If working locally, you'll need to make sure the root ``neon`` directory is in
your python path:

.. code-block:: bash

    PYTHONPATH="." bin/neon my_example_cfg.yaml


Key neon Command-line Arguments
-------------------------------

* ``-h, --help`` describe the complete list of support arguments
* ``-g GPU, --gpu GPU`` Specify a graphics card accelerated backend.  Replace
  ``GPU`` with either `nervanagpu` or `cudanet` to match the backend you have
  installed.
* ``-i ID, --device_id ID`` Run neon under the selected accelerated device id.
  Useful in systems with multiple GPU cards.  Replace ``ID`` with the
  appropriate integer value.
* ``-r SEED, --rng_seed SEED`` Seed the random number generator with the value
  passed in for ``SEED`` (should be an integer).  This aids in reproducing
  prior results


Experiment File Format
----------------------
A `YAML <http://www.yaml.org/>`_ configuration file is used to control the
design of each experiment.  Below is a fully annotated example showing the
process to train and run inference on a toy network:

.. highlight:: bash

.. literalinclude:: ../../examples/ANNOTATED_EXAMPLE.yaml
   :linenos:


Parallelization
---------------
Read through the :doc:`distributed` section to see how to run model training in
data and model parallel modes using MPI.


.. _train_models:

Training Models and Learning Parameters
---------------------------------------
The human-readable YAML based markup format used by neon makes it easy to
define and change the hyperparameter values that control how the network looks
and performs.  To efficiently find optimal parameter values it is highly
recommended to utilize an automated tuning strategy, like that described in
:doc:`hyperparameter_tuning`

To prevent overfitting of the model parameters to limited training data, neon
supports several forms of network regularization. Direct weight regularization
is supported using weight decay, which is specified in the learning rule. Using
the :class:`GradientDescentMomentumWeightDecay<neon.optimizers.gradient_descent.GradientDescentMomentumWeightDecay>`
learning rule, weight decay on individual layers can be implemented.

DropOut regularization is supported with the
:class:`DropOutLayer<neon.layers.dropout.DropOutLayer>` layer type.


Saving Models
-------------
See :ref:`saving_models` to see how to periodically save model state to disk
and restore from a given saved snapshot.


Generating Predictions
----------------------
With a trained model, we can pass data through it and have it generate
predicted output values that we can save to disk for analysis.  The process for
doing so is described in :ref:`gen_predictions`


Reporting Performance
---------------------
See the :doc:`metrics` section to learn how to add one or more performance
metrics to be computed on one or more dataset partitions.


Working Interactively
---------------------
If you'd prefer not to utilize the ``neon`` command line executable and YAML
files to generate experiments, you have the option of writing python code
directly.

Here's a basic example showing how to train a simple MLP and generate
predictions:

.. highlight:: python

.. literalinclude:: ../../examples/mlp/mnist-small-noyaml.py
   :linenos:

An example using an `iPython notebook <https://github.com/NervanaSystems/neon/tree/master/examples/mlp/mnist-notebook.ipynb>` is also provided.
