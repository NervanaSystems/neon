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

Hyperparameter optimization
===========================

Finding good hyperparameters for deep networks is quite tedious to do manually
and can be greatly accelerated by performing automated hyperparameter tuning.

To this end, third-party hyperparameter optimization packages can be integrated
with our framework. We currently offer support for
`Spearmint <https://github.com/JasperSnoek/spearmint>`_, which have forked and
slightly extended to work with neon.

Installing our Spearmint Fork
-----------------------------

First you'll need to install the following dependencies:

* `python <http://www.python.org/>`_ 2.7+
* `numpy <http://www.numpy.org/>`_ 1.6.1+
* `scipy <http://www.numpy.org/>`_ 1.6.1+
* `google protocol buffers <https://developers.google.com/protocol-buffers/>`_
* `flask <http://flask.pocoo.org/>`_ for visualizing results.

Then you'll need to checkout and install our Spearmint fork which you can do
via:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/spearmint
    cd spearmint/spearmint/spearmint
    python setup.py install


YAML file changes
-----------------

#. create a new yaml file with the top level experiment of type
   :py:class:`neon.experiments.fit_predict_err.FitPredictErrorExperiment` with
   an additional ``return_item`` argument specifying which dataset (``test``,
   ``train``, or ``validation``) to measure the objective function on.  This
   objective function value will be used to compare whether one parameter
   setting is better than another.
#. In the model specification of the yaml simply replace a given numeric
   hyper-parameter with a range of values over which to search.  The format for
   specifying this range is of the form:
   ``!hyperopt name type start_range end_range`` where name is an identifier,
   type should be one of ``FLOAT`` or ``INT`` to match the type of values being
   tuned, and the two range parameters specify the lowest and highest parameter
   values to search between.


.. code-block:: bash

    !obj:experiments.FitPredictErrorExperiment {
        return_item: test,
        # ...
    },

    !obj:models.MLP {
        # specifying a range from 0.01 to 0.1 for the learning rate
        learning_rate: !hyperopt lr FLOAT 0.01 0.1,
        # ...
    },


Experiment Initialization
-------------------------

Hyperparameter optimization requires two additional environment variables to
identify the ``spearmint/bin`` directory and the desired location to store
temporary files and results of the experiment, such as:

.. code-block:: bash

    export SPEARMINT_PATH=/path/to/spearmint/spearmint/bin
    export HYPEROPT_PATH=/path/to/hyperopt_experiment

To initialize a new experiment, you make use of the ``hyperopt`` executable
that is installed as part of neon.  To the executable we need to use the
``init`` flag and pass the ``-y`` argument to specify the yaml file containing
the hyperparameter ranges, for example:

.. code-block:: bash

    hyperopt init -y examples/mlp/iris-hyperopt-small.yaml

This creates a spearmint configuration file in protobuf format in the
experiment directory.

Running
-------

* Once an experiment has been initialized, it can be run by calling ``hyperopt``
  with the ``run`` flag and specifying a port with the ``-p`` argument where
  outputs will be generated, for example:

.. code-block:: bash

    hyperopt run -p 50000

* The output can be viewed in the browser at http://localhost:50000, or by
  directly inspecting the files in the experiment directory.
* The experiment will keep running indefinitely. It can be interrupted with
  ``Ctrl+C`` and continued by calling the ``hyperopt run`` command again.

* To start a new experiment, reset the previous one either by manually deleting
  the contents of the experiment directory, or by running:

.. code-block:: bash

    hyperopt reset

