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

Metrics
=======

Metrics are used to quantitatively measure error or some aspect of model
performance (typically against some Dataset partition).

Reporting Metric Values
-----------------------

To report metrics you need to specify them at the Experiment level and choose
FitPredictErrorExperiment as your type.  In the YAML file you'd define the 
metrics dictionary and list the metrics to be computed for each dataset
partition.  Here's an example:

.. code-block:: yaml

    metrics: {
      "train": [
        !obj:metrics.MisclassRate(),
      ],
      "test": [
        !obj:metrics.AUC(),
        !obj:metrics.LogLossMean(),
      ],
      "validation": [
        !obj:metrics.MisclassPercentage(),
      ],
    },


Available Metrics
-----------------

.. autosummary::
   :toctree: generated/

   neon.metrics.misclass.MisclassSum
   neon.metrics.misclass.MisclassRate
   neon.metrics.misclass.MisclassPercentage

   neon.metrics.roc.AUC

   neon.metrics.loss.LogLossSum
   neon.metrics.loss.LogLossMean

   neon.metrics.sqerr.SSE
   neon.metrics.sqerr.MSE
