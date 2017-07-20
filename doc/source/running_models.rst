.. ---------------------------------------------------------------------------
.. Copyright 2015-2017 Nervana Systems Inc.
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
..  ---------------------------------------------------------------------------

Running models
==============

With the virtual environment activated, there are two ways to run models
through neon. The first is to simply execute the python script
containing the model (with ``-b mkl``), as mentioned before:

.. code-block:: bash

    examples/mnist_mlp.py # equivalent to examples/mnist_mlp.py -b mkl

This will run the multilayer perceptron (MLP) model and print the final
misclassification error after 10 training epochs. On the first run, neon will download the MNIST dataset. It will create a ``~/nervana`` directory where the raw datasets are kept. The data directory can be controlled with the ``-w`` flag.

The second method is to specify the model in a YAML file.
`YAML <http://yaml.org/>`__ is a widely-used markup language. For
examples, see the YAML files in the ``examples`` folder. To run the YAML
file for the MLP example, enter from the neon repository directory:

.. code-block:: bash

    neon examples/mnist_mlp.yaml

In a YAML file, the mkl backend can be specified by adding ``backend: mkl``.

Arguments
---------

Both methods accept command line arguments to configure how you would
like to run the model. For a full list, type ``neon --help`` in the
command line. Some commonly used flags include:

.. csv-table::
   :header: "Flag", "Description"
   :widths: 20, 50
   :escape: ~

   ``-w~, --data_dir``, Path to data directory (default: ``nervana/data``)
   ``-e~, --epochs``, Number of epochs to run during training (default: ``10``)
   ``-s~, --save_path``, Path to save the model snapshots (default: ``None``)
   ``-o~, --output_file``, Path to save the metrics and callback data generated during training. Can be used by ``nvis`` for visualization  (default: ``None``)
   ``-b~, --backend {cpu,mkl,gpu}``, Which backend to use (default: ``mkl``)
   ``-z~, --batch_size``, Batch size for training (default: ``128``)
   ``-v~``, Verbose output. Displays each layer's shape information.
