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

Distributed Implementations using MPI
=====================================
In an effort to reduce the amount of time it takes to train and run models, it
is often advantageous to split the computation across several processes and
nodes so that they can be run in parallel.

In neon, we currently have some preliminary support for this via
`MPI <http://www.open-mpi.org/>`_ and
`mpi4py <https://github.com/mpi4py/mpi4py>`_ (see :ref:`mpi_install`) to
install the required dependencies.

Note that distributed processing support in neon is still very experimental.
Performance speed-ups (at increasing scale) are still forthcoming.


Available Models
----------------

Existing Models and Datasets can be parallelized by adding the ``--datapar`` or
``--modelpar`` command line parameters.

In the ``--datapar`` (data parallel) approach, data examples are partitioned
and distributed across multiple processes.  A separate model replica lives on
each process, and parameter values are synchronized across the models
to ensure each replica remains (eventually) consistent.

In the ``--modelpar`` (model parallel) approach, layer nodes are partitioned
and distributed across multiple processes.  Activations are then communicated
between processes whose nodes are connected.  At this time, we support model
parallelism on fully connected model layers only.

Parameter server based asynchronous SGD is not yet implemented, but please
contact us if this is something you need for your use case.

Examples
--------

The following example illustrates how to train a data parallel convnet on the
MNIST dataset using 4 neon processes (on the same host):

.. code-block:: bash

    mpirun -n 4 neon --datapar examples/convnet/mnist-small.yaml

