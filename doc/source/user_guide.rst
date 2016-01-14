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
..  ---------------------------------------------------------------------------

Installation
============

External Dependencies
---------------------

To install neon on a Linux or Mac OSX machine, please ensure you have recent
versions of the following system software (Ubuntu package names shown):

* ``python``, ``python-dev`` - We currently support python 2.7
* ``python-pip`` - Needed to install python dependencies.
* ``python-virtualenv`` - Needed to configure an isolated environment
* ``libhdf5-dev`` - (h5py) for callback hdf5 datasets
* ``libyaml-dev`` - (pyyaml) for YAML input file parsing
* ``libopenblas-dev`` - optional requirement, greatly enhances performance
  of the numpy based CPU backend, supports multi-threading for basic algebric
  operations
* ``libopencv-dev``, ``pkg-config`` - (imageset_decoder) optional requirement,
  used to perform decoding of image data

Though neon will run on a CPU, you'll get better performance by utilizing a
recent GPU (Maxwell based architecture).  This requires installation of the
`CUDA SDK and drivers <https://developer.nvidia.com/cuda-downloads>`_.


Virtualenv
----------

A virtualenv based install is recommended as this will ensure a self-contained
environment in which to run and develop neon.  To setup neon in this manner
run the following commands:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    make

The virtualenv will install all required files into the ``.venv`` directory.
To begin using it type the following:

.. code-block:: bash

    . .venv/bin/activate

You'll see your prompt change to highlight the venv, you can now run the neon
examples, or extend the code:

.. code-block:: bash

    cd examples
    ./mnist_mlp.py

When you have finished working on neon, you can deactivate the virtualenv via:

.. code-block:: bash

    deactivate


System-wide
-----------

The virtualenv based install is recommended to ensure an isolated
environment. As an alternative, it is possible to install neon into
your system python path.  The process for doing so is:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    make sysinstall


Anaconda
--------

If you use the `Anaconda <http://docs.continuum.io/anaconda/index>`_
distribution of python, the virtualenv based install described above will not
work.  Instead the following sequence of steps should get you going, assuming
you have already installed and configured anaconda python.

First configure and actiavte a new conda environment for neon:

.. code-block:: bash

    wget https://raw.githubusercontent.com/wleepang/sd-deep-learning/master/2015-12-02/neon-conda-environment.yaml
    conda env create -f neon-conda-environment.yaml
    source activate neon

Now clone and install neon, bypassing the Makefile:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    python setup.py develop

When complete, you can deactivate the environment:

.. code-block:: bash

    source deactivate


Docker
------

If you would prefer having a containerized installation of neon and its dependencies, the open source community has contributed the following Docker images (note that these are not supported/maintained by Nervana):

* `neon (CPU-only) <https://hub.docker.com/r/kaixhin/neon/>`_
* `cuda-neon <https://hub.docker.com/r/kaixhin/cuda-neon/>`_


Support
-------
For any bugs or feature requests please:

1. Search the open and closed
   `issues list <https://github.com/NervanaSystems/neon/issues>`_ to see if we're
   already working on what you have uncovered.
2. Check that your issue/request hasn't already been addressed in our
   `Frequently Asked Questions (FAQ) <http://neon.nervanasys.com/docs/latest/faq.html>`_
   or `neon-users`_ Google group.
3. File a new `issue <https://github.com/NervanaSystems/neon/issues>`_ or submit
   a new `pull request <https://github.com/NervanaSystems/neon/pulls>`_ if you
   have some code you'd like to contribute

For other questions and discussions please:

1. Post a message to the `neon-users`_ Google group

.. _neon-users: https://groups.google.com/forum/#!forum/neon-users
