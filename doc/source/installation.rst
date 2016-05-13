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
===============

Let's get you started using Neon to build deep learning models!

Requirements
~~~~~~~~~~~~

Neon runs on **Python 2.7** or **Python 3.4+** and we support Linux and Mac OS X machines.
Before install, please ensure you have recent versions of the following
packages (different system names shown):

.. csv-table::
   :header: "Ubuntu", "OSX", "Description"
   :widths: 20, 20, 40
   :escape: ~

   python-pip, pip, Tool to install python dependencies
   python-virtualenv (*), virtualenv (*), Allows creation of isolated environments ((*): This is required only for Python 2.7 installs. With Python3: test for presence of ``venv`` with ``python3 -m venv -h``)
   libhdf5-dev, h5py, Enables loading of hdf5 formats
   libyaml-dev, pyaml, Parses YAML format inputs
   pkg-config, pkg-config, Retrieves information about installed libraries


`OpenCV <http://opencv.org/>`__ is also a required package. We recommend installing
with a package manager (e.g. apt-get or homebrew).

Additionally, there are several optional libraries.

* To enable multi-threading operations on a CPU, install `OpenBLAS <http://www.openblas.net/>`__, then recompile numpy with links to openBLAS (see sample instructions `here <https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/>`_). While Neon will run on the CPU, you'll get far better performance using GPUs.
* Enabling Neon to use GPUs requires installation of `CUDA SDK and drivers <https://developer.nvidia.com/cuda-downloads>`__. We support both `Maxwell <http://maxwell.nvidia.com/>`__ and `Kepler <http://www.nvidia.com/object/nvidia-kepler.html>`__ GPU architectures, but our backend is optimized for Maxwell GPUs. Remember to add the CUDA path to your environment variables.

For GPU users, remember to add the CUDA path. For example, on Ubuntu:

.. code-block:: bash

    export PATH="/usr/local/cuda/bin:"$PATH
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib:"$LD_LIBRARY_PATH

Installation
~~~~~~~~~~~~

We recommend installing Neon within a `virtual
environment <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`__
to ensure a self-contained environment. To install neon within an
already existing virtual environment, see the System-wide Install section.
If you use the `Anaconda <http://docs.continuum.io/anaconda/index>`__ python
distribution, please see the Anaconda Install section. Otherwise, to
setup neon in this manner, run the following commands:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon; make

This will install the files in the ``neon/.venv/`` directory and will use the python version in the
default PATH.  To instead force a Python2 or Python3 install, supply this as an optional parameter:

.. code-block:: bash

   make python2

Or:

.. code-block:: bash

   make python3

To activate the virtual environment, type

.. code-block:: bash

    . .venv/bin/activate

You will see the prompt change to reflect the activated environment. To
start Neon and run the MNIST multi-layer perceptron example (the "Hello
World" of deep learning), enter

.. code-block:: bash

    examples/mnist_mlp.py

When you are finished, remember to deactivate the environment

.. code-block:: bash

    deactivate

Congratulations, you have installed neon! Next, we recommend you learn
how to run models in neon and walk through the MNIST multilayer
perceptron tutorial.


Virtual Environment
~~~~~~~~~~~~~~~~~~~

``Virtualenv`` is a python tool that keeps the dependencies and packages
required for different projects in separate environments. By default,
our install creates a copy of python executable files in the
``neon/.venv`` directory. To learn more about virtual environments, see
the guide at http://docs.python-guide.org/en/latest/dev/virtualenvs/.

System-wide install
~~~~~~~~~~~~~~~~~~~

If you would prefer not to use a new virtual environment, Neon can be
installed system-wide with

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon && make sysinstall

To install neon in a previously existing virtual environment, first activate
that environment, then run ``make sysinstall``. Neon will install the
dependencies in your virtual environment's python folder.

Anaconda install
~~~~~~~~~~~~~~~~

If you have already installed and configured the Anaconda distribution
of python, follow the subsequent steps.

First, configure and activate a new conda environment for neon:

.. code-block:: bash

    conda create --name neon pip
    source activate neon

Now clone and run a system-wide install. Since the install takes place
inside a conda environment, the dependencies will be installed in your
environment folder.

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon && make sysinstall

When complete, deactivate the environment:

.. code-block:: bash

    source deactivate

Docker
~~~~~~

If you would prefer having a containerized installation of neon and its
dependencies, the open source community has contributed the following
Docker images (note that these are not supported/maintained by Nervana):

-  `neon (CPU only) <https://hub.docker.com/r/kaixhin/neon/>`__
-  `neon (GPU) <https://hub.docker.com/r/kaixhin/cuda-neon/>`__

Support
~~~~~~~

For any bugs or feature requests please:

1. Search the open and closed
   `issues <https://github.com/NervanaSystems/neon/issues>`__ list to
   see if we’re already working on what you have uncovered.
2. Check that your issue/request isn't answered in our `Frequently Asked
   Questions (FAQ) <http://neon.nervanasys.com/docs/latest/faq.html>`__
   or
   `neon-users <https://groups.google.com/forum/#!forum/neon-users>`__
   Google group.
3. File a new `issue <https://github.com/NervanaSystems/neon/issues>`__
   or submit a new
   `pull <https://github.com/NervanaSystems/neon/pulls>`__ request if
   you have some code to contribute. See our `contributing
   guide <https://github.com/NervanaSystems/neon/blob/master/CONTRIBUTING.rst>`__.
4. For other questions and discussions please post a message to the
   `neon-users <https://groups.google.com/forum/#!forum/neon-users>`__
   Google group.
