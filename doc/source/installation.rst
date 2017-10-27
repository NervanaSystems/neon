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

Let's get you started using neon to build deep learning models!

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

.. note::
   To enable Aeon, neon's dataloader, several optional libraries should be installed. For image processing, install `OpenCV <http://opencv.org/>`__. For audio and video data, install `ffmpeg <https://ffmpeg.org/>`__. We recommend installing with a package manager (e.g. apt-get or homebrew). If you have encountered error messages about failing to install aeon while building neon, please visit `aeon <https://github.com/NervanaSystems/aeon>`__ page for how to install prerequisites for aeon to enable neon with aeon data loader.


Additionally, there are several other libraries.

* Neon v2.0.0+ by default comes with Intel Math Kernel Library (MKL) support, which enables multi-threading operations on Intel CPU. It is the recommended library to use for best performance on CPU. When installing neon, MKL support will be automatically enabled.
* (optional) If interested to compare multi-threading performance of MKL optimized neon, install `OpenBLAS <http://www.openblas.net/>`__, then recompile numpy with links to openBLAS (see sample instructions `here <https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/>`_). While neon will run on the CPU with OpenBLAS, you'll get better performance using MKL on CPUs or CUDA on GPUs.
* Enabling neon to use GPUs requires installation of `CUDA SDK and drivers <https://developer.nvidia.com/cuda-downloads>`__. We support `Pascal <http://developer.nvidia.com/pascal>`__ ,  `Maxwell <http://maxwell.nvidia.com/>`__ and `Kepler <http://www.nvidia.com/object/nvidia-kepler.html>`__ GPU architectures, but our backend is optimized for Maxwell GPUs. Remember to add the CUDA path to your environment variables.

For GPU users, remember to add the CUDA path. For example, on Ubuntu:

.. code-block:: bash

    export PATH="/usr/local/cuda/bin:"$PATH
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib:"$LD_LIBRARY_PATH

Or on Mac OS X:

.. code-block:: bash

    export PATH="/usr/local/cuda/bin:"$PATH
    export DYLD_LIBRARY_PATH="/usr/local/cuda/lib:"$DYLD_LIBRARY_PATH

Installation
~~~~~~~~~~~~

We recommend installing neon within a `virtual
environment <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`__
to ensure a self-contained environment. To install neon within an
already existing virtual environment, see the System-wide Install section.
If you use the `Anaconda <http://docs.continuum.io/anaconda/index>`__ python
distribution, please see the Anaconda Install section. Otherwise, to
setup neon in this manner, run the following commands:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon; git checkout latest; make

The above checks out the latest stable release (e.g. a tagged release version v2.3.0) and build neon.
Alternatively, you can check out and build the latest master branch:

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon; make


This will install the files in the ``neon/.venv/`` directory and will use the python version in the
default PATH. Note that neon would automatically download the released MKLML library that
features MKL support.

To instead force a Python2 or Python3 install, supply this as an optional parameter:

.. code-block:: bash

   make python2

Or:

.. code-block:: bash

   make python3

To activate the virtual environment, type

.. code-block:: bash

    . .venv/bin/activate

You will see the prompt change to reflect the activated environment. To
start neon and run the MNIST multi-layer perceptron example (the "Hello
World" of deep learning), enter

.. code-block:: bash

    examples/mnist_mlp.py

Note that since neon v2.1 the above is equivalent to explicitly add ``-b mkl`` for better performance on Intel CPUs. In other words, mkl backend is the default backend

.. code-block:: bash

    examples/mnist_mlp.py -b mkl

.. note::
   To achieve best performance, we recommend setting KMP_AFFINITY and OMP_NUM_THREADS in this way: ``export KMP_AFFINITY=compact,1,0,granularity=fine`` and ``export OMP_NUM_THREADS=<Number of Physical Cores>``. You can set these environment variables in bash and do ``source ~/.bashrc`` to activate it. You may need to activate the virtual environment again after sourcing bashrc. For detailed information about KMP_AFFINITY, please read here: https://software.intel.com/en-us/node/522691. We encourage users to experiment with this thread affinity configurations to achieve even better performance.

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

If you would prefer not to use a new virtual environment, neon can be
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
Docker images (note that these are not supported/maintained by Intel Nervana):

-  `neon (CPU only) <https://hub.docker.com/r/kaixhin/neon/>`__
-  `neon (MKL) <https://hub.docker.com/r/aminaka/mkl-neon/>`__
-  `neon (GPU) <https://hub.docker.com/r/kaixhin/cuda-neon/>`__
-  `neon (CPU with Jupyter Notebook) <https://hub.docker.com/r/sofianhw/docker-neon-ipython/>`__

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
