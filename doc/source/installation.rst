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

Installation
============

Overview
--------

.. code-block:: bash

    # get the latest source
    git clone https://github.com/NervanaSystems/neon.git
    cd neon

    # configure optional backends like GPU, distributed processing by editing
    # setup.cfg with a text editor.
    nano setup.cfg

    # to install system wide (we recommend first setting up a virtualenv):
    make install  # sudo make install on Linux

    # or to build for working locally in the source tree
    # (useful for active development)
    make develop  # sudo make develop on Linux
    # or
    make build  # will require updating PYTHONPATH to point at neon dir


Required Dependencies
---------------------

We expect a system with python 2.7 at minimum.  Any dependent python package
will be automatically installed as part of the ``make install`` call.

* `python <https://www.python.org/>`_  (2.7 and 3.4 are supported)
* `numpy <http://www.numpy.org/>`_ for CPU backend and on-host dataset storage
* `pyyaml <http://pyyaml.org/>`_ for configuration file parsing


Optional Dependencies
---------------------

Though not strictly required to run some basic examples, these dependencies
can be installed to unlock faster backends, simplified image dataset
handling, and tools necessary for doing neon development.

These dependencies can be installed by editing the appropriate parameters
defined in `Configuration Setup`_ (spelled out as subsection titles below):

GPU=nervanagpu
^^^^^^^^^^^^^^

* `nervanagpu <http://github.com/NervanaSystems/nervanagpu/>`_ our in-house
  developed fp16/fp32 Maxwell GPU backend.  To take advantage of this you'll
  need a CUDA capable Maxwell graphics card with CUDA drivers and
  `SDK <https://developer.nvidia.com/cuda-downloads>`_ installed.
* `pycuda <http://mathema.tician.de/software/pycuda/>`_ Required for the
  nervanagpu backend
* `maxas <https://github.com/NervanaSystems/maxas/>`_ Assembler for NVIDIA
  Maxwell architecture.  Required for installing the nervanagpu backend.

GPU=cudanet
^^^^^^^^^^^

* `Nervana's cuda-convnet2 <http://github.com/NervanaSystems/cuda-convnet2/>`_
  our updated fork of Alex Krizhevsky's
  `cuda-convnet2 <https://code.google.com/p/cuda-convnet2/>`_ that powers our
  cudanet GPU backend.  To use this you'll need a CUDA capable graphics card
  with CUDA drivers and SDK installed.

DEV=1
^^^^^

* `imgworker <https://github.com/NervanaSystems/imgworker/>`_ our in-house
  developed multithreaded image decoder.  Required for
  `neon.datasets.imageset.Imageset` based datasets.  Note that this requires
  that the `boost C++ libraries <http://www.boost.org/>`_ first be installed in
  a typical directory.
* `Pillow <http://pillow.readthedocs.org/index.html/>`_ PIL fork required for
  batch writer and doing initial processing for the Imagenet dataset.
* `nose <https://nose.readthedocs.org/en/latest/>`_ for running unit tests as
  part of the ``make test`` target
* `sphinx <http://sphinx-doc.org/>`_ for generating the documentation as part
  of the ``make doc`` target
* sphinxcontrib-napoleon for google style autodoc parsing
* `flake8 <https://flake8.readthedocs.org/>`_ for code style conformance as
  part of the ``make style`` target
* `pep8-naming <https://pypi.python.org/pypi/pep8-naming>`_ plugin for variable
  name checking
* `matplotlib <http://matplotlib.org>`_ Currently used for some basic
  visualizations like RNN features.

DIST=1
^^^^^^

* `mpi4py <https://github.com/mpi4py/mpi4py>`_ for creation of distributed
  Tensors in data and model parallel models.
* `openmpi <http://www.open-mpi.org/>`_ required for mpi4py


Configuration Setup
-------------------

Initial build type and required dependency handling can be controlled either by
editing the ``setup.cfg`` file prior to installation, or by passing arguments
to the ``make`` command.  Below is an example showing the default values for
``setup.cfg``:

.. highlight:: ini

.. literalinclude:: ../../setup.cfg
   :linenos:

As shown, the default set of options is fairly restrictive, so only the CPU
based backend will be available:

* Set ``GPU=nervanagpu`` (maxwell) or ``GPU=cudanet`` (kepler), if you have a CUDA capable GPU
* Set ``DEV=1``, if you plan to run unit tests, build documentation or develop neon 
* Set ``DIST=1``, if you would like to run your model training in parallel via MPI

To override what is defined in ``setup.cfg``, one can pass the appropriate
options on the command-line (useful when doing in-place development).  Here's
an example:

.. code-block:: bash

    make -e GPU=cudanet DEV=1 test


.. _mpi_install:

Installing MPI on an Ubuntu cluster (for distributed models)
------------------------------------------------------------
neon provides distributed implementations of convnets and MLPs in addition to the non-distributed implementations.
It has been tested with
`OpenMPI 1.8.1 <http://www.open-mpi.org/software/ompi/v1.8/>`_ and
`mpi4py <https://github.com/mpi4py/mpi4py>`_.

1. Install OpenMPI:

.. code-block:: bash

    cd <openmpi_source_dir>
    ./configure --prefix=/<path_to_install_openmpi> --with-cuda
    make all
    sudo make install

Make sure that ``PATH`` includes ``/<path_to_openmpi>/bin`` and
``LD_LIBRARY_PATH`` includes ``/<path_to_openmpi>/lib``

2. Install mpi4py:

.. code-block:: bash

  # set DIST=1 in setup.cfg then run:
  make install
  # or
  make -e DIST=1 install
  # or
  cd <mpi4py_source_dir>
	sudo python setup.py build --configure install

3. Setup ``/etc/hosts`` with IPs of the nodes.
e.g.:

.. code-block:: bash

	192.168.1.1 host1
	192.168.1.2 host2

4. Setup a hosts file to use with MPI ``-hostfile`` option.
For additional info refer to `this document <http://cs.calvin.edu/curriculum/cs/374/homework/MPI/01/multicoreHostFiles.html>`_.
e.g.:

.. code-block:: bash

	host1 slots=2
	host2 slots=2

5. Read through the :doc:`distributed` section to see how run neon in data or
   model parallel mode using MPI.


Virtualenv
----------
If you are doing work on a multi-user system, don't have sudo access, or just
want to have an isolated installation, using a
`virtualenv <https://packaging.python.org/en/latest/installing.html#creating-virtual-environments>`_
is highly recommended.

Setting up a virtual environment requires creation of directory to which you
can write. This will end up housing a copy of python, it's packaging tools,
and any python libraries you install.  In the example below we call this
directory `.venv`

.. code-block:: bash

    # if on python 2.7:
    virtualenv .venv
    # if on python3:
    pyvenv .venv

Now that you've setup a virtual environment, you need to "activate" it so that
any subsequent python or pip related commands will utilize it.

.. code-block:: bash

    . .venv/bin/activate

The above changes your python and pip executables to point inside the `.venv`
directory. Any packages installed will be in there and you can use python, pip
as normal. You’ll also see that your prompt changes to (.venv) to indicate
this.

When finished using the virtualenv you need to "deactivate" it to get back to
using the standard python paths and tools.  To do this simply issue:

.. code-block:: bash

    deactivate

You can always reactivate your virtualenv again at a later time, via
`.  .venv/bin/activate`.  If you want to completely remove the virtualenv just
delete the `.venv` directory.


Upgrading
---------

Assuming we've prepared a new release and you still have the cloned git
repository, you can issue:

.. code-block:: bash

    # get into the directory where you downloaded the neon repository
    cd neon

    # pull down source code changes
    git pull origin master

    # install as before (may need to first edit setup.cfg)
    sudo make install


Uninstalling
------------

.. code-block:: bash

    sudo pip uninstall neon
