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

Backends
========

Generating a backend
--------------------
To create a backend use the ``neon.backends.gen_backend`` function

.. code-block:: python

    from neon.backends import gen_backend
    be = gen_backend()

Without any arguments, this will return a CPU backend

Nervana GPU
-----------
.. autosummary::
   neon.backends.nervanagpu.NervanaGPU

The NervanaGPU backend consists of kernels written in MaxAs_ assembler and
Python wrappers. It includes pre-compiled kernels for matrix operations such
as GEMM and CONV. Kernels for element-wise operations are templated and build
on the fly so multiple operations can be compounded into a single kernel,
which avoids memory bandwidth bottlenecks. The backend follows the API
defined by the MOP layer.

Sequences of operations are performed using a lazy evaluation scheme where
operations are pushed onto an OpTree and only evaluated when an explicit
assignment is made using :doc:`optree` syntax (``optree[:]``), or when a
GEMM or CONV operation is performed.

OpTrees can also be passed to
:py:class:`Autodiff<neon.backends.autodiff.Autodiff>` for automatic
differentiation.

.. _MaxAs: https://github.com/NervanaSystems/maxas

Nervana CPU
-----------
.. autosummary::
   neon.backends.nervanacpu.NervanaCPU

The NervanaCPU backend is built on top of NumPy linear algebra functions for
NDarrays and supports automatic differentiation through the use of OpTrees
via :py:class:`Autodiff<neon.backends.autodiff.Autodiff>`.
