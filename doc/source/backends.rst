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

Backends
========

Backends incorporate a basic multi-dimensional
:class:`neon.backends.backend.Backend.Tensor` data structure as well the
algebraic and deep learning specific operations that can be performed on them.

Each implemented backend conforms to our :doc:`ml_operational_layer` API to
ensure a consistent behavior.

The Backend and Tensor classes share a lot in common with
`numpy <http://www.numpy.org/>`_
`ufunc's <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_ and
`ndarray's <http://docs.scipy.org/doc/numpy/reference/arrays.html>`_
respectively, so if you've done numpy programming in the past you should
already be fairly familiar with how to work with our Backend and Tensor
objects.

Our syntax differs slightly from numpy, and we also explicitly require that the
user manage and specify target output Tensor buffers (most operations have a
required ``out`` Tensor parameter -- in numpy this is usually optional).  While
this requires a bit more effort on the part of the user, the benefit is improved
efficiency and a (sometimes vastly) reduced memory footprint.  Unfortunately
numpy may make intermediate copies of Tensor data, and our forcing explicit
``out`` parameter specification avoids this.


Current Implementations
-----------------------

.. autosummary::
   :toctree: generated/

   neon.backends.cpu.CPU
   neon.backends.gpu.GPU
   neon.backends.cc2.GPU
   neon.backends.mgpu.MGPU

Adding a new Backend
--------------------

1. Generate a subclass of :class:`neon.backends.backend.Backend` including an
   associated tensor class :class:`neon.backends.backend.Backend.Tensor`.

2. Implement overloaded operators to manipulate these tensor objects, as well
   other operations.  Effectively this amounts to implementing our MOP API (see
   :doc:`ml_operational_layer`)

3. To date, these operations have attempted to more or less mimic numpy syntax
   as much as possible.
