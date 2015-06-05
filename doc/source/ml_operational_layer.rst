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
.. currentmodule:: neon

.. |Tensor| replace:: :py:class:`~neon.backends.backend.Tensor`
.. |Backend| replace:: :py:class:`~neon.backends.backend.Backend`

******************************
ML OPerational Layer (MOP) API 
******************************

We expose the following API which we refer to as our ML operational layer (aka
MOP layer). It currently consists of the functions defined in the following two
interface classes, which we detail further on the rest of this page:

.. autosummary::
   :toctree: generated/

   neon.backends.backend.Tensor
   neon.backends.backend.Backend

Basic Data Structure
====================

The |Tensor| class is used to represent an arbitrary dimensional array in which
each element is stored using a consistent underlying type.

We have the ability to instantiate and copy instances of this data
structure, as well as initialize its elements, reshape its dimensions, and 
access metadata.

|Tensor| Creation
-----------------

.. autosummary::

   neon.backends.backend.Backend.empty
   neon.backends.backend.Backend.array
   neon.backends.backend.Backend.zeros
   neon.backends.backend.Backend.ones
   neon.backends.backend.Backend.copy
   neon.backends.backend.Backend.uniform
   neon.backends.backend.Backend.normal

|Tensor| Manipulation
---------------------

.. autosummary::

   neon.backends.backend.Tensor.asnumpyarray
   neon.backends.backend.Tensor.take
   neon.backends.backend.Tensor.__getitem__
   neon.backends.backend.Tensor.__setitem__
   neon.backends.backend.Tensor.fill
   neon.backends.backend.Tensor.transpose
   neon.backends.backend.Tensor.reshape
   neon.backends.backend.Tensor.repeat
   neon.backends.backend.Tensor.free

|Tensor| Attributes
-------------------

.. autosummary::

   neon.backends.backend.Tensor.shape
   neon.backends.backend.Tensor.dtype
   neon.backends.backend.Tensor.raw

Arithmetic Operation Support
============================

Unary and binary arithmetic operations can be performed on |Tensor| objects via
appropriate |Backend| calls.  In all cases it is up to the user to pre-allocate
correctly sized output to house the result.

Element-wise Binary Operations
------------------------------
.. autosummary::

   neon.backends.backend.Backend.add
   neon.backends.backend.Backend.subtract
   neon.backends.backend.Backend.multiply
   neon.backends.backend.Backend.divide

Element-wise Unary Transcendental Functions
-------------------------------------------
.. autosummary::

   neon.backends.backend.Backend.log
   neon.backends.backend.Backend.exp
   neon.backends.backend.Backend.power

Matrix Algebra Operations
-------------------------
.. autosummary::

   neon.backends.backend.Backend.dot

Logical Operation Support
=========================

.. autosummary::

   neon.backends.backend.Backend.equal
   neon.backends.backend.Backend.not_equal
   neon.backends.backend.Backend.greater
   neon.backends.backend.Backend.greater_equal
   neon.backends.backend.Backend.less
   neon.backends.backend.Backend.less_equal

Summarization Operation Support
===============================
.. autosummary::

   neon.backends.backend.Backend.sum
   neon.backends.backend.Backend.mean
   neon.backends.backend.Backend.min
   neon.backends.backend.Backend.max
   neon.backends.backend.Backend.argmin
   neon.backends.backend.Backend.argmax
   neon.backends.backend.Backend.norm
   neon.backends.backend.Backend.variance

Initialization and Setup
========================
.. autosummary::

   neon.backends.backend.Backend.rng_init
   neon.backends.backend.Backend.err_init
   neon.backends.backend.Backend.begin
   neon.backends.backend.Backend.end

Higher Level Operation Support
==============================

Fully Connected Neural Network Layer
------------------------------------
.. autosummary::

   neon.backends.backend.Backend.fprop_fc
   neon.backends.backend.Backend.bprop_fc
   neon.backends.backend.Backend.update_fc

Convolutional Neural Network Layer
----------------------------------
.. autosummary::

   neon.backends.backend.Backend.fprop_conv
   neon.backends.backend.Backend.bprop_conv
   neon.backends.backend.Backend.update_conv

Pooling Neural Network Layer
----------------------------
.. autosummary::

   neon.backends.backend.Backend.fprop_pool
   neon.backends.backend.Backend.bprop_pool

CrossMap Neural Network Layer
-----------------------------
.. autosummary::

   neon.backends.backend.Backend.fprop_cmpool
   neon.backends.backend.Backend.bprop_cmpool
   neon.backends.backend.Backend.update_cmpool
   neon.backends.backend.Backend.fprop_cmrnorm
   neon.backends.backend.Backend.bprop_cmrnorm


***************
MOP API Changes
***************

v0.9.0
======

* begin and end functions now take two parameters: block and identifier.  The
  first requires an attribute of class Block (also defined in backend.py)
  indicating what type of computation is about to commence.  The second is a
  unique integer identifier that indicates which iteration we are in.
* new persist_values parameter added with default value True to most of the
  backend array initialization routines, as well as Tensor attribute.

* rename axes parameter to axis in summarization operations (planned)

v0.8.0
======

* new function variance to compute the variance.

v0.7.0
======

* to support 3D convolutions:

  * new parameter ofmsize has been added to fprop_conv, bprop_conv,
    update_conv, fprop_pool, bprop_pool
  * new parameter fpsize has been added to bprop_pool
  * new parameter ifmsize has been added to fprop_cmpool, bprop_cmpool,
    update_cmpool

* functions providing compiler hints:

  * new Tensor.free() function indicating Tensor no longer in use and can be
    cleaned up
  * new backend begin() and end() functions to indicate the start and end of
    repeated instructions (as would be present in loops and so forth).
  * code can be processed to inject the above hints via
    `neon.util.compiler_hints.py`.  To remove, pass the `-s` flag to the
    command invocation.  Alternatively from the top level of the project, one
    can call `make insert_compiler_hints` or `make strip_compiler_hints`

* epsilon removed as a backend parameter (now associated with specific
  functions being used)
