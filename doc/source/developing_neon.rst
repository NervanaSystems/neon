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

Contributing to the Framework
=============================

.. include:: ../../CONTRIBUTING.rst

Architecture
------------

.. figure:: _static/framework_architecture.png
   :alt: neon architecture

Extending the Framework
-----------------------

The process for adding a new type of model, layer, dataset, and so forth is
typically the same.  You start by inheriting from the base class of the
construct being extended, then fill in a handful of required functions.

The specifics for extending each type of construct can be found at the
following links:

* :ref:`extending_experiment`
* :ref:`extending_dataset`
* :ref:`extending_model`
* :ref:`extending_layer`
* :ref:`extending_learningrule`
