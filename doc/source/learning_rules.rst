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

Learning Rules
==============

Learning rules are optimizers that use backpropagated gradients to update
layer weights. The most basic form is stochastic gradient descent (SGD), which
can be augmented with momentum. AdaDelta (Zeiler 2012) is an adaptive gradient
method that does not require the learning rate to be tuned manually.


Available Learning Rules
------------------------

.. autosummary::
   :toctree: generated/

   neon.optimizers.gradient_descent.GradientDescent
   neon.optimizers.gradient_descent.GradientDescentPretrain
   neon.optimizers.gradient_descent.GradientDescentMomentum
   neon.optimizers.gradient_descent.GradientDescentMomentumWeightDecay

   neon.optimizers.adadelta.AdaDelta
   neon.optimizers.rmsprop.RMSProp

.. _extending_learningrule:

Adding a new type of Learning Rule
----------------------------------

#. subclass :class:`neon.optimizers.learning_rule.LearningRule`
#. implement :func:`neon.optimizers.learning_fule.LearningRule.apply_rule`
