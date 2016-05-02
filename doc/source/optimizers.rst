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

Optimizers
==========

Critical to any deep learning model is the optimization of weights to
perform the given task. This set of classes provide options for
selecting and customizing the appropriate optimization algorithm. Neon
supports the following optimizers:

.. csv-table::
    :header: Function, Description
    :widths: 20, 40
    :delim: |

    :py:class:`neon.optimizers.GradientDescentMomentum<neon.optimizers.optimizer.GradientDescentMomentum>` | `Stochastic gradient descent <http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/>`__ with `momentum <http://jmlr.org/proceedings/papers/v28/sutskever13.pdf>`__
    :py:class:`neon.optimizers.RMSProp<neon.optimizers.optimizer.RMSProp>` | Root Mean Square propagation (see `Hinton's slides <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`__)
    :py:class:`neon.optimizers.Adagrad<neon.optimizers.optimizer.Adagrad>` | `Adagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`__ method for adapting the learning rate
    :py:class:`neon.optimizers.Adadelta<neon.optimizers.optimizer.Adadelta>` | `Adadelta <http://arxiv.org/abs/1212.5701>`__ method for adapting the learning rate
    :py:class:`neon.optimizers.Adam<neon.optimizers.optimizer.Adam>` | `Adam <http://arxiv.org/pdf/1412.6980v8.pdf>`__ optimization algorithm
    :py:class:`neon.optimizers.MultiOptimizer<neon.optimizers.optimizer.MultiOptimizer>` | Class for assigning optimizers to different layers


Each optimization algorithm inherits from the
:py:class:`neon.optimizers.Optimizer<neon.optimizers.optimizer.Optimizer>` class and implements the ``optimize``
method:

.. code-block:: python

    """
    Given the model's layers and the current training epoch,
    iterate over each layer and update the weights.
    """
    def optimize(self, layer_list, epoch):

The :py:class:`neon.optimizers.Optimizer<neon.optimizers.optimizer.Optimizer>` base class also implements two methods:

1. ``clip_gradient_value``, which clips each gradient between :math:`-k` and :math:`k`, where :math:`k` is the argument  ``gradient_clip_value``.

2. ``clip_gradient_norm``, which scales each gradient by :math:`k`, which is the argument ``gradient_clip_norm``.

Stochastic Gradient Descent
---------------------------

Stochastic Gradient Descent (SGD) has existed for awhile, but its
usefulness in training deep neural networks has only recently been
realized. SGD is similar to traditional gradient descent, except that
the gradient updates are computed over a small subset of the total
training data (i.e., a minibatch).

Given the parameters :math:`\theta`, the learning rate :math:`\alpha`, and the gradients :math:`\nabla J(\theta; x)`
computed on the minibatch data :math:`x`, SGD updates the parameters via

.. math::

    \theta' = \theta - \alpha\nabla J(\theta; x)

Here we implement SGD with momentum. Momentum tracks the history of
gradient updates to help the system move faster through saddle points.
Given the additional parameters: momentum :math:`\gamma`, weight decay :math:`\lambda`, and current velocity
:math:`v`, we use the following update equations

.. math::

    v' &= \gamma v - \alpha(\nabla J(\theta; x) + \lambda\theta) \\
    \theta' &= \theta + v'

Example usage:

.. code-block:: python

    from neon.optimizers import GradientDescentMomentum

    # use SGD with learning rate 0.01 and momentum 0.9, while
    # clipping the gradients between -5 and 5.
    opt = GradientDescentMomentum(0.01, 0.9, gradient_clip_value = 5)

RMS propagation
---------------

Root Mean Square (RMS) propagation protects against vanishing and
exploding gradients. In RMSprop, the gradient is divided by a running
average of recent gradients. Given the parameters :math:`\theta`, gradient :math:`\nabla J`, we keep a running average
:math:`\mu` of the last :math:`1/\lambda` gradients squared. The update equations are then given by

.. math::

   \mu' &= \lambda\mu + (1-\lambda)(\nabla J)^2 \\
   \theta' &= \theta - \frac{\alpha}{\sqrt{\mu + \epsilon} + \epsilon}\nabla J

where we use :math:`\epsilon` as a (small) smoothing factor to prevent from dividing by zero.

When reaching a plateau in the error surface, the gradient is very
small, but the normalization factor here increases the update step for
faster learning (small update: :math:`\alpha\nabla J = 0.0001`, but square root of the weighted average:
:math:`\sqrt{\mu}= 0.00002`, yielding an update of 0.2). If the gradients are exploding, RMSprop also provides protection (large
update: :math:`\alpha\nabla J = 100`, but the weighted average :math:`\sqrt{\mu} = 20`, yielding a much smaller update of 5). Because of these advantages,
RMSprop is often used in recurrent neural networks to protect against vanishing or exploding gradients.

Example usage:

.. code-block:: python

    from neon.optimizers import RMSprop

    # RMSprop
    optimizer = RMSProp(decay_rate=0.95, learning_rate=2e-3)

Adagrad
-------

Adagrad is an algorithm that adapts the learning rate individually for
each parameter by dividing by the :math:`L_2`-norm of all previous gradients. Given the parameters
:math:`\theta`, gradient :math:`\nabla J`, accumulating norm :math:`G`, and smoothing factor :math:`\epsilon`,
we use the update equations:

.. math::

   G' &= G + (\nabla J)^2 \\
   \theta' &= \theta - \frac{\alpha}{\sqrt{G' + \epsilon}} \nabla J

where the smoothing factor :math:`epsilon` prevents from dividing by zero. By adjusting the learning rate
individually for each parameter, Adagrad adapts to the geometry of the
error surface. Differently scaled weights have appropriately scaled
update steps.

Example usage:

.. code-block:: python

    from neon.optimizers import Adagrad

    # use Adagrad with a learning rate of 0.01
    optimizer = Adagrad(learning_rate=0.01, epsilon=1e-6)

Adadelta
--------

Adadelta was designed to address two drawbacks of the above Adagrad
algorithm:

1. Continual decay of learning rates over training caused by the accumulation of the :math:`L_2`-norm.

2. Need for a manually tuned learning rate :math:`\alpha`

Similar to RMSprop, Adadelta tracks the running average of the
gradients, :math:`\mu_J`, over a window size :math:`1/\lambda`, where
:math:`\lambda` is the parameter ``decay``. Adadelta also tracks an average of the
recent update steps, which we denote as :math:`\mu_\theta`, and sets the learning rate as the ratio of the two averages:

.. math::
    \mu_J' &= \lambda\mu_J + (1-\lambda) (\nabla J)^2 \\
    \Delta \theta &= \sqrt{\frac{\mu_\theta + \epsilon}{\mu_J' + \epsilon}} \nabla J \\
    \mu_\theta &= \lambda \mu_\theta + (1-\rho) (\Delta \theta)^2 \\
    \theta &= \theta - \Delta \theta

Note that the learning rate is a ratio of the average updates from the
previous step, :math:`\mu_\theta`, divided by the average gradients including the current step,
:math:`\mu'_J`.

Example usage:

.. code-block:: python

    from neon.optimizers import Adadelta

    # use Adagrad with a learning rate of 0.01
    optimizer = Adadelta(decay=0.95, epsilon=1e-6)

Adam
----

The Adam optimizer combines features from RMSprop and Adagrad. We
accumulate both the first and second moments of the gradient with decay
rates :math:`\beta_1` and :math:`\beta_2` corresponding to window sizes of
:math:`1/\beta_1` and :math:`1/\beta_2`, respectively.

.. math::
    m' &= \beta_1 m + (1-\beta_1) \nabla J \\
    v' &= \beta_2 v + (1-\beta_2) (\nabla J)^2

We update the parameters by the ratio of the two moments:

.. math::
    \theta = \theta - \alpha \frac{\hat{m}'}{\sqrt{\hat{v}'}+\epsilon}

where we compute the bias-corrected moments :math:`\hat{m}'` and :math:`\hat{v}'` via

.. math::
    \hat{m}' &= m'/(1-\beta_1^t) \\
    \hat{v}' &= v'/(1-\beta_1^t)

Example usage:

.. code-block:: python

    from neon.optimizers import Adam

    # use Adam
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

Using multiple optimizers
-------------------------

Often, we may want to assign differently configured optimizers to
different layers. For example, when training AlexNet, the learning rates
and schedules for the bias layers are different from the convolutional
and pooling layers. We first define the different optimizers:

.. code-block:: python

    from neon.optimizers import GradientDescentMomentum, RMSprop

    optimizer_A = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9)
    optimizer_B = GradientDescentMomentum(learning_rate=0.05, momentum_coef=0.9)
    optimizer_C = RMSprop(learning_rate=2e-3, decay_rate=0.95)

Then, we instantiate a ``neon.optimizers.MultiOptimizer`` and pass a
dictionary mapping layers to optimizers. The keys can either be:
``default``, a layer class name (e.g. ``Bias``), or the Layer's name
attribute. The latter takes precedence for finer layer-to-layer control.

For example, if we have the following layers,

.. code-block:: python

    layers = []
    layers.append(Linear(nout = 100, init=Gaussian(), name="layer_one"))
    layers.append(Linear(nout = 50, init=Gaussian(), name="layer_two"))
    layers.append(Affine(nout = 5, init=Gaussian(), activation=Softmax()))

we can define multiple optimizers with

.. code-block:: python

    from neon.optimizers import MultiOptimizer

    # dictionary of mappings
    mapping = {'default': optimizer_A, # default optimizer
               'Linear': optimizer_B, # all layers from the Linear class
               'layer_two': optimizer_C} # this overrides the previous entry for a specific layer

    # use multiple optimizers
    opt = MultiOptimizer(mapping)

After definition, we have the following mapping

+----------------------+----------------------------+
| Layer                | Optimizer                  |
+======================+============================+
| ``layer_one``        | ``optimizer_B``            |
+----------------------+----------------------------+
| ``layer_two``        | ``optimizer_C``            |
+----------------------+----------------------------+
| ``Affine.Linear``    | ``optimizer_B``            |
+----------------------+----------------------------+
| ``Affine.Bias``      | ``optimizer_A``            |
+----------------------+----------------------------+
| ``Affine.Softmax``   | ``None (no parameters)``   |
+----------------------+----------------------------+

Creating new optimizers
-----------------------

To create new optimizers, subclass from ``neon.optimizers.Optimizer``
and implement the constructor and the ``optimize`` method:

.. code-block:: python

    """
    Constructor to include arguments for optimizer-specific parameters,
    stochastic rounding (optional), gradient clipping (optional), and gradient scaling (optional)
    """
    def __init__(self, myparam_1, stochastic_round=False, \
                 gradient_clip_value=None, gradient_clip_norm=None):

    """
    Given the model's layers and the current training epoch,
    iterate over each layer and update the weights.
    """
    def optimize(self, layer_list, epoch):

Neon provides helper methods to iterate over the layers. Here is the
skeleton for a custom ``optimize`` method.

.. code-block:: python


    def optimize(self, layer_list, epoch):

        # get a flattened list of layer weights
        param_list = get_param_list(layer_list)

        # iterate over the weights (param), gradients (grad), and
        # any accumulated variables (states)
        for (param, grad), states in param_list:

            # if states not initialized, allocate with zeros
            if len(states) == 0:
                states.append(self.be.zeros_like(grad))

            # scale gradient by size of minibatch (be.bsz)
            grad = grad / self.be.bsz

            delta_param = # enter your update equations
            param[:] = param + delta_param

For more guidance, consult the source code for the existing optimization
algorithms in ``neon/optimizers/optimizer.py``.
