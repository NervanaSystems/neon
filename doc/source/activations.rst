.. ---------------------------------------------------------------------------
.. Copyright 2016 Nervana Systems Inc.
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


Activation functions
====================

Activation functions such as the rectified linear unit (ReLu) or sigmoid
are treated as layers within neon. For convenience, these functions are
wrapped inside the :py:class:`Activation<neon.layers.layer.Activation>` layer, which takes
care of a lot of the layer-specific verbiage. Neon has the following
activation functions:

.. csv-table::
   :header: "Name", "Description"
   :widths: 20, 20
   :escape: ~

   :py:class:`neon.transforms.Identity<neon.transforms.activation.Identity>`, :math:`f(x) = x`
   :py:class:`neon.transforms.Rectlin<neon.transforms.activation.Rectlin>`, :math:`f(x) = \max(x~, 0)`
   :py:class:`neon.transforms.Explin<neon.transforms.activation.Explin>`, :math:`f(x) = \max(x~, 0) + \alpha (e^{\min(x~, 0)}-1)`
   :py:class:`neon.transforms.Normalizer<neon.transforms.activation.Normalizer>`, :math:`f(x) = x / \alpha`
   :py:class:`neon.transforms.Softmax<neon.transforms.activation.Softmax>`, :math:`f(x_j) = \frac{\exp{x_j}}{\sum_i \exp {x_i}}`
   :py:class:`neon.transforms.Tanh<neon.transforms.activation.Tanh>`, :math:`f(x) = \tanh(x)`
   :py:class:`neon.transforms.Logistic<neon.transforms.activation.Logistic>`, :math:`f(x) = \frac{1}{1+e^{-x}}`


Creating custom activations
---------------------------

To create a new activation function, subclass from :py:class:`Transform<neon.transforms.transform.Transform>` and
implement the :py:meth:`bprop()<neon.transforms.transform.Transform.bprop>` and :py:meth:`__call__()<neon.transforms.transform.Transform.__call__>` methods.

As an example, we implement the ReLu function:

.. code-block:: python

    class MyReLu(Transform):
        " ReLu activation function. Implements f(x) = max(0,x) "

        def __init__(self, name=None):
            super(MyReLu, self).__init__(name)

        # f(x) = max(0,x)
        def __call__(self, x):
            return self.be.maximum(x, 0)

        # If x > 0, gradient is 1; otherwise 0.
        def bprop(self, x):
            return self.be.greater(x, 0)

Both methods receive as input a :py:class:`.Tensor` ``x``.

In most models, activation functions are appended to a filtering (e.g.
convolution) or linear (all-to-all) layer. For this reason, Neon
provides several convenient :py:class:`CompoundLayer<neon.layers.layer.CompoundLayer>` classes (:py:class:`Affine<neon.layers.layer.Affine>`, :py:class:`Conv<neon.layers.layer.Conv>`, and :py:class:`Deconv<neon.layers.layer.Deconv>`. For example, a linear layer followed by your
ReLu function can be instantiated via

.. code-block:: python

    layers = [Affine(nout = 1000, activation=MyReLu())]
