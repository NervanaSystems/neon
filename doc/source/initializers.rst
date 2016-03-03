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

Initializers
============

Each create layer with weights should be constructed with a provided
initializer class. These classes define how the weights are initialized
before training begins. Each class implements the ``fill(param)`` method
to assign values to the input tensor ``param``. Neon supports the
following initializers:

.. csv-table::
    :header: "Function", "Description"
    :widths: 20, 40
    :delim: |

    :py:class:`neon.initializers.Constant<neon.initializers.initializer.Constant>` | Initialize all tensors with a constant value ``val``
    :py:class:`neon.initializers.Array<neon.initializers.initializer.Array>` | Initialize all tensors with array values ``val``
    :py:class:`neon.initializers.Uniform<neon.initializers.initializer.Uniform>` | Uniform distribution from ``low`` to ``high``
    :py:class:`neon.initializers.Gaussian<neon.initializers.initializer.Gaussian>` | Gaussian distribution with mean ``loc`` and std. dev. ``scale``
    :py:class:`neon.initializers.GlorotUniform<neon.initializers.initializer.GlorotUniform>` | Uniform distribution from :math:`-k` to :math:`k`, where :math:`k` is scaled by the input dimensions (:math:`k = \sqrt{6/(d_{in} + d_{out})}`), see `Glorot, 2010 <http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>`_
    :py:class:`neon.initializers.Xavier<neon.initializers.initializer.Xavier>` | Alternate form of Glorot where only the input dimension is used for scaling :math:`k = \sqrt{3/d_{in}}`)
    :py:class:`neon.initializers.Kaiming<neon.initializers.initializer.Kaiming>` | Gaussian distribution with :math:`\mu = 0` and :math:`\sigma = \sqrt{2/d_{in}}`
    :py:class:`neon.initializers.IdentityInit<neon.initializers.initializer.IdentityInit>` | Fills with identity matrix
    :py:class:`neon.initializers.Orthonormal<neon.initializers.initializer.Orthonormal>` | Uses the singular value decomposition of a gaussian random matrix, scaled by a factor ``scale``. (see `Saxe, 2014 <http://arxiv.org/pdf/1312.6120v3.pdf>`_)

In the above table, :math:`d_{in}` and :math:`d_{out}` refer to the input and output dimensions of the input tensor, respectively. Neon assumes that

.. code-block:: python

    d_in = param.shape[0]
    d_out = param.shape[1]

Custom initialization schemes should subclass from
:py:class:`neon.initializers.Initializer<neon.initializers.initializer.Initializer>` and implement

.. code-block:: python

    # Constructor to define any needed parameters
    # (e.g. fill value, moments, name, etc.)
    def __init__(self, myParam=0.0, name="myInitName"):

    # Method to assign values to the input tensor `param`
    def fill(self, param):
