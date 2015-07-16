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

Distributed Implementations using multiple GPUs
===============================================
In an effort to reduce the amount of time it takes to train and run models, it
is often advantageous to split the computation across several devices so that
it can be run in parallel.  We have implemented multi-GPU support in neon
using the strategy described by Krizhevsky in [AK2014]_.

Note that we only support parallel computation with multiple GPUs and not on
multiple CPUs.  Moreover, multi-GPU computation is only supported via our
``nervanagpu`` backend, which requires Maxwell architecture devices.

The parallel implementation used in neon has been tested on up to 8 GPUs.  All
devices must be housed in the same machine.

Parallelization Model
---------------------
The "weird trick" parallelization model implemented by Krizhevsky uses data
parallel mode in local layers (convolutional, pooling), where the activations
outnumber the model parameters, and model parallel mode in fully connected
layers, where the model parameters outnumber the activations.

In data-parallel mode, activations are fragmented and sent to different
devices, each of which contains a replica of the model parameters.  During the
model's update step, the local parameter gradients are shared with each other
to generate a total gradient that is applied using the learning rule for that
layer.

In model-parallel mode, activations are shared so that each device receives the
same input activations.  Each device only retains a fragment of the model
parameters (a slice of the weight matrix), which are used to compute a portion
of the output activations.  Output activations are then combined to generate
the replica activations that are used for the next layer.

Requirements
------------
In order to parallelize across ``N`` GPU device nodes, the following
conditions must be satisfied:

- In data parallel mode, the minibatch size must be a multiple of ``N``.
- In model parallel mode, the number of output units of each fully connected
  layer must be a multiple of ``N``.

For example, an MLP with no convolutional layers that has 3 hidden layers with
6, 200, and 20 hidden nodes can be parallelized across at most 2 GPUs (because
``GCD(6, 200, 20) == 2``).  If the first layer had 12 hidden nodes, the model
could be parallelized across 4 GPUs.

Since AlexNet [AK2012]_ has fully connected layers with outputs of 4096, 4096,
and 1000, it can be split across up to 8 GPUs (``GCD(4096, 1000) = 8``) as long
as the minibatch supplied is divisible by 8.


Usage
=====

The following example illustrates how to train a convnet on the MNIST dataset
across 2 GPUs (devices selected by default):

.. code-block:: bash

    neon --gpu nervanagpu2 examples/convnet/mnist-small.yaml


The following example illustrates how to train the same convnet with 2 GPUs,
but specifying devices 1 and 2 (Note that the device_ids specified here do not
necessarily correspond to how they appear when running ``nvidia-smi``):

.. code-block:: bash

    neon --gpu nervanagpu2 examples/convnet/mnist-small.yaml --device_id 1 2


The following example illustrates how to train a convnet on the i1k alexnet
model included with neon across 4 GPUs:

.. code-block:: bash

    neon --gpu nervanagpu4 examples/convnet/i1k-alexnet-fp32.yaml


Known Issues
============

Dropout Layers
--------------
Dropout layers occur between fully connected layers, which have replicated
activations across devices.  However, since the binary masks used for dropout
are generated on device, each activation replica undergoes a different random
masking.  This leads to slightly different results when training the same model
in parallel mode versus single GPU mode.  One way to mitigate this difference
would be to share masks during fprop, but this would introduce additional
communication overhead, and in practice we do not observe a penalty in network
performance with the current approach.


Batch Normalization
-------------------
For convolutional networks, using batch normalization with multiple GPUs leads
to faster convergence compared to using a single gpu.  This is because each
device is seeing only a portion of the overall batch, and the fragment batch
statistics are not shared during fprop.  In our implementation, we average
batch norm parameter gradients prior to updating to ensure that parameters stay
consistent across model replicas.

In fully connected layers, since activations are replicated on each device, the
batch normalization parameters should be identical without need for sharing.


References
==========

.. [AK2014] Alex Krizhevsky, One weird trick for parallelizing convolutional neural networks. http://arxiv.org/abs/1404.5997
.. [AK2012] Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton, ImageNet classification with deep convolutional neural networks. http://www.cs.toronto.edu/~kriz/imgnet-paper-2012.pdf
