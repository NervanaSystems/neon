Tutorials
=========

We present tutorials that cover the implementation of common model
architectures:

* :doc:`Tutorial 1 <mnist>`: MNIST with multi-layer perceptron (MLP) network
* :doc:`Tutorial 2 <cifar10>`: Object recognition with a convolutional neural network
* :doc:`Tutorial 3 <lstm>`: Sentiment analysis with a recurrent neural network

In addition, we include several how-to guides for customizing neon and
visualizing the results. We recommend reading the section on the neon
:doc:`backend <backends>` as a prerequisite to these tutorials:

* :doc:`Tutorial 4 <creating_new_layers>`: Creating new layers
* :doc:`Tutorial 5 <tools>`: Visualizing the results

Since neon v2.0.0+ is now released with MKL backend support, we encourage users
to use ``-b mkl`` on Intel CPUs for all the tutorial examples used. 

.. toctree::
   :hidden:
   :maxdepth: 0

   mnist.rst
   cifar10.rst
   lstm.rst
   creating_new_layers.rst
   tools.rst
