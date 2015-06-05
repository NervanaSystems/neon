# neon

[neon](https://github.com/NervanaSystems/neon) is Nervana's Python based
Deep Learning framework. We have designed it with the following
functionality in mind:

* YAML for easy model specification (inspired by [pylearn2](https://github.com/lisa-lab/pylearn2))
* Python for easily adding models and support for many data formats
* Support for commonly used models: convnets, MLPs, RNNs, LSTMs, autoencoders,
  RBMs
* Support for common learning rules, activation functions and cost functions
* Comparing performance of alternate numeric representations with 32-bit floating point (fp32) for
  Deep Learning
* Support for using [spearmint](https://github.com/JasperSnoek/spearmint) for hyperparameter optimization
* Swappable hardware backends: write code once and then deploy on CPUs, GPUs,
  or Nervana hardware

Features that are unique to neon include:

* Tight integration with
  [nervanagpu](https://github.com/NervanaSystems/nervanagpu) kernels for fp16
  and fp32 ([benchmarks](https://github.com/soumith/convnet-benchmarks)) on
  Maxwell GPUs. These are the fastest implementations of the benchmark deep
  networks.
* 4.3s/macrobatch on AlexNet on Titan X (Full run on 1 GPU ~ 45 hrs)
* Out of the box [fp16 AlexNet model](examples/convnet/i1k-alexnet-fp16.yaml)
  that has the same accuracy as [fp32](examples/convnet/i1k-alexnet-fp32.yaml)
* Integration with our fork
  ([cudanet](https://github.com/NervanaSystems/cuda-convnet2)) of Alex
  Krizhevsky's cuda-convnet2 library for Kepler GPU support
* Support for our distributed processor (Nervana Engine&trade;) for deep learning.

We use neon internally at Nervana to solve our customers' problems across many
[domains](http://www.nervanasys.com/products/). We are hiring across several
roles. Apply [here](http://www.nervanasys.com/careers/)!


## Getting started

Basic information to get started is below. Please consult the
[full documentation](http://neon.nervanasys.com/docs/latest) for more
information.


### Installation

* [Local install and dependencies](http://neon.nervanasys.com/docs/latest/installation.html)
* Cloud-based access ([email us](mailto:demo@nervanasys.com) for an account)
* [Docker images](http://neon.nervanasys.com/docs/latest/installation.html#docker-images>)
  (community provided)

### Quick Install

On a Mac OSX or Linux machine, enter the following to download and install
neon, and use it to train your first multi-layer perceptron
or convolutional neural networks below.

    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    sudo make install

The above will install neon system-wide.  If you don't have sufficient
privileges or would prefer an isolated installation, see either our
[virtualenv](http://neon.nervanasys.com/docs/latest/installation.html#virtualenv)
based install, or take a look at some of the community provided [docker
images](http://neon.nervanasys.com/docs/latest/installation.html#docker-images).

There are several examples built-in to neon in the `examples` directory for a
user to get started. The YAML format is plain-text and can be edited to change
various aspects of the model. See the
[ANNOTATED\_EXAMPLE.yaml](examples/ANNOTATED_EXAMPLE.yaml) for some of the
definitions and possible choices.


### Running a simple MNIST model (on CPU)

	neon examples/mlp/mnist-small.yaml


### Running an Alexnet model (on GPU)

In [fp32](examples/convnet/i1k-alexnet-fp32.yaml):

	# for nervangpu (requires Maxwell GPUs)
	neon --gpu nervanagpu examples/convnet/i1k-alexnet-fp32.yaml

	# for cudanet (works with Kepler or Maxwell GPUs)
	neon --gpu cudanet examples/convnet/i1k-alexnet-fp32.yaml

In [fp16](examples/convnet/i1k-alexnet-fp16.yaml):

	neon --gpu nervanagpu examples/convnet/i1k-alexnet-fp16.yaml


### Code organization

	backends    --- implementation of different hardware backends
	datasets    --- support for common datasets CIFAR-10, ImageNet, MNIST etc.
	diagnostics --- hooks to measure timing and numeric ranges
	hyperopt    --- hooks for hyperparameter optimization
	layers      --- layer code
	models      --- model code
	optimizers  --- learning rules
	transforms  --- activation & cost functions
	metrics     --- performance evaluation metrics


### Documentation

The complete documentation for neon is available
[here](http://neon.nervanasys.com/docs/latest). Some useful starting points are:

* [Using neon](http://neon.nervanasys.com/docs/latest/using_neon.html)
* [API](http://neon.nervanasys.com/docs/latest/api.html)
* [Developing for neon](http://neon.nervanasys.com/docs/latest/developing_neon.html)


### Issues

For any bugs or feature requests please:

1. Search the open and closed
   [issues list](https://github.com/NervanaSystems/neon/issues) to see if we're
   already working on what you have uncovered.
2. Check that your issue/request has already been addressed in our
   [Frequently Asked Questions (FAQ)](http://neon.nervanasys.com/docs/latest/faq.html)
3. File a new [issue](https://github.com/NervanaSystems/neon/issues) or submit
   a new [pull request](https://github.com/NervanaSystems/neon/pulls) if you
   have some code you'd like to contribute


## Machine learning OPerations (MOP) Layer

The [MOP](http://neon.nervanasys.com/docs/latest/ml_operational_layer.html)
is an abstraction layer for Nervana's system software and
hardware which includes the Nervana Engine, a custom distributed
processor for deep learning.

The MOP consists of linear algebra and other operations required by deep
learning. Some MOP operations are currently exposed in neon, while others,
such as distributed primitives, will be exposed in later versions as well as
in other forthcoming Nervana libraries.

Defining models in a MOP-compliant manner guarantees they will run on all
provided backends. It also provides a way for existing Deep Learning frameworks
such as [theano](https://github.com/Theano/Theano),
[torch](https://github.com/torch/torch7), and
[caffe](https://github.com/BVLC/caffe) to interface with the Nervana Engine.


## Upcoming libraries

We have separate, upcoming efforts on the following fronts:

* Distributed models
* Automatic differentiation
* Integration with Nervana Cloud&trade;


## License

We are releasing [neon](https://github.com/NervanaSystems/neon) and
[nervanagpu](https://github.com/NervanaSystems/nervanagpu) under an open source
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License. We welcome
you to [contact us](mailto:info@nervanasys.com) with your use cases.
