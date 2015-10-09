# neon

[neon](https://github.com/NervanaSystems/neon) is Nervana's Python based
Deep Learning framework and achieves the [fastest performance](https://github.com/soumith/convnet-benchmarks) on many common deep neural networks such as AlexNet, VGG and GoogLeNet.
We have designed it with the following
functionality in mind:

* Support for commonly used models and examples: convnets, MLPs, RNNs, LSTMs, autoencoders
* Tight integration with nervanagpu kernels for fp16 and fp32 ([benchmarks](https://github.com/soumith/convnet-benchmarks)) on Maxwell GPUs
	* 3s/macrobatch (3072 images) on AlexNet on Titan X (Full run on 1 GPU ~ 32 hrs)
	* Fast image captioning model (~200x faster than CPU based NeuralTalk)
* Basic automatic differentiation support
* Framework for visualization
* Swappable hardware backends: write code once and then deploy on CPUs, GPUs, or Nervana hardware


[New features](https://github.com/NervanaSystems/neon/blob/master/ChangeLog) in latest release.

We use neon internally at Nervana to solve our customers' problems across many
[domains](http://www.nervanasys.com/products/). We are hiring across several
roles. Apply [here](http://www.nervanasys.com/careers/)!


## Getting started

Basic information to get started is below. Please consult the
[full documentation](http://neon.nervanasys.com/docs/latest) for more
information.


### Installation

* [Local install and dependencies](http://neon.nervanasys.com/docs/latest/user_guide.html#installation)

### Quick Install

On a Mac OSX or Linux machine, enter the following to download and install
neon, and use it to train your first multi-layer perceptron
or convolutional neural networks below.

    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    make
    . .venv/bin/activate
    neon examples/mnist_mlp.yaml
    # alternatively, use a script:
    python examples/mnist_mlp.py


### Code organization

    backends    --- implementation of different hardware backends
    layers      --- layer code
    models      --- model code
    optimizers  --- learning rules
    transforms  --- activation & cost functions


### Documentation

The complete documentation for neon is available
[here](http://neon.nervanasys.com/docs/latest). Some useful starting points are:

* [Using neon](http://neon.nervanasys.com/docs/latest/user_guide.html)
* [API](http://neon.nervanasys.com/docs/latest/api.html)
* [Developing for neon](http://neon.nervanasys.com/docs/latest/developer_guide.html)


### Support

For any bugs or feature requests please:

1. Search the open and closed
   [issues list](https://github.com/NervanaSystems/neon/issues) to see if we're
   already working on what you have uncovered.
2. Check that your issue/request hasn't already been addressed in our
   [Frequently Asked Questions (FAQ)](http://neon.nervanasys.com/docs/latest/faq.html)
   or [neon-users](https://groups.google.com/forum/#!forum/neon-users) Google
   group.
3. File a new [issue](https://github.com/NervanaSystems/neon/issues) or submit
   a new [pull request](https://github.com/NervanaSystems/neon/pulls) if you
   have some code you'd like to contribute

For other questions and discussions please:

1. Post a message to the
   [neon-users](https://groups.google.com/forum/?hl=en#!forum/neon-users)
   Google group

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

## License

We are releasing [neon](https://github.com/NervanaSystems/neon) under an open source
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License. We welcome you to [contact us](mailto:info@nervanasys.com) with your use cases.


