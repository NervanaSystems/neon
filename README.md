# neon

[neon](https://github.com/NervanaSystems/neon) is Intel Nervana's reference deep learning framework committed to [best performance](https://github.com/soumith/convnet-benchmarks) on all hardware. Designed for ease-of-use and extensibility.

* [Tutorials](http://neon.nervanasys.com/docs/latest/tutorials.html) and [iPython notebooks](https://github.com/NervanaSystems/meetup) to get users started with using neon for deep learning.
* Support for commonly used layers: convolution, RNN, LSTM, GRU, BatchNorm, and more.
* [Model Zoo](https://github.com/NervanaSystems/ModelZoo) contains pre-trained weights and example scripts for start-of-the-art models, including: [VGG](https://github.com/NervanaSystems/ModelZoo/tree/master/ImageClassification/ILSVRC2012/VGG), [Reinforcement learning](https://github.com/NervanaSystems/ModelZoo/tree/master/DeepReinforcement), [Deep Residual Networks](https://github.com/NervanaSystems/ModelZoo/tree/master/SceneClassification/DeepResNet), [Image Captioning](https://github.com/NervanaSystems/ModelZoo/tree/master/ImageCaptioning), [Sentiment analysis](https://github.com/NervanaSystems/ModelZoo/tree/master/NLP/SentimentClassification/IMDB), and [more](http://neon.nervanasys.com/docs/latest/model_zoo.html).
* Swappable hardware backends: write code once and then deploy on CPUs, GPUs, or Nervana hardware

For fast iteration and model exploration, neon has the fastest performance among deep learning libraries (2x speed of cuDNNv4, see [benchmarks](https://github.com/soumith/convnet-benchmarks)).
* 2.5s/macrobatch (3072 images) on AlexNet on Titan X (Full run on 1 GPU ~ 26 hrs)
* Training VGG with 16-bit floating point on 1 Titan X takes ~10 days (original paper: 4 GPUs for 2-3 weeks)

We use neon internally at Intel Nervana to solve our customers' problems across many
[domains](http://www.nervanasys.com/solutions/). We are hiring across several
roles. Apply [here](http://www.nervanasys.com/careers/)!

See the [new features](https://github.com/NervanaSystems/neon/blob/master/ChangeLog) in our latest release.
We want to highlight that neon v2.0.0+ has been optimized for much better performance on CPUs by enabling Intel Math Kernel Library (MKL). The DNN (Deep Neural Networks) component of MKL that is used by neon is provided free of charge and downloaded automatically as part of the neon installation.

## Quick Install

* [Local install and dependencies](http://neon.nervanasys.com/docs/latest/installation.html)

On a Mac OSX or Linux machine, enter the following to download and install
neon (conda users see the [guide](http://neon.nervanasys.com/docs/latest/installation.html)), and use it to train your first multi-layer perceptron. To force a python2 or python3 install, replace `make` below with either `make python2` or `make python3`.

```bash
    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    make
    . .venv/bin/activate
```

Starting after neon v2.2.0, the master branch of neon will be updated weekly with work-in-progress toward the next release. Check out a release tag (e.g., "git checkout v2.2.0") for a stable release. Or simply check out the "latest" release tag to get the latest stable release (i.e., "git checkout latest")

**Warning**

> Between neon v2.1.0 and v2.2.0, the aeon manifest file format has been changed. When updating from neon < v2.2.0 manifests have to be recreated using ingest scripts (in examples folder) or updated using [this](neon/data/convert_manifest.py) script.

### Use a script to run an example

```bash
    python examples/mnist_mlp.py 
```

#### Selecting a backend engine from the command line

The gpu backend is selected by default, so the above command is equivalent to if a compatible GPU resource is found on the system:

```bash
    python examples/mnist_mlp.py -b gpu
```

When no GPU is available, the **optimized** CPU (MKL) backend is now selected by default as of neon v2.1.0, which means the above command is now equivalent to:

```bash
    python examples/mnist_mlp.py -b mkl
```

If you are interested in comparing the default mkl backend with the non-optimized CPU backend, use the following command:

```bash
    python examples/mnist_mlp.py -b cpu
```

### Use a yaml file to run an example

Alternatively, a yaml file may be used run an example.

```bash
    neon examples/mnist_mlp.yaml
```

To select a specific backend in a yaml file, add or modify a line that contains ``backend: mkl`` to enable mkl backend, or ``backend: cpu`` to enable cpu backend.  The gpu backend is selected by default if a GPU is available.

## Recommended Settings for neon with MKL on Intel Architectures

The Intel Math Kernel Library takes advantages of the parallelization and vectorization capabilities of Intel Xeon and Xeon Phi systems. When hyperthreading is enabled on the system, we recommend 
the following KMP_AFFINITY setting to make sure parallel threads are 1:1 mapped to the available physical cores. 

```bash
    export OMP_NUM_THREADS=<Number of Physical Cores>
    export KMP_AFFINITY=compact,1,0,granularity=fine  
```
or 
```bash
    export OMP_NUM_THREADS=<Number of Physical Cores>
    export KMP_AFFINITY=verbose,granularity=fine,proclist=[0-<Number of Physical Cores>],explicit
```
For more information about KMP_AFFINITY, please check [here](https://software.intel.com/en-us/node/522691).
We encourage users to set out trying and establishing their own best performance settings. 


## Documentation

The complete documentation for neon is available
[here](http://neon.nervanasys.com/docs/latest). Some useful starting points are:

* [Tutorials](http://neon.nervanasys.com/docs/latest/tutorials.html) for neon
* [Overview](http://neon.nervanasys.com/docs/latest/overview.html) of the neon workflow
* [API](http://neon.nervanasys.com/docs/latest/api.html) documentation
* [Resources](http://neon.nervanasys.com/docs/latest/resources.html) for neon and deep learning


## Support

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

For other questions and discussions please post a message to the
   [neon-users](https://groups.google.com/forum/?hl=en#!forum/neon-users)
   Google group

## License

We are releasing [neon](https://github.com/NervanaSystems/neon) under an open source
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License. We welcome you to [contact us](mailto:info@nervanasys.com) with your use cases.
