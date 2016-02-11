# neon

**This is an unsupported branch of neon.** Please switch to neon master branch for the officially released library.


[neon](https://github.com/NervanaSystems/neon) is Nervana's Python based
Deep Learning framework and achieves the [fastest performance](https://github.com/soumith/convnet-benchmarks) on modern deep neural networks such as AlexNet, VGG and GoogLeNet. Designed for ease-of-use and extensibility.

* [Tutorials](http://neon.nervanasys.com/docs/latest/tutorials.html) and [iPython notebooks](https://github.com/NervanaSystems/meetup) to get users started with using neon for deep learning.
* Support for commonly used layers: convolution, RNN, LSTM, GRU, BatchNorm, and more.
* [Model Zoo](https://github.com/nervanazoo/NervanaModelZoo) contains pre-trained weights and example scripts for start-of-the-art models, including: [VGG](https://github.com/nervanazoo/NervanaModelZoo/tree/master/ImageClassification/ILSVRC2012/VGG), [Reinforcement learning](https://github.com/nervanazoo/NervanaModelZoo/tree/master/DeepReinforcement), [Deep Residual Networks](https://github.com/nervanazoo/NervanaModelZoo/tree/master/SceneClassification/DeepResNet), [Image Captioning](https://github.com/nervanazoo/NervanaModelZoo/tree/master/ImageCaptioning), [Sentiment analysis](https://github.com/nervanazoo/NervanaModelZoo/tree/master/NLP/SentimentClassification/IMDB), and [more](http://neon.nervanasys.com/docs/latest/model_zoo.html).
* Swappable hardware backends: write code once and then deploy on CPUs, GPUs, or Nervana hardware

For fast iteration and model exploration, neon has the fastest performance among deep learning libraries (2x speed of cuDNNv4, see [benchmarks](https://github.com/soumith/convnet-benchmarks)).
* 2.5s/macrobatch (3072 images) on AlexNet on Titan X (Full run on 1 GPU ~ 26 hrs)
* Training VGG with 16-bit floating point on 1 Titan X takes ~10 days (original paper: 4 GPUs for 2-3 weeks)

We use neon internally at Nervana to solve our customers' problems across many
[domains](http://www.nervanasys.com/solutions/). We are hiring across several
roles. Apply [here](http://www.nervanasys.com/careers/)!

See the [new features](https://github.com/NervanaSystems/neon/blob/master/ChangeLog) in our latest release.

## Quick Install

* [Local install and dependencies](http://neon.nervanasys.com/docs/latest/installation.html)

On a Mac OSX or Linux machine, enter the following to download and install
neon (conda users see the [guide](http://neon.nervanasys.com/docs/latest/installation.html)), and use it to train your first multi-layer perceptron.

```bash
    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    make
    . .venv/bin/activate
    neon examples/mnist_mlp.yaml
    # alternatively, use a script:
    python examples/mnist_mlp.py
```

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
