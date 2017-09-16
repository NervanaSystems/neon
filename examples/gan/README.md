## Generative Adversarial Network on MNIST Data

This example demonstrates how to train generative adversarial networks as first
described by [Goodfellow et. al.][goodfellow14].  The example here uses Batch
Normalization, strided convolutions and other tricks described in
[Radford et al.][radford15] and [Arjovsky et al.][[arjovsky17]], as well as the
excellent [NIPS tutorial][goodfellow16] by Ian Goodfellow.


### Pre-requisites:
Installations of matplotlib, scipy, and lmdb are required to run the following
examples.

```bash
. .venv/bin/activate
pip install matplotlib
pip install scipy
pip install lmdb
```

### MNIST Data
This is an example of DCGAN and WGAN trained with MNIST images, which allows
quick validation of GAN models.

#### DCGAN
This simple example follows the best practices collectively considered DCGAN,
and is trained on MNIST data. It can be run with

```bash
examples/gan/mnist_dcgan.py
```

This will produce a progress bar with the discriminator cost, and periodically
produce plots of some of the data samples and model samples with filenames
`mnist_dcgan_[yyyy-mm-dd-HH-MM-SS]_?.png` after the first epoch, etc., stored
in the `examples/gan/results` directory.

#### Training WGAN ([Arjovsky et al. 2017][arjovsky17])
This simple example trains a deep-convolutional Wasserstein GAN to generate
MNIST images. Run

```bash
examples/gan/mnist_wgan.py
```

This will train the GAN for 32 epochs over the MNIST training set. At the end
of the training, one should obtain a Wasserstein distance estimate at about 0.1.

Generated images will be dumped in the `examples/gan/results` directory
similarly as described above. In addition, a learning curve (Waserstein
estimates plotted against generator iterations) will be created in the same
directory and updated at the end of each epoch.

### LSUN Data
This is an example of DCGAN and WGAN trained on the LSUN bedroom images.

#### Download and ingest LSUN images for training
For the first time, LSUN bedroom images need to be downloaded and ingested for
AEON data loader.

```bash
LSUN_DATA_PATH=<some/directory/to/hold/lsun/data>
python examples/gan/lsun_data.py --out_dir $LSUN_DATA_PATH --category bedroom --dset train --png
```
This script will first download, then unpack, and finally ingest LSUN images
for training. A manifest file will be generated in the data directory and a
configuration file `train.cfg` will be generated in the `examples/gan`
directory. *Note:* Data download and ingestion could take a long time due to
the large size.

#### Training DCGAN ([Radford et al. 2015][radford15])
Run the following script to train a DCGAN on the LSUN data:
```bash
examples/gan/lsun_dcgan.py
```

Generated images will be dumped in the `examples/gan/results`
directory similarly as described above.

#### Training WGAN ([Arjovsky et al. 2017][arjovsky17])
Run the following script to train a DC-WGAN on the LSUN data:
```bash
examples/gan/lsun_wgan.py
```
The script follows all the hyper-parameters, including training schedule as in the
[Wasserstein GAN `pytorch` repo](https://github.com/martinarjovsky/WassersteinGAN):
Number of critics defaults to `5` and changed to `100` for the initial `25` and
then every `500` generator iterations.

Generated images and learning curve will be stored in the `examples/gan/results`
directory similarly as described above.

## References
```
Generative Adversarial Nets
http://arXiv.org/abs/1406.2661
```
```
Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks
http://arxiv.org/abs/1511.06434
```
```
NIPS 2016 Tutorial: Generative Adversarial Networks
http://arXiv.org/abs/1701.00160
```
```
Wasserstein GAN
http://arxiv.org/abs/1701.07875
```
   [goodfellow14]: <http://arXiv.org/abs/1406.2661>
   [radford15]: <http://arxiv.org/abs/1511.06434>
   [goodfellow16]: <http://arXiv.org/abs/1701.00160>
   [arjovsky17]: <http://arxiv.org/abs/1701.07875>
