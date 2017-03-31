## Generative Adversarial Network on MNIST Data

This example demonstrates how to train generative adversarial networks as first
described by [Goodfellow et. al.][goodfellow14].  The example here uses Batch
Normalization, strided convolutions and other tricks described in
[Radford et al.][radford15], as well as the excellent
[NIPS tutorial][goodfellow16] by Ian Goodfellow.


### MNIST Example
This simple example follows the best practices collectively considered DCGAN,
and is trained on MNIST data. It can be run with

```bash
python examples/gan/mnist-gan.py --original_cost
```

This will produce a progress bar with the discriminator cost, and periodically
produce plots of some of the data samples and model samples with filenames
`mnist_dcgan1.png` after the first epoch, etc., stored in the `examples/gan`
directory the model is called from.

### LSUN Example
This is an example of DCGAN and WGAN trained on the LSUN bedroom images.

#### Download and ingest LSUN images for training
For the first time, LSUN bedroom images need to be downloaded and ingested for AEON data loader.
```bash
LSUN_DATA_PATH=<some/directory/to/hold/lsun/data>
python examples/gan/lsun_data.py --out_dir $LSUN_DATA_PATH --category bedroom --dset train --png
```
This script will first download, then unpack, and finally ingest LSUN images for training.
A manifest file will be generated in the data directory and a configuration file `train.cfg` will be generated in the `examples/gan` directory.
*Note:* Data download and ingestion could take a long time due to the large size.

## References
```
Generative Adversarial Nets
http://arXiv.org/abs/1406.2661
```
````
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
http://arxiv.org/abs/1511.06434
```
```
NIPS 2016 Tutorial: Generative Adversarial Networks
http://arXiv.org/abs/1701.00160
```

   [goodfellow14]: <http://arXiv.org/abs/1406.2661>
   [radford15]: <http://arxiv.org/abs/1511.06434>
   [goodfellow16]: <http://arXiv.org/abs/1701.00160>
