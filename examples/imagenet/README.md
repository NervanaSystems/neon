# Imagenet Models
Alexnet - An implementation of a deep convolutional neural network for
classification of images from the ImageNet 2012 competition based on
[Krizhevsky, Sutskever and Hinton, 2012][kriz].

MSRA - An implementation of the variations of a deep residual network as described by He, et. al. in [msra]

## Overview
All examples trained on imagenet data use the [aeon][aeon] dataloader, which requires the training and validation images to be extracted from their original tar archives locally and associated with their appropriate category labels.  This step is known as ingestion.

First, the raw data must be obtained from ILSVRC.  The required input files consist of:
 - Train set images: `ILSVRC2012_img_train.tar`
 - Validation set images: `ILSVRC2012_img_val.tar`
 - The development kit with dataset metadata: `ILSVRC2012_devkit_t12.tar.gz`

Place them all in one common directory.  Then select an output location for the extracted files to go.  For reference, we will use the local shell variable `$I1K_DATA_PATH` as the output location.  Be sure to point use a locally mounted drive for this as step as it can involve a significant amount of data being written.

## Ingestion
Ingestion can be performed using the `ingest.py` script

```
I1K_DATA_PATH=/usr/local/data/imagenet  # any locally mounted path

python examples/imagenet/ingest.py --input_dir </path/to/imagenet_archives> --out_dir $I1K_DATA_PATH
```

By default, images are resized so that their short side is less than or equal to 256 pixels.  One can change this setting by supplying an optional `--target_size <maxdim>` argument to the ingest command.

Ingest only needs to occur once, and after the files are unpacked, the original tars are not used by the training scripts.

## Training
### AlexNet
This example trains something very similar to the deep convolutional neural network original implemented by [Krizhevsky et. al.][kriz]  The model can be trained using the command,
```bash
python examples/imagenet/alexnet.py --save_path </path/to/save/weights>
```
### AllCNN
This example trains a model that uses only convolutional layers to stride down, with no pooling layers as described by [Springenberg et. al.][allcnn].  The model can be trained using the command,
```bash
python examples/imagenet/allcnn.py --save_path </path/to/save/weights>
```
### Deep Residual Network
A basic residual network for imagenet can be trained using the command,
```bash
python examples/imagenet/i1k_msra.py --save_path </path/to/save/weights>
```

## Recommended Settings for Best Performance on Non-Uniform Memory Archiecture (NUMA) Enabled Systems
The aeon data loader of neon itself is not NUMA-aware. Specific NUMA configurations can greatly improve training performance of topologies like Resnet-50 on systems that have NUMA turned on (e.g. a dual-socket system where NUMA is enabled via the BIOS). The recommended settings is to use `numactl` command combined with `--interleave=all` option, i.e.,
```bash
numactl --interleave=all python examples/imagenet/i1k_msra.py --depth 50 --save_path </path/to/save/weights>
```
`numactl` is a standard Linux utility tool and could be obtained via package managers like `apt-get` on Ubuntu or `yum` on CentOS etc. Users are encouraged to explore other `numactl` options for their specific models and datasets.

## Citation
```
ImageNet Classification with Deep Convolutional Neural Networks
http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
```

```
Striving for Simplicity: the All Convolutional Net
http://arxiv.org/abs/1412.6806
```

```
Deep Residual Learning for Image Recognition
http://arxiv.org/abs/1512.03385
```

```
Identity Mappings in Deep Residual Networks
http://arxiv.org/abs/1603.05027
```
   [kriz]: <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>
   [allcnn]: <http://arxiv.org/abs/1412.6806>
   [msra1]: <http://arxiv.org/abs/1512.03385>
   [msra2]: <http://arxiv.org/abs/1603.05027>

