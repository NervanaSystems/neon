## Overview

This example VGG directory contains scripts to perform VGG training and inference using MKL backend and GPU backend.

## Model

For GPU backend, we have ported the weights for the 16 and 19 layer VGG models from the Caffe Model Zoo (see [link](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014)): https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/VGG_D.p for VGG_D and https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/VGG_E.p for VGG_E
For MKL backend, we are planning to port the weights for the 16 and 19 layer VGG models from the Caffe Model Zoo (see [link](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014)) as well.

### Model script
The model run scripts included here ![vgg_neon_train.py](./vgg_neon_train.py) and ![vgg_neon_inference.py](./vgg_neon_inference.py) perform training and inference respectively. We are providing both the training and the inference script, they can be adapted for fine tuning this network but we have yet to test the training script because a successful training protocol may require details beyond what is available from the Caffe Model Zoo. The inference script will take the trained weight file as input: for GPU backend, supply it with the VGG_D.p or VGG_E.p. For MKL backend inference, please see below for instructions.

## Instructions 

To run this model script on the ILSVRC2012 dataset, you will need to have the data in the neon aeon format; follow
the instructions in the ![README.md](../imagenet/README.md) to setup the dataset.

If neon is installed into a `virtualenv`, make sure that it is activated before running the commands below.

### Training 

To run the training of VGG with MKL backend: 

`python -u vgg_neon_train.py -c vgg_mkl.cfg -vvv --save_path VGG16-model.prm --output_file VGG16-data.h5 --caffe`

`numactl -i all` is our recommendation to get as much performance as possible for Intel architecture-based servers which
feature multiple sockets and when NUMA is enabled. On such systems, please run the following:

`numactl -i all python -u vgg_neon_train.py -c vgg_mkl.cfg -vvv --save_path VGG16-model.prm --output_file VGG16-data.h5   --caffe`

vgg_mkl.cfg is an example configuration file where it describes the above generated dataset and also the other parameters.

To run the training of VGG with GPU backend:
modify vgg_mkl.cfg 'backend' entry or simply using the following command: 

`python -u vgg_neon_train.py -c vgg_mkl.cfg -b gpu -vvv --save_path VGG16-model.prm --output_file VGG16-data.h5 --caffe`

To run the evaluation/inference of the model:
```
# for 16 layer VGG D model 
python vgg_neon_inference.py -c vgg_mkl.cfg --vgg_ver D --model_file VGG16-model.prm -z 64 --caffe

# for 16 layer VGG E model
python vgg_neon_inference.py -c vgg_mkl.cfg --vgg_ver E --model_file VGG19-model.prm -z 64 --caffe
```
Note that the `--caffe` option is needed to match the dropout implementation used by Caffe.
Please note that VGG16.prm and VGG19.prm should be either trained using neon or ported from Caffe.

Please also note that for gpu backend, you can replace the VGG16-model.prm and VGG19-model.prm with VGG_D.p and VGG_E.p. 

### Version compatibility

Neon version: neon v2.2.

## Citation

```
Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman
arXiv:1409.1556
```

## License

For the model weight files please abide by the license posted with the Caffe weights files:
[Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).

