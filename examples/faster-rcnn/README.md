## Model

This example demonstrates how to train and test a faster R-CNN model using PASCAL VOC dataset.

The script will download the PASCAL dataset and ingest and provide the data for training and inference.

Reference:

"Faster R-CNN"\
http://arxiv.org/abs/1506.01497\
https://github.com/rbgirshick/py-faster-rcnn

### Model script
#### train.py

Trains a Fast-RCNN model to do object localization using PASCAL VOC dataset.

By default, the faster R-CNN model has several convolution and linear layers initialized from a pre-trained VGG16 model, and this script will download the VGG model from neon model zoo and load the weights for those layers. If the script is given --model_file, it will continue training the Fast R-CNN from the given model file. 

Usage:
```
python examples/faster-rcnn/train.py -e 14 -s faster_rcnn.pkl 
````

Notes:

1. The training currently runs 1 image in each minibatch.

2. The original caffe model goes through 40000 iteration (mb) of training, with
2 images per minibatch.

3. The model converges after training for 14 epochs.

4. The dataset can be cached as the preprocessed file and re-use if the same
configuration of the dataset is used again. The cached file by default is in:
    
### inference.py

Test a trained Faster-RCNN model to do object detection using PASCAL VOC dataset.
This test currently runs 1 image at a time.

Usage:
```
    python examples/fast-rcnn/test.py --model_file faster_rcnn.pkl
```
Notes:

1. For VGG16 based Faster R-CNN model, we can support testing with batch size as 1
images.

2. The dataset can be cached as the preprocessed file and re-use that if the same
configuration of the dataset is used again. 

3. The mAP evaluation script is adapted from:
https://github.com/rbgirshick/py-faster-rcnn/commit/45e0da9a246fab5fd86e8c96dc351be7f145499f


### Tests
There are a few unit tests for components of the model. It is setup based on the py.test framework. To run the tests,
```
py.test examples/faster-rcnn/tests
```
