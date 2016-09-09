## Model

This example demonstrates how to train and test a faster R-CNN model using PASCAL VOC dataset.

The script will download the PASCAL dataset and ingest and provide the data for training and inference.

Reference:

"Faster R-CNN"\
http://arxiv.org/abs/1506.01497\
https://github.com/rbgirshick/py-faster-rcnn

### Model script
#### train.py

Trains a Faster-RCNN model to do object localization using PASCAL VOC dataset.

By default, the faster R-CNN model has several convolution and linear layers initialized from a pre-trained VGG16 model, and this script will download the VGG model from neon model zoo and load the weights for those layers. If the script is given --model_file, it will continue training the Faster R-CNN from the given model file. 

Usage:
```
python examples/faster-rcnn/train.py -r0 -e7 -s faster_rcnn.pkl -vv
````

Notes:

1. The training currently runs 1 image in each minibatch.

2. The original caffe model goes through 40000 iteration (mb) of training, with
1 images per minibatch (iteration), but 2 iterations per weight updates.

3. The model converges after training for 7 epochs.

4. The dataset can be cached as the preprocessed file and re-use if the same
configuration of the dataset is used again. The cached file by default ~/nervana/data
    
### inference.py

Test a trained Faster-RCNN model to do object detection using PASCAL VOC dataset.

Usage:
```
    python examples/faster-rcnn/inference.py --model_file faster_rcnn.pkl
```
Notes:

1. This test currently runs 1 image at a time.

2. The dataset can be cached as the preprocessed file and re-use that if the same
configuration of the dataset is used again. 

3. The mAP evaluation script is adapted from:
https://github.com/rbgirshick/py-faster-rcnn/


### Tests
There are a few unit tests for components of the model. It is setup based on the py.test framework. To run the tests,
```
py.test examples/faster-rcnn/tests
```
