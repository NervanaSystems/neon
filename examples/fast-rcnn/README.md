* train.py
Trains a Fast-RCNN model on PASCAL VOC dataset.
This Fast-RCNN is based on VGG16 that was pre-trained using ImageI1K.

By default, the script will download the pre-trained VGG16 from neon model zoo and seed the convolution and pooling layers. And Fast R-CNN starts training from that. If the script is given --model_file, it will continue training the Fast R-CNN from the given model file.

Reference:
    "Fast R-CNN"
    http://arxiv.org/pdf/1504.08083v2.pdf
    https://github.com/rbgirshick/fast-rcnn

Usage:
    python examples/fast-rcnn/train.py -e 20 --save_path frcn_vgg.pkl

Notes:

    1. For VGG16 based Fast R-CNN model, we can support training/testing with small
    batch size such as, 2 or 3 images per batch. The model training will converge
    around 20 epochs. With 3 images per batch, and 64 ROIs per image, the training
    consumes about 11G memory.

    2. The original caffe model goes through 40000 iteration (mb) of training, with
    2 images per minibatch.

    3. The dataset will cache the preprocessed file and re-use that if the same
    configuration of the dataset is used again. The cached file by default is in
    ~/nervana/data/VOCDevkit/VOC<year>/train_< >.pkl or
    ~/nervana/data/VOCDevkit/VOC<year>/inference_< >.pkl
    

* inference.py
Test a trained Fast-RCNN model to do object detection using PASCAL VOC dataset.
This test currently runs 1 image at a time.

Usage:
    python examples/fast-rcnn/inference.py --model_file frcn_vgg.pkl

Notes:
    1. For VGG16 based Fast R-CNN model, we can support testing with batch size as 1
    images. The testing consumes about 7G memory.

    2. During testing/inference, all the selective search ROIs will be used to go
    through the network, so the inference time varies based on how many ROIs in each
    image. For PASCAL VOC 2007, the average number of SelectiveSearch ROIs is around
    2000.

    3. The dataset will cache the preprocessed file and re-use that if the same
    configuration of the dataset is used again. The cached file by default is in
    ~/nervana/data/VOCDevkit/VOC<year>/train_< >.pkl or
    ~/nervana/data/VOCDevkit/VOC<year>/inference_< >.pkl

The mAP evaluation script is adapted from:
https://github.com/rbgirshick/py-faster-rcnn/commit/45e0da9a246fab5fd86e8c96dc351be7f145499f
