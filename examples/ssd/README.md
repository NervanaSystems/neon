## Single Shot Multibox Detector (SSD)

This example demonstrates how to train and test the Single Shot Multibox Detector (SSD) model on the
[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php)
datasets. This object localization model learns to detect objects in natural scenes and provide bounding boxes
and category information for each object.

Reference:

"SSD: Single Shot MultiBox Detector"
[https://arxiv.org/abs/1512.02325](https://arxiv.org/abs/1512.02325)

[https://github.com/weiliu89/caffe/tree/ssd]( https://github.com/weiliu89/caffe/tree/ssd)

### Pre-requisites:
Installation of scipy is required.

```bash
. .venv/bin/activate
pip install scipy
```

### Data preparation

Note: This example requires installing the Nervana aeon dataloader:
[aeon](https://github.com/NervanaSystems/aeon). For more information, see the
[aeon documentation](http://aeon.nervanasys.com/index.html/)

#### PASCALVOC

First, download the PASCALVOC 2007 [training](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar), [testing](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar), and PASCALVOC 2012 [training](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
datasets to a local directory. These datasets consist of images of scenes and corresponding
annotation files with bounding box and category information for each object in the scene.

Then, run the `ingest_pascalvoc.py` script to decompress and process the files into an output
directoy which we use the shell variable `$DATA`:

```
python datasets/ingest_pascalvoc.py --input_dir <dir/containing/tar/files> --output_dir $DATA --height 300 --width 300
```
The above script will:

1. Decompress the tar files into the output directory, inside the folder `VOCdevkit`.

2. Convert the annotations from XML to the json format expected by our dataloader. The converted json files are saved to the folder `VOCdevkit/VOC2007/Annotations-300x300`

3. Write manifest files for the training and testing sets. These are written to `$DATA/VOCdevkit/` as `val_300x300.csv` and `train_300x300.csv`.

4. Write a SSD model config file, written to `$DATA/VOCdevkit/pascalvoc_ssd_300x300.cfg`.

5. Write a configuration file to pass to neon. The config file is written to `$DATA/VOCdevkit/` folder as `pascalvoc_300x300.cfg`. The config file contains the paths to the manifest files, as well as some other dataset-specific settings. For example:

```
height = 300
epochs = 230
width = 300
manifest_root = /usr/local/data/VOCdevkit
manifest = [train:/usr/local/data/VOCdevkit/train_300x300.csv, val:/usr/local/data/VOCdevkit/val_300x300.csv]
ssd_config = [train:/usr/local/data/VOCdevkit/pascalvoc_ssd_300x300.cfg, val:/usr/local/data/VOCdevkit/pascalvoc_ssd_300x300_val.cfg]
```



#### KITTI

First, download the [KITTI Object Detection Evaluation 2012 dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php).
Only the [left color images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip)
and the [training labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip) files need to be
downloaded.

Then, run the `ingest_kitti.py` script to decompress and process the files into an output directoy
specified by the `--output_dir` command line option.  This script will also resize the KITTI images
from the original 375 by 1242 pixels to 300 x 994 pixels.  This maintains the original aspect ratio
while reducing the size for processing with the SSD model.

```
python ingest_kitti.py --input_dir <dir/containing/tar/files> --output_dir $DATA
```

The script will unzip the data into the folder `$DATA/kitti/`, and carry out a similar procedure as above for PASCALVOC dataset. The configuration file will be saved as `$DATA/kitti/kitti_300x994.cfg`.

Note that the SSD model configuration is slightly different from the PASCAL dataset since the aspect ratio of the images are different. See the SSD configuration files for more details on the differences.


#### Spacenet

[Spacenet](https://aws.amazon.com/public-datasets/spacenet/) is a dataset of satellite imagery with corresponding building footprints. Instructions for the downloading the dataset are found [here](https://github.com/SpaceNetChallenge/utilities/tree/master/content/download_instructions). Afer downloading the extracting the archive files, you should have folders for each of the cities, located in the `$DATA` folder. Then, run the ingest script:

```
python ingest_spacenet.py --data_dir $DATA --height 512 --weight 512
```

Note the above ingest script only works on the 3-band images.

The script will preprocess the images and convert the building footprints into enclosing bounding boxes. Several pre-processing steps are done:
1. Convert images to 512x512 shape, and resave as `png` files.
2. Compute the enclosing bounding box for the building footprints.
3. Shrink the bounding box to 80% of original size.
4. Remove images with >50% blank pixels.
5. Remove buildings smaller than 1% of the image width or height.

The converted images and annotations are saved in each city's folder. For example: `$DATA/AOI_2_Vegas_Train/RGB-PanSharpen-512x512`, and the config file for the entire dataset (combined across all cities), is saved in `$DATA/spacenet_512x512.cfg`.

#### Training

To train the model, use:
```
python train.py --config <path/to/config/file> --verbose --save_path  model.prm -z 32
```

The above command will train the model and save the model to the file `model.prm`.

By default, the SSD has several convolution and linear layers that are initialized from a pre-trained VGG16 model. These are automatically downloaded by the script.


### Testing

To evaluate the trained model using the Mean Average Precision (MAP) metric, use the below command.

Usage:
```
    python inference.py --config <path/to/config/file> --model_file frcn_model.prm --output results.prm
```

