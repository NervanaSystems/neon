## Faster-RCNN

This example demonstrates how to train and test the Faster R-CNN model on the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset. This object localization model learns to detect objects in natural scenes and provide bounding boxes and category information for each object

Reference:

"Faster R-CNN"

http://arxiv.org/abs/1506.01497

https://github.com/rbgirshick/py-faster-rcnn

### Data preparation

Note: This example requires installing our new dataloader: [aeon](https://github.com/NervanaSystems/aeon). For more information, see the [aeon documentation](http://aeon.nervanasys.com/index.html/)

First, download the PASCALVOC 2007 [training](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [testing](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) datasets to a local directory. These datasets consist of images of scenes and corresponding annotation files with bounding box and category information for each object in the scene.

Then, run the `ingest_pascalvoc.py` script to decompress and process the files into an output directoy which we use the shell variable `$PASCAL_DATA_PATH`:

```
python ingest_pascalvoc.py --input_dir <dir/containing/tar/files> --output_dir $PASCAL_DATA_PATH
```
The above script will:

1. Decompress the tar files into the output directory.

2. Convert the annotations from XML to the json format expected by our dataloader. The converted json files are saved to the folders `Annotations-json` and `Annotations-json-inference`. When training the model, we exclude objects with the 'difficult' metadata tag. For evaluating the model however, the 'difficult' objects are included (following the above reference), so we create separate folders for the two conditions.

3. Write manifest files for the training and testing sets. These are written to $PASCAL_DATA_PATH

4. Write a configuration file to pass to neon. The config file is written to the `examples/faster_rcnn` folder as `pascalvoc.cfg`. The config file contains the paths to the manifest files, as well as some other dataset-specific settings. For example:

```
manifest = [train:/usr/local/data/VOCdevkit/VOC2007/trainval.csv, val:/usr/local/data/VOCdevkit/VOC2007/val.csv]
manifest_root = /usr/local/data/
epochs = 14
height = 1000
width = 1000
batch_size = 1
rng_seed = 0
```


#### Training

To train the model on the PASCALVOC 2007 dataset, use:
```
python train.py --config <path/to/config/file> --verbose --save_path frcn_model.prm
````

The above command will train the model for 14 epochs (~70K iterations) and save the model to the file `frcn_model.prm`. Note that the training uses a minibatch of 1 image.

By default, the Faster R-CNN model has several convolution and linear layers that are initialized from a pre-trained VGG16 model. These VGG weights will be automatically downloaded from the [neon model zoo](https://github.com/NervanaSystems/ModelZoo) and saved in `~/nervana/data/pascalvoc_cache/`.

Note: the config file passes its contents to the python script as command-line arguments. The equivalent command by passing in the arguments directly is:
```
python train.py --manifest train:$PASCAL_DATA_PATH/VOCdevkit/VOC2007/trainval.csv \
--manifest val:$PASCAL_DATA_PATH/VOCdevkit/VOC2007/val.csv --manifest_root $PASCAL_DATA_PATH \
-e 14 --height 1000 --width 1000 --batch_size 1 --verbose --rng_seed 0 -s frcn_model.prm
```

### Testing

To evaluate the trained model using the Mean Average Precision (MAP) metric, use the below command.

Usage:
```
    python inference.py --config <path/to/config/file> --model_file frcn_model.prm --output results.prm
```

A fully trained model should yield a MAP of >69%. The inference results are saved in the file `results.prm`, which includes the predicted boxes and the average precision. The predicted bounding boxes for each image are a `N x 6` array,
with the followng attributes: `[x_min, y_min, x_max, y_max, score, class]`.

### Other files

This folder includes several other key files, which we describe here:
- `faster_rcnn.py`: Functions for creating the Faster R-CNN network and transforming the output to bounding box predictions.
- `proposal_layer.py`: Proposal layer.
- `objectlocalization.py`: Dataset-specific configurations and settings.

Several utility functions are also included:
- `voc_eval.py`: computes the MAP on the voc dataset.
- `util.py`: Bounding box calculations and non-max suppression.
- `generate_anchors.py`: Generate anchor boxes.
- `convert_xml_to_json.py`: Converts PASCAL XML format to json format.

### Tests
There are a few unit tests for components of the model, set up using the py.test framework. To run these tests, use the below command. The unit tests require defining the environment variables `$PASCAL_MANIFEST_PATH` and `$PASCAL_MANIFEST_ROOT`.
```
py.test examples/faster-rcnn/tests
```

### Other datasets

To extend Faster-RCNN to other datasets, write a script to ingest the data by converting the annotations into json format, and generate a manifest file according to the specifications in our [aeon documentation](http://aeon.nervanasys.com/index.html/). As an example, we included the ingest script for the KITTI dataset `ingest_kitti.py` and the configuration class `KITTI` in `objectlocalization.py`.



