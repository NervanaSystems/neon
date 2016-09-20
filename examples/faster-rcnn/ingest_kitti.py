import os
import glob
import json
import numpy as np
from PIL import Image
import math
from ingest_pascalvoc import create_manifest


def convert_annot_to_json(path, im_path, out_path, difficult):
    """
    Converts the KITTI annotations to json file.

    Uses the below reference for the KITTI dataset:

    OO representation of label format used in Kitti dataset.

    Description of fields from Kitti dataset dev kit: (link)[]
    The label files contain the following information, which can be read and
    written using the matlab tools (readLabels.m, writeLabels.m) provided within
    this devkit. All values (numerical or strings) are separated via spaces,
    each row corresponds to one object. The 15 columns represent:
    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.

    Arguments:
        path (string): path to KITTI annotation file
        im_path (string): path to image
        out_path (string): path to save the json file
        difficult (bool): include difficult objects
    """
    with open(path) as f:
        labels = f.readlines()

    # start empty dictionary
    annot = {'object': []}

    # load image
    im = np.array(Image.open(im_path))
    (h, w, c) = im.shape
    annot['size'] = {'depth': c, 'height': h, 'width': w}

    for label in labels:
        vals = label.split()
        type = vals[0]
        truncated = float(vals[1])
        occluded = int(vals[2])
        bbox = tuple([float(x) for x in vals[4:8]])
        bbox_int = tuple([int(math.floor(x)) for x in bbox])

        if type == 'DontCare':
            assert truncated == -1
            assert occluded == -1
        else:
            assert occluded in (0, 1, 2, 3)

        diff = truncated > 0.5 or occluded == 2

        # add object to annotation
        obj = {'bndbox': {'xmin': bbox_int[0], 'ymin': bbox_int[1],
               'xmax': bbox_int[2], 'ymax': bbox_int[3]},
               'difficult': difficult,
               'name': type,
               'truncated': truncated > 0.5,
               'occluded': occluded
               }

        if not diff or difficult:
            annot['object'].append(obj)

    # print "Saving to {}".format(out_path)

    with open(out_path, 'w') as f:
        json.dump(annot, f, indent=4)


def ingest_kitti(data_dir, train_percent=90):
    """
    Ingests the KITTI dataset. Peforms the following ops:
    1. Convert annotations to json format
    2. Split the training data into train and validation sets
    3. Write manifest file
    4. Write configuration file

    Arguments:
        data_dir (string): path to KITTI data
        train_percent (float): percent of data to use for training.
    """
    assert os.path.exists(data_dir)

    img_path = os.path.join(data_dir, 'image_2')
    annot_path = os.path.join(data_dir, 'label_2')

    # get list of images
    ext = '.png'
    images = glob.glob(os.path.join(img_path, '*' + ext))
    images = [os.path.splitext(im)[0] for im in images]
    images = [os.path.split(im)[1] for im in images]

    print "Found {} images".format(len(images))
    assert len(images) > 0, "Did not found any images. Check your input_dir."

    # for each image, convert the annotation to json
    annot_save_dir = os.path.join(data_dir, 'label_2-json')
    annot_save_dir_difficult = os.path.join(data_dir, 'label_2-json-difficult')

    if not os.path.exists(annot_save_dir):
        os.mkdir(annot_save_dir)

    if not os.path.exists(annot_save_dir_difficult):
        os.mkdir(annot_save_dir_difficult)

    for im in images:
        path = os.path.join(annot_path, im + '.txt')
        im_path = os.path.join(img_path, im + ext)

        assert os.path.exists(im_path)

        out_path = os.path.join(annot_save_dir, im + '.json')
        convert_annot_to_json(path, im_path, out_path, difficult=False)
        print("Writing to: {}".format(out_path))

        out_path = os.path.join(annot_save_dir_difficult, im + '.json')
        convert_annot_to_json(path, im_path, out_path, difficult=True)
        print("Writing to: {}".format(out_path))

    # shuffle files and split into training and validation set.
    np.random.seed(0)
    np.random.shuffle(images)

    train_count = (len(images) * train_percent) // 100
    train = images[:train_count]
    val = images[train_count:]

    # write manifest files
    train_manifest = os.path.join(data_dir, 'train.csv')
    create_manifest(train_manifest, train, annot_save_dir, img_path, ext=ext, overwrite=True)

    val_manifest = os.path.join(data_dir, 'val.csv')
    create_manifest(val_manifest, val, annot_save_dir_difficult, img_path, ext=ext, overwrite=True)

    # write configuration file
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'kitti.cfg')
    with open(config_path, 'w') as f:
        f.write('manifest = [train:{}, val:{}]\n'.format(train_manifest, val_manifest))
        f.write('epochs = 14\n')
        f.write('height = 375\n')
        f.write('width = 1242\n')
        f.write('batch_size = 1\n')
    print("Wrote config file to: {}".format(config_path))


if __name__ == '__main__':
    from configargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='path to KITTI dataset')
    args = parser.parse_args()

    ingest_kitti(os.path.join(args.input_dir, 'training'))
