from configargparse import ArgumentParser
from convert_xml_to_json import convert_xml_to_json
import os


def ingest_pascal(datadir, year):

    root_dir = os.path.join(datadir, 'VOC' + year)
    assert os.path.exists(root_dir)

    # convert the annotations to json, exclude difficult objects (used for training)
    input_path = os.path.join(root_dir, 'Annotations')
    annot_path = os.path.join(root_dir, 'Annotations-json')
    print("Reading PASCAL XML files from {}".format(input_path))
    print("Converting XML files to json format, writing to: {}".format(annot_path))
    convert_xml_to_json(input_path, annot_path, difficult=False)

    # convert the annotations to json, including difficult objects (used for inference)
    annot_path_difficult = os.path.join(root_dir, 'Annotations-json-difficult')
    print("Converting XML files to json format (including objects with difficult flag),")
    print ("writing to: {}".format(annot_path_difficult))
    convert_xml_to_json(input_path, annot_path_difficult, difficult=True)

    img_dir = os.path.join(root_dir, 'JPEGImages')

    # write manifest file for training
    index_path = os.path.join(root_dir, 'ImageSets', 'Main', 'trainval.txt')
    manifest_path = os.path.join(root_dir, 'trainval.csv')
    create_manifest(manifest_path, index_path, annot_path, img_dir, overwrite=True)

    # write manifest file for testing
    # for testing, we inlude the difficult objects, so use annot_path_difficult
    index_path = os.path.join(root_dir, 'ImageSets', 'Main', 'test.txt')
    manifest_path_inference = os.path.join(root_dir, 'val.csv')
    create_manifest(manifest_path_inference, index_path,
                    annot_path_difficult, img_dir, overwrite=True)

    # write configuration file
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pascalvoc.cfg')
    with open(config_path, 'w') as f:
        f.write('manifest = [train:{}, val:{}]\n'.format(manifest_path, manifest_path_inference))
        f.write('epochs = 14\n')
        f.write('height = 1000\n')
        f.write('width = 1000\n')
        f.write('batch_size = 1\n')
    print("Wrote config file to: {}".format(config_path))


def create_manifest(manifest_path, index, annot_dir, img_dir, ext='.jpg', overwrite=False):
    """
    Based on a PASCALVOC index file, creates a manifest csv file. If the manifest file already
    exists, this function will skip writing, unless the overwrite argument is set to True.

    Arguments:
        manifest_path (string): path to save the manifest file
        index (string or list): list of images. if a string, the list will be read from the file.
        annot_dir (string): directory of annotations
        img_dir (string): direcoty of images
        ext (string, optional): image extension (default=.jpg)
        overwrite (bool, optional): overwrite manifest file (default=False)
    """
    if (overwrite or not os.path.exists(manifest_path)):

        if isinstance(index, basestring):
            with open(index) as f:
                images = f.readlines()
        elif isinstance(index, list):
            images = index
        else:
            raise ValueError("Expected index argument to be string or list.")

        with open(manifest_path, 'w') as f:
            for im in images:
                img = os.path.join(img_dir, im.rstrip('\n') + ext)
                annot = os.path.join(annot_dir, im.rstrip('\n') + '.json')

                assert os.path.exists(img), 'Path {} not found'.format(img)
                assert os.path.exists(annot), 'Path {} not found'.format(annot)

                f.write("{img},{annot}\n".format(img=img, annot=annot))

        print("Wrote manifest file to {}".format(manifest_path))
    else:
        print("Found existing manifest file, skipping. To overwrite, set overwrite=True")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='path to vocdevkit data')
    parser.add_argument('--year', default='2007', help='VOCdevkit year: 2007 (default) or 2012')
    args = parser.parse_args()

    ingest_pascal(args.input_dir, args.year)
