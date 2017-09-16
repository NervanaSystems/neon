from __future__ import print_function
import os
import json
import numpy as np
from zipfile import ZipFile
from scipy.ndimage import imread
from scipy.misc import imsave, imresize


def get_image_scale(im_shape, im_reshape):
    if im_reshape is None:
        scale = [1.0, 1.0]
        return scale, im_shape
    else:
        assert len(im_reshape) == 2
        scale = [float(x)/float(y) for (x, y) in zip(im_reshape, im_shape)]
        return scale, im_reshape


def scale_boxes(bbox, scale):
    assert all(isinstance(x, float) for x in bbox), "BBox coordinates must be float."

    bbox[0] *= scale[1]
    bbox[2] *= scale[1]  # xmin/xmax div W scale
    bbox[1] *= scale[0]
    bbox[3] *= scale[0]  # ymin/ymax div H scale

    return bbox


def unzip_files(zipfiles, input_dir, data_dir):
    files = [os.path.join(input_dir, zipfile) for zipfile in zipfiles]

    for fid in files:
        with ZipFile(fid, 'r') as zf:
            print("Extracting {} to {}".format(fid, data_dir))
            zf.extractall(data_dir)


def write_config(config, config_path):
    with open(config_path, 'w') as f:
        for key in config:
            f.write('{} = {}\n'.format(key, config[key]))
    print("Wrote config file to: {}".format(config_path))


def resize_image(image, img_save_path, img_reshape):
    im = imread(image)
    if img_reshape is not None:
        im = imresize(im, img_reshape)
    imsave(img_save_path, im)
    return img_save_path


def write_ssd_config(ssd_config, ssd_config_path, overwrite=False):
    if not overwrite and os.path.exists(ssd_config_path):
        raise IOError("{} already exists, remove or use --overwrite flag".format(ssd_config_path))

    json.dump(ssd_config, open(ssd_config_path, 'w'), indent=4, separators=(',', ': '))
    print("Wrote SSD config file to: {}".format(ssd_config_path))


def create_manifest(manifest_path, manifest, root_dir):
    records = [('@FILE', 'FILE')]

    for entry in manifest:

        (annot, image) = entry
        assert os.path.exists(image), 'Path {} not found'.format(image)
        assert os.path.exists(annot), 'Path {} not found'.format(annot)

        records.append((os.path.relpath(image, root_dir),
                        os.path.relpath(annot, root_dir)))
    np.savetxt(manifest_path, records, fmt='%s\t%s')
    print("Writing manifest file ({} records) to: {}".format(len(manifest), manifest_path))


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print('Creating directory: {}'.format(directory))
