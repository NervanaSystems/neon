# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Functions used to load commonly available datasets.
"""

import logging
import os
import sys
import urllib2
import zipfile

from neon import NervanaObject

logger = logging.getLogger(__name__)


class Dataset(NervanaObject):
    """
    Container class for stock datasets.

    Arguments:
        filename (str/list): name of the file to download
        url (str): url for dataset
        size (int/list): file size
        path (str): local path to place files
        subset_pct (float/int): percentage of data set to use for training
    """
    def __init__(self, filename, url, size, path='.', subset_pct=100):
        # parameters to use in dataset config serialization
        super(Dataset, self).__init__(name=None)
        self.filename = filename
        self.url = url
        self.size = size
        self.path = path
        self.subset_pct = subset_pct
        if subset_pct != 100:
            # placeholder to use partial data set
            raise NotImplemented('subset percentage feature is not yet implemented')

    def serialize(self):
        """
        Generates dictionary with the required parameters to describe this object
        """
        return self.get_description()

    def load_zip(self, filename, size):
        """
        Helper function for downloading test files
        Will download and unzip the file into the directory self.path

        Arguments:
            filename (str): name of file to download from self.url
            size (str): size of the file in bytes?

        Returns:
            str: Path to the downloaded dataset.
        """
        workdir, filepath = self._valid_path_append(self.path, '', filename)

        if not os.path.exists(filepath):
            self.fetch_dataset(self.url, filename, filepath, size)
        if '.zip' in filepath:
            zip_ref = zipfile.ZipFile(filepath)
            zip_ref.extractall(workdir)
            zip_ref.close()
            filepath = filepath.split('.zip')[0]
        return filepath

    @staticmethod
    def _valid_path_append(path, *args):
        """
        Helper to validate passed path directory and append any subsequent
        filename arguments.

        Arguments:
            path (str): Initial filesystem path.  Should expand to a valid
                        directory.
            *args (list, optional): Any filename or path suffices to append to path
                                    for returning.

            Returns:
                (list, str): path prepended list of files from args, or path alone if
                         no args specified.

        Raises:
            ValueError: if path is not a valid directory on this filesystem.
        """
        full_path = os.path.expanduser(path)
        res = []
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        if not os.path.isdir(full_path):
            raise ValueError("path: {0} is not a valid directory".format(path))
        for suffix_path in args:
            res.append(os.path.join(full_path, suffix_path))
        if len(res) == 0:
            return path
        elif len(res) == 1:
            return res[0]
        else:
            return res

    @staticmethod
    def fetch_dataset(url, sourcefile, destfile, totalsz):
        """
        Download the file specified by the given URL.

        Args:
            url (str): Base URL of the file to be downloaded.
            sourcefile (str): Name of the source file.
            destfile (str): Path to the destination.
            totalsz (int): Size of the file to be downloaded.
        """
        cloudfile = urllib2.urlopen(os.path.join(url, sourcefile))
        print("Downloading file: {}".format(destfile))
        blockchar = u'\u2588'  # character to display in progress bar
        with open(destfile, 'wb') as f:
            data_read = 0
            chunksz = 1024**2
            while 1:
                data = cloudfile.read(chunksz)
                if not data:
                    break
                data_read = min(totalsz, data_read + chunksz)
                progress_string = u'Download Progress |{:<50}| '.format(
                    blockchar * int(float(data_read) / totalsz * 50))
                sys.stdout.write('\r')
                sys.stdout.write(progress_string.encode('utf-8'))
                sys.stdout.flush()

                f.write(data)
            print("Download Complete")

    def gen_iterators(self):
        # children of this class will need to implement this method
        raise NotImplemented()


class I1Kmeta(Dataset):
    """
    Helper class for loading the I1K dataset meta data

    Not an actual dataset
    """
    def __init__(self, path='.'):
        url = 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/imagenet',
        super(I1Kmeta, self).__init__('neon_ILSVRC2012_devmeta.zip',
                                      url,
                                      758648,
                                      path=path)

    def load_data(self):
        self.file_path = Dataset.load_zip('i1kmeta', self.path)
        return self.file_path


# functions below are deprecated and will be removed in a future release
def load_i1kmeta(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_i1kmeta  # noqa
    return load_i1kmeta(path)


def _valid_path_append(path, *args):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import _valid_path_append  # noqa
    return _valid_path_append(path, *args)


def fetch_dataset(url, sourcefile, destfile, totalsz):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import fetch_dataset  # noqa
    return fetch_dataset(url, sourcefile, destfile, totalsz)


def load_mnist(path=".", normalize=True):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_mnist  # noqa
    return load_mnist(path=path, normalize=normalize)


def _compute_zca_transform(imgs, filter_bias=0.1):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import _compute_zca_transform  # noqa
    return _compute_zca_transform(imgs, filter_bias=filter_bias)


def zca_whiten(train, test, cache=None):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import zca_whiten  # noqa
    return zca_whiten(train, test, cache=cache)


def global_contrast_normalize(X, scale=1., min_divisor=1e-8):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import global_contrast_normalize
    return global_contrast_normalize(X, scale=scale, min_divisor=min_divisor)


def load_cifar10(path=".", normalize=True, contrast_normalize=False, whiten=False):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_cifar10  # noqa
    return load_cifar10(path=path,
                        normalize=normalize,
                        contrast_normalize=contrast_normalize,
                        whiten=whiten)


def load_babi(path=".", task='qa1_single-supporting-fact', subset='en'):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    raise NotImplemented('load_babi has been removed')


def load_ptb_train(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_ptb_train  # noqa
    return load_ptb_train(path)


def load_ptb_valid(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_ptb_valid  # noqa
    return load_ptb_valid(path)


def load_ptb_test(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_ptb_test  # noqa
    return load_ptb_test(path)


def load_hutter_prize(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_hutter_prize  # noqa
    return load_hutter_prize(path)


def load_shakespeare(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_shakespeare  # noqa
    return load_shakespeare(path)


def load_flickr8k(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_flickr8k  # noqa
    return load_flickr8k(path)


def load_flickr30k(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_flickr30k  # noqa
    return load_flickr30k(path)


def load_coco(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_coco  # noqa
    return load_coco(path)


def load_imdb(path):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_imdb  # noqa
    return load_imdb(path)


def load_text(dataset, path="."):
    """
    Deprecated, moved to neon.data.dataloaders.
    """
    logger.error('This function has moved, import from neon.data.dataloaders')
    from neon.data.dataloaders import load_text  # noqa
    return load_text(dataset, path=path)
