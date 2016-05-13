# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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
Deprecated, use Datasets classes instead. Functions used to load commonly available datasets.
"""
import logging

from neon.data.datasets import Dataset
from neon.data.image import MNIST, CIFAR10
from neon.data.imagecaption import Flickr8k, Flickr30k, Coco
from neon.data.text import PTB, Shakespeare, IMDB, HutterPrize

logger = logging.getLogger(__name__)


class I1Kmeta(Dataset):
    """
    Helper class for loading the I1K dataset meta data

    This is not an actual dataset but instead the meta data
    for the I1K data set
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
    i1kmeta_dataset = I1Kmeta(path=path)
    return i1kmeta_dataset.file_path


def _valid_path_append(path, *args):
    return Dataset._valid_path_append(path, *args)


def fetch_dataset(url, sourcefile, destfile, totalsz):
    Dataset.fetch_dataset(url, sourcefile, destfile, totalsz)


def load_mnist(path=".", normalize=True):
    mnist_dataset = MNIST(path=path, normalize=normalize)
    return mnist_dataset.load_data()


def _compute_zca_transform(imgs, filter_bias=0.1):
    return CIFAR10._compute_zca_transform(imgs, filter_bias=filter_bias)


def zca_whiten(train, test, cache=None):
    return CIFAR10.zca_whiten(train, test, cache=cache)


def global_contrast_normalize(X, scale=1., min_divisor=1e-8):
    return CIFAR10.global_contrast_normalize(X, scale=scale, min_divisor=min_divisor)


def load_cifar10(path=".", normalize=True, contrast_normalize=False, whiten=False):
    cfiar10_dataset = CIFAR10(path=path,
                              normalize=normalize,
                              contrast_normalize=contrast_normalize,
                              whiten=whiten)
    return cfiar10_dataset.load_data()


def load_babi(path=".", task='qa1_single-supporting-fact', subset='en'):
    raise NotImplemented('load_babi has been removed')


def load_ptb_train(path):
    dataset = PTB(timesteps=None, path=path)
    dataset.load_data()
    return dataset.file_paths['train']


def load_ptb_valid(path):
    dataset = PTB(timesteps=None, path=path)
    dataset.load_data()
    return dataset.file_paths['valid']


def load_ptb_test(path):
    dataset = PTB(timesteps=None, path=path)
    dataset.load_data()
    return dataset.file_paths['test']


def load_hutter_prize(path):
    dataset = HutterPrize(path=path)
    return dataset.load_data()


def load_shakespeare(path):
    dataset = Shakespeare(timesteps=None, path=path)
    return dataset.load_data()


def load_flickr8k(path):
    dataset = Flickr8k(path=path)
    return dataset.load_data()


def load_flickr30k(path):
    dataset = Flickr30k(path=path)
    return dataset.load_data()


def load_coco(path):
    dataset = Coco(path=path)
    return dataset.load_data()


def load_imdb(path):
    dataset = IMDB(vocab_size=None, sentence_length=None, path=path)
    return dataset.load_data()


def load_text(dataset, path="."):
    logger.error('load_text function is deprecated as is the data_meta dictionary, '
                 'please use the new Dataset classes or the specific load_ function '
                 'for the dataset being used.')
    # deprecated method for load_text uses dataset as key to index
    # dataset_meta dictionary, for temporary backward compat, placed
    # dataset_meta dictionary here
    dataset_meta = {
        'mnist': {
            'size': 15296311,
            'file': 'mnist.pkl.gz',
            'url': 'https://s3.amazonaws.com/img-datasets',
            'func': load_mnist
        },
        'cifar-10': {
            'size': 170498071,
            'file': 'cifar-10-python.tar.gz',
            'url': 'http://www.cs.toronto.edu/~kriz',
            'func': load_cifar10
        },
        'babi': {
            'size': 11745123,
            'file': 'tasks_1-20_v1-2.tar.gz',
            'url': 'http://www.thespermwhale.com/jaseweston/babi',
            'func': load_babi
        },
        'ptb-train': {
            'size': 5101618,
            'file': 'ptb.train.txt',
            'url': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data',
            'func': load_ptb_train
        },
        'ptb-valid': {
            'size': 399782,
            'file': 'ptb.valid.txt',
            'url': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data',
            'func': load_ptb_valid
        },
        'ptb-test': {
            'size': 449945,
            'file': 'ptb.test.txt',
            'url': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data',
            'func': load_ptb_test
        },
        'hutter-prize': {
            'size': 35012219,
            'file': 'enwik8.zip',
            'url': 'http://mattmahoney.net/dc',
            'func': load_hutter_prize
        },
        'shakespeare': {
            'size': 4573338,
            'file': 'shakespeare_input.txt',
            'url': 'http://cs.stanford.edu/people/karpathy/char-rnn',
            'func': load_shakespeare
        },
        'flickr8k': {
            'size': 49165563,
            'file': 'flickr8k.zip',
            'url': 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/image-caption',
            'func': load_flickr8k
        },
        'flickr30k': {
            'size': 195267563,
            'file': 'flickr30k.zip',
            'url': 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/image-caption',
            'func': load_flickr30k
        },
        'coco': {
            'size': 738051031,
            'file': 'coco.zip',
            'url': 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/image-caption',
            'func': load_coco
        },
        'i1kmeta': {
            'size': 758648,
            'file': 'neon_ILSVRC2012_devmeta.zip',
            'url': 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/imagenet',
            'func': load_i1kmeta
        },
        'imdb': {
            'size': 33213513,
            'file': 'imdb.pkl',
            'url': 'https://s3.amazonaws.com/text-datasets',
            'func': load_imdb,
        }
    }
    meta = dataset_meta[dataset]
    ds = Dataset(meta['file'], meta['url'], meta['size'], path=path)
    return ds.load_zip(ds.filename, ds.size)
