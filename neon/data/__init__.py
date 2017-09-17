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

from neon.data.dataiterator import NervanaDataIterator, ArrayIterator
from neon.data.hdf5iterator import HDF5Iterator, HDF5IteratorOneHot, HDF5IteratorAutoencoder
from neon.data.datasets import Dataset
from neon.data.text import Text, Shakespeare, PTB, HutterPrize, IMDB, SICK
from neon.data.questionanswer import BABI, QA
from neon.data.ticker import Ticker, CopyTask, RepeatCopyTask, PrioritySortTask
from neon.data.imagecaption import ImageCaption, ImageCaptionTest, Flickr8k, Flickr30k, Coco
from neon.data.image import MNIST, CIFAR10, DUMMY
