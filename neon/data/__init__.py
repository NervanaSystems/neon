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

from neon.data.dataiterator import NervanaDataIterator, DataIterator, ArrayIterator
from neon.data.datasets import Dataset
from neon.data.dataloaders import (load_mnist, load_cifar10, load_babi, load_flickr8k,
                                   load_flickr30k, load_coco, load_i1kmeta, load_text,
                                   I1Kmeta, load_shakespeare)
from neon.data.text import Text, Shakespeare, PTB, HutterPrize, IMDB
from neon.data.batch_writer import BatchWriter, BatchWriterI1K
from neon.data.dataloader import DataLoader
from neon.data.media import ImageParams, ImageIngestParams, VideoParams
from neon.data.imageloader import ImageLoader
from neon.data.questionanswer import BABI, QA
from neon.data.ticker import Ticker, CopyTask, RepeatCopyTask, PrioritySortTask
from neon.data.video import Video
from neon.data.imagecaption import ImageCaption, ImageCaptionTest, Flickr8k, Flickr30k, Coco
from neon.data.image import MNIST, CIFAR10
from neon.data.pascal_voc import PASCALVOCTrain, PASCALVOCInference
