# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
Verify that different ways of loading datasets lead to the same result.

This test utility accepts the same command line parameters as neon. It
downloads the CIFAR-10 dataset and saves it as individual JPEG files. It then
proceeds to fit and evaluate a model using two different ways of loading the
data. Macrobatches are written to disk as needed.

run as follows:
python compare.py -e 1 -r 0 -b cpu

"""
import os
import numpy as np
from neon.data import DataIterator
from neon.initializers import Uniform
from neon.layers import Affine, Conv, Pooling, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Misclassification, Rectlin, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.data import load_cifar10, ImageLoader
from neon.util.batch_writer import BatchWriter
from PIL import Image
from glob import glob

trainimgs = 'trainimgs'
testimgs = 'testimgs'


def process_dataset(data, labels, inputpath, leafdir):
    datadir = os.path.join(inputpath, leafdir)
    print('Saving images to %s' % datadir)
    os.mkdir(datadir)
    ulabels = np.unique(labels)
    for ulabel in ulabels:
        os.mkdir(os.path.join(datadir, str(ulabel)))
    for idx in range(data.shape[0]):
        im = data[idx].reshape((3, 32, 32))
        im = np.uint8(np.transpose(im, axes=[1, 2, 0]).copy())
        im = Image.fromarray(im)
        path = os.path.join(datadir, str(labels[idx][0]), str(idx) + '.jpg')
        im.save(path, format='JPEG', subsampling=0, quality=95)


def process(inputpath):
    (X_train, y_train), (X_test, y_test), nclass = load_cifar10(inputpath,
                                                                normalize=False)
    process_dataset(X_train, y_train, inputpath, trainimgs)
    process_dataset(X_test, y_test, inputpath, testimgs)


def load_dataset(basepath, datadir, shuffle):
    path = os.path.join(basepath, datadir)
    if not os.path.exists(path):
        process(basepath)
    subdirs = glob(os.path.join(path, '*'))
    labelnames = sorted(map(lambda x: os.path.basename(x), subdirs))
    inds = range(len(labelnames))
    labeldict = {key: val for key, val in zip(labelnames, inds)}
    lines = []
    for subdir in subdirs:
        subdirlabel = labeldict[os.path.basename(subdir)]
        files = glob(os.path.join(subdir, '*.jpg'))
        lines += [(filename, subdirlabel) for filename in files]
    assert(len(lines) > 0)
    data = None
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(lines)
    for idx in range(len(lines)):
        im = np.asarray(Image.open(lines[idx][0]))[:, :, ::-1]
        im = np.transpose(im, axes=[2, 0, 1]).ravel()
        if data is None:
            data = np.empty((len(lines), im.shape[0]), dtype='float32')
            labels = np.empty((len(lines), 1), dtype='int32')
        data[idx] = im
        labels[idx] = lines[idx][1]
    return (data, labels)


def load_cifar10_imgs(path):
    (X_train, y_train) = load_dataset(path, trainimgs, shuffle=True)
    (X_test, y_test) = load_dataset(path, testimgs, shuffle=False)
    return (X_train, y_train), (X_test, y_test), 10


def write_batches(args, macrodir, datadir, val_pct):
    if os.path.exists(macrodir):
        return
    print('Writing batches to %s' % macrodir)
    bw = BatchWriter(out_dir=macrodir,
                     image_dir=os.path.join(args.data_dir, datadir),
                     target_size=32, macro_size=1024,
                     file_pattern='*.jpg', validation_pct=val_pct)
    bw.run()


def run(args, train, test):
    init_uni = Uniform(low=-0.1, high=0.1)
    opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                      momentum_coef=0.9,
                                      stochastic_round=args.rounding)
    layers = [Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=True),
              Pooling((2, 2)),
              Conv((5, 5, 32), init=init_uni, activation=Rectlin(), batch_norm=True),
              Pooling((2, 2)),
              Affine(nout=500, init=init_uni, activation=Rectlin(), batch_norm=True),
              Affine(nout=10, init=init_uni, activation=Softmax())]
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    mlp = Model(layers=layers)
    callbacks = Callbacks(mlp, train, eval_set=test, **args.callback_args)
    mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
    err = mlp.eval(test, metric=Misclassification())*100
    print('Misclassification error = %.2f%%' % err)
    return err


def test_iterator():
    parser = NeonArgparser(__doc__)
    args = parser.parse_args()
    (X_train, y_train), (X_test, y_test), nclass = load_cifar10_imgs(path=args.data_dir)
    train = DataIterator(X_train, y_train, nclass=nclass, lshape=(3, 32, 32))
    test = DataIterator(X_test, y_test, nclass=nclass, lshape=(3, 32, 32))
    return run(args, train, test)


def test_loader():
    parser = NeonArgparser(__doc__)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, 'macrotrain')
    test_dir = os.path.join(args.data_dir, 'macrotest')
    write_batches(args, train_dir, trainimgs, 0)
    write_batches(args, test_dir, testimgs, 1)
    train = ImageLoader(set_name='train', do_transforms=False, inner_size=32,
                        repo_dir=train_dir)
    test = ImageLoader(set_name='validation', do_transforms=False, inner_size=32,
                       repo_dir=test_dir)
    err = run(args, train, test)
    return err


assert test_iterator() == test_loader()
