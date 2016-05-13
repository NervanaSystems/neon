#!/usr/bin/env python
# this script runs all examples and checks that they all
# run without throwing an exception
from __future__ import print_function
import os
import sys
from glob import glob
import subprocess as subp
from datetime import timedelta
from timeit import default_timer as timer

# Modify the following to suit your environment
NUM_EPOCHS = 2
BACKEND = "gpu"
SUBSET_PCT = 5
ADDITIONAL_ARGS = ""

BASE_DATA_DIR = '~/nervana/data'
I1K_BATCH_DIR = os.path.join(BASE_DATA_DIR, 'I1K/macrobatches')
CIFAR_BATCH_DIR = os.path.join(BASE_DATA_DIR, 'CIFAR10/macrobatches')

# skip; implement subset pct to fit
FILES_TO_SKIP = ['examples/timeseries_lstm.py',
                 'examples/vgg_bn.py',
                 'examples/fast-rcnn/train.py']
# skip; needs to download dataset
FILES_TO_SKIP += ['examples/deep_dream.py',
                  'examples/imdb/train.py',
                  'examples/video-c3d/train.py']

ADD_I1K_BATCH_DIR = ['alexnet.py', 'imagenet_allcnn.py', 'vgg_bn.py',
                     'i1k_msra.py']
ADD_CIFAR_BATCH_DIR = ['cifar10_msra.py']
ADD_SUBSET_PCT = ADD_I1K_BATCH_DIR + ADD_CIFAR_BATCH_DIR + ['examples/fast-rcnn/train.py']

# Jenkins environment setup
if os.getenv("EXECUTOR_NUMBER"):
    BASE_DATA_DIR = '/usr/local/data/jenkins'
    I1K_BATCH_DIR = os.path.join(BASE_DATA_DIR, 'I1K/macrobatches')
    CIFAR_BATCH_DIR = os.path.join(BASE_DATA_DIR, 'CIFAR10/macrobatches')
    ADDITIONAL_ARGS += "-i {}".format(os.getenv("EXECUTOR_NUMBER"))

if not os.path.isdir('examples'):
    raise IOError('Must run from root dir of none repo')

# check for venv activations
cmd = 'if [ -z "$VIRTUAL_ENV" ];then exit 1;else exit 0;fi'
if subp.call(cmd, shell=True) > 0:
    raise IOError('Need to activate the virtualenv')

examples = glob('examples/*.py') + glob('examples/*/train.py')

skipped = []
results = []
for ex in sorted(examples):
    if ex in FILES_TO_SKIP:
        skipped.append(ex)
        continue
    cmdargs = "-e {} -b {} --serialize 1 --no_progress_bar -s {} {}".format(
        NUM_EPOCHS, BACKEND, os.path.splitext(ex)[0] + '.prm',
        ADDITIONAL_ARGS)
    cmd = "python {} ".format(ex) + cmdargs

    ex_bn = os.path.basename(ex)
    if ex_bn in ADD_I1K_BATCH_DIR:
        cmd += ' -w {}'.format(I1K_BATCH_DIR)
    elif ex_bn in ADD_CIFAR_BATCH_DIR:
        cmd += ' -w {}'.format(CIFAR_BATCH_DIR)
    else:
        cmd += ' -w {}'.format(BASE_DATA_DIR)

    if ex_bn in ADD_SUBSET_PCT:
        cmd += ' --subset_pct {}'.format(SUBSET_PCT)
    start = timer()
    rc = subp.call(cmd, shell=True)
    end = timer()
    results.append([ex, rc, end - start])

print('\nFound {} scripts:'.format(len(examples)))
for dat in results:
    if dat[1] == 0:
        print('SUCCESS on {} in {}'.format(dat[0], timedelta(seconds=int(dat[2]))))

for ex in skipped:
    print('SKIPPED {}'.format(ex))

errors = 0
for dat in results:
    if dat[1] != 0:
        print('FAILURE on {}'.format(dat[0]))
        errors += 1

print("\nExiting with %d errors" % errors)
sys.exit(errors)
