#!/usr/bin/env python
# this script runs all examples and checks that they all
# run without throwing and exception
import os
import sys
from glob import glob
import subprocess as subp

# Modify the following to suit your environment
NUM_EPOCHS = 2
BACKEND = "gpu"
SUBSET_PCT = 5
ADDITIONAL_ARGS = ""

BASE_DATA_DIR = '~/nervana/data'
I1K_BATCH_DIR = os.path.join(BASE_DATA_DIR, 'I1K/macrobatches')
CIFAR_BATCH_DIR = os.path.join(BASE_DATA_DIR, 'CIFAR10/macrobatches')

# skip examples that won't fit on a single GTX970 (ci1 box)
FILES_TO_SKIP = ['timeseries_lstm.py', 'vgg_bn.py', 'fast_rcnn_alexnet.py',
                 'i1k_msra.py']
ADD_I1K_BATCH_DIR = ['alexnet.py', 'imagenet_allcnn.py', 'vgg_bn.py',
                     'i1k_msra.py']
ADD_CIFAR_BATCH_DIR = ['cifar10_msra.py']
ADD_SUBSET_PCT = ADD_I1K_BATCH_DIR + ADD_CIFAR_BATCH_DIR

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

examples = glob('examples/*.py')
examples.append('examples/babi/train.py')

results = []
for ex in examples:
    print(ex)
    ex_bn = os.path.basename(ex)
    if ex_bn in FILES_TO_SKIP:
        print('Skipping this example\n\n')
        continue
    cmdargs = "-e {} -b {} --no_progress_bar -s {} {}".format(
            NUM_EPOCHS, BACKEND, os.path.splitext(ex_bn)[0] + '.prm',
            ADDITIONAL_ARGS)
    cmd = "python {} ".format(ex) + cmdargs

    if ex_bn in ADD_I1K_BATCH_DIR:
        cmd += ' -w {}'.format(I1K_BATCH_DIR)
    elif ex_bn in ADD_CIFAR_BATCH_DIR:
        cmd += ' -w {}'.format(CIFAR_BATCH_DIR)
    else:
        cmd += ' -w {}'.format(BASE_DATA_DIR)

    if ex_bn in ADD_SUBSET_PCT:
        cmd += ' --subset_pct {}'.format(SUBSET_PCT)
    rc = subp.call(cmd, shell=True)

    results.append([ex, rc])
    print('\n\n')

errors = 0
for dat in results:
    if dat[1] != 0:
        print('FAILURE on {}'.format(dat[0]))
        errors += 1
sys.exit(errors)
