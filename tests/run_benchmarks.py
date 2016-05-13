#!/usr/bin/env python
# this script runs all examples and checks that they all
# run without throwing and exception
from __future__ import print_function
import os
import sys
from glob import glob
import subprocess as subp

if not os.path.isdir('examples'):
    raise IOError('Must run from root dir of none repo')

# check for venv activations
cmd = 'if [ -z "$VIRTUAL_ENV" ];then exit 1;else exit 0;fi'
if subp.call(cmd, shell=True) > 0:
    raise IOError('Need to activate the virtualenv')

benchmarks = glob('examples/convnet-benchmarks/*.py')

results = []
for ex in benchmarks:
    for dt_arg in ['f16', 'f32']:
        print((ex, dt_arg))
        ex_bn = os.path.basename(ex)
        cmd = "python {} -d {}".format(ex, dt_arg)

        rc = subp.call(cmd, shell=True)

        results.append([ex, rc])
        print('\n\n')

errors = 0
for dat in results:
    if dat[1] != 0:
        print('FAILURE on {}'.format(dat[0]))
        errors += 1
sys.exit(errors)
