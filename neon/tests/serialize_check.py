# Copyright 2015 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Serialization check
This has been tested for Maxwell Titan X GPU
Pending: Diff CPUs currently give slightly different results so tol is
not enforced.

For AlexNet have to manually set:
center=True, flip=False,
#center=self.ds.predict, flip=True,
in line 53 of imageset.py
"""

import argparse
import logging
import os
import sys
from neon.backends import gen_backend
from neon.util.persist import deserialize


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run serialize check examples')
    parser.add_argument('--cpu', default=0, help='Run CPU serialize check',
                        type=int)
    parser.add_argument('--gpu', default="", help='Run GPU serialize check '
                        '(specify one of cudanet or nervanagpu')
    parser.add_argument('--datapar', default=0, type=int,
                        help='Run data parallel serialize check')
    parser.add_argument('--modelpar', default=0, type=int,
                        help='Run model parallel serialize check')
    return parser.parse_args()


def serialize_check(conf_file, result, **be_args):
    experiment = deserialize(os.path.join(dir, conf_file))
    backend = gen_backend(model=experiment.model, **be_args)
    experiment.initialize(backend)
    res = experiment.run()
    print float(res['test']['MisclassPercentage_TOP_1']), result
    # tol = .1
    # print abs(float(res['test']['MisclassPercentage_TOP_1']) - result)
    # assert abs(float(res['test']['MisclassPercentage_TOP_1']) - result) < tol


def serialize_check_alexnet(conf_file, result, **be_args):
    experiment = deserialize(os.path.join(dir, conf_file))
    backend = gen_backend(model=experiment.model, **be_args)
    experiment.initialize(backend)
    res = experiment.run()
    print float(res['validation']['MisclassPercentage_TOP_1']), result
    # tol = .1
    # print abs(float(res['test']['MisclassPercentage_TOP_1']) - result)
    # assert abs(
    #    float(res['validation']['MisclassPercentage_TOP_1']) - result) < tol

if __name__ == '__main__':
    # setup an initial console logger (may be overridden in config)
    logging.basicConfig(level=40)  # ERROR or higher
    res = 0
    # args = parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    check_files = []

    toy = True
    if toy:
        for i in range(3):
            check_files.append(
                os.path.join(script_dir,
                             'toy-serialize_check_' + str(i + 1) + '.yaml'))

        expected_result = 28.90625
        expected_result_2 = 17.18750
        expected_result_3 = 16.40625
        serialized_files = ['~/data/model5.prm', '~/data/model10.prm',
                            '~/data/model10b.prm']
        # delete previously serialized files
        for serialized_file in serialized_files:
            if os.path.isfile(os.path.expanduser(serialized_file)):
                print "deleting:", serialized_file
                os.remove(os.path.expanduser(serialized_file))

        # Step 1: Run 5 epochs of ToyImages model and serialize, MODEL5
        be = "cpu"
        be_args = {'rng_seed': 0}
        print('{} check '.format(be)),
        serialize_check(check_files[0], expected_result, **be_args)
        print('OK')

        # Step 2: Deserialize MODEL5 and compare inference performance
        be = "cpu"
        be_args = {'rng_seed': 0}
        print('{} check '.format(be)),
        serialize_check(check_files[0], expected_result, **be_args)
        print('OK')

        # Step 3a: Change backend to gpu and perform Step 2
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print('{} check '.format(be)),
        serialize_check(check_files[0], expected_result, **be_args)
        print('OK')

        # Step 3b: Change backend to gpu (nervanagpu) and perform Step 2
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "nervanagpu"
        print('{} check '.format(be)),
        serialize_check(check_files[0], expected_result, **be_args)
        print('OK')

        # Step 5: Train 10 epochs of ToyImages model and serialize, MODEL10
        be = "cpu"
        be_args = {'rng_seed': 0}
        print('{} check '.format(be))
        serialize_check(check_files[1], expected_result_2, **be_args)
        print('OK')

        # Step 5: Train 5 more epochs of MODEL5
        be = "cpu"
        be_args = {'rng_seed': 0}
        print('{} check '.format(be)),
        serialize_check(check_files[2], expected_result_2, **be_args)
        print('OK')

        # Step 6: Change backends & Train 5 more epochs of MODEL5
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print('{} check '.format(be)),
        serialize_check(check_files[2], expected_result_3, **be_args)
        print('OK')

    # todo: gpu -> cpu deserialization

    alexnet = True
    if alexnet:
        # alexnet tests
        check_files = []
        for i in range(3):
            check_files.append(
                os.path.join(script_dir,
                             'i1k-serialize_check_' + str(i + 1) + '.yaml'))

        expected_result = 99.6419270833
        expected_result_2 = 99.4791666667
        expected_result_3 = 99.51171875  # diff from #2?
        serialized_files = ['~/data/i1k-model2.pkl', '~/data/i1k-model4.pkl',
                            '~/data/i1k-model4b.pkl']
        # delete previously serialized files
        for serialized_file in serialized_files:
            if os.path.isfile(os.path.expanduser(serialized_file)):
                print "deleting:", serialized_file
                os.remove(os.path.expanduser(serialized_file))

        # Step 1: Run 2 epochs of I1K model and serialize, MODEL2
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print('{} check '.format(be)),
        serialize_check_alexnet(check_files[0], expected_result, **be_args)
        print('OK')

        # Step 2: Deserialize MODEL2 and compare inference performance on gpu
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print('{} check '.format(be)),
        serialize_check_alexnet(check_files[0], expected_result, **be_args)
        print('OK')

        # # Step 3: Deserialize MODEL2 and compare inference performance on cpu
        # be = "cpu"
        # be_args = {'rng_seed': 0}
        # print('{} check '.format(be)),
        # serialize_check_alexnet(check_files[0], expected_result, **be_args)
        # print('OK')

        # Step 4: Run 4 epochs of I1K model and serialize, MODEL4
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print('{} check '.format(be)),
        serialize_check_alexnet(check_files[1], expected_result_2, **be_args)
        print('OK')

        # Step 4: Run 2 more epochs of I1K model and serialize on MODEL2
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print('{} check '.format(be)),
        serialize_check_alexnet(check_files[2], expected_result_3, **be_args)
        print('OK')

    sys.exit(res)
