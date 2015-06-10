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

For AlexNet if we want to remove more randomness have to manually set:
center=True, flip=False,
#center=self.ds.predict, flip=True,
in line 53 of imageset.py
"""
import logging
import os
import sys
from neon.backends import gen_backend
from neon.util.persist import deserialize


def serialize_check(conf_file, result, tol, res_string, **be_args):
    experiment = deserialize(conf_file)
    backend = gen_backend(model=experiment.model, **be_args)
    experiment.initialize(backend)
    res = experiment.run()
    print("{:f}, {:f}".format(float(res[res_string]['MisclassPercentage_TOP_1']
                                    ), result))
    assert abs(
        float(res[res_string]['MisclassPercentage_TOP_1']) - result) < tol

if __name__ == '__main__':
    try:
        import nervanagpu  # noqa
    except:
        raise RuntimeError("Can't find nervanagpu")

    try:
        import cudanet  # noqa
    except:
        raise RuntimeError("Can't find cudanet")

    # setup an initial console logger (may be overridden in config)
    logging.basicConfig(level=40)  # ERROR or higher
    res = 0
    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_dir = os.path.join(script_dir, 'tests_yamls')
    check_files = []

    toy = True
    if toy:
        for i in range(4):
            check_files.append(
                os.path.join(script_dir,
                             'toy-serialize_check_' + str(i + 1) + '.yaml'))

        expected_result = 28.90625
        expected_result_2 = 16.
        expected_result_3 = 16.40625
        serialized_files = ['~/data/model5.prm', '~/data/model10.prm',
                            '~/data/model10b.prm']
        # delete previously serialized files
        for serialized_file in serialized_files:
            if os.path.isfile(os.path.expanduser(serialized_file)):
                print("deleting: {0}".format(serialized_file))
                os.remove(os.path.expanduser(serialized_file))

        res_string = 'test'

        tol = .01
        # Step 1: Run 5 epochs of ToyImages model and serialize, MODEL5
        be = "cpu"
        be_args = {'rng_seed': 0}
        print('cpu check: train 5 epochs of ToyImages on CPU -> MODEL5 '),
        serialize_check(
            check_files[0], expected_result, tol, res_string, **be_args)
        print('OK')

        # Step 2: Deserialize MODEL5 and compare inference performance
        be = "cpu"
        be_args = {'rng_seed': 0}
        print('cpu check: load MODEL5 and calc inference perf '),
        serialize_check(
            check_files[0], expected_result, tol, res_string, **be_args)
        print('OK')

        # Step 3a: Change backend to gpu and perform Step 2
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print(
            be_args[
                be] + ' check: load MODEL5 and calc inference perf on GPU '),
        serialize_check(
            check_files[0], expected_result, tol, res_string, **be_args)
        print('OK')

        # Step 3b: Change backend to gpu (nervanagpu) and perform Step 2
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "nervanagpu"
        print(
            be_args[
                be] + ' check: load MODEL5 and calc inference perf on GPU '),
        serialize_check(
            check_files[0], expected_result, tol, res_string, **be_args)
        print('OK')

        tol = 1.2
        # Step 4: Train 10 epochs of ToyImages model and serialize, MODEL10
        be = "cpu"
        be_args = {'rng_seed': 0}
        print('cpu check: train 10 epochs of ToyImages on CPU -> MODEL10 '),
        serialize_check(
            check_files[1], expected_result_2, tol, res_string, **be_args)
        print('OK')

        # Step 5: Train 5 more epochs of MODEL5
        be = "cpu"
        be_args = {'rng_seed': 0}
        print(
            'cpu check: load MODEL5 & train 5 more epochs '),
        serialize_check(
            check_files[2], expected_result_2, tol, res_string, **be_args)
        print('OK')

        tol = .01
        # Step 6: Change backends & Train 5 more epochs of MODEL5
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print(
            be_args[
                be] + ' check: load MODEL5 & train 5 more epochs '),
        serialize_check(
            check_files[2], expected_result_3, tol, res_string, **be_args)
        print('OK')

        # Step 7: Run 5 epochs of ToyImages model and serialize, MODEL5b
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "nervanagpu"
        print(
            be_args[
                be] + ' check: train 5 epochs of ToyImages -> MODEL5b '),
        serialize_check(
            check_files[3], expected_result, tol, res_string, **be_args)
        print('OK')

        # Step 7: Deserialize MODEL5 on CPU and compare inference performance
        be = "cpu"
        be_args = {'rng_seed': 0}
        print('cpu check: load MODEL5b and calc inference perf on CPU '),
        serialize_check(
            check_files[3], expected_result, tol, res_string, **be_args)
        print('OK')

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
                print("deleting: {0}".format(serialized_file))
                os.remove(os.path.expanduser(serialized_file))

        res_string = 'validation'

        tol = .3
        # Step 1: Run 2 epochs of I1K model and serialize, MODEL2
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print(
            be_args[
                be] + ' check: train 2 epochs of ImageNet on GPU -> MODEL2'),
        serialize_check(
            check_files[0], expected_result, tol, res_string, **be_args)
        print('OK')

        # Step 2: Deserialize MODEL2 and compare inference performance on gpu
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print(
            be_args[
                be] + ' check: load MODEL2 on GPU and calc inference perf '),
        serialize_check(
            check_files[0], expected_result, tol, res_string, **be_args)
        print('OK')

        # Step 3: Run 4 epochs of I1K model and serialize, MODEL4
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print(
            be_args[
                be] + ' check: train 4 epochs of ImageNet on GPU '),
        serialize_check(
            check_files[1], expected_result_2, tol, res_string, **be_args)
        print('OK')

        # Step 4: Run 2 more epochs of I1K model and serialize on MODEL2
        be = "gpu"
        be_args = {'rng_seed': 0}
        be_args[be] = "cudanet"
        print(
            be_args[
                be] + ' check: load MODEL2 & train for 2 more epochs '),
        serialize_check(
            check_files[2], expected_result_3, tol, res_string, **be_args)
        print('OK')

    sys.exit(res)
