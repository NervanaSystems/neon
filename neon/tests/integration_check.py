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
Integration check
Run several example networks and make sure their accuracy is within a tolerance
"""
import logging
import os
import sys
from neon.backends import gen_backend
from neon.util.persist import deserialize


def integration_check(conf_file, result, tol, **be_args):
    experiment = deserialize(conf_file)
    backend = gen_backend(model=experiment.model, **be_args)
    experiment.initialize(backend)
    res = experiment.run()
    print(float(res['train']['MisclassPercentage_TOP_1'])), result,
    assert abs(float(res['train']['MisclassPercentage_TOP_1']) - result) < tol


if __name__ == '__main__':
    # setup an initial console logger (may be overridden in config)
    logging.basicConfig(level=40)  # ERROR or higher
    res = 0
    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_dir = os.path.join(script_dir, 'tests_yamls')

    # Test 1: i1k-alexnet-fp16 (nervanagpu only)
    check_file = os.path.join(script_dir, 'i1k-alexnet-fp16.yaml')
    expected_result = 99.21875
    tol = .3
    be_args = {'rng_seed': 0}
    gpu = "nervanagpu"
    be_args["gpu"] = gpu
    print(gpu + ' check i1k-alexnet-fp16 '),
    integration_check(check_file, expected_result, tol, **be_args)
    print('OK')

    # Test 2: i1k-alexnet-fp32 (nervanagpu, cudanet)
    check_file = os.path.join(script_dir, 'i1k-alexnet-fp32.yaml')
    expected_result = 99.
    tol = .3
    be_args = {'rng_seed': 0}
    for gpu in ["nervanagpu", "cudanet"]:
        be_args["gpu"] = gpu
        print(gpu + ' check i1k-alexnet-fp32 '),
        integration_check(check_file, expected_result, tol, **be_args)
        print('OK')

    # Test 3: mnist cnn (cpu, nervanagpu, cudanet)
    check_file = os.path.join(script_dir, 'mnist-convnet.yaml')
    expected_result = 2.8
    tol = .1
    for be in ["cpu", "gpu"]:
        be_args = {'rng_seed': 0}
        if be == "gpu":
            for gpu in ["nervanagpu", "cudanet"]:
                be_args[be] = gpu
                print(gpu + ' check mnist-convnet '),
                integration_check(check_file, expected_result, tol, **be_args)
                print('OK')
        else:
            print('cpu check mnist-convnet '),
            integration_check(check_file, expected_result, tol, **be_args)
            print('OK')

    # Test 4: cifar cnn
    check_file = os.path.join(script_dir, 'cifar10-convnet.yaml')
    expected_result = 61.
    tol = 1.2
    for be in ["cpu", "gpu"]:
        be_args = {'rng_seed': 0}
        if be == "gpu":
            for gpu in ["nervanagpu", "cudanet"]:
                be_args[be] = gpu
                print(gpu + ' check cifar-convnet '),
                integration_check(check_file, expected_result, tol, **be_args)
                print('OK')
        else:
            print('cpu check cifar-convnet '),
            integration_check(check_file, expected_result, tol, **be_args)
            print('OK')

    # Test 5: mnist mlp
    check_file = os.path.join(script_dir, 'mnist-mlp.yaml')
    expected_result = 7.35394
    tol = .01
    for be in ["cpu", "gpu"]:
        be_args = {'rng_seed': 0}
        if be == "gpu":
            for gpu in ["nervanagpu", "cudanet"]:
                be_args[be] = gpu
                print(gpu + ' check mnist-mlp '),
                integration_check(check_file, expected_result, tol, **be_args)
                print('OK')
        else:
            print('cpu check mnist-mlp '),
            integration_check(check_file, expected_result, tol, **be_args)
            print('OK')

    # Test 6: cifar mlp
    check_file = os.path.join(script_dir, 'cifar10-mlp.yaml')
    expected_result = 69.
    tol = 2.
    for be in ["cpu", "gpu"]:
        be_args = {'rng_seed': 0}
        if be == "gpu":
            for gpu in ["nervanagpu", "cudanet"]:
                be_args[be] = gpu
                print(gpu + ' check cifar-mlp '),
                integration_check(check_file, expected_result, tol, **be_args)
                print('OK')
        else:
            print('cpu check cifar-mlp '),
            integration_check(check_file, expected_result, tol, **be_args)
            print('OK')

    # Test 7: rnn, cpu only for now
    check_file = os.path.join(script_dir, 'moby-rnn.yaml')
    expected_result = 0.18162
    tol = .01
    for be in ["cpu"]:  # todo: "gpu" support
        be_args = {'rng_seed': 0}
        if be == "gpu":
            for gpu in ["nervanagpu", "cudanet"]:
                be_args[be] = gpu
                print(gpu + ' check moby-rnn '),
                integration_check(check_file, expected_result, tol, **be_args)
                print('OK')
        else:
            print('cpu check moby-rnn '),
            integration_check(check_file, expected_result, tol, **be_args)
            print('OK')

    # Test 8: lstm
    check_file = os.path.join(script_dir, 'moby-lstm.yaml')
    expected_result = 0.16
    tol = .01
    for be in ["cpu"]:  # todo: "gpu" support
        be_args = {'rng_seed': 0}
        if be == "gpu":
            for gpu in ["nervanagpu", "cudanet"]:
                be_args[be] = gpu
                print(gpu + ' check moby-lstm '),
                integration_check(check_file, expected_result, tol, **be_args)
                print('OK')
        else:
            print('cpu check moby-lstm '),
            integration_check(check_file, expected_result, tol, **be_args)
            print('OK')

    sys.exit(res)
