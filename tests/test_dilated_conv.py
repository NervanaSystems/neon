# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
Dilated convolution layer tests
"""
from __future__ import print_function
import itertools as itt
import numpy as np
import os
import subprocess as subp
from neon.backends import gen_backend
from neon.layers import Conv, Affine, GeneralizedCost
from neon.models import Model
from neon.initializers.initializer import Gaussian
from neon.transforms import CrossEntropyBinary
from neon.optimizers import GradientDescentMomentum
from utils import allclose_with_out
from neon import NervanaObject
try:
    from neon.backends.nervanagpu import NervanaGPU
except:
    # stub out the class
    class NervanaGPU(object):
        pass


def pytest_generate_tests(metafunc):
    """
    Build a list of test arguments.

    """
    fsz = [3, 4, 7]
    dil = [2, 3]
    strides = [1, 2, 3, 6]

    if 'fargs_tests' in metafunc.fixturenames:
        fargs = itt.product(fsz, dil, strides)
        metafunc.parametrize("fargs_tests", fargs)


def fprop(model, inputs):
    layers = model.layers
    for l in layers._layers:
        l.be.convert_data(inputs, l.get_is_mklop())
        inputs = l.fprop(inputs)
    # Return outputs from the last layer.
    return layers._layers[-1].outputs


def bprop(model, delta):
    layers = model.layers
    for l in reversed(layers._layers):
        l.be.convert_data(delta, l.get_is_mklop())
        delta = l.bprop(delta)
    # Return weights from the first layer.
    return layers._layers[0].W


def dilated_fsz(fsz, dil):
    return ((fsz - 1) * dil) + 1


def dilate(weights, K, fsz, dil):
    # Dilate filters with a dilation factor of dil.
    # A shape of (K, fsz, fsz, K) is assumed for the input filters.
    new_fsz = dilated_fsz(fsz, dil)
    new_weights = np.zeros((K*new_fsz*new_fsz, K), dtype=np.float32)

    dst = new_weights.reshape((K, new_fsz, new_fsz, K))
    src = weights.reshape((K, fsz, fsz, K))

    for x in range(fsz):
        for y in range(fsz):
            dst[:, y * dil, x * dil] = src[:, y, x]

    return new_weights


def save(model):
    weights = {}
    index = 0
    layers = model.layers
    for layer in layers._layers:
        if hasattr(layer, 'W'):
            weights[index] = layer.W.get()
            index += 1
    return weights


def load(weights, model, K, fsz, dil):
    index = 0
    layers = model.layers
    for layer in layers._layers:
        if hasattr(layer, 'W'):
            if layer.W.shape == weights[index].shape:
                layer.W[:] = weights[index]
            else:
                layer.W[:] = dilate(weights[index], K, fsz, dil)
            index += 1


def out_shape(W, S, stride, dil, pad):
    Q = W - 4
    Q = ((Q + (2 * pad)) - (((S - 1) * dil) + 1)) // stride
    return Q


def run(be, fake_dilation, fsz, stride, pad, dilation):
    K = 8
    strides = stride
    padding = pad
    be.rng = be.gen_rng(be.rng_seed)

    in_shape = 16
    while out_shape(in_shape, fsz, stride, dilation, pad) < 3:
        in_shape *= 2
    train_shape = (1, in_shape, in_shape)

    inp = be.array(be.rng.randn(np.prod(train_shape), be.bsz))
    init = Gaussian()

    layers = [Conv((5, 5, K), init=init),
              Conv((fsz, fsz, K), strides=strides, padding=padding, init=init,
                   dilation=dict(dil_d=1, dil_h=dilation, dil_w=dilation)),
              Conv((3, 3, K), init=init),
              Affine(nout=1, init=init)]
    model = Model(layers=layers)
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())
    model.initialize(train_shape, cost)

    if fake_dilation:
        # Perform regular convolution with an expanded filter.
        weights = save(model)
        new_layers = layers
        # Replace the middle layers.
        new_fsz = dilated_fsz(fsz, dilation)
        new_layers[1] = Conv((new_fsz, new_fsz, K), strides=strides, padding=padding, init=init)
        model = Model(layers=new_layers)
        cost = GeneralizedCost(costfunc=CrossEntropyBinary())
        model.initialize(train_shape, cost)
        load(weights, model, K, fsz, dilation)

    print(model)
    model.optimizer = GradientDescentMomentum(learning_rate=0.01,
                                              momentum_coef=0.9)
    outputs = fprop(model, inp)
    weights = bprop(model, outputs)
    model.optimizer.optimize(model.layers_to_optimize, epoch=0)
    return outputs.get(), weights.get()


def test_dilated_conv(backend_default, fargs_tests):

    fsz = fargs_tests[0]
    dil = fargs_tests[1]
    stride = fargs_tests[2]
    be = backend_default

    o1, w1 = run(be, False, fsz, stride, 1, dil)
    o2, w2 = run(be, True, fsz, stride, 1, dil)
    # Verify that the results of faked dilation match those of actual dilation.
    assert allclose_with_out(o1, o2, atol=0, rtol=3e-3)
    try:
        assert allclose_with_out(w1, w2, atol=0, rtol=1e-3)
    except:
        # xfail for cpu/mkl backends on KNM/KNL platforms
        if not isinstance(NervanaObject.be, NervanaGPU):
            if os.getenv("PLATFORM"):
                platform = os.getenv("PLATFORM")
            else:
                if os.path.exists("/proc/cpuinfo"):
                    cat_cmd = 'cat /proc/cpuinfo | grep "model name" | tail -1 | cut -f 2 -d \':\' | \
                           cut -f 3 -d \')\' | cut -f 1 -d \'@\' | cut -f 2,3 -d \' \''
                    cpu_model_name = subp.check_output(cat_cmd, shell=True)
                    print('CPU model name = {}'.format(cpu_model_name))
                else:
                    cpu_model_name = "unknown"

            if cpu_model_name == 'CPU E5-2699\n':
                platform = "BDW"
            elif cpu_model_name == 'CPU 7250\n':
                platform = "KNL"
            # temporary identification for KNM model name
            elif cpu_model_name == 'CPU 0000\n':
                platform = "KNM"
            else:
                platform = "unknown"

            print('Test platform = {}'.format(platform))
            assert allclose_with_out(w1, w2, atol=1e-1, rtol=1e-3)


if __name__ == '__main__':
    be_cpu = gen_backend(backend='cpu', rng_seed=0, batch_size=128)
    fargs_tests = [3, 2, 1]
    test_dilated_conv(be_cpu)
    print('OK')

    be_mkl = gen_backend(backend='mkl', rng_seed=0, batch_size=128)
    fargs_tests = [3, 2, 1]
    test_dilated_conv(be_mkl)
    print('OK')
