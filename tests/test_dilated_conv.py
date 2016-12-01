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

import numpy as np
from neon.backends import gen_backend
from neon.layers import Conv, Affine, GeneralizedCost
from neon.models import Model
from neon.initializers.initializer import Gaussian
from neon.transforms import CrossEntropyBinary
from neon.optimizers import GradientDescentMomentum


def fprop(model, inputs):
    layers = model.layers
    for l in layers._layers:
        inputs = l.fprop(inputs)
    # Return outputs from the last layer.
    return layers._layers[-1].outputs


def bprop(model, delta):
    layers = model.layers
    for l in reversed(layers._layers):
        delta = l.bprop(delta)
    # Return weights from the first layer.
    return layers._layers[0].W


def dilate(weights, K):
    # Dilate filters with a dilation factor of 2.
    # A shape of (K, 2, 2, K) is assumed for the input filters.
    new_weights = np.zeros((K*9, K), dtype=np.float32)
    dst = new_weights.reshape((K, 3, 3, K))
    src = weights.reshape((K, 2, 2, K))
    dst[:, 0, 0] = src[:, 0, 0]
    dst[:, 0, 2] = src[:, 0, 1]
    dst[:, 2, 0] = src[:, 1, 0]
    dst[:, 2, 2] = src[:, 1, 1]
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


def load(weights, model, K):
    index = 0
    layers = model.layers
    for layer in layers._layers:
        if hasattr(layer, 'W'):
            if layer.W.shape == weights[index].shape:
                layer.W[:] = weights[index]
            else:
                layer.W[:] = dilate(weights[index], K)
            index += 1


def run(be, fake_dilation):
    K = 8
    be.rng = be.gen_rng(be.rng_seed)
    train_shape = (1, 7, 7)

    inp = be.array(be.rng.randn(np.prod(train_shape), be.bsz))
    init = Gaussian()

    dil_dict = dict(dil_d=2, dil_h=2, dil_w=2)
    layers = [Conv((5, 5, K), init=init),
              Conv((2, 2, K), init=init, dilation=dil_dict),
              Affine(nout=1, init=init)]
    model = Model(layers=layers)
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())
    model.initialize(train_shape, cost)

    if fake_dilation:
        # Perform regular convolution with an expanded filter.
        weights = save(model)
        new_layers = layers
        # Replace the middle layer.
        new_layers[1] = Conv((3, 3, K), init=init)
        model = Model(layers=new_layers)
        cost = GeneralizedCost(costfunc=CrossEntropyBinary())
        model.initialize(train_shape, cost)
        load(weights, model, K)

    model.optimizer = GradientDescentMomentum(learning_rate=0.01,
                                              momentum_coef=0.9)
    outputs = fprop(model, inp)
    weights = bprop(model, outputs)
    model.optimizer.optimize(model.layers_to_optimize, epoch=0)
    return outputs.get(), weights.get()


def test_dilated_conv(backend_default):
    be = backend_default
    o1, w1 = run(be, fake_dilation=False)
    o2, w2 = run(be, fake_dilation=True)
    # Verify that the results of faked dilation match those of actual dilation.
    assert np.allclose(o1, o2)
    assert np.allclose(w1, w2)


if __name__ == '__main__':
    be = gen_backend(backend='cpu', rng_seed=0, batch_size=128)
    test_dilated_conv(be)
    print('OK')
