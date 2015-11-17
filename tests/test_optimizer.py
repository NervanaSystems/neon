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

'''
Test of the optimizers
'''

import numpy as np
import copy

from neon import NervanaObject
from neon.backends import gen_backend
from neon.optimizers import GradientDescentMomentum, RMSProp, Adadelta, Adam, Adagrad
from neon.optimizers import MultiOptimizer
from neon.layers import Conv, Affine, LSTM, GRU
from neon.initializers import Gaussian, Constant
from neon.transforms import Rectlin, Logistic, Tanh


class DummyLayer(object):

    def __init__(self, p):
        self.p = p[0]

    def get_params(self):
        return self.p


def compare_tensors(func, param_list, param2, tol=0., epoch=1):
    func.optimize([DummyLayer(param_list)], epoch=epoch)
    (param, grad), states = param_list[0]
    cond = np.sum(np.abs(param.get() - param2) <= tol)
    assert cond == np.prod(param2.shape)


def wrap(x):
    be = NervanaObject.be
    dtypeu = np.float32
    return be.array(dtypeu(x))


def test_gdm(backend_default):
    lrate, mom, wdecay = 0.1, 0.9, 0.005
    gdm = GradientDescentMomentum(
        learning_rate=lrate, momentum_coef=mom, wdecay=wdecay)
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128)]
    velocity = states[0]
    param2[:] = param2 + velocity * mom - grad2 * lrate - wdecay * lrate * param
    param_list = [((wrap(param), wrap(grad)), [wrap(states[0])])]
    compare_tensors(gdm, param_list, param2, tol=1e-7)


def test_rmsprop(backend_default):
    rms = RMSProp()
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128)]
    state = states[0]
    decay = rms.decay_rate
    denom = np.sqrt(decay * state + np.square(grad2) * (1.0 - decay) + rms.epsilon) + rms.epsilon
    param2[:] -= grad2 * rms.learning_rate / denom
    param_list = [((wrap(param), wrap(grad)), [wrap(states[0])])]
    compare_tensors(rms, param_list, param2, tol=1e-7)


def test_adadelta(backend_default):
    ada = Adadelta()
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128),
              0.01 * np.random.rand(200, 128),
              0.01 * np.random.rand(200, 128)]
    states2 = [copy.deepcopy(states[0]),
               copy.deepcopy(states[1]),
               copy.deepcopy(states[2])]
    decay = ada.decay
    states2[0][:] = states2[0] * decay + (1. - decay) * grad2 * grad2
    states2[2][:] = np.sqrt(
        (states2[1] + ada.epsilon) / (states2[0] + ada.epsilon)) * grad2
    states2[1][:] = states2[1] * decay + (1. - decay) * states2[2] * states2[2]
    param2[:] -= states2[2]
    param_list = [
        ((wrap(param), wrap(grad)), [wrap(states[0]), wrap(states[1]), wrap(states[2])])]
    compare_tensors(ada, param_list, param2, tol=1e-7)


def test_adagrad(backend_default):
    ada = Adagrad()
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128)]
    states2 = [copy.deepcopy(states[0])]
    states2[0][:] = states2[0] + np.square(grad2)
    denom = np.sqrt(states2[0] + ada.epsilon)
    param2[:] -= grad2 * ada.learning_rate / denom
    param_list = [
        ((wrap(param), wrap(grad)), [wrap(states[0])])]
    compare_tensors(ada, param_list, param2, tol=1e-7)


def test_adam(backend_default):
    adam = Adam()
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128),
              0.01 * np.random.rand(200, 128)]
    states2 = [copy.deepcopy(states[0]),
               copy.deepcopy(states[1])]
    epoch = 1
    t = epoch + 1
    l = adam.learning_rate * np.sqrt(1 - adam.beta_2 ** t) / (1 - adam.beta_1 ** t)
    m, v = states2
    m[:] = m * adam.beta_1 + (1. - adam.beta_1) * grad2
    v[:] = v * adam.beta_2 + (1. - adam.beta_2) * grad2 * grad2
    param2[:] -= l * m / (np.sqrt(v) + adam.epsilon)
    param_list = [
        ((wrap(param), wrap(grad)), [wrap(states[0]), wrap(states[1])])]
    compare_tensors(adam, param_list, param2, tol=1e-7, epoch=epoch)


def test_multi_optimizer(backend_default):
    opt_gdm = GradientDescentMomentum(
        learning_rate=0.001, momentum_coef=0.9, wdecay=0.005)
    opt_ada = Adadelta()
    opt_adam = Adam()
    opt_rms = RMSProp()
    opt_rms_1 = RMSProp(gradient_clip_value=5)
    init_one = Gaussian(scale=0.01)

    l1 = Conv((11, 11, 64), strides=4, padding=3,
              init=init_one, bias=Constant(0), activation=Rectlin())
    l2 = Affine(nout=4096, init=init_one,
                bias=Constant(1), activation=Rectlin())
    l3 = LSTM(output_size=1000, init=init_one, activation=Logistic(), gate_activation=Tanh())
    l4 = GRU(output_size=100, init=init_one, activation=Logistic(), gate_activation=Tanh())
    layers = [l1, l2, l3, l4]
    layer_list = []
    for layer in layers:
        if isinstance(layer, list):
            layer_list.extend(layer)
        else:
            layer_list.append(layer)

    opt = MultiOptimizer({'default': opt_gdm,
                          'Bias': opt_ada,
                          'Convolution': opt_adam,
                          'Linear': opt_rms,
                          'LSTM': opt_rms_1,
                          'GRU': opt_rms_1})

    map_list = opt.map_optimizers(layer_list)
    assert map_list[opt_adam][0].__class__.__name__ == 'Convolution'
    assert map_list[opt_ada][0].__class__.__name__ == 'Bias'
    assert map_list[opt_rms][0].__class__.__name__ == 'Linear'
    assert map_list[opt_gdm][0].__class__.__name__ == 'Activation'
    assert map_list[opt_rms_1][0].__class__.__name__ == 'LSTM'
    assert map_list[opt_rms_1][1].__class__.__name__ == 'GRU'

if __name__ == '__main__':
    be = gen_backend(backend='gpu', batch_size=50)
    test_multi_optimizer(be)
