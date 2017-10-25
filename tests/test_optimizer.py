# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
from neon.optimizers import (GradientDescentMomentum, RMSProp, Adadelta, Adam, Adagrad,
                             ShiftAdaMax)
from neon.optimizers import MultiOptimizer
from neon.layers import Conv, Affine, LSTM, GRU
from neon.layers.layer import ParameterLayer
from neon.initializers import Gaussian, Constant
from neon.transforms import Rectlin, Logistic, Tanh


class DummyLayer(object):

    def __init__(self, p):
        self.p = p[0]

    def get_params(self):
        return self.p


def compare_tensors(func, param_list, param2, tol=0., epoch=1):
    """
    Compare parameters updated by optimizer againt those manually computed, using a dummy layer.

    Arguments:
        func (instance of Optimizer): optimizer
        param_list (list): list of parameters
        param2 (tensor): manually computed parameters on host
        tol (float): tolerance
        epoch (int): dummy epoch
    """
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


def test_gdm_wclip(backend_default):
    lrate, mom, wdecay, wclip = 0.1, 0.9, 0.005, 0.5
    gdm = GradientDescentMomentum(
        learning_rate=lrate, momentum_coef=mom, wdecay=wdecay, param_clip_value=wclip)
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128)]
    velocity = states[0]
    param2[:] = param2 + velocity * mom - grad2 * lrate - wdecay * lrate * param
    np.clip(param2, -wclip, wclip, param2)
    param_list = [((wrap(param), wrap(grad)), [wrap(states[0])])]
    compare_tensors(gdm, param_list, param2, tol=1e-7)


def test_gdm_nesterov(backend_default):
    lrate, mom, wdecay = 0.1, 0.9, 0.005
    gdm = GradientDescentMomentum(learning_rate=lrate, momentum_coef=mom,
                                  wdecay=wdecay, nesterov=True)
    data_shape = (200, 128)

    # params to be updated using GDM
    np_param = np.random.rand(*data_shape)
    param = wrap(np_param)

    # Optimizer states
    velocity = 0.01 * np.random.rand(*data_shape)
    states = [wrap(velocity)]

    # Check a few iterations in a row
    for ii in range(20):
        # Choose a gradient
        np_grad = 0.01 * np.random.rand(*data_shape)
        grad = wrap(np_grad)

        # Update manually
        np_grad = np_grad / data_shape[1]
        velocity[:] = mom * velocity - lrate * (np_grad + wdecay * np_param)
        np_param[:] = np_param + mom * velocity - lrate * (np_grad + wdecay * np_param)
        param_list = [((param, grad),
                       states)]
        compare_tensors(gdm, param_list, np_param, tol=1e-6)


def test_gdm_nesterov_wclip(backend_default):
    lrate, mom, wdecay, wclip = 0.1, 0.9, 0.005, 0.5
    gdm = GradientDescentMomentum(learning_rate=lrate, momentum_coef=mom,
                                  wdecay=wdecay, nesterov=True,
                                  param_clip_value=wclip)
    data_shape = (200, 128)

    # params to be updated using GDM
    np_param = np.random.rand(*data_shape)
    param = wrap(np_param)

    # Optimizer states
    velocity = 0.01 * np.random.rand(*data_shape)
    states = [wrap(velocity)]

    # Check a few iterations in a row
    for ii in range(20):
        # Choose a gradient
        np_grad = 0.01 * np.random.rand(*data_shape)
        grad = wrap(np_grad)

        # Update manually
        np_grad = np_grad / data_shape[1]
        velocity[:] = mom * velocity - lrate * (np_grad + wdecay * np_param)
        np_param[:] = np_param + mom * velocity - lrate * (np_grad + wdecay * np_param)
        np.clip(np_param, -wclip, wclip, np_param)
        param_list = [((param, grad),
                       states)]
        compare_tensors(gdm, param_list, np_param, tol=1e-6)


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
    param2[:] -= grad2 * float(rms.learning_rate) / denom
    param_list = [((wrap(param), wrap(grad)), [wrap(states[0])])]
    compare_tensors(rms, param_list, param2, tol=1e-7)


def test_rmsprop_wclip(backend_default):
    wclip = 0.5
    rms = RMSProp(param_clip_value=wclip)
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128)]
    state = states[0]
    decay = rms.decay_rate
    denom = np.sqrt(decay * state + np.square(grad2) * (1.0 - decay) + rms.epsilon) + rms.epsilon
    param2[:] -= grad2 * float(rms.learning_rate) / denom
    np.clip(param2, -wclip, wclip, param2)
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
        (states2[1] + float(ada.epsilon)) / (states2[0] + ada.epsilon)) * grad2
    states2[1][:] = states2[1] * decay + (1. - decay) * states2[2] * states2[2]
    param2[:] -= states2[2]
    param_list = [
        ((wrap(param), wrap(grad)), [wrap(states[0]), wrap(states[1]), wrap(states[2])])]
    compare_tensors(ada, param_list, param2, tol=1e-7)


def test_adadelta_wclip(backend_default):
    wclip = 0.5
    ada = Adadelta(param_clip_value=wclip)
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
        (states2[1] + float(ada.epsilon)) / (states2[0] + ada.epsilon)) * grad2
    states2[1][:] = states2[1] * decay + (1. - decay) * states2[2] * states2[2]
    param2[:] -= states2[2]
    np.clip(param2, -wclip, wclip, param2)
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
    param2[:] -= grad2 * float(ada.learning_rate) / denom
    param_list = [
        ((wrap(param), wrap(grad)), [wrap(states[0])])]
    compare_tensors(ada, param_list, param2, tol=1e-7)


def test_adagrad_wclip(backend_default):
    wclip = 0.5
    ada = Adagrad(param_clip_value=wclip)
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128)]
    states2 = [copy.deepcopy(states[0])]
    states2[0][:] = states2[0] + np.square(grad2)
    denom = np.sqrt(states2[0] + ada.epsilon)
    param2[:] -= grad2 * float(ada.learning_rate) / denom
    np.clip(param2, -wclip, wclip, param2)
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
    t = 1
    l = adam.learning_rate * np.sqrt(1. - adam.beta_2 ** t) / (1. - adam.beta_1 ** t)
    m, v = states2
    m[:] = m * adam.beta_1 + (1. - adam.beta_1) * grad2
    v[:] = v * adam.beta_2 + (1. - adam.beta_2) * grad2 * grad2
    param2[:] -= l * m / (np.sqrt(v) + adam.epsilon)
    param_list = [
        ((wrap(param), wrap(grad)), [wrap(states[0]), wrap(states[1])])]
    compare_tensors(adam, param_list, param2, tol=1e-7, epoch=epoch)


def test_adam_wclip(backend_default):
    wclip = 0.5
    adam = Adam(param_clip_value=wclip)
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128),
              0.01 * np.random.rand(200, 128)]
    states2 = [copy.deepcopy(states[0]),
               copy.deepcopy(states[1])]
    epoch = 1
    t = 1
    l = adam.learning_rate * np.sqrt(1. - adam.beta_2 ** t) / (1. - adam.beta_1 ** t)
    m, v = states2
    m[:] = m * adam.beta_1 + (1. - adam.beta_1) * grad2
    v[:] = v * adam.beta_2 + (1. - adam.beta_2) * grad2 * grad2
    param2[:] -= l * m / (np.sqrt(v) + adam.epsilon)
    np.clip(param2, -wclip, wclip, param2)
    param_list = [
        ((wrap(param), wrap(grad)), [wrap(states[0]), wrap(states[1])])]
    compare_tensors(adam, param_list, param2, tol=1e-7, epoch=epoch)


def test_shift_adamax(backend_default):
    shiftadamax = ShiftAdaMax()
    param = np.random.rand(200, 128)
    param2 = copy.deepcopy(param)
    grad = 0.01 * np.random.rand(200, 128)
    grad2 = grad / 128.
    states = [0.01 * np.random.rand(200, 128),
              0.01 * np.random.rand(200, 128),
              0.01 * np.random.rand(200, 128)]
    states2 = [copy.deepcopy(states[0]),
               copy.deepcopy(states[1])]
    epoch = 1
    t = epoch + 1
    l = shiftadamax.learning_rate / (1 - shiftadamax.beta_1 ** t)
    m, v = states2
    m[:] = m * shiftadamax.beta_1 + (1. - shiftadamax.beta_1) * grad2
    v[:] = np.maximum(v * shiftadamax.beta_2, np.absolute(grad2))
    inv_v = np.random.rand(200, 128)
    inv_v[:] = 1.0 / (v + shiftadamax.epsilon)

    def safelog(left):
        return np.log(np.maximum(left, np.exp(-50.)))

    def shift(ary, shift_ary):
        exp = np.rint(safelog(np.absolute(shift_ary))/np.log(2))
        ap2 = np.multiply(np.sign(shift_ary), np.exp2(exp))
        return np.multiply(ary, ap2)

    param2[:] = param2 - shift(shift(m, inv_v), l)
    np.clip(param2, -1, 1, param2)
    param_list = [
        ((wrap(param), wrap(grad)), [wrap(states[0]), wrap(states[1]), wrap(states[2])])]
    compare_tensors(shiftadamax, param_list, param2, tol=1e-4, epoch=epoch)


def test_multi_optimizer(backend_default_mkl):
    """
    A test for MultiOptimizer.
    """
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
    for l in layer_list:
        l.configure(in_obj=(16, 28, 28))
        l.allocate()
    # separate layer_list into two, the last two recurrent layers and the rest
    layer_list1, layer_list2 = layer_list[:-2], layer_list[-2:]
    opt = MultiOptimizer({'default': opt_gdm,
                          'Bias': opt_ada,
                          'Convolution': opt_adam,
                          'Convolution_bias': opt_adam,
                          'Linear': opt_rms,
                          'LSTM': opt_rms_1,
                          'GRU': opt_rms_1})
    layers_to_optimize1 = [l for l in layer_list1 if isinstance(l, ParameterLayer)]
    layers_to_optimize2 = [l for l in layer_list2 if isinstance(l, ParameterLayer)]
    opt.optimize(layers_to_optimize1, 0)
    assert opt.map_list[opt_adam][0].__class__.__name__ is 'Convolution_bias'
    assert opt.map_list[opt_rms][0].__class__.__name__ == 'Linear'
    opt.optimize(layers_to_optimize2, 0)
    assert opt.map_list[opt_rms_1][0].__class__.__name__ == 'LSTM'
    assert opt.map_list[opt_rms_1][1].__class__.__name__ == 'GRU'


if __name__ == '__main__':
    be = gen_backend(backend='gpu', batch_size=128)
    # test_multi_optimizer(be)
    test_gdm_nesterov(be)
