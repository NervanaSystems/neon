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
Test of the cost functions
'''
import itertools as itt
import numpy as np
from neon import NervanaObject
from neon.backends import gen_backend
from neon.transforms import (CrossEntropyBinary, CrossEntropyMulti, SumSquared,
                             MeanSquared, Misclassification, PrecisionRecall,
                             SmoothL1Loss, SquareHingeLoss)


def pytest_generate_tests(metafunc):

    if 'fargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            shape1_rng = [2, 3, 4, 5]
            shape2_rng = [3, 5, 10, 20]
            mag_rng = [3, 5, 10, 20]
            sigma_rng = [1.0, 3.0, 5.0]
        else:
            shape1_rng = [3]
            shape2_rng = [5]
            mag_rng = [10]
            sigma_rng = [1.0, 3.0]
        fargs = itt.product(shape1_rng, shape2_rng, mag_rng, sigma_rng)
        metafunc.parametrize('fargs', fargs)


def compare_tensors(func, y, t, outputs, deriv=False, tol=0.):
    be = NervanaObject.be
    temp = be.empty(outputs.shape)
    dtypeu = np.float32
    if deriv is True:
        temp[:] = func.bprop(be.array(dtypeu(y)), be.array(dtypeu(t)))
    else:
        # try:
        temp[:] = func(be.array(dtypeu(y)), be.array(dtypeu(t)))
        # except:
        #     import ipdb; ipdb.set_trace()
    cond = np.sum(np.abs(temp.get() - outputs) <= tol)
    assert cond == np.prod(outputs.shape)

"""
    CrossEntropyBinary
"""


def test_cross_entropy_binary(backend_default):
    outputs = np.array([0.5, 0.9, 0.1, 0.0001]).reshape((4, 1))
    targets = np.array([0.5, 0.99, 0.01, 0.2]).reshape((4, 1))
    eps = np.exp(-50)
    expected_log = np.log(np.maximum(outputs, eps))
    expected_mlog = np.log(np.maximum(1 - outputs, eps))
    expected_result = np.sum((-targets * expected_log) - (1 - targets) * expected_mlog,
                             keepdims=True)
    compare_tensors(CrossEntropyBinary(),
                    outputs, targets, expected_result, tol=1e-6)


def test_cross_entropy_binary_limits(backend_default):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001]).reshape((4, 1))
    targets = np.array(([0.5, 0.0, 1.0, 0.2])).reshape((4, 1))
    eps = np.exp(-50)
    expected_log = np.log(np.maximum(outputs, eps))
    expected_mlog = np.log(np.maximum(1 - outputs, eps))
    expected_result = np.sum((-targets * expected_log) - (1 - targets) * expected_mlog,
                             keepdims=True)
    compare_tensors(CrossEntropyBinary(),
                    outputs, targets, expected_result, tol=1e-5)


def test_cross_entropy_binary_derivative(backend_default):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001]).reshape((4, 1))
    targets = np.array(([0.5, 0.0, 1.0, 0.2])).reshape((4, 1))
    # bprop assumes shortcut
    expected_result = ((outputs - targets) / outputs.shape[1])
    compare_tensors(
        CrossEntropyBinary(), outputs, targets, expected_result, deriv=True,
        tol=1e-6)

"""
    CrossEntropyMulti
"""


def test_cross_entropy_multi(backend_default):
    outputs = np.array([0.5, 0.9, 0.1, 0.0001]).reshape((4, 1))
    targets = np.array([0.5, 0.99, 0.01, 0.2]).reshape((4, 1))
    eps = np.exp(-50)
    expected_log = np.log(np.maximum(outputs, eps))
    expected_result = np.sum(-targets * expected_log, axis=0, keepdims=True)
    compare_tensors(CrossEntropyMulti(),
                    outputs, targets, expected_result, tol=1e-6)


def test_cross_entropy_multi_limits(backend_default):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001]).reshape((4, 1))
    targets = np.array(([0.5, 0.0, 1.0, 0.2])).reshape((4, 1))
    eps = np.exp(-50)
    expected_log = np.log(np.maximum(outputs, eps))
    expected_result = np.sum(-targets * expected_log, axis=0, keepdims=True)
    compare_tensors(CrossEntropyMulti(),
                    outputs, targets, expected_result, tol=1e-5)


def test_cross_entropy_multi_derivative(backend_default):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001]).reshape((4, 1))
    targets = np.array(([0.5, 0.0, 1.0, 0.2])).reshape((4, 1))
    expected_result = ((outputs - targets) / outputs.shape[1])
    compare_tensors(CrossEntropyMulti(), outputs, targets, expected_result,
                    deriv=True, tol=1e-6)

"""
    SumSquared
"""


def test_sum_squared(backend_default):
    outputs = np.array([0.5, 0.9, 0.1, 0.0001]).reshape((4, 1))
    targets = np.array([0.5, 0.99, 0.01, 0.2]).reshape((4, 1))
    expected_result = np.sum((outputs - targets) ** 2, axis=0, keepdims=True) / 2.
    compare_tensors(SumSquared(), outputs, targets, expected_result, tol=1e-8)


def test_sum_squared_limits(backend_default):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001]).reshape((4, 1))
    targets = np.array(([0.5, 0.0, 1.0, 0.2])).reshape((4, 1))
    expected_result = np.sum((outputs - targets) ** 2, axis=0, keepdims=True) / 2.
    compare_tensors(SumSquared(), outputs, targets, expected_result, tol=1e-7)


def test_sum_squared_derivative(backend_default):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001]).reshape((4, 1))
    targets = np.array(([0.5, 0.0, 1.0, 0.2])).reshape((4, 1))
    expected_result = (outputs - targets) / outputs.shape[1]
    compare_tensors(SumSquared(), outputs,
                    targets, expected_result, deriv=True, tol=1e-8)

"""
    MeanSquared
"""


def test_mean_squared(backend_default):
    outputs = np.array([0.5, 0.9, 0.1, 0.0001]).reshape((4, 1))
    targets = np.array([0.5, 0.99, 0.01, 0.2]).reshape((4, 1))
    expected_result = np.mean((outputs - targets) ** 2, axis=0, keepdims=True) / 2.
    compare_tensors(MeanSquared(), outputs, targets, expected_result, tol=1e-8)


def test_mean_squared_limits(backend_default):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001]).reshape((4, 1))
    targets = np.array(([0.5, 0.0, 1.0, 0.2])).reshape((4, 1))
    expected_result = np.mean((outputs - targets) ** 2, axis=0, keepdims=True) / 2.
    compare_tensors(MeanSquared(), outputs, targets, expected_result, tol=1e-7)


def test_mean_squared_derivative(backend_default):
    outputs = np.array([0.5, 1.0, 0.0, 0.0001]).reshape((4, 1))
    targets = np.array(([0.5, 0.0, 1.0, 0.2])).reshape((4, 1))
    expected_result = (outputs - targets) / outputs.shape[1] / outputs.shape[0]
    compare_tensors(MeanSquared(), outputs,
                    targets, expected_result, deriv=True, tol=1e-8)

"""
    Misclassification
"""


def compare_metric(func, y, t, outputs, deriv=False, tol=0.):
    be = NervanaObject.be
    dtypeu = np.float32
    temp = func(be.array(dtypeu(y)), be.array(dtypeu(t)))
    cond = np.sum(np.abs(temp - outputs) <= tol)
    assert cond == np.prod(outputs.shape)


def test_misclassification(backend_default):
    NervanaObject.be.bsz = 3
    outputs = np.array(
        [[0.25, 0.99, 0.33], [0.5, 0.005, 0.32], [0.25, 0.005, 0.34]])
    targets = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])
    expected_result = np.ones((1, 1)) / 3.
    compare_metric(Misclassification(),
                   outputs, targets, expected_result, tol=1e-7)

"""
    Precision / Recall
"""


def test_precision_recall(backend_default):
    be = NervanaObject.be
    be.bsz = 4
    preds = np.array([[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    targets = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0]])
    expected_result = np.array([1 + 1 + 0, 1 + 0.5 + 0]) / 3.
    compare_metric(PrecisionRecall(3), preds, targets, expected_result, tol=1e-6)


def test_precision_recall_binarize(backend_default):
    be = NervanaObject.be
    be.bsz = 4
    preds = np.array([[0.2, 0.9, 0.01, 1],
                      [0.75, 0.05, 0.44, 0],
                      [0.05, 0.05, 0.55, 0]])
    targets = np.array([[0, 1, 0, 1],
                        [1, 0, 1, 0],
                        [0, 0, 0, 0]])
    expected_result = np.array([1 + 1 + 0, 1 + 0.5 + 0]) / 3.
    compare_metric(PrecisionRecall(3, binarize=True), preds, targets,
                   expected_result, tol=1e-6)

"""
    Smooth L1 loss
"""


def test_smoothL1_random(backend_default, fargs):
    s1, s2, m, sigma = fargs
    sigma2 = sigma**2
    shape = (s1, s2)
    magnitude = m
    outputs = (np.random.random(shape) - 0.5) * magnitude
    targets = np.random.random(shape)
    x = outputs - targets
    expected_result = np.zeros(shape)
    I1, J1 = np.where(abs(x) < 1.0 / sigma2)
    I2, J2 = np.where(abs(x) >= 1.0 / sigma2)
    expected_result[I1, J1] = 0.5 * x[I1, J1]**2 * sigma2
    expected_result[I2, J2] = abs(x[I2, J2]) - 0.5 / sigma2
    expected_result = np.sum(expected_result, axis=0, keepdims=True)
    compare_tensors(SmoothL1Loss(sigma=sigma), outputs,
                    targets, expected_result,
                    deriv=False, tol=1e-5)


def test_smoothL1_zeros(backend_default, fargs):
    s1, s2, m, sigma = fargs
    sigma2 = sigma**2
    shape = (s1, s2)
    outputs = np.zeros(shape)
    targets = np.zeros(shape)
    x = outputs - targets
    expected_result = np.zeros(shape)
    I1, J1 = np.where(abs(x) < 1.0 / sigma2)
    I2, J2 = np.where(abs(x) >= 1.0 / sigma2)
    expected_result[I1, J1] = 0.5 * x[I1, J1]**2 * sigma2
    expected_result[I2, J2] = abs(x[I2, J2]) - 0.5 / sigma2
    expected_result = np.sum(expected_result, axis=0, keepdims=True)
    compare_tensors(SmoothL1Loss(sigma=sigma), outputs,
                    targets, expected_result,
                    deriv=False, tol=1e-5)


def test_smoothL1_ones(backend_default, fargs):
    s1, s2, m, sigma = fargs
    sigma2 = sigma**2
    shape = (s1, s2)
    outputs = np.ones(shape) * m
    targets = np.ones(shape) * m
    x = outputs - targets
    expected_result = np.zeros(shape)
    I1, J1 = np.where(abs(x) < 1.0 / sigma2)
    I2, J2 = np.where(abs(x) >= 1.0 / sigma2)
    expected_result[I1, J1] = 0.5 * x[I1, J1]**2 * sigma2
    expected_result[I2, J2] = abs(x[I2, J2]) - 0.5 / sigma2
    expected_result = np.sum(expected_result, axis=0, keepdims=True)
    compare_tensors(SmoothL1Loss(sigma=sigma), outputs,
                    targets, expected_result,
                    deriv=False, tol=1e-5)


def test_smoothL1_random_derivative(backend_default, fargs):
    s1, s2, m, sigma = fargs
    sigma2 = sigma**2
    shape = (s1, s2)
    magnitude = m
    outputs = (np.random.random(shape) - 0.5) * magnitude
    targets = np.random.random(shape)
    x = outputs - targets
    expected_result = np.zeros(shape)
    I1, J1 = np.where(abs(x) < 1.0 / sigma2)
    I2, J2 = np.where(abs(x) >= 1.0 / sigma2)
    expected_result[I1, J1] = x[I1, J1] * sigma2
    expected_result[I2, J2] = np.sign(x[I2, J2])
    compare_tensors(SmoothL1Loss(sigma=sigma), outputs,
                    targets, expected_result,
                    deriv=True, tol=1e-5)

"""
    SquareHingeLoss
"""


def test_square_hinge(backend_default):
    outputs = np.array([0.3, 0.7]).reshape((2, 1))
    targets = np.array([0, 1]).reshape((2, 1))
    shifted_targets = np.array([-1, 1]).reshape((2, 1))
    expected_result = np.mean(np.square(np.maximum(1 - shifted_targets * outputs, 0)), axis=0)
    compare_tensors(SquareHingeLoss(), outputs, targets, expected_result, tol=1e-7)


def test_square_hinge_derivative(backend_default):
    outputs = np.array([0.3, 0.7]).reshape((2, 1))
    targets = np.array([0, 1]).reshape((2, 1))
    shifted_targets = np.array([-1, 1]).reshape((2, 1))
    expected_result = -1 * shifted_targets * np.maximum(1 - shifted_targets * outputs, 0)
    compare_tensors(SquareHingeLoss(), outputs,
                    targets, expected_result, deriv=True, tol=1e-7)


if __name__ == '__main__':
    be = gen_backend(backend='gpu', batch_size=50)
    fargs = (4, 10, 20)
    test_smoothL1_random(be, fargs)
    test_smoothL1_random_derivative(be, fargs)
