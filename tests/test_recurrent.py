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
"""
This test compares the NEON recurrent layer against a numpy reference recurrent
implementation and compares the NEON recurrent bprop deltas to the gradients
estimated by finite differences.
The numpy reference recurrent layer contains static methods for forward pass
and backward pass.
The test runs a SINGLE layer of recurrent layer and compare numerical values
The reference model handles batch_size as 1 only

The following are made sure to be the same in both recurrent layers
    -   initial h values (all zeros)
    -   initial W, b (ones or random values)
    -   input data (random data matrix)
    -   input error (random data matrix)
    -   the data shape inside recurrent_ref is seq_len, input_size, 1
    -   the data shape inside recurrent (neon) is feature, seq_len * batch_size
"""

import itertools as itt
import numpy as np

from neon import NervanaObject, logger as neon_logger
from neon.initializers.initializer import Constant, Gaussian
from neon.layers import Recurrent
from neon.layers.container import DeltasTree
from neon.transforms import Tanh
from recurrent_ref import Recurrent as RefRecurrent
from utils import allclose_with_out


def pytest_generate_tests(metafunc):
    bsz_rng = [1]

    if 'refgruargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            seq_rng = [2, 3, 4]
            inp_rng = [3, 5, 10]
            out_rng = [3, 5, 10]
        else:
            seq_rng = [3]
            inp_rng = [5]
            out_rng = [10]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng)
        metafunc.parametrize('refgruargs', fargs)

    if 'gradgruargs' in metafunc.fixturenames:
        fargs = []
        if metafunc.config.option.all:
            seq_rng = [2, 3]
            inp_rng = [5, 10]
            out_rng = [3, 5, 10]
        else:
            seq_rng = [3]
            inp_rng = [5]
            out_rng = [10]
        fargs = itt.product(seq_rng, inp_rng, out_rng, bsz_rng)
        metafunc.parametrize('gradgruargs', fargs)


def test_ref_compare_ones(backend_default, refgruargs):
    # run comparison with reference code
    # for all ones init
    seq_len, input_size, hidden_size, batch_size = refgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    check_rnn(seq_len, input_size, hidden_size,
              batch_size, Constant(val=1.0), [1.0, 0.0])


def test_ref_compare_rand(backend_default, refgruargs):
    # run comparison with reference code
    # for Gaussian random init
    seq_len, input_size, hidden_size, batch_size = refgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    check_rnn(seq_len, input_size, hidden_size, batch_size,
              Gaussian())


# compare neon RNN to reference RNN implementation
def check_rnn(seq_len, input_size, hidden_size,
              batch_size, init_func, inp_moms=[0.0, 1.0]):
    # init_func is the initializer for the model params
    # inp_moms is the [ mean, std dev] of the random input
    input_shape = (input_size, seq_len * batch_size)
    output_shape = (hidden_size, seq_len * batch_size)
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    # ======== create models ========
    # neon RNN
    rnn = Recurrent(hidden_size, init_func, activation=Tanh())

    # reference numpy RNN
    rnn_ref = RefRecurrent(input_size, hidden_size)
    Wxh = rnn_ref.Wxh
    Whh = rnn_ref.Whh
    bh = rnn_ref.bh

    # ========= generate data =================
    # generate random input tensor
    inp = np.random.rand(*input_shape) * inp_moms[1] + inp_moms[0]
    inpa = rnn.be.array(inp)
    # generate random deltas tensor
    deltas = np.random.randn(*output_shape)

    # the reference code expects these shapes:
    # input_shape: (seq_len, input_size, batch_size)
    # output_shape: (seq_len, hidden_size, batch_size)
    inp_ref = inp.copy().T.reshape(
        seq_len, batch_size, input_size).swapaxes(1, 2)
    deltas_ref = deltas.copy().T.reshape(
        seq_len, batch_size, hidden_size).swapaxes(1, 2)

    # ========= running models ==========
    # run neon fprop
    rnn.configure((input_size, seq_len))
    rnn.prev_layer = True
    rnn.allocate()

    dtree = DeltasTree()
    rnn.allocate_deltas(dtree)
    dtree.allocate_buffers()
    rnn.set_deltas(dtree)

    rnn.fprop(inpa)

    # weights are only initialized after doing fprop, so now
    # make ref weights and biases the same with neon model
    Wxh[:] = rnn.W_input.get()
    Whh[:] = rnn.W_recur.get()
    bh[:] = rnn.b.get()

    (dWxh_ref, dWhh_ref, db_ref, h_ref_list,
     dh_ref_list, d_out_ref) = rnn_ref.lossFun(inp_ref, deltas_ref)

    # now test the bprop
    rnn.bprop(rnn.be.array(deltas))
    # grab the delta W from gradient buffer
    dWxh_neon = rnn.dW_input.get()
    dWhh_neon = rnn.dW_recur.get()
    db_neon = rnn.db.get()

    # comparing outputs
    neon_logger.display('====Verifying hidden states====')
    assert allclose_with_out(rnn.outputs.get(),
                             h_ref_list,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('fprop is verified')

    neon_logger.display('====Verifying update on W and b ====')
    neon_logger.display('dWxh')
    assert allclose_with_out(dWxh_neon,
                             dWxh_ref,
                             rtol=0.0,
                             atol=1.0e-5)
    neon_logger.display('dWhh')
    assert allclose_with_out(dWhh_neon,
                             dWhh_ref,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('====Verifying update on bias====')
    neon_logger.display('db')
    assert allclose_with_out(db_neon,
                             db_ref,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('bprop is verified')

    return


def reset_rnn(rnn):
    # in order to run fprop multiple times
    # for the gradient check tests the
    # rnn internal variables need to be
    # cleared
    rnn.x = None
    rnn.xs = None  # just in case
    rnn.outputs = None
    return


def test_gradient_neon_gru(backend_default, gradgruargs):
    seq_len, input_size, hidden_size, batch_size = gradgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    gradient_check(seq_len, input_size, hidden_size, batch_size)


def gradient_check(seq_len, input_size, hidden_size, batch_size,
                   threshold=1.0e-3):
    # 'threshold' is the max fractional difference
    #             between gradient estimate and
    #             bprop deltas (def is 5%)
    # for a given set of layer parameters calculate
    # the gradients and compare to the derivatives
    # obtained with the bprop function.  repeat this
    # for a range of perturbations and use the
    # perturbation size with the best results.
    # This is necessary for 32 bit computations

    min_max_err = -1.0  # minimum max error
    neon_logger.display('Perturb mag, max grad diff')
    for pert_exp in range(-5, 0):
        # need to generate the scaling and input outside
        # having an issue with the random number generator
        # when these are generated inside the gradient_calc
        # function
        input_shape = (input_size, seq_len * batch_size)
        output_shape = (hidden_size, seq_len * batch_size)

        rand_scale = np.random.random(output_shape) * 2.0 - 1.0
        inp = np.random.randn(*input_shape)

        pert_mag = 10.0**pert_exp
        (grad_est, deltas) = gradient_calc(seq_len,
                                           input_size,
                                           hidden_size,
                                           batch_size,
                                           epsilon=pert_mag,
                                           rand_scale=rand_scale,
                                           inp_bl=inp)
        dd = np.max(np.abs(grad_est - deltas))
        neon_logger.display('%e, %e' % (pert_mag, dd))
        if min_max_err < 0.0 or dd < min_max_err:
            min_max_err = dd
        # reset the seed so models are same in each run
        allclose_with_out(grad_est, deltas, rtol=0.0, atol=0.0)
        NervanaObject.be.rng_reset()

    # check that best value of worst case error is less than threshold
    neon_logger.display('Worst case error %e with perturbation %e' % (min_max_err, pert_mag))
    neon_logger.display('Threshold %e' % (threshold))
    assert min_max_err < threshold


def gradient_calc(seq_len, input_size, hidden_size, batch_size,
                  epsilon=None, rand_scale=None, inp_bl=None):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    input_shape = (input_size, seq_len * batch_size)

    # generate input if one is not given
    if inp_bl is None:
        inp_bl = np.random.randn(*input_shape)

    # neon rnn instance
    rnn = Recurrent(hidden_size, Gaussian(), activation=Tanh())
    inpa = rnn.be.array(np.copy(inp_bl))

    # run fprop on the baseline input
    rnn.configure((input_size, seq_len))
    rnn.prev_layer = True
    rnn.allocate()

    dtree = DeltasTree()
    rnn.allocate_deltas(dtree)
    dtree.allocate_buffers()
    rnn.set_deltas(dtree)

    out_bl = rnn.fprop(inpa).get()

    # random scaling/hash to generate fake loss
    if rand_scale is None:
        rand_scale = np.random.random(out_bl.shape) * 2.0 - 1.0
    # loss function would be:
    # loss_bl = np.sum(rand_scale * out_bl)

    # run back prop with rand_scale as the errors
    # use copy to avoid any interactions
    deltas_neon = rnn.bprop(rnn.be.array(np.copy(rand_scale))).get()

    # add a perturbation to each input element
    grads_est = np.zeros(inpa.shape)
    inp_pert = inp_bl.copy()
    for pert_ind in range(inpa.size):
        save_val = inp_pert.flat[pert_ind]

        inp_pert.flat[pert_ind] = save_val + epsilon
        reset_rnn(rnn)
        rnn.allocate()
        out_pos = rnn.fprop(rnn.be.array(inp_pert)).get()

        inp_pert.flat[pert_ind] = save_val - epsilon
        reset_rnn(rnn)
        rnn.allocate()
        out_neg = rnn.fprop(rnn.be.array(inp_pert)).get()

        # calculate the loss with perturbations
        loss_pos = np.sum(rand_scale * out_pos)
        loss_neg = np.sum(rand_scale * out_neg)
        # compute the gradient estimate
        grad = 0.5 * (loss_pos - loss_neg) / epsilon

        grads_est.flat[pert_ind] = grad

        # reset the perturbed input element
        inp_pert.flat[pert_ind] = save_val

    del rnn
    return (grads_est, deltas_neon)


if __name__ == '__main__':
    from neon.backends import gen_backend
    bsz = 1
    be = gen_backend(backend='gpu', batch_size=bsz)

    fargs = (30, 5, 10, bsz)

    # test_ref_compare_ones(be, fargs)
    test_ref_compare_rand(be, fargs)
