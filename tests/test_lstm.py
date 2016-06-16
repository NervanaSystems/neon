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
This test compares the NEON LSTM layer against a numpy reference LSTM
implementation and compares the NEON LSTM bprop deltas to the gradients
estimated by finite differences.
The numpy reference LSTM contains static methods for forward pass
and backward pass.
It runs a SINGLE layer of LSTM and compare numerical values

The following are made sure to be the same in both LSTMs
    -   initial c, h values (all zeros)
    -   initial W, b (random values)
    -   input data (random data matrix)
    -   input error (random data matrix)
    -   the data shape inside LSTM_np is seq_len, batch_size, input_size.
        Need transpose
    -   the data shape inside LSTM (neon) is input_size, seq_len * batch_size

"""
import itertools as itt
import numpy as np

from neon import NervanaObject, logger as neon_logger
from neon.initializers.initializer import Constant, Gaussian
from neon.layers.recurrent import LSTM
from neon.layers.container import DeltasTree
from neon.transforms import Logistic, Tanh
from lstm_ref import LSTM as RefLSTM
from utils import sparse_rand, allclose_with_out


def pytest_generate_tests(metafunc):
    if metafunc.config.option.all:
        bsz_rng = [16, 32]
    else:
        bsz_rng = [16]

    if 'reflstmargs' in metafunc.fixturenames:
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
        metafunc.parametrize('reflstmargs', fargs)

    if 'gradlstmargs' in metafunc.fixturenames:
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
        metafunc.parametrize('gradlstmargs', fargs)


def test_ref_compare_ones(backend_default, reflstmargs):
        # run comparison with reference code
        # for all ones init
        np.random.seed(seed=0)
        seq_len, input_size, hidden_size, batch_size = reflstmargs
        NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

        check_lstm(seq_len, input_size, hidden_size,
                   batch_size, Constant(val=1.0), [1.0, 0.0])


def test_ref_compare_rand(backend_default, reflstmargs):
        # run comparison with reference code
        # for Gaussian random init
        np.random.seed(seed=0)
        seq_len, input_size, hidden_size, batch_size = reflstmargs
        NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
        check_lstm(seq_len, input_size, hidden_size, batch_size,
                   Gaussian())


# compare neon LSTM to reference LSTM implementation
def check_lstm(seq_len, input_size, hidden_size,
               batch_size, init_func, inp_moms=[0.0, 1.0]):
    # init_func is the initializer for the model params
    # inp_moms is the [ mean, std dev] of the random input
    input_shape = (input_size, seq_len * batch_size)
    hidden_shape = (hidden_size, seq_len * batch_size)
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    # neon LSTM
    lstm = LSTM(hidden_size,
                init_func,
                activation=Tanh(),
                gate_activation=Logistic())

    inp = np.random.rand(*input_shape) * inp_moms[1] + inp_moms[0]
    inpa = lstm.be.array(inp)
    # run neon fprop
    lstm.configure((input_size, seq_len))
    lstm.prev_layer = True  # Hack to force allocating a delta buffer
    lstm.allocate()

    dtree = DeltasTree()
    lstm.allocate_deltas(dtree)
    dtree.allocate_buffers()
    lstm.set_deltas(dtree)

    lstm.fprop(inpa)

    # reference numpy LSTM
    lstm_ref = RefLSTM()
    WLSTM = lstm_ref.init(input_size, hidden_size)

    # make ref weights and biases with neon model
    WLSTM[0, :] = lstm.b.get().T
    WLSTM[1:input_size + 1, :] = lstm.W_input.get().T
    WLSTM[input_size + 1:] = lstm.W_recur.get().T

    # transpose input X and do fprop
    inp_ref = inp.copy().T.reshape(seq_len, batch_size, input_size)
    (Hout_ref, cprev, hprev, batch_cache) = lstm_ref.forward(inp_ref,
                                                             WLSTM)

    # the output needs transpose as well
    Hout_ref = Hout_ref.reshape(seq_len * batch_size, hidden_size).T
    IFOGf_ref = batch_cache['IFOGf'].reshape(seq_len * batch_size, hidden_size * 4).T
    Ct_ref = batch_cache['Ct'].reshape(seq_len * batch_size, hidden_size).T

    # compare results
    neon_logger.display('====Verifying IFOG====')
    assert allclose_with_out(lstm.ifog_buffer.get(),
                             IFOGf_ref,
                             rtol=0.0,
                             atol=1.5e-5)

    neon_logger.display('====Verifying cell states====')
    assert allclose_with_out(lstm.c_act_buffer.get(),
                             Ct_ref,
                             rtol=0.0,
                             atol=1.5e-5)

    neon_logger.display('====Verifying hidden states====')
    assert allclose_with_out(lstm.outputs.get(),
                             Hout_ref,
                             rtol=0.0,
                             atol=1.5e-5)

    neon_logger.display('fprop is verified')

    # now test the bprop
    # generate random deltas tensor
    deltas = np.random.randn(*hidden_shape)

    lstm.bprop(lstm.be.array(deltas))
    # grab the delta W from gradient buffer
    dWinput_neon = lstm.dW_input.get()
    dWrecur_neon = lstm.dW_recur.get()
    db_neon = lstm.db.get()

    deltas_ref = deltas.copy().T.reshape(seq_len, batch_size, hidden_size)
    (dX_ref, dWLSTM_ref, dc0_ref, dh0_ref) = lstm_ref.backward(deltas_ref,
                                                               batch_cache)
    dWrecur_ref = dWLSTM_ref[-hidden_size:, :]
    dWinput_ref = dWLSTM_ref[1:input_size + 1, :]
    db_ref = dWLSTM_ref[0, :]
    dX_ref = dX_ref.reshape(seq_len * batch_size, input_size).T

    # compare results
    neon_logger.display('Making sure neon LSTM match numpy LSTM in bprop')
    neon_logger.display('====Verifying update on W_recur====')

    assert allclose_with_out(dWrecur_neon,
                             dWrecur_ref.T,
                             rtol=0.0,
                             atol=1.5e-5)

    neon_logger.display('====Verifying update on W_input====')
    assert allclose_with_out(dWinput_neon,
                             dWinput_ref.T,
                             rtol=0.0,
                             atol=1.5e-5)

    neon_logger.display('====Verifying update on bias====')
    assert allclose_with_out(db_neon.flatten(),
                             db_ref,
                             rtol=0.0,
                             atol=1.5e-5)

    neon_logger.display('====Verifying output delta====')
    assert allclose_with_out(lstm.out_deltas_buffer.get(),
                             dX_ref,
                             rtol=0.0,
                             atol=1.5e-5)

    neon_logger.display('bprop is verified')

    return


def reset_lstm(lstm):
    # in order to run fprop multiple times
    # for the gradient check tests the
    # lstm internal variables need to be
    # cleared
    lstm.x = None
    lstm.xs = None  # just in case
    lstm.outputs = None
    return


def test_gradient_ref_lstm(backend_default, gradlstmargs):
    seq_len, input_size, hidden_size, batch_size = gradlstmargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    gradient_check_ref(seq_len, input_size, hidden_size, batch_size)


def test_gradient_neon_lstm(backend_default, gradlstmargs):
    seq_len, input_size, hidden_size, batch_size = gradlstmargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    gradient_check(seq_len, input_size, hidden_size, batch_size)


def gradient_check_ref(seq_len, input_size, hidden_size, batch_size,
                       epsilon=1.0e-5, dtypeu=np.float64, threshold=1e-4):
    # this is a check of the reference code itself
    # estimates the gradients by adding perturbations
    # to the input and the weights and compares to
    # the values calculated in bprop

    # generate sparse random input matrix
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    input_shape = (seq_len, input_size, batch_size)
    # hidden_shape = (seq_len, hidden_size, batch_size)
    (inp_bl, nz_inds) = sparse_rand(input_shape, frac=1.0 / float(input_shape[1]))
    inp_bl = np.random.randn(*input_shape)

    # convert input matrix from neon to ref code format
    inp_bl = inp_bl.swapaxes(1, 2).astype(dtypeu)

    # generate reference LSTM
    lstm_ref = RefLSTM()
    WLSTM = lstm_ref.init(input_size, hidden_size).astype(dtypeu)

    # init parameters as done for neon
    WLSTM = np.random.randn(*WLSTM.shape)

    (Hout, cprev, hprev, cache) = lstm_ref.forward(inp_bl, WLSTM)

    # scale Hout by random matrix...
    rand_scale = np.random.random(Hout.shape) * 2.0 - 1.0
    rand_scale = dtypeu(rand_scale)

    # line below would be the loss function
    # loss_bl = np.sum(rand_scale * Hout)

    # run bprop, input deltas is rand_scale
    (dX_bl, dWLSTM_bl, dc0, dh0) = lstm_ref.backward(rand_scale, cache)

    grads_est = np.zeros(dX_bl.shape)
    inp_pert = inp_bl.copy()
    for pert_ind in range(inp_bl.size):
        save_val = inp_pert.flat[pert_ind]

        # add/subtract perturbations to input
        inp_pert.flat[pert_ind] = save_val + epsilon
        # and run fprop on perturbed input
        (Hout_pos, cprev, hprev, cache) = lstm_ref.forward(inp_pert, WLSTM)

        inp_pert.flat[pert_ind] = save_val - epsilon
        (Hout_neg, cprev, hprev, cache) = lstm_ref.forward(inp_pert, WLSTM)

        # calculate the loss on outputs
        loss_pos = np.sum(rand_scale * Hout_pos)
        loss_neg = np.sum(rand_scale * Hout_neg)

        grads_est.flat[pert_ind] = 0.5 / float(epsilon) * (loss_pos - loss_neg)

        # reset input
        inp_pert.flat[pert_ind] = save_val

    # assert that gradient estimates within rel threshold of
    # bprop calculated deltas
    assert allclose_with_out(grads_est, dX_bl, rtol=threshold, atol=0.0)
    return


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
        # allclose_with_out(grad_est,deltas, rtol=0.0, atol=0.0)
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

    # neon lstm instance
    lstm = LSTM(hidden_size, Gaussian(), activation=Tanh(), gate_activation=Logistic())
    inpa = lstm.be.array(np.copy(inp_bl))

    # run fprop on the baseline input
    lstm.configure((input_size, seq_len))
    lstm.prev_layer = True  # Hack to force allocating a delta buffer

    lstm.allocate()

    dtree = DeltasTree()
    lstm.allocate_deltas(dtree)
    dtree.allocate_buffers()
    lstm.set_deltas(dtree)

    out_bl = lstm.fprop(inpa).get()

    # random scaling/hash to generate fake loss
    if rand_scale is None:
        rand_scale = np.random.random(out_bl.shape) * 2.0 - 1.0
    # loss function would be:
    # loss_bl = np.sum(rand_scale * out_bl)

    # run back prop with rand_scale as the errors
    # use copy to avoid any interactions
    deltas_neon = lstm.bprop(lstm.be.array(np.copy(rand_scale))).get()

    # add a perturbation to each input element
    grads_est = np.zeros(inpa.shape)
    inp_pert = inp_bl.copy()
    for pert_ind in range(inpa.size):
        save_val = inp_pert.flat[pert_ind]

        inp_pert.flat[pert_ind] = save_val + epsilon
        reset_lstm(lstm)
        lstm.allocate()
        out_pos = lstm.fprop(lstm.be.array(inp_pert)).get()

        inp_pert.flat[pert_ind] = save_val - epsilon
        reset_lstm(lstm)
        lstm.allocate()
        out_neg = lstm.fprop(lstm.be.array(inp_pert)).get()

        # calculate the loss with perturbations
        loss_pos = np.sum(rand_scale * out_pos)
        loss_neg = np.sum(rand_scale * out_neg)
        # compute the gradient estimate
        grad = 0.5 / float(epsilon) * (loss_pos - loss_neg)

        grads_est.flat[pert_ind] = grad

        # reset the perturbed input element
        inp_pert.flat[pert_ind] = save_val

    del lstm
    return (grads_est, deltas_neon)
