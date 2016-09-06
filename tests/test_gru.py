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
This test compares the NEON GRU layer against a numpy reference GRU
implementation and compares the NEON GRU bprop deltas to the gradients
estimated by finite differences.
The numpy reference GRU contains static methods for forward pass
and backward pass.
It runs a SINGLE layer of GRU and compare numerical values

The following are made sure to be the same in both GRUs
    -   initial h values (all zeros)
    -   initial W, b (ones or random values)
    -   input data (random data matrix)
    -   input error (random data matrix)
    -   the data shape inside GRU_ref is seq_len, 1, input_size.
        Need transpose
    -   the data shape inside GRU (neon) is is batch_size, seq_len * batch_size

"""
import itertools as itt
import numpy as np

from neon import NervanaObject, logger as neon_logger
from neon.initializers.initializer import Constant, Gaussian
from neon.layers import GRU
from neon.transforms import Logistic, Tanh
from neon.layers.container import DeltasTree

from gru_ref import GRU as RefGRU
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

    check_gru(seq_len, input_size, hidden_size,
              batch_size, Constant(val=1.0), [1.0, 0.0])


def test_ref_compare_rand(backend_default, refgruargs):
    # run comparison with reference code
    # for all ones init
    seq_len, input_size, hidden_size, batch_size = refgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    check_gru(seq_len, input_size, hidden_size, batch_size,
              Gaussian())


def test_ref_compare_rand_init_state(backend_default, refgruargs):
    seq_len, input_size, hidden_size, batch_size = refgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    check_gru(seq_len, input_size, hidden_size, batch_size,
              Gaussian(), add_init_state=True)


# compare neon GRU to reference GRU implementation
def check_gru(seq_len, input_size, hidden_size,
              batch_size, init_func, inp_moms=[0.0, 1.0], add_init_state=False):
    # init_func is the initializer for the model params
    # inp_moms is the [ mean, std dev] of the random input
    input_shape = (input_size, seq_len * batch_size)
    output_shape = (hidden_size, seq_len * batch_size)
    slice_shape = (hidden_size, batch_size)

    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    # neon GRU
    gru = GRU(hidden_size,
              init_func,
              activation=Tanh(),
              gate_activation=Logistic())

    # generate random input tensor
    inp = np.random.rand(*input_shape) * inp_moms[1] + inp_moms[0]
    inp_dev = gru.be.array(inp)
    # generate random deltas tensor
    deltas = np.random.randn(*output_shape)

    # run neon fprop
    gru.configure((input_size, seq_len))
    gru.prev_layer = True
    gru.allocate()

    test_buffer = DeltasTree()
    gru.allocate_deltas(test_buffer)
    test_buffer.allocate_buffers()
    gru.set_deltas(test_buffer)

    if add_init_state:
        init_state = np.random.rand(*slice_shape)*inp_moms[1] + inp_moms[0]
        init_state_dev = gru.be.array(init_state)
        gru.fprop(inp_dev, init_state=init_state_dev)
    else:
        gru.fprop(inp_dev)

    # reference numpy GRU
    gru_ref = RefGRU(input_size, hidden_size)
    WGRU = gru_ref.weights

    # make ref weights and biases the same with neon model
    r_range = list(range(hidden_size))
    z_range = list(range(hidden_size, hidden_size * 2))
    c_range = list(range(hidden_size * 2, hidden_size * 3))

    WGRU[gru_ref.weights_ind_br][:] = gru.b.get()[r_range]
    WGRU[gru_ref.weights_ind_bz][:] = gru.b.get()[z_range]
    WGRU[gru_ref.weights_ind_bc][:] = gru.b.get()[c_range]

    WGRU[gru_ref.weights_ind_Wxr][:] = gru.W_input.get()[r_range]
    WGRU[gru_ref.weights_ind_Wxz][:] = gru.W_input.get()[z_range]
    WGRU[gru_ref.weights_ind_Wxc][:] = gru.W_input.get()[c_range]

    WGRU[gru_ref.weights_ind_Rhr][:] = gru.W_recur.get()[r_range]
    WGRU[gru_ref.weights_ind_Rhz][:] = gru.W_recur.get()[z_range]
    WGRU[gru_ref.weights_ind_Rhc][:] = gru.W_recur.get()[c_range]

    # transpose input X and do fprop
    # the reference code expects these shapes:
    # input_shape: (seq_len, input_size, batch_size)
    # output_shape: (seq_len, hidden_size, batch_size)
    inp_ref = inp.copy().T.reshape(
        seq_len, batch_size, input_size).swapaxes(1, 2)
    deltas_ref = deltas.copy().T.reshape(
        seq_len, batch_size, hidden_size).swapaxes(1, 2)

    if add_init_state:
        init_state_ref = init_state.copy()
        (dWGRU_ref, h_ref_list, dh_ref_list,
            dr_ref_list, dz_ref_list, dc_ref_list) = gru_ref.lossFun(inp_ref,
                                                                     deltas_ref,
                                                                     init_state_ref)
    else:
        (dWGRU_ref, h_ref_list, dh_ref_list,
            dr_ref_list, dz_ref_list, dc_ref_list) = gru_ref.lossFun(inp_ref,
                                                                     deltas_ref)

    neon_logger.display('====Verifying hidden states====')
    assert allclose_with_out(gru.outputs.get(),
                             h_ref_list,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('fprop is verified')

    # now test the bprop
    neon_logger.display('Making sure neon GRU matches numpy GRU in bprop')
    gru.bprop(gru.be.array(deltas))
    # grab the delta W from gradient buffer
    dWinput_neon = gru.dW_input.get()
    dWrecur_neon = gru.dW_recur.get()
    db_neon = gru.db.get()
    dWxr_neon = dWinput_neon[r_range]
    dWxz_neon = dWinput_neon[z_range]
    dWxc_neon = dWinput_neon[c_range]
    dWrr_neon = dWrecur_neon[r_range]
    dWrz_neon = dWrecur_neon[z_range]
    dWrc_neon = dWrecur_neon[c_range]
    dbr_neon = db_neon[r_range]
    dbz_neon = db_neon[z_range]
    dbc_neon = db_neon[c_range]

    drzc_neon = gru.rzhcan_delta_buffer.get()
    dr_neon = drzc_neon[r_range]
    dz_neon = drzc_neon[z_range]
    dc_neon = drzc_neon[c_range]

    dWxr_ref = dWGRU_ref[gru_ref.dW_ind_Wxr]
    dWxz_ref = dWGRU_ref[gru_ref.dW_ind_Wxz]
    dWxc_ref = dWGRU_ref[gru_ref.dW_ind_Wxc]
    dWrr_ref = dWGRU_ref[gru_ref.dW_ind_Rhr]
    dWrz_ref = dWGRU_ref[gru_ref.dW_ind_Rhz]
    dWrc_ref = dWGRU_ref[gru_ref.dW_ind_Rhc]
    dbr_ref = dWGRU_ref[gru_ref.dW_ind_br]
    dbz_ref = dWGRU_ref[gru_ref.dW_ind_bz]
    dbc_ref = dWGRU_ref[gru_ref.dW_ind_bc]

    # neon_logger.display '====Verifying hidden deltas ===='
    neon_logger.display('====Verifying r deltas ====')
    assert allclose_with_out(dr_neon,
                             dr_ref_list,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('====Verifying z deltas ====')
    assert allclose_with_out(dz_neon,
                             dz_ref_list,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('====Verifying hcan deltas ====')
    assert allclose_with_out(dc_neon,
                             dc_ref_list,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('====Verifying update on W_input====')
    neon_logger.display('dWxr')
    assert allclose_with_out(dWxr_neon,
                             dWxr_ref,
                             rtol=0.0,
                             atol=1.0e-5)
    neon_logger.display('dWxz')
    assert allclose_with_out(dWxz_neon,
                             dWxz_ref,
                             rtol=0.0,
                             atol=1.0e-5)
    neon_logger.display('dWxc')
    assert allclose_with_out(dWxc_neon,
                             dWxc_ref,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('====Verifying update on W_recur====')

    neon_logger.display('dWrr')
    assert allclose_with_out(dWrr_neon,
                             dWrr_ref,
                             rtol=0.0,
                             atol=1.0e-5)
    neon_logger.display('dWrz')
    assert allclose_with_out(dWrz_neon,
                             dWrz_ref,
                             rtol=0.0,
                             atol=1.0e-5)
    neon_logger.display('dWrc')
    assert allclose_with_out(dWrc_neon,
                             dWrc_ref,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('====Verifying update on bias====')
    neon_logger.display('dbr')
    assert allclose_with_out(dbr_neon,
                             dbr_ref,
                             rtol=0.0,
                             atol=1.0e-5)
    neon_logger.display('dbz')
    assert allclose_with_out(dbz_neon,
                             dbz_ref,
                             rtol=0.0,
                             atol=1.0e-5)
    neon_logger.display('dbc')
    assert allclose_with_out(dbc_neon,
                             dbc_ref,
                             rtol=0.0,
                             atol=1.0e-5)

    neon_logger.display('bprop is verified')

    return


def reset_gru(gru):
    # in order to run fprop multiple times
    # for the gradient check tests the
    # gru internal variables need to be
    # cleared
    gru.x = None
    gru.xs = None  # just in case
    gru.outputs = None
    return


def test_gradient_neon_gru(backend_default, gradgruargs):
    seq_len, input_size, hidden_size, batch_size = gradgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    gradient_check(seq_len, input_size, hidden_size, batch_size)


def test_gradient_neon_gru_init_state(backend_default, gradgruargs):
    seq_len, input_size, hidden_size, batch_size = gradgruargs
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size
    gradient_check(seq_len, input_size, hidden_size, batch_size, True)


def gradient_check(seq_len, input_size, hidden_size, batch_size,
                   add_init_state=False, threshold=1.0e-3):
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
                                           add_init_state=add_init_state,
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


def gradient_calc(seq_len, input_size, hidden_size, batch_size, add_init_state=False,
                  epsilon=None, rand_scale=None, inp_bl=None):
    NervanaObject.be.bsz = NervanaObject.be.batch_size = batch_size

    input_shape = (input_size, seq_len * batch_size)

    # generate input if one is not given
    if inp_bl is None:
        inp_bl = np.random.randn(*input_shape)

    # neon gru instance
    gru = GRU(hidden_size, init=Gaussian(), activation=Tanh(), gate_activation=Logistic())
    inpa = gru.be.array(np.copy(inp_bl))

    # run fprop on the baseline input
    gru.configure((input_size, seq_len))
    gru.prev_layer = True
    gru.allocate()

    test_buffer = DeltasTree()
    gru.allocate_deltas(test_buffer)
    test_buffer.allocate_buffers()
    gru.set_deltas(test_buffer)

    if add_init_state is True:
        slice_shape = (hidden_size, batch_size)
        ini_s = np.random.randn(*slice_shape)
        ini_s_dev = gru.be.array(ini_s.copy())
        out_bl = gru.fprop(inpa, ini_s_dev).get()
    else:
        out_bl = gru.fprop(inpa).get()

    # random scaling/hash to generate fake loss
    if rand_scale is None:
        rand_scale = np.random.random(out_bl.shape) * 2.0 - 1.0
    # loss function would be:
    # loss_bl = np.sum(rand_scale * out_bl)

    # run back prop with rand_scale as the errors
    # use copy to avoid any interactions
    deltas_neon = gru.bprop(gru.be.array(np.copy(rand_scale))).get()

    # add a perturbation to each input element
    grads_est = np.zeros(inpa.shape)
    inp_pert = inp_bl.copy()
    for pert_ind in range(inpa.size):
        save_val = inp_pert.flat[pert_ind]

        inp_pert.flat[pert_ind] = save_val + epsilon
        reset_gru(gru)
        gru.allocate()
        if add_init_state is True:
            ini_s_dev = gru.be.array(ini_s.copy())
            out_pos = gru.fprop(gru.be.array(inp_pert), ini_s_dev).get()
        else:
            out_pos = gru.fprop(gru.be.array(inp_pert)).get()

        inp_pert.flat[pert_ind] = save_val - epsilon
        reset_gru(gru)
        gru.allocate()
        if add_init_state is True:
            ini_s_dev = gru.be.array(ini_s.copy())
            out_neg = gru.fprop(gru.be.array(inp_pert), ini_s_dev).get()
        else:
            out_neg = gru.fprop(gru.be.array(inp_pert)).get()

        # calculate the loss with perturbations
        loss_pos = np.sum(rand_scale * out_pos)
        loss_neg = np.sum(rand_scale * out_neg)
        # compute the gradient estimate
        grad = 0.5 / float(epsilon) * (loss_pos - loss_neg)

        grads_est.flat[pert_ind] = grad

        # reset the perturbed input element
        inp_pert.flat[pert_ind] = save_val

    del gru
    return (grads_est, deltas_neon)
