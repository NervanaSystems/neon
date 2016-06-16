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
Generalized gradient testing applied to different layers and activations
"""
import numpy as np
from neon import logger as neon_logger
from neon.layers.container import DeltasTree


def sweep_epsilon(layer, inp, pert_rng, out_shape=None, lshape=None,
                  pert_frac=0.1):
    # sweep the given list of perturbations (pert_rng)
    # return perturbation magnitude which gave best
    # match between brop deltas and finite diff grads

    if out_shape is None:
        # if the output shape is not provided
        # get it by running fprop
        inpa = layer.be.array(inp.copy())
        in_shape = lshape if lshape is not None else inpa.shape[0]
        layer.configure(in_shape)
        out_shape = layer.out_shape

    # generate loss_scale outside so that the the
    # model/be state can be reset and the computation
    # is repeatable...for some reason the first run is
    # not repeatable if loss_scale is generated inside
    # self.general_gradient_comp()
    loss_scale = np.random.random(out_shape) * 2.0 - 1.0

    # select pert_frac fraction of inps to perturb
    pert_cnt = int(np.ceil(inpa.size * pert_frac))
    pert_inds = np.random.permutation(inpa.size)[0:pert_cnt]

    layer.be.rng_reset()  # reset to same initial rng state

    min_max_diff = -1.0
    min_max_pert = None
    neon_logger.display('epsilon, max diff')
    for epsilon in pert_rng:
        (max_abs, max_rel) = general_gradient_comp(layer,
                                                   inp,
                                                   epsilon=epsilon,
                                                   loss_scale=loss_scale,
                                                   lshape=lshape,
                                                   pert_inds=pert_inds)
        layer.be.rng_reset()  # reset to same initial rng state
        if min_max_diff < 0 or max_abs < min_max_diff:
            min_max_diff = max_abs
            min_max_pert = epsilon
        neon_logger.display('%e %e %e' % (epsilon, max_abs, max_rel))
        neon_logger.display('Min max diff : %e at Pert. Mag. %e' % (min_max_diff, min_max_pert))
    return (min_max_pert, min_max_diff)


def general_gradient_comp(layer,
                          inp,
                          epsilon=1.0e-5,
                          loss_scale=None,
                          lshape=None,
                          pert_inds=None,
                          pooling=False):
    # given a layer, test the bprop
    # using finite differences

    # run neon fprop
    layer.reset()
    inpa = layer.be.array(inp.copy())
    in_shape = lshape if lshape is not None else inpa.shape[0]
    layer.configure(in_shape)
    if layer.owns_delta:
        layer.prev_layer = True
    layer.allocate()

    dtree = DeltasTree()
    layer.allocate_deltas(dtree)
    dtree.allocate_buffers()
    layer.set_deltas(dtree)

    out = layer.fprop(inpa).get()

    out_shape = out.shape

    # scale out by random matrix...
    if loss_scale is None:
        loss_scale = np.random.random(out_shape) * 2.0 - 1.0

    # the loss function is:
    # loss_bl = np.sum(loss_scale * out)

    # run bprop, input deltas is rand_scale
    bprop_deltas = layer.bprop(layer.be.array(loss_scale.copy())).get()

    max_abs_err = -1.0
    max_rel_err = -1.0

    inp_pert = inp.copy()
    if pert_inds is None:
        pert_inds = list(range(inp.size))
    for pert_ind in pert_inds:
        save_val = inp_pert.flat[pert_ind]
        # add/subtract perturbations to input
        inp_pert.flat[pert_ind] = save_val + epsilon
        # and run fprop on perturbed input
        layer.reset()
        layer.configure(in_shape)
        layer.allocate()
        inpa = layer.be.array(inp_pert.copy())
        out_pos = layer.fprop(inpa).get().copy()

        inp_pert.flat[pert_ind] = save_val - epsilon
        inpa = layer.be.array(inp_pert.copy())
        layer.reset()
        layer.configure(in_shape)
        layer.allocate()
        out_neg = layer.fprop(inpa).get().copy()

        # calculate the loss on outputs
        loss_pos = np.sum(loss_scale * out_pos)
        loss_neg = np.sum(loss_scale * out_neg)
        grad_est = 0.5 * (loss_pos - loss_neg) / epsilon

        # reset input
        inp_pert.flat[pert_ind] = save_val

        bprop_val = bprop_deltas.flat[pert_ind]

        abs_err = abs(grad_est - bprop_val)
        if abs_err > max_abs_err:
            max_abs_err = abs_err
            max_abs_vals = [grad_est, bprop_val]

        if (abs(grad_est) + abs(bprop_val)) == 0.0:
            rel_err = 0.0
        else:
            rel_err = float(abs_err) / (abs(grad_est) + abs(bprop_val))
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            max_rel_vals = [grad_est, bprop_val]

    neon_logger.display('Worst case diff %e, vals grad: %e, bprop: %e' % (max_abs_err,
                                                                          max_abs_vals[0],
                                                                          max_abs_vals[1]))
    neon_logger.display('Worst case diff %e, vals grad: %e, bprop: %e' % (max_rel_err,
                                                                          max_rel_vals[0],
                                                                          max_rel_vals[1]))
    return (max_abs_err, max_rel_err)
