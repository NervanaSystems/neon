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
from __future__ import division
from builtins import range, zip
import numpy as np
from neon.layers.layer import ParameterLayer, Layer


def get_steps(x, shape):
    """
    Convert a (feature_size, steps * batch_size) array
    into a [(feature_size, batch_size)] * steps list of views.
    """
    steps = shape[1]
    if x is None:
        return [None for step in range(steps)]

    xs = x.reshape(shape + (-1,))
    return [xs[:, step, :] for step in range(steps)]


def interpret_in_shape(xshape):
    """
    Helper function to interpret the tensor layout of preceding layer for input
    to a recurrent layer. Handles non-recurrent, recurrent, and local layers
    """
    if isinstance(xshape, int):
        nin, nsteps = xshape, 1
    else:
        if len(xshape) == 2:
            nin, nsteps = xshape
        else:
            nin = np.prod(xshape[:-1])
            nsteps = xshape[-1]
    return (nin, nsteps)


class Recurrent(ParameterLayer):

    """
    Basic recurrent layer.

    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless
        name (str, optional): name to refer to this layer as.

    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (Tensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """

    def __init__(self, output_size, init, init_inner=None, activation=None,
                 reset_cells=False, name=None):
        assert activation is not None, "missing activation function for Recurrent"
        super(Recurrent, self).__init__(init, name)
        self.x = None
        self.in_deltas = None
        self.nout = output_size
        self.output_size = output_size
        self.h_nout = output_size
        self.activation = activation
        self.outputs = None
        self.W_input = None
        self.ngates = 1
        self.reset_cells = reset_cells
        self.init_inner = init_inner

    def __str__(self):
        return "Recurrent Layer '%s': %d inputs, %d outputs, %d steps" % (
            self.name, self.nin, self.nout, self.nsteps)

    def configure(self, in_obj):
        """
        Set shape based parameters of this layer given an input tuple, int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer, Tensor or dataset): object that provides shape
                                                           information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Recurrent, self).configure(in_obj)

        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.i_shape = (self.nin, self.nsteps)

        self.out_shape = (self.nout, self.nsteps)
        self.gate_shape = (self.nout * self.ngates, self.nsteps)
        if self.weight_shape is None:
            self.weight_shape = (self.nout, self.nin)
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(Recurrent, self).allocate(shared_outputs)
        self.h_ff_buffer = self.be.zeros_like(self.outputs)
        self.final_state_buffer = self.be.iobuf(self.out_shape[0])
        self.h_ff = get_steps(self.h_ff_buffer, self.out_shape)
        self.h = get_steps(self.outputs, self.out_shape)
        self.h_prev = self.h[-1:] + self.h[:-1]
        # State deltas
        self.h_delta_buffer = self.be.iobuf(self.out_shape)
        self.h_delta = get_steps(self.h_delta_buffer, self.out_shape)
        self.final_hidden_error = self.be.zeros(self.h_delta[0].shape)
        self.bufs_to_reset = [self.outputs]

        if self.W_input is None:
            self.init_params(self.weight_shape)

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        super(Recurrent, self).set_deltas(delta_buffers)
        self.out_deltas_buffer = self.deltas
        self.out_delta = get_steps(self.out_deltas_buffer, self.i_shape)

    def init_buffers(self, inputs):
        """
        Initialize buffers for recurrent internal units and outputs.
        Buffers are initialized as 2D tensors with second dimension being steps * batch_size.
        The second dimension is ordered as [s1b1, s1b2, ..., s1bn, s2b1, s2b2, ..., s2bn, ...]
        A list of views are created on the buffer for easy manipulation of data
        related to a certain time step

        Arguments:
            inputs (Tensor): input data as 2D tensor. The dimension is
                             (input_size, sequence_length * batch_size)

        """
        if self.x is None or self.x is not inputs:
            if self.x is not None:
                for buf in self.bufs_to_reset:
                    buf[:] = 0
            self.x = inputs.reshape(self.nin, self.nsteps * self.be.bsz)
            self.xs = get_steps(inputs, self.i_shape)

    def init_params(self, shape):
        """
        Initialize params including weights and biases.
        The weight matrix and bias matrix are concatenated from the weights
        for inputs and weights for recurrent inputs and bias.

        Arguments:
            shape (Tuple): contains number of outputs and number of inputs

        """
        (nout, nin) = shape
        g_nout = self.ngates * nout
        doFill = False
        weight_dim = (nout + nin + 1)

        if self.W is None:
            self.W = self.be.empty((weight_dim, g_nout))
            self.dW = self.be.zeros_like(self.W)
            doFill = True
        else:
            # Deserialized weights and empty grad
            if self.W.shape != (weight_dim, g_nout):
                raise ValueError('expected {} found {}'.format(
                    self.W.shape, (weight_dim, g_nout)))
            assert self.dW.shape == (weight_dim, g_nout)

        self.W_input = self.W[:nin].reshape((g_nout, nin))
        self.W_recur = self.W[nin:-1].reshape((g_nout, nout))

        self.b = self.W[-1:].reshape((g_nout, 1))

        if doFill:
            wtlist = ('W_input', 'W_recur')
            gatelist = [g * nout for g in range(0, self.ngates + 1)]
            for wtnm in wtlist:
                wtmat = getattr(self, wtnm)
                if wtnm == 'W_recur' and self.init_inner is not None:
                    initfunc = self.init_inner
                else:
                    initfunc = self.init

                for gb, ge in zip(gatelist[:-1], gatelist[1:]):
                    initfunc.fill(wtmat[gb:ge])
            self.b.fill(0.)

        self.dW_input = self.dW[:nin].reshape(self.W_input.shape)
        self.dW_recur = self.dW[nin:-1].reshape(self.W_recur.shape)
        self.db = self.dW[-1:].reshape(self.b.shape)

    def fprop(self, inputs, inference=False, init_state=None):
        """
        Forward propagation of input to recurrent layer.

        Arguments:
            inputs (Tensor): input to the model for each time step of
                             unrolling for each input in minibatch
                             shape: (feature_size, sequence_length * batch_size)
                             where:

                             * feature_size: input size
                             * sequence_length: degree of model unrolling
                             * batch_size: number of inputs in each mini-batch

            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: layer output activations for each time step of
                unrolling and for each input in the minibatch
                shape: (output_size, sequence_length * batch_size)
        """
        self.init_buffers(inputs)

        if self.reset_cells:
            self.h[-1][:] = 0

        # recurrent layer needs a h_prev buffer for bprop
        self.h_prev_bprop = [0] + self.h[:-1]

        if init_state:
            self.h_prev_bprop[0] = init_state
            self.h[-1][:] = init_state

        # feedforward input
        self.be.compound_dot(self.W_input, self.x, self.h_ff_buffer)

        for (h, h_prev, h_ff) in zip(self.h, self.h_prev, self.h_ff):
            self.be.compound_dot(self.W_recur, h_prev, h)
            h[:] = self.activation(h + h_ff + self.b)

        self.final_state_buffer[:] = self.h[-1]
        return self.outputs

    def bprop(self, deltas, alpha=1.0, beta=0.0):
        """
        Backward propagation of errors through recurrent layer.

        Arguments:
            deltas (Tensor): tensors containing the errors for
                             each step of model unrolling.  Expected 2D shape
                             is (output_size, sequence_length * batch_size)
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: back propagated errors for each step of time unrolling
                for each mini-batch element
                shape: (input_size, sequence_length * batch_size)
        """
        self.dW[:] = 0

        if self.in_deltas is None:
            self.in_deltas = get_steps(deltas, self.out_shape)
            self.prev_in_deltas = self.in_deltas[-1:] + self.in_deltas[:-1]

        params = (self.xs, self.h, self.h_prev_bprop, self.h_delta,
                  self.in_deltas, self.prev_in_deltas, self.out_delta)

        for (xs, hs, h_prev, h_delta, in_deltas,
             prev_in_deltas, out_delta) in reversed(list(zip(*params))):

            in_deltas[:] = self.activation.bprop(hs) * in_deltas
            self.be.compound_dot(self.W_recur.T, in_deltas, h_delta)
            prev_in_deltas[:] = prev_in_deltas + h_delta
            if h_prev != 0:
                self.be.compound_dot(in_deltas, h_prev.T, self.dW_recur, beta=1.0)
            self.be.compound_dot(in_deltas, xs.T, self.dW_input, beta=1.0)
            self.db[:] = self.db + self.be.sum(in_deltas, axis=1)
            # save a bit of computation if not bpropping activation gradients
            if out_delta:
                self.be.compound_dot(self.W_input.T, in_deltas, out_delta, alpha=alpha, beta=beta)

        self.final_hidden_error[:] = self.h_delta[0]
        return self.out_deltas_buffer

    def final_state(self):
        """
        Return final state for sequence to sequence models
        """
        return self.final_state_buffer

    def get_final_hidden_error(self):
        """
        Return hidden delta after bprop and adjusting for bprop from decoder
        to encoder in sequence to sequence models.
        """
        return self.final_hidden_error


class LSTM(Recurrent):

    """
    Long Short-Term Memory (LSTM) layer based on
    Hochreiter and Schmidhuber, Neural Computation 9(8): 1735-80 (1997).

    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        gate_activation (Transform): Activation function for the gates
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless
        name (str, optional): name to refer to this layer as.

    Attributes:
        x (Tensor): input data as 2D tensor. The dimension is
                    (input_size, sequence_length * batch_size)
        W_input (Tensor): Weights on the input units
            (out size * 4, input size)
        W_recur (Tensor): Weights on the recursive inputs
            (out size * 4, out size)
        b (Tensor): Biases (out size * 4 , 1)
    """
    def __init__(self, output_size, init, init_inner=None, activation=None,
                 gate_activation=None, reset_cells=False, name=None):
        super(LSTM, self).__init__(output_size, init, init_inner,
                                   activation, reset_cells, name)
        assert gate_activation is not None, ("LSTM layer requires " +
                                             "gate_activation to be specified")
        assert activation is not None, "missing activation function for LSTM"
        self.gate_activation = gate_activation
        self.ngates = 4  # Input, Output, Forget, Cell

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(LSTM, self).allocate(shared_outputs)
        # indices for slicing gate buffers
        (ifo1, ifo2) = (0, self.nout * 3)
        (i1, i2) = (0, self.nout)
        (f1, f2) = (self.nout, self.nout * 2)
        (o1, o2) = (self.nout * 2, self.nout * 3)
        (g1, g2) = (self.nout * 3, self.nout * 4)

        # States: hidden, cell, previous hidden, previous cell
        self.c_buffer = self.be.iobuf(self.out_shape)
        self.c = get_steps(self.c_buffer, self.out_shape)
        self.c_prev = self.c[-1:] + self.c[:-1]
        self.c_prev_bprop = [0] + self.c[:-1]

        self.c_act_buffer = self.be.iobuf(self.out_shape)
        self.c_act = get_steps(self.c_act_buffer, self.out_shape)

        # Gates: input, forget, output, input modulation
        self.ifog_buffer = self.be.iobuf(self.gate_shape)
        self.ifog = get_steps(self.ifog_buffer, self.gate_shape)
        self.ifo = [gate[ifo1:ifo2] for gate in self.ifog]
        self.i = [gate[i1:i2] for gate in self.ifog]
        self.f = [gate[f1:f2] for gate in self.ifog]
        self.o = [gate[o1:o2] for gate in self.ifog]
        self.g = [gate[g1:g2] for gate in self.ifog]

        # State deltas
        self.c_delta_buffer = self.be.iobuf((self.out_shape))
        self.c_delta = get_steps(self.c_delta_buffer, self.out_shape)
        self.c_delta_prev = [None] + self.c_delta[:-1]

        # Pre activation gate deltas
        self.ifog_delta_buffer = self.be.iobuf(self.gate_shape)
        self.ifog_delta = get_steps(self.ifog_delta_buffer, self.gate_shape)
        self.i_delta = [gate[i1:i2] for gate in self.ifog_delta]
        self.f_delta = [gate[f1:f2] for gate in self.ifog_delta]
        self.o_delta = [gate[o1:o2] for gate in self.ifog_delta]
        self.g_delta = [gate[g1:g2] for gate in self.ifog_delta]
        self.bufs_to_reset.append(self.c_buffer)

    def fprop(self, inputs, inference=False, init_state=None):
        """
        Apply the forward pass transformation to the input data.  The input
            data is a list of inputs with an element for each time step of
            model unrolling.

        Arguments:
            inputs (Tensor): input data as 2D tensors, then being converted into a
                             list of 2D slices. The dimension is
                             (input_size, sequence_length * batch_size)
            init_state (Tensor, optional): starting cell values, if not None.
                                           For sequence to sequence models.
            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: LSTM output for each model time step
        """
        self.init_buffers(inputs)

        if self.reset_cells:
            self.h[-1][:] = 0
            self.c[-1][:] = 0

        if init_state is not None:
            self.h[-1][:] = init_state

        params = (self.h, self.h_prev, self.ifog, self.ifo,
                  self.i, self.f, self.o, self.g, self.c, self.c_prev, self.c_act)

        self.be.compound_dot(self.W_input, self.x, self.ifog_buffer)

        for (h, h_prev, ifog, ifo, i, f, o, g, c, c_prev, c_act) in zip(*params):
            self.be.compound_dot(self.W_recur, h_prev, ifog, beta=1.0)

            ifog[:] = ifog + self.b
            ifo[:] = self.gate_activation(ifo)
            g[:] = self.activation(g)

            c[:] = f * c_prev + i * g
            c_act[:] = self.activation(c)
            h[:] = o * c_act

        self.final_state_buffer[:] = self.h[-1]
        return self.outputs

    def bprop(self, deltas, alpha=1.0, beta=0.0):
        """
        Backpropagation of errors, output delta for previous layer, and
        calculate the update on model params

        Arguments:
            deltas (Tensor): tensors containing the errors for
                             each step of model unrolling.
                             Expected 2D shape is
                             (output_size, sequence_length * batch_size)
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Attributes:
            dW_input (Tensor): input weight gradients
            dW_recur (Tensor): recursive weight gradients
            db (Tensor): bias gradients

        Returns:
            Tensor: Backpropagated errors for each time step
                    of model unrolling
        """
        self.c_delta_buffer[:] = 0
        self.dW[:] = 0

        if self.in_deltas is None:
            self.in_deltas = get_steps(deltas, self.out_shape)
            self.prev_in_deltas = self.in_deltas[-1:] + self.in_deltas[:-1]
            self.ifog_delta_last_steps = self.ifog_delta_buffer[:, self.be.bsz:]
            self.h_first_steps = self.outputs[:, :-self.be.bsz]

        params = (self.h_delta, self.in_deltas, self.prev_in_deltas,
                  self.i, self.f, self.o, self.g, self.ifog_delta,
                  self.i_delta, self.f_delta, self.o_delta, self.g_delta,
                  self.c_delta, self.c_delta_prev, self.c_prev_bprop, self.c_act)

        for (h_delta, in_deltas, prev_in_deltas,
             i, f, o, g, ifog_delta, i_delta, f_delta, o_delta, g_delta,
             c_delta, c_delta_prev, c_prev, c_act) in reversed(list(zip(*params))):

            # current cell delta
            c_delta[:] = c_delta + self.activation.bprop(c_act) * (o * in_deltas)
            i_delta[:] = self.gate_activation.bprop(i) * c_delta * g
            f_delta[:] = self.gate_activation.bprop(f) * c_delta * c_prev
            o_delta[:] = self.gate_activation.bprop(o) * in_deltas * c_act
            g_delta[:] = self.activation.bprop(g) * c_delta * i

            # out deltas
            self.be.compound_dot(self.W_recur.T, ifog_delta, h_delta)

            if c_delta_prev is not None:
                c_delta_prev[:] = c_delta * f

            prev_in_deltas[:] = prev_in_deltas + h_delta

        # Weight deltas and accumulate
        self.be.compound_dot(self.ifog_delta_last_steps, self.h_first_steps.T, self.dW_recur)
        self.be.compound_dot(self.ifog_delta_buffer, self.x.T, self.dW_input)

        # Bias delta and accumulate
        self.db[:] = self.be.sum(self.ifog_delta_buffer, axis=1)

        # out deltas
        if self.out_deltas_buffer:  # save a bit of computation
            self.be.compound_dot(self.W_input.T, self.ifog_delta_buffer,
                                 self.out_deltas_buffer.reshape(self.nin, -1),
                                 alpha=alpha, beta=beta)

        self.final_hidden_error[:] = self.h_delta[0]
        return self.out_deltas_buffer


class GRU(Recurrent):

    """
    Implementation of the Gated Recurrent Unit based on [Cho2014]_.

    - It uses two gates: reset gate (r) and update gate (z)
    - The update gate (z) decides how much the activation is updated
    - The reset gate (r) decides how much to reset (when r = 0) from the previous activation
    - Activation (h_t) is a linear interpolation (by z) between the previous
      activation (h_t-1) and the new candidate activation ( h_can )
    - r and z are computed the same way, using different weights
    - gate activation function and unit activation function are usually different
    - gate activation is usually logistic
    - unit activation is usually tanh
    - consider there are 3 gates: r, z, h_can

    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model's input to hidden weights.  By
                            default, this initializer will also be used for recurrent parameters
                            unless init_inner is also specified.  Biases will always be
                            initialized to zero.
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation.
        gate_activation (Transform): Activation function for the gates.
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless.
        name (str, optional): name to refer to this layer as.

    Attributes:
        x (Tensor): Input data tensor (input_size, sequence_length * batch_size)
        W_input (Tensor): Weights on the input units
            (out size * 3, input size)
        W_recur (Tensor): Weights on the recursive inputs
            (out size * 3, out size)
        b (Tensor): Biases (out size * 3 , 1)

    References:

        * Learning phrase representations using rnn encoder-decoder for
          statistical machine translation [Cho2014]_
        * Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
          [Chung2014]_

    .. [Cho2014] http://arxiv.org/abs/1406.1078
    .. [Chung2014] http://arxiv.org/pdf/1412.3555v1.pdf
    """

    def __init__(self, output_size, init, init_inner=None, activation=None,
                 gate_activation=None, reset_cells=False, name=None):
        super(GRU, self).__init__(output_size, init, init_inner,
                                  activation, reset_cells, name)
        self.gate_activation = gate_activation
        self.ngates = 3  # r, z, h candidate

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(GRU, self).allocate(shared_outputs)
        self.h_prev_bprop = [0] + self.h[:-1]

        # indices for slicing gate buffers
        (rz1, rz2) = (0, self.nout * 2)
        (r1, r2) = (0, self.nout)
        (z1, z2) = (self.nout, self.nout * 2)
        (c1, c2) = (self.nout * 2, self.nout * 3)

        # buffers for:
        # rh_prev_buffer: previous hidden multiply with r;
        # wrc_T_dc: wc_recur.T dot with hcan_delta
        self.rh_prev_buffer = self.be.iobuf(self.out_shape)
        self.rh_prev = get_steps(self.rh_prev_buffer, self.out_shape)
        self.wrc_T_dc = self.be.iobuf(self.nout)

        # Gates: reset: r; update: z; candidate h: hcan
        self.rzhcan_buffer = self.be.iobuf(self.gate_shape)
        self.rzhcan = get_steps(self.rzhcan_buffer, self.gate_shape)
        self.rz = [gate[rz1:rz2] for gate in self.rzhcan]
        self.r = [gate[r1:r2] for gate in self.rzhcan]
        self.z = [gate[z1:z2] for gate in self.rzhcan]
        self.hcan = [gate[c1:c2] for gate in self.rzhcan]

        # the buffer only deals with recurrent inputs to the gates
        self.rzhcan_rec_buffer = self.be.iobuf(self.gate_shape)
        self.rzhcan_rec = get_steps(self.rzhcan_rec_buffer, self.gate_shape)
        self.rz_rec = [gate[rz1:rz2] for gate in self.rzhcan_rec]
        self.hcan_rec = [gate[c1:c2] for gate in self.rzhcan_rec]

        # Pre activation gate deltas
        self.rzhcan_delta_buffer = self.be.iobuf(self.gate_shape)
        self.rzhcan_delta = get_steps(self.rzhcan_delta_buffer, self.gate_shape)
        self.rz_delta = [gate[rz1:rz2] for gate in self.rzhcan_delta]
        self.r_delta = [gate[r1:r2] for gate in self.rzhcan_delta]
        self.z_delta = [gate[z1:z2] for gate in self.rzhcan_delta]
        self.hcan_delta = [gate[c1:c2] for gate in self.rzhcan_delta]

    def init_params(self, shape):
        """
        Initialize params for GRU including weights and biases.
        The weight matrix and bias matrix are concatenated from the weights
        for inputs and weights for recurrent inputs and bias.
        The shape of the weights are (number of inputs + number of outputs +1 )
        by (number of outputs * 3)

        Arguments:
            shape (Tuple): contains number of outputs and number of inputs

        """
        super(GRU, self).init_params(shape)
        (nout, nin) = shape

        # indices for slicing gate buffers
        (rz1, rz2) = (0, nout * 2)
        (c1, c2) = (nout * 2, nout * 3)

        self.Wrz_recur = self.W_recur[rz1:rz2]
        self.Whcan_recur = self.W_recur[c1:c2]

        self.b_rz = self.b[rz1:rz2]
        self.b_hcan = self.b[c1:c2]

        self.dWrz_recur = self.dW_recur[rz1:rz2]
        self.dWhcan_recur = self.dW_recur[c1:c2]

    def fprop(self, inputs, inference=False, init_state=None):
        """
        Apply the forward pass transformation to the input data.  The input data is a list of
            inputs with an element for each time step of model unrolling.

        Arguments:
            inputs (Tensor): input data as 3D tensors, then converted into a list of 2D tensors.
                              The dimension is (input_size, sequence_length * batch_size)
            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: GRU output for each model time step
        """
        self.init_buffers(inputs)

        if self.reset_cells:
            self.h[-1][:] = 0
            self.rz[-1][:] = 0
            self.hcan[-1][:] = 0

        if init_state is not None:
            if init_state.shape != self.h[-1].shape:
                raise ValueError((
                    "init_state shape mismatch.  Expected: {expected}, "
                    "found: {found}."
                ).format(
                    expected=self.h[-1].shape,
                    found=init_state.shape,
                ))

            self.h[-1][:] = init_state
            self.h_prev_bprop[0] = init_state

        self.be.compound_dot(self.W_input, self.x, self.rzhcan_buffer)

        for (h, h_prev, rh_prev, xs, rz, r, z, hcan, rz_rec, hcan_rec, rzhcan) in zip(
                self.h, self.h_prev, self.rh_prev, self.xs, self.rz, self.r,
                self.z, self.hcan, self.rz_rec, self.hcan_rec, self.rzhcan):

            # computes r, z, hcan from recurrents
            self.be.compound_dot(self.Wrz_recur, h_prev, rz_rec)
            rz[:] = self.gate_activation(rz + rz_rec + self.b_rz)
            rh_prev[:] = r * h_prev
            self.be.compound_dot(self.Whcan_recur, rh_prev, hcan_rec)

            hcan[:] = self.activation(hcan_rec + hcan + self.b_hcan)
            h[:] = (1 - z) * h_prev + z * hcan

        self.final_state_buffer[:] = self.h[-1]
        return self.outputs

    def bprop(self, deltas, alpha=1.0, beta=0.0):
        """
        Backpropagation of errors, output delta for previous layer, and calculate the update on
            model params.

        Arguments:
            deltas (Tensor): error tensors for each time step of unrolling
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Attributes:
            dW_input (Tensor): input weight gradients
            dW_recur (Tensor): recurrent weight gradients
            db (Tensor): bias gradients

        Returns:
            Tensor: Backpropagated errors for each time step of model unrolling
        """

        self.dW[:] = 0

        if self.in_deltas is None:
            self.in_deltas = get_steps(deltas, self.out_shape)
            self.prev_in_deltas = self.in_deltas[-1:] + self.in_deltas[:-1]

        params = (self.r, self.z, self.hcan, self.rh_prev, self.h_prev_bprop,
                  self.r_delta, self.z_delta, self.hcan_delta, self.rz_delta, self.rzhcan_delta,
                  self.h_delta, self.in_deltas, self.prev_in_deltas)

        for (r, z, hcan, rh_prev, h_prev, r_delta, z_delta, hcan_delta, rz_delta,
             rzhcan_delta, h_delta, in_deltas, prev_in_deltas) in reversed(list(zip(*params))):

            # hcan_delta
            hcan_delta[:] = self.activation.bprop(hcan) * in_deltas * z
            z_delta[:] = self.gate_activation.bprop(z) * in_deltas * (hcan - h_prev)

            # r_delta
            self.be.compound_dot(self.Whcan_recur.T, hcan_delta, r_delta)
            r_delta[:] = self.gate_activation.bprop(r) * r_delta * h_prev

            # out hidden delta
            h_delta[:] = in_deltas * (1 - z)
            self.be.compound_dot(self.Wrz_recur.T, rz_delta, h_delta, beta=1.0)
            self.be.compound_dot(self.Whcan_recur.T, hcan_delta, self.wrc_T_dc)
            h_delta[:] = h_delta + r * self.wrc_T_dc

            if h_prev != 0:
                self.be.compound_dot(rz_delta, h_prev.T, self.dWrz_recur, beta=1.0)
                self.be.compound_dot(hcan_delta, rh_prev.T, self.dWhcan_recur, beta=1.0)

            prev_in_deltas[:] = prev_in_deltas + h_delta

        # Weight deltas and accumulate
        self.be.compound_dot(self.rzhcan_delta_buffer, self.x.T, self.dW_input)  # batch
        self.db[:] = self.be.sum(self.rzhcan_delta_buffer, axis=1)

        # out deltas
        if self.out_deltas_buffer:  # save a bit of computation
            self.be.compound_dot(self.W_input.T, self.rzhcan_delta_buffer,
                                 self.out_deltas_buffer.reshape(self.nin, -1),
                                 alpha=alpha, beta=beta)

        self.final_hidden_error[:] = self.h_delta[0]
        return self.out_deltas_buffer


class RecurrentOutput(Layer):

    """
    A layer to combine the recurrent layer outputs over time steps. It will
    collapse the time dimension in several ways. These layers do not have
    parameters and do not optimize during training.

    Options derived from this include:
        RecurrentSum, RecurrentMean, RecurrentLast

    """

    def __init__(self, name=None):
        name = name if name else self.classnm
        super(RecurrentOutput, self).__init__(name)
        self.owns_output = self.owns_delta = True
        self.x = None

    def __str__(self):
        return "RecurrentOutput choice %s : (%d, %d) inputs, %d outputs" % (
            self.name, self.nin, self.nsteps, self.nin)

    def configure(self, in_obj):
        """
        Set shape based parameters of this layer given an input tuple, int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer, Tensor or dataset): object that provides shape
                                                           information for layer

        Returns:
            (tuple): shape of output data
        """
        super(RecurrentOutput, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.out_shape = (self.nin, 1)
        return self

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        super(RecurrentOutput, self).set_deltas(delta_buffers)
        self.deltas_buffer = self.deltas
        if self.deltas:
            self.deltas = get_steps(self.deltas_buffer, self.in_shape)
        else:
            self.deltas = []  # for simplifying bprop notation

    def init_buffers(self, inputs):
        """
        Initialize buffers for recurrent internal units and outputs.
        Buffers are initialized as 2D tensors with second dimension being steps * batch_size
        A list of views are created on the buffer for easy manipulation of data
        related to a certain time step

        Arguments:
            inputs (Tensor): input data as 2D tensor. The dimension is
                             (input_size, sequence_length * batch_size)

        """
        if self.x is None or self.x is not inputs:
            self.x = inputs
            self.xs = get_steps(inputs, self.in_shape)


class RecurrentSum(RecurrentOutput):

    """
    A layer that sums over the recurrent layer outputs over time.
    """
    def configure(self, in_obj):
        """
        Set shape based parameters of this layer given an input tuple, int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer, Tensor or dataset): object that provides shape
                                                           information for layer

        Returns:
            (tuple): shape of output data
        """
        super(RecurrentSum, self).configure(in_obj)
        self.sumscale = 1.
        return self

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (int):  (Default value = 0.0)

        Returns:
            Tensor: output data
        """
        self.init_buffers(inputs)
        self.outputs.fill(0)
        for x in self.xs:
            self.outputs[:] = self.outputs + self.sumscale * x
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        for delta in self.deltas:
            delta[:] = alpha * self.sumscale * error + delta * beta
        return self.deltas_buffer


class RecurrentMean(RecurrentSum):

    """
    A layer that gets the averaged recurrent layer outputs over time.
    """
    def configure(self, in_obj):
        """
        Set shape based parameters of this layer given an input tuple, int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer, Tensor or dataset): object that provides shape
                                                           information for layer

        Returns:
            (tuple): shape of output data
        """
        super(RecurrentMean, self).configure(in_obj)
        self.sumscale = 1. / self.nsteps
        return self


class RecurrentLast(RecurrentOutput):

    """
    A layer that only keeps the recurrent layer output at the last time step.
    """

    def fprop(self, inputs, inference=False):
        """
        Passes output from preceding layer on without modification.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.init_buffers(inputs)
        self.outputs[:] = self.xs[-1]
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            # RNN/LSTM layers don't allocate new hidden units delta buffers and they overwrite it
            # while doing bprop. So, init with zeros here.
            self.deltas_buffer.fill(0)
            self.deltas[-1][:] = alpha * error
        return self.deltas_buffer


class BiRNN(ParameterLayer):

    """
    Basic Bi Directional Recurrent layer.

    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model parameters
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless.
        split_inputs (bool): to expect the input coming from the same source of separate
                             sources
        name (str, optional): name to refer to this layer as.

    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (Tensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """

    def __init__(self, output_size, init, init_inner=None, activation=None,
                 reset_cells=False, split_inputs=False, name=None,
                 parallelism="Data"):
        super(BiRNN, self).__init__(init, name, parallelism=parallelism)
        self.in_deltas_f = None
        self.in_deltas_b = None
        self.nout = output_size
        self.h_nout = output_size
        self.output_size = output_size
        self.activation = activation
        self.h_buffer = None
        self.W_input = None
        self.ngates = 1
        self.split_inputs = split_inputs
        self.reset_cells = reset_cells
        self.init_inner = init_inner
        self.x = None

    def __str__(self):
        if self.split_inputs:
            return "BiRNN Layer '%s': (%d inputs) * 2, (%d outputs) * 2, %d steps" % (
                self.name, self.nin // 2, self.nout, self.nsteps)
        else:
            return "BiRNN Layer '%s': %d inputs, (%d outputs) * 2, %d steps" % (
                self.name, self.nin, self.nout, self.nsteps)

    def configure(self, in_obj):
        """
        Set shape based parameters of this layer given an input tuple, int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer, Tensor or dataset): object that provides shape
                                                           information for layer

        Returns:
            (tuple): shape of output data
        """
        super(BiRNN, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)

        self.out_shape = (2 * self.nout, self.nsteps)
        self.gate_shape = (2 * self.nout * self.ngates, self.nsteps)
        self.hidden_shape = (2 * self.nout, self.nsteps + 2)

        if self.split_inputs is True and self.nin % 2 == 1:
            raise ValueError("# inputs units is odd and split_inputs is True ")

        self.o_shape = (self.nout, self.nsteps)
        self.g_shape = (self.nout * self.ngates, self.nsteps)
        self.i_shape = (
            self.nin // 2, self.nsteps) if self.split_inputs else (self.nin, self.nsteps)

        if self.weight_shape is None:
            self.weight_shape = (self.nout, self.nin)
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        assert self.owns_output

        # Allocate a hidden state buffer with an extra time step of padding on either end
        o_shape_pad = (self.o_shape[0], self.o_shape[1] + 2)
        self.h_buffer_all = self.be.iobuf(self.hidden_shape,
                                          shared=shared_outputs,
                                          parallelism=self.parallelism)

        step_size = self.h_buffer_all.shape[1] // (self.nsteps + 2)
        self.outputs = self.h_buffer_all[:, step_size:-step_size]
        super(BiRNN, self).allocate(shared_outputs)

        nout = self.o_shape[0]
        self.h_buffer = self.outputs
        self.out_deltas_buffer = self.deltas

        # forward
        self.h_buffer_f = self.h_buffer[:nout]
        self.h_prev_buffer = self.h_buffer_all[:nout, :-(2 * step_size)]
        self.h_f_last = self.h_prev_buffer[:, :step_size]

        # Slice padded buffer to a list of step tensors, remove padding steps
        self.h_f = get_steps(self.h_buffer_all[:nout, :], o_shape_pad)[1:-1]

        # Slice padded buffer to a list of step tensors, remove the last step and end padding
        self.h_prev = get_steps(self.h_buffer_all[:nout, :], o_shape_pad)[:-2]

        # backward
        self.h_buffer_b = self.h_buffer[nout:]
        self.h_next_buffer = self.h_buffer_all[nout:, (2 * step_size):]
        self.h_b_last = self.h_next_buffer[:, -step_size:]

        # Slice padded buffer to a list of step tensors, remove padding steps
        self.h_b = get_steps(self.h_buffer_all[nout:, :], o_shape_pad)[1:-1]

        # Slice padded buffer to a list of step tensors, remove the beginning padding and 1st step
        self.h_next = get_steps(self.h_buffer_all[nout:, :], o_shape_pad)[2:]

        self.bufs_to_reset = [self.h_buffer]

        # Last time step for in deltas
        self.prev_in_deltas_last = self.be.empty_like(self.h_f[-1])
        self.next_in_deltas_last = self.be.empty_like(self.h_b[0])

        if self.W_input is None:
            self.init_params(self.weight_shape)

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        super(BiRNN, self).set_deltas(delta_buffers)
        self.out_deltas_buffer = self.deltas
        nin = self.i_shape[0]
        if self.split_inputs:
            self.out_deltas_buffer_f = self.out_deltas_buffer[:nin]
            self.out_deltas_buffer_b = self.out_deltas_buffer[nin:]
        else:
            self.out_deltas_buffer_f = self.out_deltas_buffer
            self.out_deltas_buffer_b = self.out_deltas_buffer
        self.out_delta_f = get_steps(self.out_deltas_buffer_f, self.i_shape)
        self.out_delta_b = get_steps(self.out_deltas_buffer_b, self.i_shape)
        self.out_deltas_buffer_f_v = self.out_deltas_buffer_f.reshape(nin, -1)
        self.out_deltas_buffer_b_v = self.out_deltas_buffer_b.reshape(nin, -1)

    def init_buffers(self, inputs):
        """
        Initialize buffers for recurrent internal units and outputs.
        Buffers are initialized as 2D tensors with second dimension being steps * batch_size
        A list of views are created on the buffer for easy manipulation of data
        related to a certain time step

        Arguments:
            inputs (Tensor): input data as 2D tensor. The dimension is
                             (input_size, sequence_length * batch_size)

        """
        if self.x is None or self.x.base is not inputs:
            if self.x:
                for buf in self.bufs_to_reset:
                    buf[:] = 0

            assert inputs.size == self.nin * self.nsteps * self.be.bsz

            self.x = inputs.reshape(self.nin, self.nsteps * self.be.bsz)
            nin = self.i_shape[0]
            if self.split_inputs:
                self.x_f = self.x[:nin]
                self.x_b = self.x[nin:]
            else:
                self.x_f = self.x
                self.x_b = self.x

            self.xs_f = get_steps(self.x_f, self.i_shape)
            self.xs_b = get_steps(self.x_b, self.i_shape)
            self.x_f_v = self.x_f.reshape(nin, -1)
            self.x_b_v = self.x_b.reshape(nin, -1)

    def init_params(self, shape):
        """
        Initialize params for LSTM including weights and biases.
        The weight matrix and bias matrix are concatenated from the weights
        for inputs and weights for recurrent inputs and bias.
        The shape of the weights are (number of inputs + number of outputs +1 )
        by (number of outputs * 4)

        Arguments:
            shape (Tuple): contains number of outputs and number of inputs

        """
        (nout, nin) = (self.o_shape[0], self.i_shape[0])
        self.g_nout = self.ngates * nout
        Wshape = (2 * (nin + nout + 1), self.g_nout)
        doFill = False

        # Weights: input, recurrent, bias
        if self.W is None:
            self.W = self.be.empty(Wshape, **self.get_param_attrs())
            self.dW = self.be.zeros_like(self.W)
            doFill = True
        else:
            # Deserialized weights and empty grad
            assert self.W.shape == Wshape
            assert self.dW.shape == Wshape

        self.W_input_f = self.W[:nin].reshape((self.g_nout, nin))
        self.W_input_b = self.W[nin:2 * nin].reshape((self.g_nout, nin))

        self.W_recur_f = self.W[2 * nin:2 * nin + nout].reshape((self.g_nout, nout))
        self.W_recur_b = self.W[
            2 * nin + nout:2 * nin + 2 * nout].reshape((self.g_nout, nout))

        self.b_f = self.W[-2:-1].reshape((self.g_nout, 1))
        self.b_b = self.W[-1:].reshape((self.g_nout, 1))

        self.dW_input_f = self.dW[:nin].reshape(self.W_input_f.shape)
        self.dW_input_b = self.dW[nin:2 * nin].reshape(self.W_input_b.shape)

        self.dW_recur_f = self.dW[
            2 * nin:2 * nin + nout].reshape(self.W_recur_f.shape)
        self.dW_recur_b = self.dW[
            2 * nin + nout:2 * nin + 2 * nout].reshape(self.W_recur_b.shape)

        self.db_f = self.dW[-2:-1].reshape(self.b_f.shape)
        self.db_b = self.dW[-1:].reshape(self.b_b.shape)

        if doFill:
            gatelist = [g * nout for g in range(0, self.ngates + 1)]
            for wtnm in ('W_input_f', 'W_input_b', 'W_recur_f', 'W_recur_b'):
                wtmat = getattr(self, wtnm)
                if 'W_recur' in wtnm and self.init_inner is not None:
                    initfunc = self.init_inner
                else:
                    initfunc = self.init

                for gb, ge in zip(gatelist[:-1], gatelist[1:]):
                    initfunc.fill(wtmat[gb:ge])
            self.b_f.fill(0.)
            self.b_b.fill(0.)

    def fprop(self, inputs, inference=False):
        """
        Forward propagation of input to bi-directional recurrent layer.

        Arguments:
            inputs (Tensor): input to the model for each time step of
                             unrolling for each input in minibatch
                             shape: (feature_size, sequence_length * batch_size)
                             where:

                             * feature_size: input size
                             * sequence_length: degree of model unrolling
                             * batch_size: number of inputs in each mini-batch

            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: layer output activations for each time step of
                unrolling and for each input in the minibatch
                shape: (output_size, sequence_length * batch_size)
        """

        self.init_buffers(inputs)

        if self.reset_cells:
            self.h_f_last[:] = 0
            self.h_b_last[:] = 0
        else:
            self.h_f_last[:] = self.h_f[-1]
            self.h_b_last[:] = self.h_b[0]

        # Use single multiply for W_input
        self.be.compound_dot(self.W_input_f, self.x_f_v, self.h_buffer_f)
        self.be.compound_dot(self.W_input_b, self.x_b_v, self.h_buffer_b)

        self.be.compound_rnn_unroll_fprop(self.W_recur_f, self.h_prev,
                                          self.h_f, self.h_f, self.b_f,
                                          self.nout, self.nsteps,
                                          self.nsteps,
                                          self.activation, False)

        self.be.compound_rnn_unroll_fprop(self.W_recur_b, self.h_next,
                                          self.h_b, self.h_b, self.b_b,
                                          self.nout, self.nsteps,
                                          self.nsteps,
                                          self.activation, True)

        return self.h_buffer

    def bprop(self, error, alpha=1.0, beta=1.0):
        """
        Backward propagation of errors through bi-directional recurrent layer.

        Arguments:
            deltas (Tensor): tensors containing the errors for
                each step of model unrolling.
                shape: (output_size, sequence_length * batch_size)
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: back propagated errors for each step of time unrolling
                for each mini-batch element
                shape: (input_size, sequence_length * batch_size)
        """

        if self.in_deltas_f is None:
            self.in_deltas_f = get_steps(error[:self.nout], self.o_shape)
            self.prev_in_deltas = [self.prev_in_deltas_last] + self.in_deltas_f[:-1]

        if self.in_deltas_b is None:
            self.in_deltas_b = get_steps(error[self.nout:], self.o_shape)
            self.next_in_deltas = self.in_deltas_b[1:] + [self.next_in_deltas_last]

        self.out_deltas_buffer[:] = 0

        self.be.compound_rnn_unroll_bprop(self.W_recur_f.T, self.prev_in_deltas,
                                          self.in_deltas_f, self.h_f,
                                          self.nout, self.nsteps,
                                          self.nsteps,
                                          self.activation, True)

        self.be.compound_rnn_unroll_bprop(self.W_recur_b.T, self.next_in_deltas,
                                          self.in_deltas_b, self.h_b,
                                          self.nout, self.nsteps,
                                          self.nsteps,
                                          self.activation, False)

        # Update gradients and output deltas for forward pass
        in_deltas_all_f = error[:self.nout]
        in_deltas_cur_f = in_deltas_all_f[:, self.be.bsz:]
        h_prev_all = self.h_buffer_f[:, :-self.be.bsz]
        self.be.compound_dot(in_deltas_cur_f, h_prev_all.T,
                             self.dW_recur_f)

        # Update gradients and output deltas for backward pass
        in_deltas_all_b = error[self.nout:]
        in_deltas_cur_b = in_deltas_all_b[:, :-self.be.bsz]
        h_next_all = self.h_buffer_b[:, self.be.bsz:]
        self.be.compound_dot(in_deltas_cur_b, h_next_all.T, self.dW_recur_b)

        self.be.compound_dot(in_deltas_all_f, self.x_f_v.T, self.dW_input_f)
        self.db_f[:] = self.be.sum(in_deltas_all_f, axis=1)
        if self.out_deltas_buffer_f:
            self.be.compound_dot(self.W_input_f.T, in_deltas_all_f,
                                 self.out_deltas_buffer_f_v,
                                 alpha=alpha, beta=beta)

        self.be.compound_dot(in_deltas_all_b, self.x_b_v.T, self.dW_input_b)
        self.db_b[:] = self.be.sum(in_deltas_all_b, axis=1)
        if self.out_deltas_buffer_b:
            self.be.compound_dot(self.W_input_b.T, in_deltas_all_b,
                                 self.out_deltas_buffer_b_v,
                                 alpha=alpha, beta=beta)

        return self.out_deltas_buffer


class BiSum(Layer):

    """
    A layer to combine the forward and backward passes for bi-directional RNNs
    """

    def __init__(self, name=None):
        super(BiSum, self).__init__(name)
        self.owns_output = self.owns_delta = True
        self.prev_layer = True  # usually it is used to sum the bidirectional output
        self.x = None

    def __str__(self):
        return "Sum layer to combine the forward and backward passes for BiRNN"

    def configure(self, in_obj):
        super(BiSum, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        assert self.nin % 2 == 0, 'The input feature dimension must be mulitple of 2'
        self.out_shape = (self.nin // 2, self.nsteps)
        return self

    def init_buffers(self, inputs):
        """
        Initialize buffers for recurrent internal units and outputs.
        Buffers are initialized as 2D tensors with second dimension being steps * batch_size
        A list of views are created on the buffer for easy manipulation of data
        related to a certain time step

        Arguments:
            inputs (Tensor): input data as 2D tensor. The dimension is
                             (input_size, sequence_length * batch_size)
        """
        if self.x is None or self.x is not inputs:
            self.x = inputs

    def fprop(self, inputs, inference=False):
        self.init_buffers(inputs)
        self.outputs[:] = self.x[:self.nin // 2] + self.x[self.nin // 2:]
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        self.deltas[:self.nin // 2] = error
        self.deltas[self.nin // 2:] = error
        return self.deltas


class BiBNRNN(BiRNN):

    """
    Basic Bi Directional Recurrent layer with sequence-wise batch norm, based on:
    http://arxiv.org/abs/1510.01378

    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model parameters
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless.
        split_inputs (bool): to expect the input coming from the same source of separate
                             sources

    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (TTensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """

    def __init__(self, output_size, init, init_inner=None, activation=None,
                 reset_cells=False, split_inputs=False, name=None):
        super(BiBNRNN, self).__init__(output_size, init, init_inner, activation,
                                      reset_cells, split_inputs, name)
        self.rho = 0.9
        self.eps = 1e-3
        self.beta = None
        self.gamma = None
        self.gmean = None
        self.gvar = None
        self.stats_dtype = np.float64 if self.be.default_dtype is np.float64 else np.float32

    def __str__(self):
        if self.split_inputs:
            return "BiBNRNN Layer '%s': (%d inputs) * 2, (%d outputs) * 2, %d steps" % (
                self.name, self.nin // 2, self.nout, self.nsteps)
        else:
            return "BiBNRNN Layer '%s': %d inputs, (%d outputs) * 2, %d steps" % (
                self.name, self.nin, self.nout, self.nsteps)

    def allocate(self, shared_outputs=None):
        super(BiBNRNN, self).allocate(shared_outputs)
        # self.out_shape is used to allocate hidden buffer, is nout*2
        # self.o_shape is nout
        nout = self.o_shape[0]
        self.h_ff_buffer = self.be.zeros_like(self.outputs)
        self.h_ff_buffer_f = self.h_ff_buffer[:nout]
        self.h_ff_buffer_b = self.h_ff_buffer[nout:]
        self.h_ff_f = get_steps(self.h_ff_buffer_f, self.o_shape)
        self.h_ff_b = get_steps(self.h_ff_buffer_b, self.o_shape)
        self.prev_in_deltas = None
        self.next_in_deltas = None
        # initialize temp buffer for internel calculation, only for MKL backend
        self.ngLayer = self.be.bibnrnn_layer(self.h_buffer_all, self.h_ff_buffer, self.W_recur_f,
                                             self.W_recur_b, self.nsteps, self.nout)

    def init_params(self, shape):
        super(BiBNRNN, self).init_params(shape)
        nf = self.out_shape[0]

        if self.gmean is None:
            self.gmean = self.be.zeros((nf, 1), dtype=self.stats_dtype,
                                       **self.get_param_attrs())
        if self.gvar is None:
            self.gvar = self.be.zeros((nf, 1), dtype=self.stats_dtype,
                                      **self.get_param_attrs())
        self.inf_params = [self.gmean, self.gvar]

        self.xmean = self.be.empty((nf, 1), dtype=self.stats_dtype,
                                   **self.get_param_attrs())
        self.xvar = self.be.empty((nf, 1), dtype=self.stats_dtype,
                                  **self.get_param_attrs())
        self.beta = self.be.zeros((nf, 1), dtype=self.stats_dtype,
                                  **self.get_param_attrs())
        self.gamma = self.be.ones((nf, 1), dtype=self.stats_dtype,
                                  **self.get_param_attrs())

        self.params = [self.beta, self.gamma]
        self.grad_params = [self.be.zeros_like(p) for p in self.params]
        (self.grad_beta, self.grad_gamma) = self.grad_params

        self.allparams = self.params + self.inf_params
        self.states_bn = [[] for gradp in self.grad_params]
        self.plist_bn = [((p, g), s) for p, g, s in zip(self.params,
                                                        self.grad_params,
                                                        self.states_bn)]
        self.plist = [((self.W, self.dW), self.states)] + self.plist_bn

        try:
            self.xmean.auto_reduce = False
            self.xvar.auto_reduce = False
            self.beta.auto_reduce = False
            self.gamma.auto_reduce = False
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception:
            pass

    def set_deltas(self, delta_buffers):
        super(BiBNRNN, self).set_deltas(delta_buffers)

        self.grad_gamma = self.be.zeros_like(self.gamma)
        self.grad_beta = self.be.zeros_like(self.beta)

    def get_params(self):
        return self.plist

    def get_description(self, get_weights=False, keep_states=True):
        serial_dict = super(BiBNRNN, self).get_description(get_weights, keep_states)

        if get_weights:
            for key in ['gmean', 'gvar']:
                serial_dict['params'][key] = getattr(self, key).get()

        return serial_dict

    def fprop(self, inputs, inference=False):
        """
        Forward propagation of input to bi-directional recurrent layer.

        Arguments:
            inputs (Tensor): input to the model for each time step of
                             unrolling for each input in minibatch
                             shape: (feature_size, sequence_length * batch_size)
                             where:

                             * feature_size: input size
                             * sequence_length: degree of model unrolling
                             * batch_size: number of inputs in each mini-batch

            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: layer output activations for each time step of
                unrolling and for each input in the minibatch
                shape: (output_size, sequence_length * batch_size)
        """
        self.init_buffers(inputs)

        if self.reset_cells:
            self.h_f_last[:] = 0
            self.h_b_last[:] = 0
        else:
            self.h_f_last[:] = self.h_f[-1]
            self.h_b_last[:] = self.h_b[0]

        # Use single multiply for W_input
        self.be.compound_dot(self.W_input_f, self.x_f_v, self.h_ff_buffer_f)
        self.be.compound_dot(self.W_input_b, self.x_b_v, self.h_ff_buffer_b)

        self._fprop_bn(self.h_ff_buffer, inference)

        self.be.compound_rnn_unroll_fprop_bibnrnn(self.ngLayer, self.h_buffer_all,
                                                  self.h_ff_buffer, self.W_recur_f, self.h_prev,
                                                  self.h_ff_f, self.h_f, self.b_f, self.W_recur_b,
                                                  self.h_next, self.h_ff_b, self.h_b, self.b_b,
                                                  self.nout, self.nsteps, self.nsteps,
                                                  self.activation)

        return self.h_buffer

    def _fprop_bn(self, inputs, inference=False):
        if inference:
            hhat = (inputs - self.gmean) / self.be.sqrt(self.gvar + self.eps)
            inputs[:] = hhat * self.gamma + self.beta
        else:
            # TODO: remember to change this to be devided by non-padded element
            self.xmean[:] = self.be.mean(inputs, axis=1)
            self.xvar[:] = self.be.var(inputs, axis=1)
            hhat = (inputs - self.xmean) / self.be.sqrt(self.xvar + self.eps)
            inputs[:] = hhat * self.gamma + self.beta

            # update history (will be used for inference)
            self.gmean[:] = self.gmean * self.rho + (1.0 - self.rho) * self.xmean
            self.gvar[:] = self.gvar * self.rho + (1.0 - self.rho) * self.xvar

        return inputs

    def bprop(self, error, alpha=1.0, beta=1.0):
        """
        Backward propagation of errors through bi-directional recurrent layer.

        Arguments:
            deltas (Tensor): tensors containing the errors for
                each step of model unrolling.
                shape: (output_size, sequence_length * batch_size)

        Returns:
            Tensor: back propagated errors for each step of time unrolling
                for each mini-batch element
                shape: (input_size, sequence_length * batch_size)
        """
        if self.in_deltas_f is None:
            self.in_deltas_f = get_steps(error[:self.nout], self.o_shape)
            self.prev_in_deltas = self.in_deltas_f[-1:] + self.in_deltas_f[:-1]

        if self.in_deltas_b is None:
            self.in_deltas_b = get_steps(error[self.nout:], self.o_shape)
            self.next_in_deltas = self.in_deltas_b[1:] + self.in_deltas_b[:1]

        self.out_deltas_buffer[:] = 0

        self.be.compound_rnn_unroll_bprop_bibnrnn(self.ngLayer, error, self.in_deltas_f,
                                                  self.prev_in_deltas, self.in_deltas_b,
                                                  self.next_in_deltas, self.W_recur_f,
                                                  self.W_recur_b, self.h_f, self.h_b,
                                                  self.nout, self.nsteps, self.nsteps,
                                                  self.activation, self.h_buffer_all)

        # Update gradients and output deltas for forward pass
        in_deltas_all_f = error[:self.nout]
        in_deltas_cur_f = in_deltas_all_f[:, self.be.bsz:]
        h_prev_all = self.h_buffer_f[:, :-self.be.bsz]
        self.be.compound_dot(in_deltas_cur_f, h_prev_all.T, self.dW_recur_f)

        # Update gradients and output deltas for backward pass
        in_deltas_all_b = error[self.nout:]
        in_deltas_cur_b = in_deltas_all_b[:, :-self.be.bsz]
        h_next_all = self.h_buffer_b[:, self.be.bsz:]
        self.be.compound_dot(in_deltas_cur_b, h_next_all.T, self.dW_recur_b)

        # bprop through the BN
        self._bprop_bn(error, self.h_ff_buffer)

        # bprop through the ff
        self.be.compound_dot(in_deltas_all_f, self.x_f_v.T, self.dW_input_f)
        self.db_f[:] = self.be.sum(in_deltas_all_f, axis=1)
        if self.out_deltas_buffer_f:
            self.be.compound_dot(self.W_input_f.T, in_deltas_all_f,
                                 self.out_deltas_buffer_f_v,
                                 alpha=alpha, beta=beta)

        self.be.compound_dot(in_deltas_all_b, self.x_b_v.T, self.dW_input_b)
        self.db_b[:] = self.be.sum(in_deltas_all_b, axis=1)
        if self.out_deltas_buffer_b:
            self.be.compound_dot(self.W_input_b.T, in_deltas_all_b,
                                 self.out_deltas_buffer_b_v,
                                 alpha=alpha, beta=beta if self.split_inputs else 1.0)

        return self.out_deltas_buffer

    def _bprop_bn(self, error, input_post_bn):
        hhat = (input_post_bn - self.beta) / self.gamma
        self.grad_gamma[:] = self.be.sum(hhat * error, axis=1)
        self.grad_beta[:] = self.be.sum(error, axis=1)
        htmp = (hhat * self.grad_gamma + self.grad_beta) / float(input_post_bn.shape[1])

        error[:] = self.gamma * (error - htmp) / self.be.sqrt(self.xvar + self.eps)

        return error


class BiLSTM(BiRNN):

    """
    Long Short-Term Memory (LSTM).

    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model parameters
        init_inner (Initializer, optional): Function for initializing the model's recurrent
                                            parameters.  If absent, will default to using same
                                            initializer provided to init.
        activation (Transform): Activation function for the input modulation
        gate_activation (Transform): Activation function for the gates
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless.
        split_inputs (bool): to expect the input coming from the same source of separate
                             sources
        name (str, optional): name to refer to this layer as.

    Attributes:
        x (Tensor): input data as 2D tensor. The dimension is
                    (input_size, sequence_length * batch_size)
        W_input (Tensor): Weights on the input units
                          (out size * 4, input size)
        W_recur (Tensor): Weights on the recursive inputs
                          (out size * 4, out size)
        b (Tensor): Biases (out size * 4 , 1)

    """

    def __init__(self, output_size, init, init_inner=None, activation=None,
                 gate_activation=None, reset_cells=False, split_inputs=False, name=None):
        super(BiLSTM, self).__init__(
            output_size, init, init_inner, activation, reset_cells, split_inputs, name)
        self.gate_activation = gate_activation
        self.ngates = 4  # Input, Output, Forget, Cell
        self.reset_cells = reset_cells

    def __str__(self):
        return "BiLSTM Layer '%s': %d inputs, (%d outputs) * 2, %d steps" % (
               self.name, self.nin, self.nout, self.nsteps)

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(BiLSTM, self).allocate(shared_outputs)
        nout = self.o_shape[0]
        # indices for slicing gate buffers
        (ifo1, ifo2) = (0, self.nout * 3)
        (i1, i2) = (0, self.nout)
        (f1, f2) = (self.nout, self.nout * 2)
        (o1, o2) = (self.nout * 2, self.nout * 3)
        (g1, g2) = (self.nout * 3, self.nout * 4)

        # States: hidden, cell, previous hidden, previous cell, forward cell
        self.c_buffer = self.be.iobuf(self.out_shape)
        self.c_f = get_steps(self.c_buffer[:nout], self.o_shape)
        self.c_prev = self.c_f[-1:] + self.c_f[:-1]
        self.c_prev_bprop = [0] + self.c_f[:-1]
        self.c_b = get_steps(self.c_buffer[nout:], self.o_shape)
        self.c_next = self.c_b[1:] + self.c_b[:1]
        self.c_next_bprop = self.c_b[1:] + [0]

        self.c_act_buffer = self.be.iobuf(self.out_shape)
        self.c_act_f = get_steps(self.c_act_buffer[:nout], self.o_shape)
        self.c_act_b = get_steps(self.c_act_buffer[nout:], self.o_shape)

        # hidden delta

        # Forward Gates : input, forget, output, input modulation
        self.ifog_buffer = self.be.iobuf(self.gate_shape)
        self.ifog_buffer_f = self.ifog_buffer[:self.ngates * nout]
        self.ifog_buffer_b = self.ifog_buffer[self.ngates * nout:]

        self.ifog_f = get_steps(self.ifog_buffer_f, self.g_shape)
        self.ifo_f = [gate[ifo1:ifo2] for gate in self.ifog_f]
        self.i_f = [gate[i1:i2] for gate in self.ifog_f]
        self.f_f = [gate[f1:f2] for gate in self.ifog_f]
        self.o_f = [gate[o1:o2] for gate in self.ifog_f]
        self.g_f = [gate[g1:g2] for gate in self.ifog_f]
        # Backwards gates : input, forget, output, input modulation
        self.ifog_b = get_steps(self.ifog_buffer_b, self.g_shape)
        self.ifo_b = [gate[ifo1:ifo2] for gate in self.ifog_b]
        self.i_b = [gate[i1:i2] for gate in self.ifog_b]
        self.f_b = [gate[f1:f2] for gate in self.ifog_b]
        self.o_b = [gate[o1:o2] for gate in self.ifog_b]
        self.g_b = [gate[g1:g2] for gate in self.ifog_b]

        # State deltas
        self.c_delta_buffer = self.be.iobuf((self.o_shape))
        self.c_delta = get_steps(self.c_delta_buffer, self.o_shape)
        self.c_delta_prev = [None] + self.c_delta[:-1]
        self.c_delta_next = self.c_delta[1:] + [None]

        # Pre activation gate deltas
        self.ifog_delta_buffer = self.be.iobuf(self.g_shape)
        self.ifog_delta = get_steps(self.ifog_delta_buffer, self.g_shape)
        self.i_delta = [gate[i1:i2] for gate in self.ifog_delta]
        self.f_delta = [gate[f1:f2] for gate in self.ifog_delta]
        self.o_delta = [gate[o1:o2] for gate in self.ifog_delta]
        self.g_delta = [gate[g1:g2] for gate in self.ifog_delta]
        self.bufs_to_reset.append(self.c_buffer)

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (list): list of Tensors with one such tensor for each time
                           step of model unrolling.
            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: LSTM output for each model time step
        """
        self.init_buffers(inputs)  # calls the BiRNN init_buffers() code

        if self.reset_cells:
            self.h_f[-1][:] = 0
            self.c_f[-1][:] = 0
            self.h_b[0][:] = 0
            self.c_b[0][:] = 0

        params_f = (self.h_f, self.h_prev, self.xs_f, self.ifog_f, self.ifo_f,
                    self.i_f, self.f_f, self.o_f, self.g_f, self.c_f, self.c_prev, self.c_act_f)
        params_b = (self.h_b, self.h_next, self.xs_b, self.ifog_b, self.ifo_b,
                    self.i_b, self.f_b, self.o_b, self.g_b, self.c_b, self.c_next, self.c_act_b)

        self.be.compound_dot(self.W_input_f, self.x_f, self.ifog_buffer_f)
        self.be.compound_dot(self.W_input_b, self.x_b, self.ifog_buffer_b)

        for (h, h_prev, xs, ifog, ifo, i, f, o, g, c, c_prev, c_act) in zip(*params_f):
            self.be.compound_dot(self.W_input_f, xs, ifog)
            self.be.compound_dot(self.W_recur_f, h_prev, ifog, beta=1.0)
            ifog[:] = ifog + self.b_f

            ifo[:] = self.gate_activation(ifo)
            g[:] = self.activation(g)

            c[:] = f * c_prev + i * g
            c_act[:] = self.activation(c)
            h[:] = o * c_act

        for (h, h_next, xs, ifog, ifo, i, f, o, g, c, c_next, c_act) in \
                reversed(list(zip(*params_b))):
            self.be.compound_dot(self.W_recur_b, h_next, ifog)
            self.be.compound_dot(self.W_input_b, xs, ifog, beta=1.0)
            ifog[:] = ifog + self.b_b

            ifo[:] = self.gate_activation(ifo)
            g[:] = self.activation(g)
            c[:] = f * c_next + i * g
            c_act[:] = self.activation(c)
            h[:] = o * c_act

        return self.h_buffer

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Backpropagation of errors, output delta for previous layer, and
        calculate the update on model params

        Arguments:
            error (list[Tensor]): error tensors for each time step
                                  of unrolling
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: Backpropagated errors for each time step of model unrolling
        """
        self.dW[:] = 0

        if self.in_deltas_f is None:
            self.in_deltas_f = get_steps(error[:self.o_shape[0]], self.o_shape)
            self.prev_in_deltas = self.in_deltas_f[-1:] + self.in_deltas_f[:-1]
            self.ifog_delta_last_steps = self.ifog_delta_buffer[:, self.be.bsz:]
            self.h_first_steps = self.h_buffer_f[:, :-self.be.bsz]
            # h_delta[5] * h[4] + h_delta[4] * h[3] + ... + h_delta[1] * h[0]

        if self.in_deltas_b is None:
            self.in_deltas_b = get_steps(error[self.o_shape[0]:], self.o_shape)
            self.next_in_deltas = self.in_deltas_b[1:] + self.in_deltas_b[:1]
            self.ifog_delta_first_steps = self.ifog_delta_buffer[:,
                                                                 :-self.be.bsz]
            self.h_last_steps = self.h_buffer_b[:, self.be.bsz:]
            # h_delta[0] * h[1] + h_delta[1] * h[2] + ... + h_delta[4] * h[5]

        params_f = (self.in_deltas_f, self.prev_in_deltas,
                    self.i_f, self.f_f, self.o_f, self.g_f,
                    self.ifog_delta, self.i_delta, self.f_delta, self.o_delta, self.g_delta,
                    self.c_delta, self.c_delta_prev, self.c_prev_bprop, self.c_act_f)

        params_b = (self.in_deltas_b, self.next_in_deltas,
                    self.i_b, self.f_b, self.o_b, self.g_b,
                    self.ifog_delta, self.i_delta, self.f_delta, self.o_delta, self.g_delta,
                    self.c_delta, self.c_delta_next, self.c_next_bprop, self.c_act_b)

        # bprop for forward direction connections . Error flow from right to left
        self.c_delta_buffer[:] = 0
        self.ifog_delta_buffer[:] = 0
        self.ifog_delta_f = None
        self.ifog_delta_b = None
        for (in_deltas, prev_in_deltas,
             i, f, o, g,
             ifog_delta, i_delta, f_delta, o_delta, g_delta,
             c_delta, c_delta_prev, c_prev, c_act) in reversed(list(zip(*params_f))):

            # current cell delta
            c_delta[:] = c_delta + \
                self.activation.bprop(c_act) * (o * in_deltas)
            i_delta[:] = self.gate_activation.bprop(i) * c_delta * g
            f_delta[:] = self.gate_activation.bprop(f) * c_delta * c_prev
            o_delta[:] = self.gate_activation.bprop(o) * in_deltas * c_act
            g_delta[:] = self.activation.bprop(g) * c_delta * i

            # bprop the errors to prev_in_delta and c_delta_prev
            self.be.compound_dot(
                self.W_recur_f.T, ifog_delta, prev_in_deltas, beta=1.0)
            if c_delta_prev is not None:
                c_delta_prev[:] = c_delta * f

        # Weight deltas and accumulate
        self.be.compound_dot(
            self.ifog_delta_last_steps, self.h_first_steps.T, self.dW_recur_f)
        self.be.compound_dot(
            self.ifog_delta_buffer, self.x_f.T, self.dW_input_f)
        self.db_f[:] = self.be.sum(self.ifog_delta_buffer, axis=1)
        # out deltas to input units
        if self.out_deltas_buffer:
            self.be.compound_dot(
                self.W_input_f.T, self.ifog_delta_buffer,
                self.out_deltas_buffer_f_v,
                alpha=alpha, beta=beta)

        # bprop for backward direction connections. Error flow from left to right
        self.c_delta_buffer[:] = 0
        self.ifog_delta_buffer[:] = 0
        for (in_deltas, next_in_deltas,
             i, f, o, g,
             ifog_delta, i_delta, f_delta, o_delta, g_delta,
             c_delta, c_delta_next, c_next, c_act) in zip(*params_b):

            # current cell delta
            c_delta[:] = c_delta[:] + \
                self.activation.bprop(c_act) * (o * in_deltas)
            i_delta[:] = self.gate_activation.bprop(i) * c_delta * g
            f_delta[:] = self.gate_activation.bprop(f) * c_delta * c_next
            o_delta[:] = self.gate_activation.bprop(o) * in_deltas * c_act
            g_delta[:] = self.activation.bprop(g) * c_delta * i

            # bprop the errors to next_in_delta and c_next_delta
            self.be.compound_dot(
                self.W_recur_b.T, ifog_delta, next_in_deltas, beta=1.0)
            if c_delta_next is not None:
                c_delta_next[:] = c_delta * f

        # Weight deltas and accumulate
        self.be.compound_dot(
            self.ifog_delta_first_steps, self.h_last_steps.T, self.dW_recur_b)
        self.be.compound_dot(
            self.ifog_delta_buffer, self.x_b.T, self.dW_input_b)
        self.db_b[:] = self.be.sum(self.ifog_delta_buffer, axis=1)
        # out deltas to input units. bprop to the same inputs if
        # split_inputs=False
        if self.out_deltas_buffer:
            self.be.compound_dot(self.W_input_b.T, self.ifog_delta_buffer,
                                 self.out_deltas_buffer_b_v,
                                 alpha=alpha,
                                 beta=beta if self.inputs else 1.0)

        return self.out_deltas_buffer


class DeepBiRNN(list):

    """
    A stacked Bi-directional recurrent layer.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer): Initializer object to use for initializing weights
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless.
        depth(int, optional): Number of layers of BiRNN
    """

    def __init__(self, nout, init, init_inner=None, activation=None,
                 reset_cells=False, depth=1, batch_norm=False, bi_sum=False):
        list.__init__(self)
        if depth <= 0:
            raise ValueError("Depth is <= 0.")

        if bi_sum is True:
            split_inputs_first = False
            split_inputs_second = False
        else:
            split_inputs_first = False
            split_inputs_second = True

        if batch_norm is False:
            self.append(BiRNN(nout, init, init_inner, activation,
                              reset_cells, split_inputs=split_inputs_first))
            if bi_sum:
                self.append(BiSum())
            for i in range(depth-1):
                self.append(BiRNN(nout, init, init_inner, activation,
                            reset_cells, split_inputs=split_inputs_second))
                if bi_sum:
                    self.append(BiSum())
        else:
            self.append(BiBNRNN(nout, init, init_inner, activation,
                                reset_cells, split_inputs=split_inputs_first))
            if bi_sum:
                self.append(BiSum())
            for i in range(depth-1):
                self.append(BiBNRNN(nout, init, init_inner, activation,
                            reset_cells, split_inputs=split_inputs_second))
                if bi_sum:
                    self.append(BiSum())


class DeepBiLSTM(list):

    """
    A stacked Bi-directional LSTM layer.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer): Initializer object to use for initializing weights
        activation (Transform): Activation function for the input modulation
        reset_cells (bool): default to be False to make the layer stateful,
                            set to True to be stateless.
        depth(int, optional): Number of layers of BiRNN

    """

    def __init__(self, nout, init, init_inner=None, activation=None, gate_activation=None,
                 reset_cells=False, depth=1):
        list.__init__(self)
        if depth <= 0:
            raise ValueError("Depth is <= 0.")
        self.append(
            BiLSTM(nout, init, init_inner, activation, gate_activation,
                   reset_cells, split_inputs=False))
        for i in range(depth - 1):
            self.append(
                BiLSTM(nout, init, init_inner, activation, gate_activation,
                       reset_cells, split_inputs=True))
