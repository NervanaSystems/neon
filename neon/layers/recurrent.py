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
from neon.layers.layer import ParameterLayer, Layer


def get_steps(x, shape):
    """
    Convert a (vocab_size, steps * batch_size) array
    into a [(vocab_size, batch_size)] * steps list of views
    """
    steps = shape[1]
    if x is None:
        return [None for step in range(steps)]
    xs = x.reshape(shape + (-1,))
    return [xs[:, step, :] for step in range(steps)]


class Recurrent(ParameterLayer):

    """
    Basic recurrent layer

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

    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (TTensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """

    def __init__(self, output_size, init, init_inner=None, activation=None,
                 reset_cells=False, name="RecurrentLayer"):
        super(Recurrent, self).__init__(init, name)
        self.x = None
        self.in_deltas = None
        self.nout = output_size
        self.h_nout = output_size
        self.activation = activation
        self.outputs = None
        self.W_input = None
        self.ngates = 1
        self.reset_cells = reset_cells
        self.init_inner = init_inner

    def configure(self, in_obj):
        super(Recurrent, self).configure(in_obj)
        (self.nin, self.nsteps) = self.in_shape
        self.out_shape = (self.nout, self.nsteps)
        self.gate_shape = (self.nout * self.ngates, self.nsteps)
        if self.weight_shape is None:
            self.weight_shape = (self.nout, self.nin)
        return self

    def allocate(self, shared_outputs=None):
        super(Recurrent, self).allocate(shared_outputs)
        self.h = get_steps(self.outputs, self.out_shape)
        self.h_prev = self.h[-1:] + self.h[:-1]
        # State deltas
        self.h_delta = get_steps(self.be.iobuf(self.out_shape), self.out_shape)
        self.bufs_to_reset = [self.outputs]

        if self.W_input is None:
            self.init_params(self.weight_shape)

    def set_deltas(self, delta_buffers):
        super(Recurrent, self).set_deltas(delta_buffers)
        self.out_deltas_buffer = self.deltas
        self.out_delta = get_steps(self.out_deltas_buffer, self.in_shape)

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
            if self.x is not None:
                for buf in self.bufs_to_reset:
                    buf[:] = 0
            self.x = inputs
            self.xs = get_steps(inputs, self.in_shape)

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

        if self.W is None:
            self.W = self.be.empty((nout + nin + 1, g_nout))
            self.dW = self.be.zeros_like(self.W)
            doFill = True
        else:
            # Deserialized weights and empty grad
            assert self.W.shape == (nout + nin + 1, g_nout)
            assert self.dW.shape == (nout + nin + 1, g_nout)

        self.W_input = self.W[:nin].reshape((g_nout, nin))
        self.W_recur = self.W[nin:-1].reshape((g_nout, nout))
        self.b = self.W[-1:].reshape((g_nout, 1))

        if doFill:
            gatelist = [g * nout for g in range(0, self.ngates + 1)]
            for wtnm in ('W_input', 'W_recur'):
                wtmat = getattr(self, wtnm)
                if wtnm is 'W_recur' and self.init_inner is not None:
                    initfunc = self.init_inner
                else:
                    initfunc = self.init

                for gb, ge in zip(gatelist[:-1], gatelist[1:]):
                    initfunc.fill(wtmat[gb:ge])
            self.b.fill(0.)

        self.dW_input = self.dW[:nin].reshape(self.W_input.shape)
        self.dW_recur = self.dW[nin:-1].reshape(self.W_recur.shape)
        self.db = self.dW[-1:].reshape(self.b.shape)

    def fprop(self, inputs, inference=False):
        """
        Forward propagation of input to recurrent layer.

        Arguments:
            inputs (Tensor): input to the model for each time step of
                             unrolling for each input in minibatch
                             shape: (vocab_size * steps, batch_size)
                             where:

                             * vocab_size: input size
                             * steps: degree of model unrolling
                             * batch_size: number of inputs in each mini-batch

            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: layer output activations for each time step of
                unrolling and for each input in the minibatch
                shape: (output_size * steps, batch_size)
        """
        self.init_buffers(inputs)

        if self.reset_cells:
            self.h[-1][:] = 0

        # recurrent layer needs a h_prev buffer for bprop
        self.h_prev_bprop = [0] + self.h[:-1]

        for (h, h_prev, xs) in zip(self.h, self.h_prev, self.xs):
            self.be.compound_dot(self.W_input, xs, h)
            self.be.compound_dot(self.W_recur, h_prev, h, beta=1.0)
            h[:] = self.activation(h + self.b)

        return self.outputs

    def bprop(self, deltas, alpha=1.0, beta=0.0):
        """
        Backward propagation of errors through recurrent layer.

        Arguments:
            deltas (Tensor): tensors containing the errors for
                each step of model unrolling.
                shape: (output_size, * steps, batch_size)

        Returns:
            Tensor: back propagated errors for each step of time unrolling
                for each mini-batch element
                shape: (input_size * steps, batch_size)
        """
        self.dW[:] = 0

        if self.in_deltas is None:
            self.in_deltas = get_steps(deltas, self.out_shape)
            self.prev_in_deltas = self.in_deltas[-1:] + self.in_deltas[:-1]

        params = (self.xs, self.h, self.h_prev_bprop, self.h_delta,
                  self.in_deltas, self.prev_in_deltas, self.out_delta)

        for (xs, hs, h_prev, h_delta, in_deltas,
             prev_in_deltas, out_delta) in reversed(zip(*params)):

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

        return self.out_deltas_buffer


class LSTM(Recurrent):

    """
    Long Short-Term Memory (LSTM) layer based on
    Hochreiter, S. and J. Schmidhuber, Neural Computation 9(8): 1735-80 (1997).

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
                 gate_activation=None, reset_cells=False, name="LstmLayer"):
        super(LSTM, self).__init__(output_size, init, init_inner,
                                   activation, reset_cells, name)
        self.gate_activation = gate_activation
        self.ngates = 4  # Input, Output, Forget, Cell

    def allocate(self, shared_outputs=None):
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

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.  The input
            data is a list of inputs with an element for each time step of
            model unrolling.

        Arguments:
            inputs (Tensor): input data as 2D tensors, then being converted into a
                             list of 2D slices

        Returns:
            Tensor: LSTM output for each model time step
        """
        self.init_buffers(inputs)

        if self.reset_cells:
            self.h[-1][:] = 0
            self.c[-1][:] = 0

        params = (self.h, self.h_prev, self.xs, self.ifog, self.ifo,
                  self.i, self.f, self.o, self.g, self.c, self.c_prev, self.c_act)

        for (h, h_prev, xs, ifog, ifo, i, f, o, g, c, c_prev, c_act) in zip(*params):
            self.be.compound_dot(self.W_recur, h_prev, ifog)
            self.be.compound_dot(self.W_input, xs, ifog, beta=1.0)
            ifog[:] = ifog + self.b

            ifo[:] = self.gate_activation(ifo)
            g[:] = self.activation(g)

            c[:] = f * c_prev + i * g
            c_act[:] = self.activation(c)
            h[:] = o * c_act

        return self.outputs

    def bprop(self, deltas, alpha=1.0, beta=0.0):
        """
        Backpropagation of errors, output delta for previous layer, and
        calculate the update on model parmas

        Arguments:
            deltas (list[Tensor]): error tensors for each time step
                of unrolling
            do_acts (bool, optional): Carry out activations.  Defaults to True

        Attributes:
            dW_input (Tensor): input weight gradients
            dW_recur (Tensor): revursive weight gradients
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
             c_delta, c_delta_prev, c_prev, c_act) in reversed(zip(*params)):

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
            self.be.compound_dot(self.W_input.T, self.ifog_delta_buffer, self.out_deltas_buffer,
                                 alpha=alpha, beta=beta)

        return self.out_deltas_buffer


class GRU(Recurrent):

    """
    Implementation of the Gated Recurrent Unit based on [Cho2014]

    - It uses two gates: reset gate (r) and update gate (z)
    - The update gate (z) decides how much the activation is updated
    - The reset gate (r) decides how much to reset (when r = 0) from the previous activation
    - Activation (h_t) is a linear interpolation (by z) between the previous
        activation (h_t-1) and the new candidate activation ( h_can )
    - r and z are compuated the same way, using different weights
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
        activation (Transform): Activiation function for the input modulation
        gate_activation (Transform): Activation function for the gates

    Attributes:
        x (Tensor): Input data tensor (seq len, inp size, batch size)
        W_input (Tensor): Weights on the input units
            (out size * 3, input size)
        W_recur (Tensor): Weights on the recursive inputs
            (out size * 3, out size)
        b (Tensor): Biases (out size * 3 , 1)

    References:

        * Learning phrase representations using rnn encoder-decoder for
          statistical machine translation `[Cho2014]`_
        * Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
          `[Chung2014]`_

    .. _[Cho2014]: http://arxiv.org/abs/1406.1078
    .. _[Chung2014]: http://arxiv.org/pdf/1412.3555v1.pdf
    """

    def __init__(self, output_size, init, init_inner=None, activation=None,
                 gate_activation=None, reset_cells=False, name="GruLayer"):
        super(GRU, self).__init__(output_size, init, init_inner,
                                  activation, reset_cells, name)
        self.gate_activation = gate_activation
        self.ngates = 3  # r, z, hcandidate

    def allocate(self, shared_outputs=None):
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

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.  The input data is a list of
            inputs with an element for each time step of model unrolling.

        Arguments:
            inputs (Tensor): input data as 3D tensors, then converted into a list of 2D tensors

        Returns:
            Tensor: GRU output for each model time step
        """
        self.init_buffers(inputs)

        if self.reset_cells:
            self.h[-1][:] = 0
            self.rz[-1][:] = 0
            self.hcan[-1][:] = 0

        for (h, h_prev, rh_prev, xs, rz, r, z, hcan, rz_rec, hcan_rec, rzhcan) in zip(
                self.h, self.h_prev, self.rh_prev, self.xs, self.rz, self.r,
                self.z, self.hcan, self.rz_rec, self.hcan_rec, self.rzhcan):

            # computes r, z, hcan from inputs
            self.be.compound_dot(self.W_input, xs, rzhcan)

            # computes r, z, hcan from recurrents
            self.be.compound_dot(self.Wrz_recur, h_prev, rz_rec)
            rz[:] = self.gate_activation(rz + rz_rec + self.b_rz)
            rh_prev[:] = r * h_prev
            self.be.compound_dot(self.Whcan_recur, rh_prev, hcan_rec)

            hcan[:] = self.activation(hcan_rec + hcan + self.b_hcan)
            h[:] = (1 - z) * h_prev + z * hcan

        return self.outputs

    def bprop(self, deltas, alpha=1.0, beta=0.0):
        """
        Backpropagation of errors, output delta for previous layer, and calculate the update on
            model parmas

        Arguments:
            deltas (Tensor): error tensors for each time step of unrolling
            do_acts (bool, optional): Carry out activations.  Defaults to True

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
             rzhcan_delta, h_delta, in_deltas, prev_in_deltas) in reversed(zip(*params)):

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
            self.be.compound_dot(self.W_input.T, self.rzhcan_delta_buffer, self.out_deltas_buffer,
                                 alpha=alpha, beta=beta)

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
        super(RecurrentOutput, self).configure(in_obj)  # gives self.in_shape
        (self.nin, self.nsteps) = self.in_shape
        self.out_shape = (self.nin, 1)
        return self

    def set_deltas(self, delta_buffers):
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
    A layer that sums over the recurrent layer outputs over time
    """
    def configure(self, in_obj):
        super(RecurrentSum, self).configure(in_obj)  # gives self.in_shape
        self.sumscale = 1.
        return self

    def fprop(self, inputs, inference=False):
        self.init_buffers(inputs)
        self.outputs.fill(0)
        for x in self.xs:
            self.outputs[:] = self.outputs + self.sumscale * x
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        for delta in self.deltas:
            delta[:] = alpha * self.sumscale * error + delta * beta
        return self.deltas_buffer


class RecurrentMean(RecurrentSum):

    """
    A layer that gets the averaged recurrent layer outputs over time
    """
    def configure(self, in_obj):
        super(RecurrentMean, self).configure(in_obj)  # gives self.in_shape
        self.sumscale = 1. / self.nsteps
        return self


class RecurrentLast(RecurrentOutput):

    """
    A layer that only keeps the recurrent layer output at the last time step
    """

    def fprop(self, inputs, inference=False):
        self.init_buffers(inputs)
        self.outputs[:] = self.xs[-1]
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        if self.deltas:
            # RNN/LSTM layers don't allocate new hidden units delta buffers and they overwrite it
            # while doing bprop. So, init with zeros here.
            self.deltas_buffer.fill(0)
            self.deltas[-1][:] = alpha * error
        return self.deltas_buffer


class BiRNN(ParameterLayer):

    """
    Basic Bi Directional Recurrent layer

    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model parameters
        activation (Transform): Activation function for the input modulation

    Attributes:
        W_input (Tensor): weights from inputs to output units
            (input_size, output_size)
        W_recur (TTensor): weights for recurrent connections
            (output_size, output_size)
        b (Tensor): Biases on output units (output_size, 1)
    """

    def __init__(self, output_size, init, activation, split_inputs=False, name="BiRNNLayer"):
        super(BiRNN, self).__init__(init, name)
        self.in_deltas_f = None
        self.in_deltas_b = None
        self.nout = output_size
        self.h_nout = output_size
        self.activation = activation
        self.h_buffer = None
        self.W_input = None
        self.ngates = 1
        self.split_inputs = split_inputs

    def __str__(self):
        if self.split_inputs:
            return "BiRNN Layer '%s': (%d inputs) * 2, (%d outputs) * 2, %d steps" % (
                self.name, self.nin/2, self.nout, self.nsteps)
        else:
            return "BiRNN Layer '%s': %d inputs, (%d outputs) * 2, %d steps" % (
                self.name, self.nin, self.nout, self.nsteps)

    def configure(self, in_obj):
        super(BiRNN, self).configure(in_obj)
        (self.nin, self.nsteps) = self.in_shape

        self.out_shape = (2 * self.nout, self.nsteps)
        self.gate_shape = (2 * self.nout * self.ngates, self.nsteps)

        if self.split_inputs is True and self.nin % 2 == 1:
            raise ValueError("# inputs units is odd and split_inputs is True ")

        self.o_shape = (self.nout, self.nsteps)
        self.g_shape = (self.nout * self.ngates, self.nsteps)
        self.i_shape = (
            self.nin/2, self.nsteps) if self.split_inputs else (self.nin, self.nsteps)

        if self.weight_shape is None:
            self.weight_shape = (self.nout, self.nin)
        return self

    def allocate(self, shared_outputs=None):
        super(BiRNN, self).allocate(shared_outputs)

        nout = self.o_shape[0]
        self.h_buffer = self.outputs
        self.out_deltas_buffer = self.deltas

        # forward
        self.h_buffer_f = self.h_buffer[:nout]
        self.h_f = get_steps(self.h_buffer_f, self.o_shape)
        self.h_prev = self.h_f[-1:] + self.h_f[:-1]

        # backward
        self.h_buffer_b = self.h_buffer[nout:]
        self.h_b = get_steps(self.h_buffer_b, self.o_shape)
        self.h_next = self.h_b[1:] + self.h_b[:1]
        self.bufs_to_reset = [self.h_buffer]

        if self.W_input is None:
            self.init_params(self.weight_shape)

    def set_deltas(self, delta_buffers):
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
        if self.split_inputs:
            # set in 2nd layer. x_f connects to h_forward and x_b connects to
            # h_b
            self.x_f = inputs[:self.i_shape[0]]
            self.x_b = inputs[self.i_shape[0]:]
        else:
            self.x_f = inputs
            self.x_b = inputs
        self.xs_f = get_steps(self.x_f, self.i_shape)
        self.xs_b = get_steps(self.x_b, self.i_shape)
        for buf in self.bufs_to_reset:
            buf[:] = 0

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
        Wshape = (2*(nin+nout+1), self.g_nout)
        # Weights: input, recurrent, bias
        if self.W is None:
            self.W = self.be.empty(Wshape)
            self.dW = self.be.zeros_like(self.W)
        else:
            # Deserialized weights and empty grad
            assert self.W.shape == Wshape
            assert self.dW.shape == Wshape

        self.W_input_f = self.W[:nin].reshape((self.g_nout, nin))
        self.W_input_b = self.W[nin:2*nin].reshape((self.g_nout, nin))

        self.W_recur_f = self.W[2*nin:2*nin+nout].reshape((self.g_nout, nout))
        self.W_recur_b = self.W[
            2*nin+nout:2*nin+2*nout].reshape((self.g_nout, nout))

        self.b_f = self.W[-2:-1].reshape((self.g_nout, 1))
        self.b_b = self.W[-1:].reshape((self.g_nout, 1))

        self.dW_input_f = self.dW[:nin].reshape(self.W_input_f.shape)
        self.dW_input_b = self.dW[nin:2*nin].reshape(self.W_input_b.shape)

        self.dW_recur_f = self.dW[
            2*nin:2*nin+nout].reshape(self.W_recur_f.shape)
        self.dW_recur_b = self.dW[
            2*nin+nout:2*nin+2*nout].reshape(self.W_recur_b.shape)

        self.db_f = self.dW[-2:-1].reshape(self.b_f.shape)
        self.db_b = self.dW[-1:].reshape(self.b_b.shape)
        weights = (self.W_input_f, self.W_input_b,
                   self.W_recur_f, self.W_recur_b)

        # initialize the baises to zero
        self.b_f[:] = 0
        self.b_b[:] = 0
        for w in weights:
            self.init.fill(w)

    def fprop(self, inputs, inference=False):
        """
        Forward propagation of input to bi-directional recurrent layer.

        Arguments:
            inputs (Tensor): input to the model for each time step of
                             unrolling for each input in minibatch
                             shape: (vocab_size * steps, batch_size)
                             where:

                             * vocab_size: input size
                             * steps: degree of model unrolling
                             * batch_size: number of inputs in each mini-batch

            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: layer output activations for each time step of
                unrolling and for each input in the minibatch
                shape: (output_size * steps, batch_size)
        """
        self.init_buffers(inputs)

        # recurrent layer needs a h_prev buffer for bprop
        self.h_prev_bprop = [None] + self.h_f[:-1]
        self.h_next_bprop = self.h_b[1:] + [None]
        for (h, h_prev, xs) in zip(self.h_f, self.h_prev, self.xs_f):
            self.be.compound_dot(self.W_input_f, xs, h)
            self.be.compound_dot(self.W_recur_f, h_prev, h, beta=1.0)
            h[:] = self.activation(h + self.b_f)

        for (h, h_next, xs) in reversed(zip(self.h_b, self.h_next, self.xs_b)):
            self.be.compound_dot(self.W_input_b, xs, h)
            self.be.compound_dot(self.W_recur_b, h_next, h, beta=1.0)
            h[:] = self.activation(h + self.b_b)

        return self.h_buffer

    def bprop(self, error, alpha=1.0, beta=1.0):
        """
        Backward propagation of errors through recurrent layer.

        Arguments:
            deltas (Tensor): tensors containing the errors for
                each step of model unrolling.
                shape: (output_size, * steps, batch_size)

        Returns:
            Tensor: back propagated errors for each step of time unrolling
                for each mini-batch element
                shape: (input_size * steps, batch_size)
        """
        self.dW[:] = 0

        if self.in_deltas_f is None:
            self.in_deltas_f = get_steps(error[:self.nout], self.o_shape)
            self.prev_in_deltas = self.in_deltas_f[-1:] + self.in_deltas_f[:-1]

        if self.in_deltas_b is None:
            self.in_deltas_b = get_steps(error[self.nout:], self.o_shape)
            self.next_in_deltas = self.in_deltas_b[1:] + self.in_deltas_b[:1]

        params_f = (self.xs_f, self.h_f, self.h_prev_bprop, self.in_deltas_f,
                    self.prev_in_deltas, self.out_delta_f)

        params_b = (self.xs_b, self.h_b, self.h_next_bprop, self.in_deltas_b,
                    self.next_in_deltas, self.out_delta_b)

        self.out_deltas_buffer[:] = 0
        # errors propogate from right to left
        for (xs, hs, h_prev, in_deltas,
             prev_in_deltas, out_delta) in reversed(zip(*params_f)):

            in_deltas[:] = self.activation.bprop(hs) * in_deltas
            self.be.compound_dot(
                self.W_recur_f.T, in_deltas, prev_in_deltas, beta=1.0)
            if h_prev is not None:
                self.be.compound_dot(
                    in_deltas, h_prev.T, self.dW_recur_f, beta=1.0)
            self.be.compound_dot(in_deltas, xs.T, self.dW_input_f, beta=1.0)
            self.db_f[:] = self.db_f + self.be.sum(in_deltas, axis=1)
            if out_delta:
                self.be.compound_dot(self.W_input_f.T, in_deltas, out_delta,
                                     alpha=alpha, beta=beta)

        # errors propogate left to right
        for (xs, hs, h_next, in_deltas,
             next_in_deltas, out_delta) in zip(*params_b):

            in_deltas[:] = self.activation.bprop(hs) * in_deltas
            self.be.compound_dot(
                self.W_recur_b.T, in_deltas, next_in_deltas, beta=1.0)
            if h_next is not None:
                self.be.compound_dot(
                    in_deltas, h_next.T, self.dW_recur_b, beta=1.0)
            self.be.compound_dot(in_deltas, xs.T, self.dW_input_b, beta=1.0)
            self.db_b[:] = self.db_b + self.be.sum(in_deltas, axis=1)
            if out_delta:
                # propogate the errors to same input if split_inputs is False
                self.be.compound_dot(self.W_input_b.T, in_deltas, out_delta,
                                     alpha=alpha, beta=beta if self.split_inputs else 1.0)

        return self.out_deltas_buffer


class BiLSTM(BiRNN):

    """
    Long Short-Term Memory (LSTM)
    Arguments:
        output_size (int): Number of hidden/output units
        init (Initializer): Function for initializing the model parameters
        activation (Transform): Activation function for the input modulation
        gate_activation (Transform): Activation function for the gates

    Attributes:
        x (Tensor): input data as 2D tensor. The dimension is
                    (input_size, sequence_length * batch_size)
        W_input (Tensor): Weights on the input units
            (out size * 4, input size)
        W_recur (Tensor): Weights on the recursive inputs
            (out size * 4, out size)
        b (Tensor): Biases (out size * 4 , 1)
    """

    def __init__(self, output_size, init, activation, gate_activation,
                 reset_cells=False, split_inputs=False, name="BiLstmLayer"):
        super(BiLSTM, self).__init__(
            output_size, init, activation, split_inputs, name)
        self.gate_activation = gate_activation
        self.ngates = 4  # Input, Output, Forget, Cell
        self.reset_cells = reset_cells

    def __str__(self):
        return "BiLSTM Layer '%s': %d inputs, (%d outputs) * 2, %d steps" % (
               self.name, self.nin, self.nout, self.nsteps)

    def allocate(self, shared_outputs=None):
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
        self.ifog_f = get_steps(
            self.ifog_buffer[:self.ngates * nout], self.g_shape)
        self.ifo_f = [gate[ifo1:ifo2] for gate in self.ifog_f]
        self.i_f = [gate[i1:i2] for gate in self.ifog_f]
        self.f_f = [gate[f1:f2] for gate in self.ifog_f]
        self.o_f = [gate[o1:o2] for gate in self.ifog_f]
        self.g_f = [gate[g1:g2] for gate in self.ifog_f]
        # Backwards gates : input, forget, output, input modulation
        self.ifog_b = get_steps(
            self.ifog_buffer[self.ngates * nout:], self.g_shape)
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

    def init_params(self, shape):
        super(BiLSTM, self).init_params(shape)
        (i1, i2) = (0, self.nout)
        (f1, f2) = (self.nout, self.nout * 2)
        (o1, o2) = (self.nout * 2, self.nout * 3)
        (g1, g2) = (self.nout * 3, self.nout * 4)

        ranges = [(i1, i2), (f1, f2), (o1, o2), (g1, g2)]
        weights = [
            self.W_input_f, self.W_input_b, self.W_recur_f, self.W_recur_b]
        for ((r1, r2), weight) in zip(ranges, weights):
            self.init.fill(weight[r1:r2])

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.  The input
            data is a list of inputs with an element for each time step of
            model unrolling.

        Returns :
            Tensor: LSTM output for each model time step
        """
        self.init_buffers(inputs)  # calls the BiRNN init_buffers() code

        if self.reset_cells:
            self.h_f[-1][:] = 0
            self.c_f[-1][:] = 0
            self.h_b[0][:] = 0
            self.c_b[0][:] = 0

        params_f = (self.h_f, self.h_prev, self.xs_f, self.ifog_f, self.ifo_f,
                    self.i_f, self.f_f, self.o_f,  self.g_f, self.c_f, self.c_prev, self.c_act_f)
        params_b = (self.h_b, self.h_next, self.xs_b, self.ifog_b, self.ifo_b,
                    self.i_b, self.f_b, self.o_b, self.g_b, self.c_b, self.c_next, self.c_act_b)

        for (h, h_prev, xs, ifog, ifo, i, f, o, g, c, c_prev, c_act) in zip(*params_f):
            self.be.compound_dot(self.W_recur_f, h_prev, ifog)
            self.be.compound_dot(self.W_input_f, xs, ifog, beta=1.0)
            ifog[:] = ifog + self.b_f

            ifo[:] = self.gate_activation(ifo)
            g[:] = self.activation(g)

            c[:] = f * c_prev + i * g
            c_act[:] = self.activation(c)
            h[:] = o * c_act

        for (h, h_next, xs, ifog, ifo, i, f, o, g, c, c_next, c_act) in reversed(zip(*params_b)):
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
        calculate the update on model parmas

        Arguments:
            deltas (list[Tensor]): error tensors for each time step
                of unrolling
            do_acts (bool, optional): Carry out activations.  Defaults to True


        Returns:
            Tensor: Backpropagated errors for each time step
                of model unrolling
        """
        self.dW[:] = 0

        if self.in_deltas_f is None:
            self.in_deltas_f = get_steps(error[:self.o_shape[0]], self.o_shape)
            self.prev_in_deltas = self.in_deltas_f[-1:] + self.in_deltas_f[:-1]
            self.ifog_delta_last_steps = self.ifog_delta_buffer[
                :, self.be.bsz:]
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

        """  bprop for forward direction connections . Error flow from right to left """
        self.c_delta_buffer[:] = 0
        self.ifog_delta_buffer[:] = 0
        self.ifog_delta_f = None
        self.ifog_delta_b = None
        for (in_deltas, prev_in_deltas,
             i, f, o, g,
             ifog_delta, i_delta, f_delta, o_delta, g_delta,
             c_delta, c_delta_prev, c_prev, c_act) in reversed(zip(*params_f)):

            # current cell delta
            c_delta[:] = c_delta + \
                self.activation.bprop(c_act) * (o * in_deltas)
            i_delta[:] = self.gate_activation.bprop(i) * c_delta * g
            f_delta[:] = self.gate_activation.bprop(f) * c_delta * c_prev
            o_delta[:] = self.gate_activation.bprop(o) * in_deltas * c_act
            g_delta[:] = self.activation.bprop(g) * c_delta * i

            # bprop the erros to prev_in_delta and c_delta_prev
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
                self.W_input_f.T, self.ifog_delta_buffer, self.out_deltas_buffer_f,
                alpha=alpha, beta=beta)

        """  bprop for backward direction connections. Error flow from left to right """
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
                                 self.out_deltas_buffer_b, alpha=alpha,
                                 beta=beta if self.inputs else 1.0)
        return self.out_deltas_buffer


class DeepBiRNN(list):

    """
    A stacked Bi-directional recurrent layer

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer): Initializer object to use for initializing weights
        activation (Transform): Activation function for the input modulation
        depth(int, optional): Number of layers of BiRNN

    """

    def __init__(self, nout, init, activation, depth=1):
        list.__init__(self)
        if depth <= 0:
            raise ValueError("Depth is <= 0.")

        self.append(BiRNN(nout, init, activation, split_inputs=False))
        for i in range(depth-1):
            self.append(BiRNN(nout, init, activation, split_inputs=True))


class DeepBiLSTM(list):

    """
    A stacked Bi-directional LSTM layer

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer): Initializer object to use for initializing weights
        activation (Transform): Activation function for the input modulation
        depth(int, optional): Number of layers of BiRNN

    """

    def __init__(self, nout, init, activation, gate_activation, reset_cells=False, depth=1):
        list.__init__(self)
        if depth <= 0:
            raise ValueError("Depth is <= 0.")
        self.append(
            BiLSTM(nout, init, activation, gate_activation, reset_cells, split_inputs=False))
        for i in range(depth-1):
            self.append(
                BiLSTM(nout, init, activation, gate_activation, reset_cells, split_inputs=True))
