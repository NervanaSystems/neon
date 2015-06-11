# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
import logging
import numpy as np
from neon.layers.layer import WeightLayer, CostLayer
from neon.util.compat import range
from neon.util.param import req_param, opt_param
logger = logging.getLogger(__name__)


class RecurrentLayer(WeightLayer):
    """
    WeightLayer baseclass for recurrent networks. Provides buffer allocation
    for pre-activations and outputs at different time steps.
    """

    def allocate_output_bufs(self):
        make_zbuf = self.backend.zeros
        opt_param(self, ['out_shape'], (self.nout, self.batch_size))
        self.output = make_zbuf(self.out_shape, self.output_dtype)

        self.pre_act = self.activation.pre_act_buffer(self.backend,
                                                      self.output,
                                                      self.pre_act_dtype)

        # TODO: Get rid of output and pre_act. But they seem to be used in the
        # cost to set a buffer size.
        self.pre_act_list = [self.pre_act] + \
                            [make_zbuf(self.out_shape, self.pre_act_dtype)
                             for k in range(1, self.unrolls)]
        self.output_list = [self.output] + \
                           [make_zbuf(self.out_shape, self.output_dtype)
                            for k in range(1, self.unrolls)]

    def set_deltas_buf(self, delta_pool, offset):
        # create deltas buffer no matter what position relative to the data
        # layer we are. In the RNN even the first layer needs deltas.
        self.deltas = self.backend.zeros(self.delta_shape, self.deltas_dtype)

    def grad_log(self, ng, val):
        logger.info("%s.bprop inc '%s' by %f", self.__class__.__name__, ng,
                    val.asnumpyarray())


class RecurrentCostLayer(CostLayer):
    """
    CostLayer for RNN. get_cost is adapted to use the last time step of
    targets.
    """

    def __init__(self, **kwargs):
        self.is_cost = True
        self.nout = 1
        super(RecurrentCostLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        super(RecurrentCostLayer, self).initialize(kwargs)
        req_param(self, ['cost', 'ref_layer'])
        opt_param(self, ['ref_label'], 'targets')
        self.targets = None
        self.cost.olayer = self.prev_layer
        self.cost.initialize(kwargs)
        self.deltas = self.cost.get_deltabuf()

    def __str__(self):
        return ("Layer {lyr_nm}: {nin} nodes, {cost_nm} cost_fn, "
                "utilizing {be_nm} backend\n\t".format
                (lyr_nm=self.name, nin=self.nin,
                 cost_nm=self.cost.__class__.__name__,
                 be_nm=self.backend.__class__.__name__))

    def fprop(self, inputs):
        pass

    def bprop(self, error, tau):
        # Since self.deltas already pointing to destination of act gradient
        # we just have to scale by mini-batch size
        if self.ref_layer is not None:
            self.targets = getattr(self.ref_layer, self.ref_label)

        self.cost.apply_derivative(self.targets[tau])
        self.backend.divide(self.deltas, self.batch_size, out=self.deltas)

    def set_targets(self):
        # used for num_grad. normally this is done inside bprop
        self.targets = getattr(self.ref_layer, self.ref_label)

    def get_cost(self):
        result = self.cost.apply_function(self.targets[-1])
        return self.backend.divide(result, self.batch_size, result)


class RecurrentOutputLayer(RecurrentLayer):
    """
    Connects to a RecurrentHiddenLayer and computes output at time step tau.
    It stores outputs and preactivations in a list indexed by tau.
    """

    def initialize(self, kwargs):
        req_param(self, ['nout', 'nin', 'unrolls', 'activation'])
        super(RecurrentOutputLayer, self).initialize(kwargs)
        self.weight_shape = (self.nout, self.nin)
        self.bias_shape = (self.nout, 1)

        opt_param(self, ['delta_shape'], (self.nin, self.batch_size))  # moved
        self.allocate_output_bufs()
        self.allocate_param_bufs()

    def allocate_output_bufs(self):
        make_zbuf = self.backend.zeros
        super(RecurrentOutputLayer, self).allocate_output_bufs()
        # super allocate will set the correct sizes for pre_act, output, berr
        self.temp_out = make_zbuf(self.weight_shape, self.weight_dtype)

    def fprop(self, inputs, tau):
        self.backend.fprop_fc(self.pre_act_list[tau], inputs, self.weights)
        self.activation.fprop_func(self.backend,
                                   self.pre_act_list[tau],
                                   self.output_list[tau])

    def bprop(self, error, tau, numgrad=False):
        inputs = self.prev_layer.output_list[tau - 1]
        if self.skip_act is False:
            self.backend.multiply(error, self.pre_act_list[tau - 1], error)

        self.backend.bprop_fc(self.deltas, self.weights, error)
        self.backend.update_fc(out=self.temp_out, inputs=inputs, deltas=error)

        if numgrad and (numgrad['name'] == "output"):
            self.grad_log(numgrad['name'], self.temp_out[numgrad['x'],
                                                         numgrad['v']])

        self.backend.add(self.weight_updates, self.temp_out,
                         self.weight_updates)


class RecurrentHiddenLayer(RecurrentLayer):
    """
    RecurrentHiddenLayer has two sets of weights, the connections received from
    the input `weights` and the recurrent connections `weights_rec`.
    """

    def initialize(self, kwargs):
        req_param(self, ['weight_init_rec'])
        self.weight_rec_shape = (self.nout, self.nout)
        super(RecurrentHiddenLayer, self).initialize(kwargs)

        self.weight_shape = (self.nout, self.nin)
        self.bias_shape = (self.nout, 1)
        opt_param(self, ['delta_shape'], (self.nout, self.batch_size))
        self.allocate_output_bufs()
        self.allocate_param_bufs()

    def allocate_output_bufs(self):
        make_zbuf = self.backend.zeros
        super(RecurrentHiddenLayer, self).allocate_output_bufs()

        # these buffers are specific to RHL:
        # might want self.Wx_up_part=temp_out, to save a buffer.
        self.Wx_up_part = make_zbuf(self.weight_shape, self.weight_dtype)
        self.Wh_up_part = make_zbuf(self.weight_rec_shape, self.weight_dtype)
        # Extra temp buffers z[0]=w*x and z[1]=w*input.
        self.preact_rec = make_zbuf(self.out_shape, self.weight_dtype)
        self.preact_in = make_zbuf(self.out_shape, self.weight_dtype)

    def allocate_param_bufs(self):
        super(RecurrentHiddenLayer, self).allocate_param_bufs()
        weight_gen = self.weight_init_rec.generate
        self.weights_rec = weight_gen(self.weight_rec_shape, self.weight_dtype)
        self.Wh_updates = self.backend.zeros(self.weight_rec_shape,
                                             self.updates_dtype)
        self.params.append(self.weights_rec)
        self.updates.append(self.Wh_updates)

        # Not ideal, since we just allocated this in the parent function, but
        # we can change the calling order later
        self.learning_rule.allocate_state(self.updates)

    def fprop(self, y, c, inputs, tau):
        self.backend.fprop_fc(out=self.preact_rec,
                              inputs=y,
                              weights=self.weights_rec)
        self.backend.fprop_fc(out=self.preact_in,
                              inputs=inputs,
                              weights=self.weights)
        self.backend.add(left=self.preact_rec,
                         right=self.preact_in,
                         out=self.pre_act_list[tau])
        self.activation.fprop_func(self.backend,
                                   self.pre_act_list[tau],
                                   self.output_list[tau])

    def bprop(self, error, error_c, tau, numgrad=False):
        """
        This function has been refactored:
        [done] remove duplicate code
        [done] remove the loop altogether.
        [todo] If the if statement can't be supported, revert to duplicated
        code
        """

        if self.prev_layer.is_data:
            inputs = self.prev_layer.output[tau]
        else:
            inputs = self.prev_layer.output_list[tau]

        if self.skip_act is False:
            self.backend.multiply(error, self.pre_act_list[tau], out=error)

        # input weight update (apply curr. delta)
        self.backend.update_fc(out=self.Wx_up_part,
                               inputs=inputs,
                               deltas=error)
        self.backend.add(self.weight_updates, self.Wx_up_part,
                         self.weight_updates)

        if (tau > 0):
            # recurrent weight update (apply prev. delta)
            self.backend.update_fc(out=self.Wh_up_part,
                                   inputs=self.output_list[tau - 1],
                                   deltas=error)
            self.backend.add(self.Wh_updates, self.Wh_up_part, self.Wh_updates)

            self.backend.bprop_fc(out=self.deltas,
                                  weights=self.weights_rec,
                                  deltas=error)
        if numgrad and (numgrad['name'] == "input"):
            self.grad_log(numgrad['name'], self.Wx_up_part[numgrad['x'],
                          numgrad['y']])
        if numgrad and (numgrad['name'] == "rec"):
            self.grad_log(numgrad['name'], self.Wh_up_part[numgrad['x'],
                          numgrad['z']])


class RecurrentLSTMLayer(RecurrentLayer):
    """
    Hidden layer with LSTM gates. Has the same interface as
    RecurrentHiddenLayer.
    """

    def initialize(self, kwargs):
        req_param(self, ['weight_init_rec'])
        self.weight_rec_shape = (self.nout, self.nout)
        super(RecurrentLSTMLayer, self).initialize(kwargs)
        self.weight_shape = (self.nout, self.nin)
        self.bias_shape = (self.nout, 1)

        opt_param(self, ['delta_shape'], (self.nout, self.batch_size))
        self.allocate_output_bufs()
        self.allocate_param_bufs()

    def allocate_output_bufs(self):
        """
        all the activations, deltas and temp buffers live here::

            activations:       {i,f,o,g}_t
            preactivations:    net_{i,f,o,g}
            gate level deltas: self.d_dh1{i,f,o,c}
            cell level deltas: self.dc_d_dh1{i,f,c}
            final deltas:      errs{hh, hc, ch, cc}
            cell state:        c_t, c_phi, c_phip

        """
        super(RecurrentLSTMLayer, self).allocate_output_bufs()

        # things that are not initalized by the super class
        be = self.backend
        net_sze = (self.nout, self.batch_size)  # tuple with activation size.

        # buffers for gate activation
        for a in ['i', 'f', 'o', 'g']:
            setattr(self, a + '_t',
                    [be.zeros(net_sze) for k in range(self.unrolls)])
            setattr(self, 'net_' + a,
                    [be.zeros(net_sze) for k in range(self.unrolls)])

        # outputs: pre-allocate for d{i,f,o,c}_dh1
        self.d_dh1 = {gateid: be.zeros(net_sze) for
                      gateid in ['i', 'f', 'o', 'c']}
        self.dc_d_dh1 = {gateid: be.zeros(net_sze) for
                         gateid in ['i', 'f', 'c']}
        self.errs = {hcval: be.zeros(net_sze) for
                     hcval in ['hh', 'hc', 'ch', 'cc']}
        self.gatedic = {}
        self.gatedic_u = {}

        # buffers for cell and output
        self.c_t = [be.zeros(net_sze) for k in range(self.unrolls)]
        self.c_phi = [be.zeros(net_sze) for k in range(self.unrolls)]
        self.c_phip = [be.zeros(net_sze) for k in range(self.unrolls)]
        self.output_list = [be.zeros(net_sze) for k in range(self.unrolls)]

        # pre-allocate preactivation buffers
        self.temp_x = [be.zeros(net_sze) for k in range(self.unrolls)]
        self.temp_h = [be.zeros(net_sze) for k in range(self.unrolls)]

        # pre-allocate derivative buffers
        self.dh_dwx_buf = be.zeros((self.nout, self.nin))
        self.dh_dwh_buf = be.zeros((self.nout, self.nout))

        self.delta_buf = be.zeros(net_sze)
        self.bsum_buf = be.zeros((self.nout, 1))

        # This quantity seems to be computed repeatedly
        # error_h * self.o_t[tau] * self.c_phip[tau]
        self.eh_ot_cphip = be.zeros(net_sze)

        # error buffers
        self.deltas = be.zeros((self.nout, self.batch_size))
        self.celtas = be.zeros((self.nout, self.batch_size))

        # temp buffer for numerical gradient
        self.temp_t = 0

    def allocate_param_bufs(self):
        """
        params and updates are dictionaries passed to the learning rule.
        self.params is the collection of 12 tensors::

            W_{i,f,o,c}x
            W_{i,f,o,c}h
            b_{i,f,o,c}

        self.updates is a similar collection of 12 tensors::

            W_{i,f,o,c}x_updates
            W_{i,f,o,c}h_updates
            b_{i,f,o,c}_updates

        these are further subdivided into per gate dictionaries containing
        ``gatedic['c'] = [Wcx, Wch, b_c, net_g, g_t]``
        """
        super(RecurrentLSTMLayer, self).allocate_param_bufs()

        be = self.backend

        # set params (self.Wix = gen_weights) -- gen_weights is now
        for a in ['i', 'f', 'o', 'c']:
            setattr(self, 'W' + a + 'x', self.weight_init_rec.generate(
                    self.weight_shape, self.weight_dtype))
            setattr(self, 'W' + a + 'h', self.weight_init_rec.generate(
                    self.weight_rec_shape, self.weight_dtype))
            setattr(self, 'b_' + a, be.zeros((self.nout, 1)))
            setattr(self, 'W' + a + 'x_updates', be.zeros(self.weight_shape))
            setattr(self, 'W' + a + 'h_updates',
                    be.zeros(self.weight_rec_shape))
            setattr(self, 'b_' + a + '_updates', be.zeros((self.nout, 1)))

        # pack params (stuff like Wix, Wix_updates)
        for a in ['i', 'f', 'o', 'c']:
            gateid = 'g' if a is 'c' else a
            self.gatedic[a] = [getattr(self, 'W' + a + 'x'),
                               getattr(self, 'W' + a + 'h'),
                               getattr(self, 'b_' + a),
                               getattr(self, 'net_' + gateid),
                               getattr(self, gateid + '_t')]
            self.gatedic_u[a] = [getattr(self, 'W' + a + 'x_updates'),
                                 getattr(self, 'W' + a + 'h_updates'),
                                 getattr(self, 'b_' + a + '_updates')]

        # param: If this isn't initialized correctly, get NaNs pretty quickly.
        be.add(self.b_i, 1, self.b_i)  # sigmoid(1) opens the gate
        # +5 following clockwork RNN paper "to encourage long term memory"
        be.add(self.b_f, -1, self.b_f)  # sigmoid(-1) closes gate.
        be.add(self.b_o, 1, self.b_o)   # sigmoid(1) open

        # pack params
        self.param_names = ['input', 'forget', 'output', 'cell']
        self.params = [self.Wix, self.Wfx, self.Wox, self.Wcx, self.Wih,
                       self.Wfh, self.Woh, self.Wch, self.b_i, self.b_f,
                       self.b_o, self.b_c]
        self.updates = [self.Wix_updates, self.Wfx_updates, self.Wox_updates,
                        self.Wcx_updates, self.Wih_updates, self.Wfh_updates,
                        self.Woh_updates, self.Wch_updates, self.b_i_updates,
                        self.b_f_updates, self.b_o_updates, self.b_c_updates]

        self.learning_rule.allocate_state(self.updates)
        for upm in self.updates:
            upm.fill(0.0)

    def list_product(self, target, plist):
        """
        Computes the product of the items in list and puts it into target
        """
        target.fill(1.0)
        reduce(lambda x, y: self.backend.multiply(x, y, x), [target] + plist)

    def list_sum(self, target, slist):
        """
        Computes the sum of the items in slist and puts it into target
        """
        target.fill(0.0)
        reduce(lambda x, y: self.backend.add(x, y, x), [target] + slist)

    def cell_bprop(self, delta_buf, xx, yy, tau, gate, dh1_out):
        be = self.backend
        [wx, wh, b] = self.gatedic[gate][:3]
        [wxu, whu, bu] = self.gatedic_u[gate]

        be.bprop_fc(out=dh1_out, weights=wh, deltas=delta_buf)
        be.update_fc(out=self.dh_dwx_buf, inputs=xx, deltas=delta_buf)
        be.update_fc(out=self.dh_dwh_buf, inputs=yy, deltas=delta_buf)
        if (tau > 0):
            # was h only, but changed this to skip the last x as well
            be.add(wxu, self.dh_dwx_buf, wxu)
            be.add(whu, self.dh_dwh_buf, whu)
        be.sum(delta_buf, 1, self.bsum_buf)
        be.add(bu, self.bsum_buf, bu)

    def cell_fprop(self, xx, yy, tau, gate, actfunc):
        be = self.backend
        [wx, wh, b, netl, tl] = self.gatedic[gate]

        be.fprop_fc(self.temp_x[tau], xx, wx)
        be.fprop_fc(self.temp_h[tau], yy, wh)
        be.add(self.temp_x[tau], self.temp_h[tau], netl[tau])
        be.add(netl[tau], b, netl[tau])
        actfunc.fprop_func(be, netl[tau], tl[tau])

    def fprop(self, y, cell, inputs, tau):
        """
        Forward pass for the google-style LSTM cell with forget gates, no
        peepholes.  cell (``self.c_t``) and hidden (``self.output_list``)
        activity variables will be updated as a result.

        Arguments:
            y:      input from prev. time step (eg. one batch of (64, 50) size)
            cell:   state of memory cell from prev. time step (shape as y)
            inputs: input from data (eg. one batch of (128, 50) size)
            tau:    unrolling step for BPTT

        Notes:
            In math notation, forward pass::

                i_t = s(Wix*x + Wih*h +b_i)
                f_t = s(Wpx*x + Wfh*h +b_f)
                o_t = s(Wox*x + Woh*h +b_o)
                g_t = s(Wcx*x + Wch*h +b_c)
                c_t = f_t .* c_t-1 + i_t .* g_t
                h_t = o_t .* phi(c_t)
                ------ output layer -----
                y_t = s(W_yh * h_t)
                e_t = xEnt(y, t)

            The values are computed and stored for all unrolls so they can be
            used in bprop. [TODO] check for redundant buffers
            self.activation is tanh
            self.gate_activation is logistic

            Visualization of the LSTM cell::

                 c(t)   h(t)
                __|______|___________
                |  \     x---       |  multiplicative gate: output
                |   \    |   \      |
                |    \   O    \     |  nonlinearity
                |     \ /      \    |
                |     _+ __     \   |  memory cell
                |    /     \     \  |
                |   x       x     \ |  multiplicative gates: forget and input
                |  / \     / \    | |
                | |   |   |   |   | |
                | |   O   O   O   O |  gate nonlinearities
                |_|___|___|___|___|_|
                  |   f   g   i   o
                 c(t-1)    h(t-1)

            this depicts the unrolled LSTM cell where loop connecting the cell
            to itself via the forget gate
        """
        be = self.backend  # shorthand

        # input gate
        self.cell_fprop(inputs, y, tau, 'i', self.gate_activation)
        # forget gate
        self.cell_fprop(inputs, y, tau, 'f', self.gate_activation)
        # output gate
        self.cell_fprop(inputs, y, tau, 'o', self.gate_activation)
        # classic RNN cell
        self.cell_fprop(inputs, y, tau, 'c', self.activation)

        # combine the parts and compute output.
        # c_phip = c_t = f_t * cell + i_t * g_t
        be.multiply(self.f_t[tau], cell, self.c_t[tau])
        be.multiply(self.i_t[tau], self.g_t[tau], self.c_phip[tau])
        be.add(self.c_t[tau], self.c_phip[tau], self.c_t[tau])
        # Hack to avoid creating a new copy for c_phip, just want assign vals
        be.add(self.c_t[tau], 0.0, self.c_phip[tau])

        self.activation.fprop_func(be, self.c_phip[tau], self.c_phi[tau])
        be.multiply(self.o_t[tau], self.c_phi[tau], self.output_list[tau])

    def bprop(self, error_h, error_c, tau, numgrad=False):
        """
        For LSTM, inject h-error and c-error, get 8 w's and h, c out. It's
        more complicated than bprop thorugh a standard layer mostly because
        we have two outputs that we inject errors into, each leading to an
        error on the two inputs (4 errors total), and each of the weight
        updates has a contribution from the error to the cell and the hidden.

        Arguments:
            error_h2: error injected into hidden
            error_c2: error injected directly into cell

        Notes:

            Outputs::

                error_h1: from h2 and c2: dh2/dh1 + dc2/dh1
                                          existing  new
                error_c1: from h2 and c2: dh2/dc1 + dc2/dc1
                                          new       new

            [TODO] Two new terms to compute!

            Basic derivation
            In math, backward pass::

                de_dJ = d/dJ CE(y,t)
                dy_dJ = d/dJ sigm(wyh*h)
                ------ hidden layer -----
                dh_dJ = d/dJ o .* tanh(c)
                dp_dJ = d/dJ phi(c)
                dc_dJ = d/dJ (f.*c_ + i.*g)
                di_dJ = d/dJ s(wix*x+wih*h+b)
                df_dJ = d/dJ s(wfx*x+wfh*h+b)
                do_dJ = d/dJ s(wcx*x+wch*h+b)
                dg_dJ = d/dJ s(wcx*x+wch*h+b)

            Over multiple time-steps, deltas feeds back in as error.
            [TODO] Currently using a bunch of if statements to catch
            propagating into outputs[-1], which should not wrap but be 0.
        """
        be = self.backend

        # Only change to layer2: input now comes direct from datalayer
        if self.prev_layer.is_data:
            cur_input = self.prev_layer.output[tau]
        else:
            cur_input = self.prev_layer.output_list[tau]

        cur_output = self.output_list[tau - 1]

        numtemp = {}
        for ifoc in ['i', 'f', 'o', 'c']:
            for hx in ['h', 'x']:
                numtemp[ifoc + hx] = np.zeros((2, 1), dtype=np.float32)

        """--------------------------
        PART 1: original dh2/dh1 terms
        --------------------------"""
        # Precalculate error_h * self.o_t[tau] * self.c_phip[tau]
        self.list_product(self.eh_ot_cphip,
                          [error_h, self.o_t[tau], self.c_phip[tau]])

        # a. Input gate
        # self.delta_buf = error_h * self.o_t[tau] * self.c_phip[tau] \
        #                  * self.g_t[tau] * self.net_i[tau]
        # b. forget gate
        # self.delta_buf = error_h * self.o_t[tau] * self.c_phip[tau] \
        #                  * self.c_t[tau-1] * self.net_f[tau]
        # c. output gate
        # self.delta_buf = error_h * self.c_phi[tau] * self.net_o[tau]
        #
        # d. cell
        # self.delta_buf = error_h * self.o_t[tau] * self.c_phip[tau]
        #                  * self.i_t[tau] * self.net_g[tau]

        deltargs = {'i': [self.eh_ot_cphip, self.g_t[tau], self.net_i[tau]],
                    'f': [self.eh_ot_cphip,
                          self.c_t[tau - 1], self.net_f[tau]],
                    'o': [error_h, self.c_phi[tau], self.net_o[tau]],
                    'c': [self.eh_ot_cphip, self.i_t[tau], self.net_g[tau]]}

        for ifoc in ['i', 'f', 'o', 'c']:
            self.list_product(self.delta_buf, deltargs[ifoc])
            self.cell_bprop(self.delta_buf, cur_input, cur_output, tau,
                            ifoc, self.d_dh1[ifoc])
            if numgrad:
                tc = self.dh_dwh_buf[numgrad['x'], numgrad['w']].asnumpyarray()
                numtemp[ifoc + 'h'][0] = tc
                tc = self.dh_dwx_buf[numgrad['x'], numgrad['y']].asnumpyarray()
                numtemp[ifoc + 'x'][0] = tc

        # e. collect terms
        self.list_sum(self.errs['hh'], self.d_dh1.values())

        """ --------------------------
        PART 2: New dc2/dc1 dc2/dh1 and dh2/dc1 terms
        ---------------------------"""
        # a. Input gate
        # self.delta_buf = error_c * self.g_t[tau] * self.net_i[tau]
        # b. Forget gate
        # self.delta_buf = error_c * self.c_t[tau-1] * self.net_f[tau]
        # c. cell
        # self.delta_buf = error_c * self.i_t[tau] * self.net_g[tau]
        deltargs = {'i': [error_c, self.g_t[tau], self.net_i[tau]],
                    'f': [error_c, self.c_t[tau - 1], self.net_f[tau]],
                    'c': [error_c, self.i_t[tau], self.net_g[tau]]}

        for ifc in ['i', 'f', 'c']:
            self.list_product(self.delta_buf, deltargs[ifc])
            self.cell_bprop(self.delta_buf, cur_input, cur_output, tau,
                            ifc, self.dc_d_dh1[ifc])
            if numgrad:
                tc = self.dh_dwh_buf[numgrad['x'], numgrad['w']].asnumpyarray()
                numtemp[ifc + 'h'][1] = tc
                tc = self.dh_dwx_buf[numgrad['x'], numgrad['y']].asnumpyarray()
                numtemp[ifc + 'x'][1] = tc

        # errs['ch'] = sum of dc_d{i,f,g}_dh1 terms
        # errs['hc'] = error_h * self.o_t * self.c_phip * self.f_t @ tau
        # errs['cc'] = error_c * self.f_t[tau]
        self.list_sum(self.errs['ch'], self.dc_d_dh1.values())
        self.list_product(self.errs['hc'], [self.eh_ot_cphip, self.f_t[tau]])
        be.multiply(error_c, self.f_t[tau], self.errs['cc'])

        # wrap up:
        be.add(self.errs['hh'], self.errs['ch'], self.deltas)
        be.add(self.errs['cc'], self.errs['hc'], self.celtas)
        if numgrad and (numgrad['name'] is not None) \
                and numgrad['name'].startswith("lstm"):
            ifoc_hx = numgrad['name'][5:7]
            logger.info("LSTM.bprop: analytic dh_dw%s[%d]= %e + %e = %e",
                        ifoc_hx, tau, numtemp[ifoc_hx][0], numtemp[ifoc_hx][1],
                        numtemp[ifoc_hx][0] + numtemp[ifoc_hx][1])
