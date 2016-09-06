"""
This is a Minimal single layer GRU implementation adapted from DoctorTeeth's code:
https://github.com/DoctorTeeth/gru/blob/master/gru.py

The adaptation includes
  - remove the GRU to output affine transformation
  - provide inputs for forward pass and errors for backward pass
  - being able to provide init_state
  - initialize weights and biases into zeros, as the main test script will externally
    initialize the weights and biases
  - being able to read out hashable values to compare with another GRU
    implementation
"""

import numpy as np


class GRU(object):

    def __init__(self, in_size, hidden_size):
        """
        This class implements a GRU.
        """
        # TODO: go back to 0.01 initialization
        # TODO: use glorot initialization?
        # input weights
        self.hidden_size = hidden_size
        self.in_size = in_size

        # input to candidate
        self.Wxc = np.zeros((hidden_size, in_size))
        # input to reset
        self.Wxr = np.zeros((hidden_size, in_size))
        # input to interpolate
        self.Wxz = np.zeros((hidden_size, in_size))

        # Recurrent weights
        # hidden to candidate
        self.Rhc = np.zeros((hidden_size, hidden_size))
        # hidden to reset
        self.Rhr = np.zeros((hidden_size, hidden_size))
        # hidden to interpolate
        self.Rhz = np.zeros((hidden_size, hidden_size))

        # biases
        self.bc = np.zeros((hidden_size, 1))  # bias for candidate
        self.br = np.zeros((hidden_size, 1))  # bias for reset
        self.bz = np.zeros((hidden_size, 1))  # bias for interpolate

        self.weights = [self.Wxc, self.Wxr, self.Wxz, self.Rhc, self.Rhr,
                        self.Rhz, self.bc, self.br, self.bz]

        # I used this for grad checking, but I should clean up
        self.names = ["Wxc", "Wxr", "Wxz", "Rhc", "Rhr", "Rhz", "bc", "br", "bz"]

        self.weights_ind_Wxc = 0
        self.weights_ind_Wxr = 1
        self.weights_ind_Wxz = 2
        self.weights_ind_Rhc = 3
        self.weights_ind_Rhr = 4
        self.weights_ind_Rhz = 5
        self.weights_ind_bc = 6
        self.weights_ind_br = 7
        self.weights_ind_bz = 8

    def lossFun(self, inputs, errors, init_state=None):
        """
        Does a forward and backward pass on the network using (inputs, errors)
        inputs is a bit-vector of seq-length
        errors is a bit-vector of seq-length
        """

        xs, rbars, rs, zbars, zs, cbars, cs, hs = {}, {}, {}, {}, {}, {}, {}, {}

        # xs are inputs
        # hs are hiddens

        # This resets the hidden state after every new sequence
        # TODO: maybe we don't need to
        if init_state is None:
            hs[-1] = np.zeros((self.hidden_size, 1))
        else:
            hs[-1] = init_state

        seq_len = len(inputs)

        hs_list = np.zeros((self.hidden_size, seq_len))

        # forward pass, compute outputs, t indexes time
        for t in range(seq_len):
            # For every variable V, Vbar represents the pre-activation version
            # For every variable Q, Qnext represents that variable at time t+1
            # where t is understood from context

            # xs is the input vector at this time
            xs[t] = np.matrix(inputs[t])

            # The r gate, which modulates how much signal from h[t-1] goes to
            # the candidate
            rbars[t] = np.dot(self.Wxr, xs[t]) + \
                np.dot(self.Rhr, hs[t - 1]) + self.br
            rs[t] = 1.0 / (1 + np.exp(-rbars[t]))
            # TODO: use an already existing sigmoid function

            # The z gate, which interpolates between candidate and h[t-1] to
            # compute h[t]
            zbars[t] = np.dot(self.Wxz, xs[t]) + \
                np.dot(self.Rhz, hs[t - 1]) + self.bz
            zs[t] = 1.0 / (1 + np.exp(-zbars[t]))

            # The candidate, which is computed and used as described above.
            cbars[t] = np.dot(
                self.Wxc, xs[t]) + np.dot(self.Rhc, np.multiply(rs[t], hs[t - 1])) + self.bc
            cs[t] = np.tanh(cbars[t])

            # TODO: get rid of this
            ones = np.ones_like(zs[t])

            # Compute new h by interpolating between candidate and old h
            hs[t] = np.multiply(
                cs[t], zs[t]) + np.multiply(hs[t - 1], ones - zs[t])

            hs_list[:, t] = hs[t].flatten()

        # allocate space for the grads of loss with respect to the weights
        dWxc = np.zeros_like(self.Wxc)
        dWxr = np.zeros_like(self.Wxr)
        dWxz = np.zeros_like(self.Wxz)
        dRhc = np.zeros_like(self.Rhc)
        dRhr = np.zeros_like(self.Rhr)
        dRhz = np.zeros_like(self.Rhz)

        # allocate space for the grads of loss with respect to biases
        dbc = np.zeros_like(self.bc)
        dbr = np.zeros_like(self.br)
        dbz = np.zeros_like(self.bz)

        # no error is received from beyond the end of the sequence
        dhnext = np.zeros_like(hs[0])
        drbarnext = np.zeros_like(rbars[0])
        dzbarnext = np.zeros_like(zbars[0])
        dcbarnext = np.zeros_like(cbars[0])
        zs[len(inputs)] = np.zeros_like(zs[0])
        rs[len(inputs)] = np.zeros_like(rs[0])

        dh_list = errors
        dh_list_out = np.zeros_like(dh_list)
        dr_list = np.zeros((self.hidden_size, seq_len))
        dz_list = np.zeros((self.hidden_size, seq_len))
        dc_list = np.zeros((self.hidden_size, seq_len))

        # go backwards through time
        for t in reversed(range(seq_len)):

            # For every variable X, dX represents dC/dX
            # For variables that influence C at multiple time steps,
            # such as the weights, the dw is a sum of dw at multiple
            # time steps

            # h[t] influences the cost in 5 ways:
            # through the interpolation using z at t+1
            dha = np.multiply(dhnext, ones - zs[t + 1])

            # through transformation by weights into rbar
            dhb = np.dot(self.Rhr.T, drbarnext)

            # through transformation by weights into zbar
            dhc = np.dot(self.Rhz.T, dzbarnext)

            # through transformation by weights into cbar
            dhd = np.multiply(rs[t + 1], np.dot(self.Rhc.T, dcbarnext))

            # through bp errors
            dhe = dh_list[t]

            dh = dha + dhb + dhc + dhd + dhe

            dh_list_out[t] = dh

            dc = np.multiply(dh, zs[t])

            # backprop through tanh
            dcbar = np.multiply(dc, ones - np.square(cs[t]))

            dr = np.multiply(hs[t - 1], np.dot(self.Rhc.T, dcbar))
            dz = np.multiply(dh, (cs[t] - hs[t - 1]))

            # backprop through sigmoids
            drbar = np.multiply(dr, np.multiply(rs[t], (ones - rs[t])))
            dzbar = np.multiply(dz, np.multiply(zs[t], (ones - zs[t])))

            dWxr += np.dot(drbar, xs[t].T)
            dWxz += np.dot(dzbar, xs[t].T)
            dWxc += np.dot(dcbar, xs[t].T)

            dRhr += np.dot(drbar, hs[t - 1].T)
            dRhz += np.dot(dzbar, hs[t - 1].T)
            dRhc += np.dot(dcbar, np.multiply(rs[t], hs[t - 1]).T)

            dbr += drbar
            dbc += dcbar
            dbz += dzbar

            dhnext = dh

            drbarnext = drbar
            dzbarnext = dzbar
            dcbarnext = dcbar

            dr_list[:, t] = drbar.flatten()
            dz_list[:, t] = dzbar.flatten()
            dc_list[:, t] = dcbar.flatten()

        dw = [dWxc, dWxr, dWxz, dRhc, dRhr, dRhz, dbc, dbr, dbz
              ]

        self.dW_ind_Wxc = 0
        self.dW_ind_Wxr = 1
        self.dW_ind_Wxz = 2
        self.dW_ind_Rhc = 3
        self.dW_ind_Rhr = 4
        self.dW_ind_Rhz = 5

        self.dW_ind_bc = 6
        self.dW_ind_br = 7
        self.dW_ind_bz = 8

        return dw, hs_list, dh_list_out, dr_list, dz_list, dc_list
