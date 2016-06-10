"""
This is a minimal single layer RNN implementation adapted from Andrej Karpathy's
code -- Minimal character-level Vanilla RNN model. BSD License
https://gist.github.com/karpathy/d4dee566867f8291f086

The adaptation includes
  - remove the file I/O
  - remove the recurrent to output affine layer
  - remove the sampling part
  - add a class container for the sizes and weights
  - keep only the lossFun function with provided inputs and errors
  - initialize weights and biases into empty, as the main test script will externally
    initialize the weights and biases
  - being able to read out hashable values to compare with another recurrent
    implementation
"""

import numpy as np


class Recurrent(object):

    def __init__(self, in_size, hidden_size):
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.Wxh = np.zeros((hidden_size, in_size))  # input to hidden
        self.Whh = np.zeros((hidden_size, hidden_size))  # hidden to hidden
        self.bh = np.zeros((hidden_size, 1))  # hidden bias

    def lossFun(self, inputs, errors):
        """
        inputs,errors are both list of integers.
        returns the hidden states and gradients on model parameters
        """

        xs, hs = {}, {}
        hs[-1] = np.zeros((self.hidden_size, 1))
        seq_len = len(inputs)
        hs_list = np.zeros((self.hidden_size, seq_len))
        nin = inputs[0].shape[0]

        # forward pass
        for t in range(seq_len):
            xs[t] = np.matrix(inputs[t])
            # hidden state
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)
            hs_list[:, t] = hs[t].flatten()

        # backward pass: compute gradients going backwards
        dhnext = np.zeros_like(hs[0])
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        dh_list = errors
        dh_list_out = np.zeros_like(dh_list)
        dout_list = np.zeros((nin, seq_len))

        for t in reversed(range(seq_len)):
            dh = dh_list[t] + dhnext  # backprop into h
            dh_list_out[t] = dh
            # import pdb;pdb.set_trace()
            # backprop through tanh nonlinearity
            dhraw = np.multiply(dh, (1 - np.square(hs[t])))
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
            dout = np.dot(self.Wxh.T, dhraw)
            dout_list[:, t] = dout.flatten()

        return dWxh, dWhh, dbh, hs_list, dh_list_out, dout_list
