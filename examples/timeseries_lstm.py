#!/usr/bin/env python
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
Train on synthetic multi-dimensional time series data.
After training, the network is able to generate the sequences

Usage:

    python examples/timeseries_lstm.py -e 10 -eval 1

Then look at the PNG plots generated.

"""

from builtins import object
import numpy as np
import math
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, RecurrentLast
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Logistic, Tanh, Identity, MeanSquared
from neon.callbacks.callbacks import Callbacks
from neon import NervanaObject, logger as neon_logger
from neon.util.argparser import NeonArgparser, extract_valid_args


do_plots = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    neon_logger.display('matplotlib needs to be installed manually to generate plots needed '
                        'for this example.  Skipping plot generation')
    do_plots = False


def rolling_window(a, lag):
    """
    Convert a into time-lagged vectors

    a    : (n, p)
    lag  : time steps used for prediction

    returns  (n-lag+1, lag, p)  array

    (Building time-lagged vectors is not necessary for neon.)
    """
    assert a.shape[0] > lag

    shape = [a.shape[0] - lag + 1, lag, a.shape[-1]]
    strides = [a.strides[0], a.strides[0], a.strides[-1]]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class TimeSeries(object):

    def __init__(self, npoints=30, ncycles=3, divide=0.2, amplitude=1, curvetype='Lissajous1'):
        """
        curvetype (str, optional): 'Lissajous1' or 'Lissajous2'
        """
        self.nsamples = npoints * ncycles
        self.x = np.linspace(0, ncycles * 2 * math.pi, self.nsamples)

        if curvetype not in ('Lissajous1', 'Lissajous2'):
            raise NotImplementedError()

        sin_scale = 2 if curvetype is 'Lissajous1' else 1

        def y_x(x):
            return 4.0 / 5 * math.sin(x / sin_scale)

        def y_y(x):
            return 4.0 / 5 * math.cos(x / 2)

        self.data = np.zeros((self.nsamples, 2))
        self.data[:, 0] = np.asarray([y_x(xs)
                                      for xs in self.x]).astype(np.float32)
        self.data[:, 1] = np.asarray([y_y(xs)
                                      for xs in self.x]).astype(np.float32)

        L = len(self.data)
        c = int(L * (1 - divide))
        self.train = self.data[:c]
        self.test = self.data[c:]


class DataIteratorSequence(NervanaObject):

    """
    This class takes a sequence and returns an iterator providing data in batches suitable for RNN
    prediction.  Meant for use when the entire dataset is small enough to fit in memory.
    """

    def __init__(self, X, time_steps, forward=1, return_sequences=True):
        """
        Implements loading of given data into backend tensor objects. If the backend is specific
        to an accelerator device, the data is copied over to that device.

        Args:
            X (ndarray): Input sequence with feature size within the dataset.
                         Shape should be specified as (num examples, feature size]
            time_steps (int): The number of examples to be put into one sequence.
            forward (int, optional): how many forward steps the sequence should predict. default
                                     is 1, which is the next example
            return_sequences (boolean, optional): whether the target is a sequence or single step.
                                                  Also determines whether data will be formatted
                                                  as strides or rolling windows.
                                                  If true, target value be a sequence, input data
                                                  will be reshaped as strides.  If false, target
                                                  value will be a single step, input data will be
                                                  a rolling_window
        """
        self.seq_length = time_steps
        self.forward = forward
        self.batch_index = 0
        self.nfeatures = self.nclass = X.shape[1]
        self.nsamples = X.shape[0]
        self.shape = (self.nfeatures, time_steps)
        self.return_sequences = return_sequences

        target_steps = time_steps if return_sequences else 1
        # pre-allocate the device buffer to provide data for each minibatch
        # buffer size is nfeatures x (times * batch_size), which is handled by
        # backend.iobuf()
        self.X_dev = self.be.iobuf((self.nfeatures, time_steps))
        self.y_dev = self.be.iobuf((self.nfeatures, target_steps))

        if return_sequences is True:
            # truncate to make the data fit into multiples of batches
            extra_examples = self.nsamples % (self.be.bsz * time_steps)
            if extra_examples:
                X = X[:-extra_examples]

            # calculate how many batches
            self.nsamples -= extra_examples
            self.nbatches = self.nsamples // (self.be.bsz * time_steps)
            self.ndata = self.nbatches * self.be.bsz * time_steps  # no leftovers

            # y is the lagged version of X
            y = np.concatenate((X[forward:], X[:forward]))
            self.y_series = y
            # reshape this way so sequence is continuous along the batches
            self.X = X.reshape(self.be.bsz, self.nbatches,
                               time_steps, self.nfeatures)
            self.y = y.reshape(self.be.bsz, self.nbatches,
                               time_steps, self.nfeatures)
        else:
            self.X = rolling_window(X, time_steps)
            self.X = self.X[:-1]
            self.y = X[time_steps:]

            self.nsamples = self.X.shape[0]
            extra_examples = self.nsamples % (self.be.bsz)
            if extra_examples:
                self.X = self.X[:-extra_examples]
                self.y = self.y[:-extra_examples]

            # calculate how many batches
            self.nsamples -= extra_examples
            self.nbatches = self.nsamples // self.be.bsz
            self.ndata = self.nbatches * self.be.bsz
            self.y_series = self.y

            Xshape = (self.nbatches, self.be.bsz, time_steps, self.nfeatures)
            Yshape = (self.nbatches, self.be.bsz, 1, self.nfeatures)
            self.X = self.X.reshape(Xshape).transpose(1, 0, 2, 3)
            self.y = self.y.reshape(Yshape).transpose(1, 0, 2, 3)

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        """
        self.batch_index = 0

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuple : the next minibatch of data.
        """
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            # get the data for this batch and reshape to fit the device buffer
            # shape
            X_batch = self.X[:, self.batch_index].T.reshape(
                self.X_dev.shape).copy()
            y_batch = self.y[:, self.batch_index].T.reshape(
                self.y_dev.shape).copy()

            # make the data for this batch as backend tensor
            self.X_dev.set(X_batch)
            self.y_dev.set(y_batch)

            self.batch_index += 1

            yield self.X_dev, self.y_dev


# replicate neon's mse error metric
def err(y, t):
    feature_axis = 1
    return (0.5 * np.square(y - t).mean(axis=feature_axis).mean())

if __name__ == '__main__':

    # parse the command line arguments
    parser = NeonArgparser(__doc__)
    parser.add_argument('--curvetype', default='Lissajous1', choices=['Lissajous1', 'Lissajous2'],
                        help='type of input curve data to use (Lissajous1 or Lissajous2)')
    args = parser.parse_args(gen_be=False)

    # network hyperparameters
    hidden = 32
    args.batch_size = 1

    # The following flag will switch between 2 training strategies:
    # 1. return_sequence True:
    #       Inputs are sequences, and target outputs will be sequences.
    #       The RNN layer's output at EVERY step will be used for errors and optimized.
    #       The RNN model contains a RNN layer and an Affine layer
    #       The data iterator will format the data accordingly, and will stride along the
    #           whole series with no overlap
    # 2. return_sequence False:
    #       Inputs are sequences, and target output will be a single step.
    #       The RNN layer's output at LAST step will be used for errors and optimized.
    #       The RNN model contains a RNN layer and RNN-output layer (i.g. RecurrentLast, etc.)
    #           and an Affine layer
    #       The data iterator will format the data accordingly, using a rolling window to go
    #           through the data
    return_sequences = False

    # Note that when the time series has higher or lower frequency, it requires different amounts
    # of data to learn the temporal pattern, the sequence length and the batch size for the
    # training process also makes a difference on learning performance.

    seq_len = 30
    npoints = 10
    ncycles = 100
    num_predict = 200
    seed_seq_len = 30

    # ================= Main neon script ====================

    be = gen_backend(**extract_valid_args(args, gen_backend))

    # a file to save the trained model
    if args.save_path is None:
        args.save_path = 'timeseries.pkl'

    if args.callback_args['save_path'] is None:
        args.callback_args['save_path'] = args.save_path

    if args.callback_args['serialize'] is None:
        args.callback_args['serialize'] = 1

    # create synthetic data as a whole series
    time_series = TimeSeries(npoints, ncycles=ncycles,
                             curvetype=args.curvetype)

    # use data iterator to feed X, Y. return_sequence determines training
    # strategy
    train_set = DataIteratorSequence(
        time_series.train, seq_len, return_sequences=return_sequences)
    valid_set = DataIteratorSequence(
        time_series.test, seq_len, return_sequences=return_sequences)

    # define weights initialization
    init = GlorotUniform()  # Uniform(low=-0.08, high=0.08)

    # define model: model is different for the 2 strategies (sequence target
    # or not)
    if return_sequences is True:
        layers = [
            LSTM(hidden, init, activation=Logistic(),
                 gate_activation=Tanh(), reset_cells=False),
            Affine(train_set.nfeatures, init, bias=init, activation=Identity())
        ]
    else:
        layers = [
            LSTM(hidden, init, activation=Logistic(),
                 gate_activation=Tanh(), reset_cells=True),
            RecurrentLast(),
            Affine(train_set.nfeatures, init, bias=init, activation=Identity())
        ]

    model = Model(layers=layers)
    cost = GeneralizedCost(MeanSquared())
    optimizer = RMSProp(stochastic_round=args.rounding)

    callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

    # fit model
    model.fit(train_set,
              optimizer=optimizer,
              num_epochs=args.epochs,
              cost=cost,
              callbacks=callbacks)

    # =======visualize how the model does on validation set==============
    # run the trained model on train and valid dataset and see how the outputs
    # match
    train_output = model.get_outputs(
        train_set).reshape(-1, train_set.nfeatures)
    valid_output = model.get_outputs(
        valid_set).reshape(-1, valid_set.nfeatures)
    train_target = train_set.y_series
    valid_target = valid_set.y_series

    # calculate accuracy
    terr = err(train_output, train_target)
    verr = err(valid_output, valid_target)

    neon_logger.display('terr = %g, verr = %g' % (terr, verr))

    if do_plots:
        plt.figure()
        plt.plot(train_output[:, 0], train_output[
                 :, 1], 'bo', label='prediction')
        plt.plot(train_target[:, 0], train_target[:, 1], 'r.', label='target')
        plt.legend()
        plt.title('Neon on training set')
        plt.savefig('neon_series_training_output.png')

        plt.figure()
        plt.plot(valid_output[:, 0], valid_output[
                 :, 1], 'bo', label='prediction')
        plt.plot(valid_target[:, 0], valid_target[:, 1], 'r.', label='target')
        plt.legend()
        plt.title('Neon on validation set')
        plt.savefig('neon_series_validation_output.png')

    # =====================generate sequence ==================================
    # when generating sequence, set sequence length to 1, since it doesn't
    # make a difference
    be.bsz = 1
    seq_len = 1

    if return_sequences is True:
        layers = [
            LSTM(hidden, init, activation=Logistic(),
                 gate_activation=Tanh(), reset_cells=False),
            Affine(train_set.nfeatures, init, bias=init, activation=Identity())
        ]
    else:
        layers = [
            LSTM(hidden, init, activation=Logistic(),
                 gate_activation=Tanh(), reset_cells=False),
            RecurrentLast(),
            Affine(train_set.nfeatures, init, bias=init, activation=Identity())
        ]

    model_new = Model(layers=layers)
    model_new.load_params(args.save_path)
    model_new.initialize(dataset=(train_set.nfeatures, seq_len))

    output = np.zeros((train_set.nfeatures, num_predict))
    seed = time_series.train[:seed_seq_len]

    x = model_new.be.empty((train_set.nfeatures, seq_len))
    for s_in in seed:
        x.set(s_in.reshape(train_set.nfeatures, seq_len))
        y = model_new.fprop(x, inference=False)

    for i in range(num_predict):
        # Take last prediction and feed into next fprop
        pred = y.get()[:, -1]
        output[:, i] = pred
        x[:] = pred.reshape(train_set.nfeatures, seq_len)
        y = model_new.fprop(x, inference=False)

    output_seq = np.vstack([seed, output.T])

    if do_plots:
        plt.figure()
        plt.plot(output_seq[:, 0], output_seq[:, 1],
                 'b.-', label='generated sequence')
        plt.plot(seed[:, 0], seed[:, 1], 'r.', label='seed sequence')
        plt.legend()
        plt.title('neon generated sequence')
        plt.savefig('neon_generated_sequence_2d.png')

        plt.figure()
        plt.plot(output_seq, 'b.-', label='generated sequence')
        plt.plot(seed, 'r.', label='seed sequence')
        plt.legend()
        plt.title('neon generated sequence')
        plt.savefig('neon_generated_sequence.png')
