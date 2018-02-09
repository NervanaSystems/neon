#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""
Usage:
    python timeseries_lstm.py -e 10 -eval 1
    python timeseries_lstm.py -e 10 -eval 1 --predict_seq

Builds an LSTM network to predict the next value in a timeseries
Lissajous curves are used as the timeseries to test the network
In addition, generates future values of the sequence based on an initial seed
    (See the sequence generation section for details, found later in the code)

The following flag will switch between 2 training strategies:
1. predict_seq True:
      Inputs are sequences, and target outputs will be sequences.
      The RNN layer's output at EVERY step will be used for errors and optimized.
      The RNN model contains a RNN layer and an Affine layer
      The data iterator will format the data accordingly, and will stride along the
          whole series with no overlap
2. predict_seq False:
      Inputs are sequences, and target output will be a single step.
      The RNN layer's output at LAST step will be used for errors and optimized.
      The RNN model contains a RNN layer and RNN-output layer (i.g. RecurrentLast, etc.)
          and an Affine layer
      The data iterator will format the data accordingly, using a rolling window to go
          through the data

Note that when the time series has higher or lower frequency, it requires different amounts
of data to learn the temporal pattern, the sequence length and the batch size for the
training process also makes a difference on learning performance.
"""

from __future__ import division, print_function
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, RecurrentLast
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Logistic, Tanh, Identity, MeanSquared
from neon.callbacks.callbacks import Callbacks
from neon import logger as neon_logger
from neon.util.argparser import NeonArgparser, extract_valid_args
from utils import SyntheticTimeSeries, DataIteratorSequence, err
import numpy as np


# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--curvetype', default='Lissajous1', choices=['Lissajous1', 'Lissajous2'],
                    help='type of input curve data to use (Lissajous1 or Lissajous2)')
parser.add_argument('--predict_seq', default=False, dest='predict_seq', action='store_true',
                    help='If given, seq_len future timepoints are predicted')
parser.add_argument('--seq_len', type=int,
                    help="Number of time points in each input sequence",
                    default=32)
args = parser.parse_args(gen_be=False)

batch_size = args.batch_size
seq_len = args.seq_len
predict_seq = args.predict_seq
no_epochs = args.epochs  # Total epochs of training

be = gen_backend(**extract_valid_args(args, gen_backend))

# a file to save the trained model
if args.save_path is None:
    args.save_path = 'timeseries.pkl'

if args.callback_args['save_path'] is None:
    args.callback_args['save_path'] = args.save_path

if args.callback_args['serialize'] is None:
    args.callback_args['serialize'] = 1

# Plot results
do_plots = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    do_plots = False

# Feature dimension of the input (for Lissajous curve, this is 2)
feature_dim = 2
# Output feature dimension (for Lissajous curve, this is 2)
output_dim = 2
# number of sinusoidal cycles
no_cycles = 2000
# number of data points per cycle
no_points = 13

# Generate Lissajous Curve
time_series = SyntheticTimeSeries(npoints=no_points, ncycles=no_cycles,
                                  curvetype=args.curvetype)

if do_plots:
    plt.figure()
    plt.plot(time_series.data[:, 0], time_series.data[:, 1])
    plt.title('%s' % args.curvetype)
    plt.savefig('%s' % args.curvetype + ".png")

    plt.figure()
    plt.plot(time_series.data[0:100, 0], 'b', label='%s' % (args.curvetype) + "_x")
    plt.plot(time_series.data[0:100, 1], 'r', label='%s' % (args.curvetype) + "_y")
    plt.legend()
    plt.title('%s' % args.curvetype)
    plt.savefig('%s' % args.curvetype + "_x_y.png")


# use data iterator to feed X, Y. return_sequence determines training
# strategy
train_set = DataIteratorSequence(
    time_series.train, seq_len, return_sequences=args.predict_seq)
valid_set = DataIteratorSequence(
    time_series.test, seq_len, return_sequences=args.predict_seq)


# define weights initialization
init = GlorotUniform()  # Uniform(low=-0.08, high=0.08)

# Number of recurrent units in the network
recurrent_units = 32

# define model: model is different for the 2 strategies (sequence target
# or not)
if args.predict_seq:
    layers = [
        LSTM(recurrent_units, init, activation=Logistic(),
             gate_activation=Tanh(), reset_cells=False),
        Affine(train_set.nfeatures, init, bias=init, activation=Identity())
    ]
else:
    layers = [
        LSTM(recurrent_units, init, activation=Logistic(),
             gate_activation=Tanh(), reset_cells=True),
        RecurrentLast(),
        Affine(train_set.nfeatures, init, bias=init, activation=Identity())
    ]

model = Model(layers=layers)

# cost and optimizer
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
    plt.plot(train_output[:, 0], train_output[:, 1], 'bo', label='prediction')
    plt.plot(train_target[:, 0], train_target[:, 1], 'r.', label='target')
    plt.legend()
    plt.title('Neon on training set')
    plt.savefig('neon_series_training_output.png')

    plt.figure()
    plt.plot(valid_output[:, 0], valid_output[:, 1], 'bo', label='prediction')
    plt.plot(valid_target[:, 0], valid_target[:, 1], 'r.', label='target')
    plt.legend()
    plt.title('Neon on validation set')
    plt.savefig('neon_series_validation_output.png')

# =====================generate sequence ==================================
# when generating sequence, set sequence length to 1, since it doesn't
# make a difference
be.bsz = 1
seq_len = 1

if args.predict_seq:
    layers = [LSTM(recurrent_units, init, activation=Logistic(),
                   gate_activation=Tanh(), reset_cells=False),
              Affine(train_set.nfeatures, init, bias=init,
                     activation=Identity())]
else:
    layers = [LSTM(recurrent_units, init, activation=Logistic(),
                   gate_activation=Tanh(), reset_cells=False),
              RecurrentLast(),
              Affine(train_set.nfeatures, init, bias=init,
                     activation=Identity())]

model_new = Model(layers=layers)
model_new.load_params(args.save_path)
model_new.initialize(dataset=(train_set.nfeatures, seq_len))

num_predict = 200
seed_seq_len = 30

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
    plt.plot(output_seq[:, 0], output_seq[:, 1], 'b.-', label='generated sequence')
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
