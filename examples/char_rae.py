#!/usr/bin/env python
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
"""
Character-level recurrent autoencoder. This model shows how to use the
Seq2Seq container class to build an Encoder-Decoder style RNN.

The model uses a sequence from the PTB dataset as input, and learns to output
the same sequence in reverse order.

Usage:

    python examples/char_rae.py -e2

"""
from builtins import str
from builtins import range

from neon.backends import gen_backend
from neon.data import PTB
from neon.initializers import Uniform
from neon.layers import GeneralizedCost, Affine, GRU, Seq2Seq
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Tanh, Logistic, Softmax, CrossEntropyMulti, Misclassification
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon import logger as neon_logger

import numpy as np

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

args.batch_size = 128
time_steps = 20
hidden_size = 512
gradient_clip_value = 5


def get_predictions(model, valid_set, time_steps, beam_search=True, num_beams=5):
    """
    Get model outputs for displaying.
    """
    shape = (valid_set.nbatches, model.be.bsz, time_steps)
    if beam_search:
        ypred = model.get_outputs_beam(valid_set, num_beams=num_beams)
        prediction = ypred.reshape(shape).transpose(1, 0, 2)
    else:
        ypred = model.get_outputs(valid_set)
        prediction = ypred.argmax(2).reshape(shape).transpose(1, 0, 2)

    groundtruth = valid_set.X[:, :valid_set.nbatches, ::-1]
    prediction = prediction[:, :, ::-1].flatten()
    groundtruth = groundtruth[:, :, ::-1].flatten()

    return prediction, groundtruth


def display_text(index_to_token, gt, pr):
    """
    Print out some example strings of input - output pairs.
    """
    index_to_token[0] = '|'  # remove actual line breaks

    display_len = 3 * time_steps

    # sample 3 sentences and their start and end time steps
    (s1_s, s1_e) = (0, time_steps)
    (s2_s, s2_e) = (time_steps, 2*time_steps)
    (s3_s, s3_e) = (2*time_steps, 3*time_steps)

    gt_string = "".join([index_to_token[gt[k]] for k in range(display_len)])
    pr_string = "".join([index_to_token[pr[k]] for k in range(display_len)])

    match = np.where([gt_string[k] == pr_string[k] for k in range(display_len)])

    di_string = "".join([gt_string[k] if k in match[0] else '.'
                        for k in range(display_len)])

    neon_logger.display('GT:   [' + gt_string[s1_s:s1_e] + '] '
                        '[' + gt_string[s2_s:s2_e] + '] '
                        '[' + gt_string[s3_s:s3_e] + '] ')

    neon_logger.display('Pred: [' + pr_string[s1_s:s1_e] + '] '
                        '[' + pr_string[s2_s:s2_e] + '] '
                        '[' + pr_string[s3_s:s3_e] + '] ')

    neon_logger.display('Difference indicated by .')
    neon_logger.display('Diff: [' + di_string[s1_s:s1_e] + '] '
                        '[' + di_string[s2_s:s2_e] + '] '
                        '[' + di_string[s3_s:s3_e] + '] ')


# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# instantiate dataset
dataset = PTB(time_steps, path=args.data_dir, reverse_target=True, get_prev_target=True)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

# weight initialization
init = Uniform(low=-0.08, high=0.08)

# conditional recurrent autoencoder model
num_layers = 1
encoder, decoder = [], []

# decoder_connections indicates the encoder layer indicies to receive conditional inputs from
decoder_connections = []
for ii in range(num_layers):
    name = "GRU" + str(ii+1)
    encoder.append(GRU(hidden_size, init, activation=Tanh(), gate_activation=Logistic(),
                       reset_cells=True, name=name+"Enc"))
    decoder.append(GRU(hidden_size, init, activation=Tanh(), gate_activation=Logistic(),
                       reset_cells=True, name=name+"Dec"))
    decoder_connections.append(ii)
decoder.append(Affine(train_set.nout, init, bias=init, activation=Softmax(), name="AffOut"))

layers = Seq2Seq([encoder, decoder],
                 decoder_connections=decoder_connections,
                 name="Seq2Seq")

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
model = Model(layers=layers)
optimizer = RMSProp(gradient_clip_value=gradient_clip_value, stochastic_round=args.rounding)
callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=args.epochs,
          cost=cost, callbacks=callbacks)

# Misclassification rate on validation set
error_rate = model.eval(valid_set, metric=Misclassification(steps=time_steps))
neon_logger.display('Misclassification error = %.2f%%' % (error_rate * 100))

# Print some example predictions.
prediction, groundtruth = get_predictions(model, valid_set, time_steps)

# convert them into text and display
display_text(valid_set.index_to_token, groundtruth, prediction)
