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
Seqquence2Sequence container class to build an Encoder-Decoder style RNN.
"""

from neon.backends import gen_backend
from neon.data import PTBAE
from neon.initializers import Uniform, IdentityInit
from neon.layers import GeneralizedCost, Affine, GRU, Seq2Seq
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Tanh, Logistic, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon import logger as neon_logger

import numpy as np

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

args.batch_size = 128
time_steps = 50
hidden_size = 512
gradient_clip_value = 5
num_beams = 0  # set to 0 to turn off beamsearch

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# instanciate dataset
dataset = PTBAE(time_steps, path=args.data_dir, conditional=True)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

# weight initialization
init = Uniform(low=-0.08, high=0.08)
init_eye = IdentityInit()


# conditional recurrent autoencoder model
num_layers = 1
encoder, decoder = [], []
init_from_layers = []
for ii in range(num_layers):
    name = "GRU" + str(ii+1)
    encoder.append(GRU(hidden_size, init, activation=Tanh(), gate_activation=Logistic(),
                       reset_cells=True, name=name+"Enc"))
    decoder.append(GRU(hidden_size, init, activation=Tanh(), gate_activation=Logistic(),
                       reset_cells=True, name=name+"Dec"))
    init_from_layers.append(ii)
decoder.append(Affine(train_set.nout, init, bias=init, activation=Softmax(), name="AffOut"))

layers = Seq2Seq([encoder, decoder],
                 conditional=True,
                 init_from_layers=init_from_layers,
                 name="Seq2Sec")

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
model = Model(layers=layers)
optimizer = RMSProp(gradient_clip_value=gradient_clip_value, stochastic_round=args.rounding)
callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=args.epochs,
          cost=cost, callbacks=callbacks)

# get predictions
# ypred = model.get_outputs(valid_set, num_beams=num_beams, eos=50)  # beamsearch coming soon
ypred = model.get_outputs(valid_set)
shape = (valid_set.nbatches, args.batch_size, time_steps)
if ypred.shape[2] == 1:
    # beamsearch
    prediction = ypred.reshape(shape).transpose(1, 0, 2)
else:
    # no beamsearch
    prediction = ypred.argmax(2).reshape(shape).transpose(1, 0, 2)
groundtruth = valid_set.X[:, :valid_set.nbatches, ::-1]
fraction_correct = (prediction == groundtruth).mean()

neon_logger.display('Misclassification error = %.3f%%' % ((1-fraction_correct)*100))

# Print some example predictions.
valid_set.index_to_token[0] = '|'  # remove actual line breaks

gt_string = "".join([valid_set.index_to_token[k]
                     for k in groundtruth[:, :, ::-1].transpose((0, 1, 2)).flatten()])
pr_string = "".join([valid_set.index_to_token[k]
                     for k in prediction[:, :, ::-1].transpose((0, 1, 2)).flatten()])
locas = np.where([gt_string[k] == pr_string[k] for k in range(500)])
di_string = "".join([gt_string[k] if k in locas[0] else '.' for k in range(500)])

neon_logger.display('GT:   [' + gt_string[0*time_steps:1*time_steps] + '] [' +
                    gt_string[1*time_steps:2*time_steps] + '] [' +
                    gt_string[2*time_steps:3*time_steps] + ']')

neon_logger.display('Pred: [' + pr_string[0*time_steps:1*time_steps] + '] [' +
                    pr_string[1*time_steps:2*time_steps] + '] [' +
                    pr_string[2*time_steps:3*time_steps] + ']')

neon_logger.display('Diff: [' + di_string[0*time_steps:1*time_steps] + '] [' +
                    di_string[1*time_steps:2*time_steps] + '] [' +
                    di_string[2*time_steps:3*time_steps] + ']')
