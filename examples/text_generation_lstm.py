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
Example that trains an LSTM or GRU based recurrent networks.
The dataset uses Penn Treebank dataset parsing on character-level.

Reference:
    Generating sequences with recurrent neural networks `[Graves2014]`_
.. _[Graves2014]: http://arxiv.org/pdf/1308.0850.pdf
"""
import numpy as np

from neon.backends import gen_backend
from neon.data import Text
from neon.data import load_text
from neon.initializers import Uniform
from neon.layers import GeneralizedCost, LSTM, Affine
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

batch_size = 64
num_epochs = args.epochs

# Override save path if None
if args.save_path is None:
    args.save_path = 'rnn_text_gen.pickle'

# hyperparameters
time_steps = 64
hidden_size = 512
clip_gradients = True

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype)

# download shakespeare text
data_path = load_text('shakespeare', path=args.data_dir)
train_path, valid_path = Text.create_valid_file(data_path)

# load data and parse on character-level
train_set = Text(time_steps, train_path)
valid_set = Text(time_steps, valid_path, vocab=train_set.vocab)

# weight initialization
init = Uniform(low=-0.08, high=0.08)

# model initialization
layers = [
    LSTM(hidden_size, init, Logistic(), Tanh()),
    Affine(len(train_set.vocab), init, bias=init, activation=Softmax())
]
model = Model(layers=layers)

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))

optimizer = RMSProp(clip_gradients=clip_gradients, stochastic_round=args.rounding)

# configure callbacks
callbacks = Callbacks(model, train_set, output_file=args.output_file,
                      valid_set=valid_set, valid_freq=1,
                      progress_bar=args.progress_bar)
callbacks.add_serialize_callback(1, args.save_path)

# fit and validate
model.fit(train_set, optimizer=optimizer, num_epochs=num_epochs, cost=cost, callbacks=callbacks)


def sample(prob):
    """
    Sample index from probability distribution
    """
    prob = prob / (prob.sum() + 1e-6)
    return np.argmax(np.random.multinomial(1, prob, 1))

# Set batch size and time_steps to 1 for generation and reset buffers
be.bsz = 1
time_steps = 1
num_predict = 1000

layers = [
    LSTM(hidden_size, init, Logistic(), Tanh()),
    Affine(len(train_set.vocab), init, bias=init, activation=Softmax())
]
model = Model(layers=layers)
model.load_weights(args.save_path)

# Generate text
text = []
seed_tokens = list('ROMEO:')

x = be.zeros((len(train_set.vocab), time_steps))

for s in seed_tokens:
    x.fill(0)
    x[train_set.token_to_index[s], 0] = 1
    y = model.fprop(x)

for i in range(num_predict):
    # Take last prediction and feed into next fprop
    pred = sample(y.get()[:, -1])
    text.append(train_set.index_to_token[int(pred)])

    x.fill(0)
    x[int(pred), 0] = 1
    y = model.fprop(x)

print ''.join(seed_tokens + text)
