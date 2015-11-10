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
Example that trains on Facebook Q&A datatset: bAbI

Task Number                  | FB LSTM Baseline | Neon QA GRU
---                          | ---              | ---
QA1 - Single Supporting Fact | 50               | 49.3
QA2 - Two Supporting Facts   | 20               | 28.9
QA3 - Three Supporting Facts | 20               | 23.4
QA4 - Two Arg. Relations     | 61               | 69.7
QA5 - Three Arg. Relations   | 70               | 55.7
QA6 - Yes/No Questions       | 48               | 49.3
QA7 - Counting               | 49               | 75.7
QA8 - Lists/Sets             | 45               | 69.3
QA9 - Simple Negation        | 64               | 62.7
QA10 - Indefinite Knowledge  | 44               | 44.7
QA11 - Basic Coreference     | 72               | 69.3
QA12 - Conjunction           | 74               | 66.0
QA13 - Compound Coreference  | 94               | 91.5
QA14 - Time Reasoning        | 27               | 36.6
QA15 - Basic Deduction       | 21               | 52.2
QA16 - Basic Induction       | 23               | 50.8
QA17 - Positional Reasoning  | 51               | 50.5
QA18 - Size Reasoning        | 52               | 91.6
QA19 - Path Finding          | 8                | 8.7
QA20 - Agent's Motivations   | 91               | 96.2

Reference:
    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    http://arxiv.org/abs/1502.05698

Usage:
    use -t to specify which bAbI task to run
    python examples/babi_lstm.py -e 10 -eval 1 -t 1 --rlayer_type gru
"""

from neon.backends import gen_backend
from neon.data import BABI, QA
from neon.initializers import GlorotUniform, Uniform, Orthonormal
from neon.layers import (Affine, GeneralizedCost, GRU, LookupTable,
                         MergeMultistream, LSTM)
from neon.models import Model
from neon.optimizers import Adam
from neon.transforms import Accuracy, CrossEntropyMulti, Logistic, Softmax, Tanh
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# list of bAbI task
subset = 'en'
task_list = [
    'qa1_single-supporting-fact',
    'qa2_two-supporting-facts',
    'qa3_three-supporting-facts',
    'qa4_two-arg-relations',
    'qa5_three-arg-relations',
    'qa6_yes-no-questions',
    'qa7_counting',
    'qa8_lists-sets',
    'qa9_simple-negation',
    'qa10_indefinite-knowledge',
    'qa11_basic-coreference',
    'qa12_conjunction',
    'qa13_compound-coreference',
    'qa14_time-reasoning',
    'qa15_basic-deduction',
    'qa16_basic-induction',
    'qa17_positional-reasoning',
    'qa18_size-reasoning',
    'qa19_path-finding',
    'qa20_agents-motivations',
]

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('-t', '--task', type=int, default='1', choices=xrange(1, 21),
                    help='the task ID to train/test on from bAbI dataset (1-20)')
parser.add_argument('--rlayer_type', default='gru', choices=['gru', 'lstm'],
                    help='type of recurrent layer to use (gru or lstm)')
args = parser.parse_args()

task = task_list[args.task - 1]
batch_size = 32

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype)

# load the bAbI dataset
babi = BABI(path=args.data_dir, task=task, subset=subset)
train_set = QA(*babi.train)
valid_set = QA(*babi.test)

# recurrent layer parameters
rlayer_params = dict(output_size=100, init=GlorotUniform(),
                     init_inner=Orthonormal(0.5), reset_cells=True)

rlayer_params['activation'] = Tanh() if args.rlayer_type == 'gru' else Logistic()
rlayer_params['gate_activation'] = Logistic() if args.rlayer_type == 'gru' else Tanh()

rlayer_obj = GRU if args.rlayer_type == 'gru' else LSTM

# lookup layer parameters
lookup_params = dict(
    vocab_size=babi.vocab_size, embedding_dim=50, init=Uniform(-0.05, 0.05))

# Model construction
story_path = [
    LookupTable(**lookup_params), rlayer_obj(**rlayer_params)]
query_path = [
    LookupTable(**lookup_params), rlayer_obj(**rlayer_params)]

layers = [MergeMultistream(layers=[story_path, query_path], merge="stack"),
          Affine(babi.vocab_size, init=GlorotUniform(), activation=Softmax())]

model = Model(layers=layers)

# cost function and optimizer
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
optimizer = Adam()

# setup callbacks
callbacks = Callbacks(model, train_set, eval_set=valid_set, **args.callback_args)

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=args.epochs,
          cost=cost,
          callbacks=callbacks)

# output accuracies
print('Train Accuracy = %.1f%%' % (model.eval(train_set, metric=Accuracy())*100))
print('Test Accuracy = %.1f%%' % (model.eval(valid_set, metric=Accuracy())*100))
