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

Task Number                  | FB LSTM Baseline | Neon QA
---                          | ---              | ---
QA1 - Single Supporting Fact | 50               | 48.9
QA2 - Two Supporting Facts   | 20               | 20.7
QA3 - Three Supporting Facts | 20               | 22.6
QA4 - Two Arg. Relations     | 61               | 60.7
QA5 - Three Arg. Relations   | 70               | 54.2
QA6 - Yes/No Questions       | 48               | 48.9
QA7 - Counting               | 49               | 76.0
QA8 - Lists/Sets             | 45               | 68.6
QA9 - Simple Negation        | 64               | 62.7
QA10 - Indefinite Knowledge  | 44               | 46.7
QA11 - Basic Coreference     | 72               | 69.7
QA12 - Conjunction           | 74               | 68.3
QA13 - Compound Coreference  | 94               | 93.5
QA14 - Time Reasoning        | 27               | 29.3
QA15 - Basic Deduction       | 21               | 51.9
QA16 - Basic Induction       | 23               | 50.3
QA17 - Positional Reasoning  | 51               | 50.4
QA18 - Size Reasoning        | 52               | 91.2
QA19 - Path Finding          | 8                | 9.1
QA20 - Agent's Motivations   | 91               | 92.2

Reference:
    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    http://arxiv.org/abs/1502.05698

Usage:
    use -t to specify which bAbI task to run
    python examples/babi_lstm.py -e 20 -eval 1 -t 1
"""

from neon.backends import gen_backend
from neon.data import BABI, QA
from neon.initializers import GlorotUniform, Uniform
from neon.layers import (Affine, GeneralizedCost, GRU, LookupTable, 
                        MergeMultistream, Sequential, LSTM)
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
parser.add_argument('-t', '--task',
                type=int, default='1', choices=xrange(1, 21),
                help='the task ID to train/test on from bAbI dataset (1-20)')

args = parser.parse_args()

num_epochs = args.epochs
task_id = args.task-1

task = task_list[task_id]

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
test_set = QA(*babi.test)

# weight initialization
uniform = Uniform(-.05, .05)
glorot = GlorotUniform()

# model parameters
embedding_dim = 50
hidden_size = 100

# model construction
rlayer_type = 'gru'
if rlayer_type == 'lstm':
    rlayer_story = LSTM(hidden_size, glorot, Logistic(), Tanh(), reset_cells=True)
    rlayer_query = LSTM(hidden_size, glorot, Logistic(), Tanh(), reset_cells=True)
elif rlayer_type == 'gru':
    rlayer_story = GRU(hidden_size, glorot, activation=Tanh(),
                        gate_activation=Logistic(), reset_cells=True)
    rlayer_query = GRU(hidden_size, glorot, activation=Tanh(),
                        gate_activation=Logistic(), reset_cells=True)
else:
    raise NotImplementedError('%s layer not implemented' % rlayer_type)


story_path = Sequential([
	LookupTable(vocab_size=babi.vocab_size, embedding_dim=embedding_dim, init=uniform),
	rlayer_story
])
query_path = Sequential([
	LookupTable(vocab_size=babi.vocab_size, embedding_dim=embedding_dim, init=uniform),
	rlayer_query
])

layers = [
    MergeMultistream(layers=[story_path, query_path], merge="stack"),
    Affine(babi.vocab_size, glorot, activation=Softmax())
]

model = Model(layers=layers)

# cost function and optimizer
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
optimizer = Adam()

# setup callbacks
callbacks = Callbacks(model, train_set, args, eval_set=test_set)

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=num_epochs,
          cost=cost,
          callbacks=callbacks)

# output accuracies
print('Train Accuracy = %.1f%%' % (model.eval(train_set, metric=Accuracy())*100))
print('Test Accuracy = %.1f%%' % (model.eval(test_set, metric=Accuracy())*100))
