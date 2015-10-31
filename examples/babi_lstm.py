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
"""

from neon.backends import gen_backend
from neon.data import BABI, QA
from neon.initializers import GlorotUniform, Uniform
from neon.layers import Affine, GeneralizedCost, GRU, LookupTable, MergeMultistream, Sequential 
from neon.models import Model
from neon.optimizers import Adam
from neon.transforms import Accuracy, CrossEntropyMulti, Logistic, Softmax, Tanh
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

num_epochs = args.epochs

batch_size = 32

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype)

# select bAbI task
subset = 'en'
task = 'qa1_single-supporting-fact'
# task = 'qa2_two-supporting-facts'
# task = 'qa3_three-supporting-facts'
# task = 'qa4_two-arg-relations'
# task = 'qa5_three-arg-relations'
# task = 'qa6_yes-no-questions'
# task = 'qa7_counting'
# task = 'qa8_lists-sets'
# task = 'qa9_simple-negation'
# task = 'qa10_indefinite-knowledge'
# task = 'qa11_basic-coreference'
# task = 'qa12_conjunction'
# task = 'qa13_compound-coreference'
# task = 'qa14_time-reasoning'
# task = 'qa15_basic-deduction'
# task = 'qa16_basic-induction'
# task = 'qa17_positional-reasoning'
# task = 'qa18_size-reasoning'
# task = 'qa19_path-finding'
# task = 'qa20_agents-motivations'

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
story_path = Sequential([
	LookupTable(vocab_size=babi.vocab_size, embedding_dim=embedding_dim, init=uniform),
	GRU(hidden_size, glorot, activation=Tanh(), gate_activation=Logistic())
])
query_path = Sequential([
	LookupTable(vocab_size=babi.vocab_size, embedding_dim=embedding_dim, init=uniform),
	GRU(hidden_size, glorot, activation=Tanh(), gate_activation=Logistic())
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
