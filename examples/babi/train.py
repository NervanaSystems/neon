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
Example that trains on Facebook Q&A dataset: bAbI

Task Number                  | FB LSTM Baseline | Neon QA GRU
---                          | ---              | ---
QA1 - Single Supporting Fact | 50               |  47.9
QA2 - Two Supporting Facts   | 20               |  29.8
QA3 - Three Supporting Facts | 20               |  20.0
QA4 - Two Arg. Relations     | 61               |  69.8
QA5 - Three Arg. Relations   | 70               |  56.4
QA6 - Yes/No Questions       | 48               |  49.1
QA7 - Counting               | 49               |  76.5
QA8 - Lists/Sets             | 45               |  68.9
QA9 - Simple Negation        | 64               |  62.8
QA10 - Indefinite Knowledge  | 44               |  45.3
QA11 - Basic Coreference     | 72               |  67.6
QA12 - Conjunction           | 74               |  63.9
QA13 - Compound Coreference  | 94               |  91.9
QA14 - Time Reasoning        | 27               |  36.8
QA15 - Basic Deduction       | 21               |  51.4
QA16 - Basic Induction       | 23               |  50.1
QA17 - Positional Reasoning  | 51               |  49.0
QA18 - Size Reasoning        | 52               |  90.5
QA19 - Path Finding          | 8                |   9.0
QA20 - Agent's Motivations   | 91               |  95.6

Reference:
    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    http://arxiv.org/abs/1502.05698

Usage:
    use -t to specify which bAbI task to run
    python examples/babi/train.py -e 20 --rlayer_type gru --save_path babi_lstm.p -t 1
"""
from util import create_model, babi_handler
from neon.backends import gen_backend
from neon.data import QA
from neon.layers import GeneralizedCost
from neon.optimizers import Adam
from neon.transforms import Accuracy, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('-t', '--task', type=int, default='1', choices=xrange(1, 21),
                    help='the task ID to train/test on from bAbI dataset (1-20)')
parser.add_argument('--rlayer_type', default='gru', choices=['gru', 'lstm'],
                    help='type of recurrent layer to use (gru or lstm)')
args = parser.parse_args(gen_be=False)

# Override save path if None
if args.save_path is None:
    args.save_path = 'babi.p'

if args.callback_args['save_path'] is None:
    args.callback_args['save_path'] = args.save_path

# setup backend
args.batch_size = 32
be = gen_backend(**extract_valid_args(args, gen_backend))

# load the bAbI dataset
babi = babi_handler(args.data_dir, args.task)
train_set = QA(*babi.train)
valid_set = QA(*babi.test)

# create model
model = create_model(babi.vocab_size, args.rlayer_type)

# setup callbacks
callbacks = Callbacks(model, train_set, eval_set=valid_set, **args.callback_args)

# train model
model.fit(train_set,
          optimizer=Adam(),
          num_epochs=args.epochs,
          cost=GeneralizedCost(costfunc=CrossEntropyMulti()),
          callbacks=callbacks)

# output accuracies
print('Train Accuracy = %.1f%%' % (model.eval(train_set, metric=Accuracy())*100))
print('Test Accuracy = %.1f%%' % (model.eval(valid_set, metric=Accuracy())*100))
