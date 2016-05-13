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
Interactive demo based on Facebook Q&A dataset: bAbI

Reference:
    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    http://arxiv.org/abs/1502.05698

Usage:
    use -t to specify which bAbI task to run
    python examples/babi/demo.py -t 1 --rlayer_type gru --model_weights babi.p
"""

from builtins import input, range
import numpy as np
from util import create_model, babi_handler
from neon import logger as neon_logger
from neon.backends import gen_backend
from neon.data import BABI, QA
from neon.data.text import Text
from neon.util.argparser import NeonArgparser, extract_valid_args

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('-t', '--task', type=int, default='1', choices=range(1, 21),
                    help='the task ID to train/test on from bAbI dataset (1-20)')
parser.add_argument('--rlayer_type', default='gru', choices=['gru', 'lstm'],
                    help='type of recurrent layer to use (gru or lstm)')
parser.add_argument('--model_weights', required=True,
                    help='pickle file of trained weights')
args = parser.parse_args(gen_be=False)

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))
be.bsz = 1

# load the bAbI dataset
babi = babi_handler(args.data_dir, args.task)
valid_set = QA(*babi.test)

# create model
model_inference = create_model(babi.vocab_size, args.rlayer_type)
model_inference.load_params(args.model_weights)
model_inference.initialize(dataset=valid_set)

ex_story, ex_question, ex_answer = babi.test_parsed[0]


def stitch_sentence(words):
    return " ".join(words).replace(" ?", "?").replace(" .", ".\n") \
              .replace("\n ", "\n")


def vectorize(words, max_len):
    return be.array(Text.pad_sentences([babi.words_to_vector(BABI.tokenize(words))],
                                       max_len))


neon_logger.display(
    "\nThe vocabulary set from this task has {} words:".format(babi.vocab_size))
neon_logger.display(stitch_sentence(babi.vocab))
neon_logger.display("\nExample from test set:")
neon_logger.display("\nStory")
neon_logger.display(stitch_sentence(ex_story))
neon_logger.display("Question")
neon_logger.display(stitch_sentence(ex_question))
neon_logger.display("\nAnswer")
neon_logger.display(ex_answer)

while True:
    # ask user for story and question
    story_lines = []
    line = input("\nPlease enter a story:\n")
    while line != "":
        story_lines.append(line)
        line = input()
    story = ("\n".join(story_lines)).strip()

    question = input("Please enter a question:\n")

    # convert user input into a suitable network input
    s = vectorize(story, babi.story_maxlen)
    q = vectorize(question, babi.query_maxlen)

    # get prediction probabilities with forward propagation
    probs = model_inference.fprop(x=(s, q), inference=True).get()

    # get top k answers
    top_k = -min(5, babi.vocab_size)
    max_indices = np.argpartition(probs, top_k, axis=0)[top_k:]
    max_probs = probs[max_indices]
    sorted_idx = max_indices[np.argsort(max_probs, axis=0)]

    neon_logger.display("\nAnswer:")
    for idx in reversed(sorted_idx):
        idx = int(idx)
        neon_logger.display(babi.index_to_word[idx], float(probs[idx]))
