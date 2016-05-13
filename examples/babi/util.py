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
Utility functions for bAbI example and demo.
"""

from neon.data import BABI
from neon.initializers import GlorotUniform, Uniform, Orthonormal
from neon.layers import Affine, GRU, LookupTable, MergeMultistream, LSTM
from neon.models import Model
from neon.transforms import Logistic, Softmax, Tanh

# list of bAbI tasks
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


def babi_handler(data_dir, task_number):
    """
    Handle for bAbI task.

    Args:
        data_dir (string) : Path to bAbI data directory.
        task_number (int) : The task ID from the bAbI dataset (1-20).

    Returns:
        BABI : Handler for bAbI task.
    """
    task = task_list[task_number - 1]
    return BABI(path=data_dir, task=task, subset=subset)


def create_model(vocab_size, rlayer_type):
    """
    Create LSTM/GRU model for bAbI dataset.

    Args:
        vocab_size (int) : String of bAbI data.
        rlayer_type (string) : Type of recurrent layer to use (gru or lstm).

    Returns:
        Model : Model of the created network
    """
    # recurrent layer parameters (default gru)
    rlayer_obj = GRU if rlayer_type == 'gru' else LSTM
    rlayer_params = dict(output_size=100, reset_cells=True,
                         init=GlorotUniform(), init_inner=Orthonormal(0.5),
                         activation=Tanh(), gate_activation=Logistic())

    # if using lstm, swap the activation functions
    if rlayer_type == 'lstm':
        rlayer_params.update(dict(activation=Logistic(), gate_activation=Tanh()))

    # lookup layer parameters
    lookup_params = dict(vocab_size=vocab_size, embedding_dim=50, init=Uniform(-0.05, 0.05))

    # Model construction
    story_path = [LookupTable(**lookup_params), rlayer_obj(**rlayer_params)]
    query_path = [LookupTable(**lookup_params), rlayer_obj(**rlayer_params)]

    layers = [MergeMultistream(layers=[story_path, query_path], merge="stack"),
              Affine(vocab_size, init=GlorotUniform(), activation=Softmax())]

    return Model(layers=layers)
