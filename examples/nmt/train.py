#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
Machine translation example using recurrent encoder-decoder.

Uses the subset of sentences that are shorter than the sequence length.
"""
from neon.backends import gen_backend
from neon.data.text import TextNMT
from neon.initializers import Uniform
from neon.layers import GeneralizedCost, Affine, GRU, Seq2Seq, LookupTable
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Tanh, Logistic, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.transforms.cost import BLEUScore
import os
import itertools as itt


def bleu_format(examples, tgt_dict, num_batches, batch_size, eos=0):
    """
    BLEU score
    """
    sents = [examples[ex, b, :] for b, ex in itt.product(range(num_batches), range(batch_size))]
    return [" ".join([tgt_dict[k].decode("utf-8") for k in sent if k != eos]) for sent in sents]


def print_sample(ex_source, ex_reference, ex_prediction, src_dict, tgt_dict):
    """
    Print some example predictions.
    """
    sample_output = u"""
    Source Sentence: {source}
    Reference Translation: {reference}
    Predicted Translation: {prediction}
    """.format(source=" ".join([src_dict[k].decode("utf-8") for k in ex_source]),
               reference=" ".join([tgt_dict[k].decode("utf-8") for k in ex_reference]),
               prediction=" ".join([tgt_dict[k].decode("utf-8") for k in ex_prediction]))
    print(sample_output.encode("utf-8"))


# parse the command line arguments
default_overrides = dict(batch_size=128)
parser = NeonArgparser(__doc__, default_overrides=default_overrides)
parser.add_argument('--num_layers', default=2, type=int,
                    help='number of recurrent layers in encoder and decoder')
parser.add_argument('--num_hidden', default=512, type=int,
                    help='number of hidden units in each recurrent layer')
parser.add_argument('--embedding_dim', default=512, type=int,
                    help='word embedding dimension')
parser.add_argument('--num_beams', default=10, type=int,
                    help='number of beam search beams (0 for no beam search)')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args(gen_be=False)

# parameters
hidden_size = args.num_hidden
embedding_dim = args.embedding_dim
num_beams = args.num_beams
num_layers = args.num_layers
gradient_clip_value = 5
time_steps = 20
dataset = 'un2000'

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# load data
train_path = os.path.join(args.data_dir, 'nmt', dataset)
train_set = TextNMT(time_steps, train_path, get_prev_target=True, onehot_input=False,
                    split='train', dataset=dataset, subset_pct=args.subset_pct)
valid_set = TextNMT(time_steps, train_path, get_prev_target=False, onehot_input=False,
                    split='valid', dataset=dataset)

# weight initialization
init = Uniform(low=-0.08, high=0.08)

# Standard or Conditional encoder / decoder:
encoder = [LookupTable(vocab_size=len(train_set.s_vocab), embedding_dim=embedding_dim,
                       init=init, name="LUT_en")]
decoder = [LookupTable(vocab_size=len(train_set.t_vocab), embedding_dim=embedding_dim,
                       init=init, name="LUT_de")]
decoder_connections = []  # link up recurrent layers
for ii in range(num_layers):
    encoder.append(GRU(hidden_size, init, activation=Tanh(), gate_activation=Logistic(),
                       reset_cells=True, name="GRU1Enc"))
    decoder.append(GRU(hidden_size, init, activation=Tanh(), gate_activation=Logistic(),
                       reset_cells=True, name="GRU1Dec"))
    decoder_connections.append(ii)
decoder.append(Affine(train_set.nout, init, bias=init, activation=Softmax(), name="Affout"))

layers = Seq2Seq([encoder, decoder],
                 decoder_connections=decoder_connections,
                 name="Seq2Seq")

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))

model = Model(layers=layers)

optimizer = RMSProp(gradient_clip_value=gradient_clip_value, stochastic_round=args.rounding)

# configure callbacks
callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=args.epochs,
          cost=cost, callbacks=callbacks)

# obtain predictions
shape = (valid_set.nbatches, args.batch_size, time_steps)
if num_beams == 0:
    ypred = model.get_outputs(valid_set)
    # flip the reversed predictions back to normal sentence order
    prediction = ypred.argmax(2).reshape(shape).transpose(1, 0, 2)[:, :, ::-1]
else:
    ypred = model.get_outputs_beam(valid_set, num_beams=num_beams)
    prediction = ypred.reshape(shape).transpose(1, 0, 2)[:, :, ::-1]

# print some examples
src_dict, tgt_dict = valid_set.s_index_to_token, valid_set.t_index_to_token
for i in range(3):
    print_sample(ex_source=valid_set.X[i, 0, :],
                 ex_reference=valid_set.y[i, 0, :],
                 ex_prediction=prediction[i, 0, :],
                 src_dict=src_dict, tgt_dict=tgt_dict)

# compute BLEU scores
inputs = valid_set.X[:, :valid_set.nbatches, :]
source_sentences = bleu_format(inputs, tgt_dict, valid_set.nbatches, args.batch_size)

generated = bleu_format(prediction, tgt_dict, valid_set.nbatches, args.batch_size)
references = bleu_format(valid_set.y, tgt_dict, valid_set.nbatches, args.batch_size)
references = [[r] for r in references]
bleu_score = BLEUScore()
bleu4 = bleu_score(generated, references, N=4, brevity_penalty=False, lower_case=True)
