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
Example that trains an image captioning model on precomputed image features from
a CNN and reference sentences. Uses standard image captioning datasets
[flickr8k, flickr30k, coco] from http://cs.stanford.edu/people/karpathy/deepimagesent/
that have been stored in pkl format. The model then transforms the image features
and sentences to the same hidden dimension size and prepends the image to be the
first word of the sequence which is then fed to a LSTM.

Referece:
https://github.com/karpathy/neuraltalk

"""
from neon.backends import gen_backend
from neon.data import load_flickr8k, ImageCaption, ImageCaptionTest
from neon.initializers import Uniform, Constant
from neon.layers import GeneralizedCostMask, LSTM, Affine, Dropout, Sequential, MergeMultistream
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

# hyperparameters
hidden_size = 512
num_epochs = args.epochs

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# download dataset
data_path = load_flickr8k(path=args.data_dir)  # Other setnames are flickr30k and coco

# load data
train_set = ImageCaption(path=data_path, max_images=-1)

# weight initialization
init = Uniform(low=-0.08, high=0.08)
init2 = Constant(val=train_set.be.array(train_set.bias_init))

# model initialization
image_path = Sequential([Affine(hidden_size, init, bias=Constant(val=0.0))])
sent_path = Sequential([Affine(hidden_size, init, linear_name='sent')])

layers = [
    MergeMultistream(layers=[image_path, sent_path], merge="recurrent"),
    Dropout(keep=0.5),
    LSTM(hidden_size, init, activation=Logistic(), gate_activation=Tanh(), reset_cells=True),
    Affine(train_set.vocab_size, init, bias=init2, activation=Softmax())
]

cost = GeneralizedCostMask(costfunc=CrossEntropyMulti(usebits=True))

# configure callbacks
checkpoint_model_path = "~/image_caption2.pickle"
if args.callback_args['save_path'] is None:
    args.callback_args['save_path'] = checkpoint_model_path

if args.callback_args['serialize'] is None:
    args.callback_args['serialize'] = 1

model = Model(layers=layers)

callbacks = Callbacks(model, train_set, **args.callback_args)

opt = RMSProp(decay_rate=0.997, learning_rate=0.0005, epsilon=1e-8, gradient_clip_value=1)

# train model
model.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

# load model (if exited) and evaluate bleu score on test set
model.load_params(checkpoint_model_path)
test_set = ImageCaptionTest(path=data_path)
sents, targets = test_set.predict(model)
test_set.bleu_score(sents, targets)
