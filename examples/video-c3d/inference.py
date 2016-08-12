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
import numpy as np
from neon import logger as neon_logger
from neon.models import Model
from neon.util.argparser import NeonArgparser
from data import make_test_loader, make_category_map, accumulate_video_pred


# parse the command line arguments
default_overrides = dict(batch_size=32)
parser = NeonArgparser(__doc__, default_overrides=default_overrides)
args = parser.parse_args()
assert args.model_file is not None, "need a model file for testing"

model = Model(args.model_file)

category_map = make_category_map()

test = make_test_loader(model.be)

clip_pred = model.get_outputs(test)

video_pred = accumulate_video_pred(clip_pred)

correct = np.zeros((len(video_pred), 2))

TOP1, TOP5 = 0, 1  # indices in correct count array (for readability)

for idx, (video_name, (label, prob_list)) in enumerate(list(video_pred.items())):
    # Average probabilities for each clip
    tot_prob = np.sum(prob_list, axis=0)
    label_idx = category_map[label]

    correct[idx, TOP1] = (label_idx == tot_prob.argmax())

    # Get top 5 predictions
    top5pred = np.argsort(tot_prob)[-5:]
    correct[idx, TOP5] = (label_idx in set(top5pred))

neon_logger.display("Top 1 Accuracy: {:3.2f}% Top 5 Accuracy: {:3.2f}%".format(
    *tuple(correct.mean(axis=0) * 100)))
