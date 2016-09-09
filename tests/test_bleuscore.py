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
'''
Test BLEUScore metric against reference
'''
from neon.transforms.cost import BLEUScore


def test_bleuscore():
    # dataset with two sentences
    sentences = ["a quick brown fox jumped",
                 "the rain in spain falls mainly on the plains"]

    references = [["a fast brown fox jumped",
                   "a quick brown fox vaulted",
                   "a rapid fox of brown color jumped",
                   "the dog is running on the grass"],
                  ["the precipitation in spain falls on the plains",
                   "spanish rain falls for the most part on the plains",
                   "the rain in spain falls in the plains most of the time",
                   "it is raining today"]]

    # reference scores for the given set of reference sentences
    bleu_score_references = [92.9, 88.0, 81.5, 67.1]    # bleu1, bleu2, bleu3, bleu4

    # compute scores
    bleu_metric = BLEUScore()
    bleu_metric(sentences, references)

    # check against references
    for score, reference in zip(bleu_metric.bleu_n, bleu_score_references):
        assert round(score, 1) == reference

if __name__ == '__main__':
    test_bleuscore()
