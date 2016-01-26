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
Shell script to execute alexnet model serialization test

Runs the model on the GPU backend for 4 epochs, serializing at every epoch.  Then
the run is continued for 1 epoch using the serialized file at epoch 3 to initialize
the model.  The output of both runs are compared.
"""

python tests/serialization/alexnet.py -b gpu -s alexnet_run1.prm --serialize 1 -e 3 \
        -H 2 -r 1 -w /usr/local/data/I1K/imageset_batches_dw/ --determin

python tests/serialization/alexnet.py -b gpu -s alexnet_run2.prm --serialize 1 \
        -e 3 -H 2 -r 1 -w /usr/local/data/I1K/imageset_batches_dw/ \
        --model_file alexnet_run1_1.prm --determin

python tests/serialization/compare_models.py alexnet_run1_2.prm alexnet_run2_2.prm
exit $?
