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
Shell script to execute cifar10 conv model serialization test

Runs the model on the CPU backend for 4 epochs, serializing at every epoch.  Then
the run is continued for 1 epoch using the serialized file at epoch 3 to initialize
the model.  The output of both runs are compared.
"""

python tests/serialization/cifar10_conv.py -b cpu -s cifar10_run1.prm --serialize 1 -e 4 -r 1 --determin -H 2 -eval 1
python tests/serialization/cifar10_conv.py -b cpu -s cifar10_run2.prm --serialize 1 -e 4 -r 1 --determin -eval 1 \
            --model_file cifar10_run1_2.prm 

python tests/serialization/compare_models.py cifar10_run1_3.prm cifar10_run2.prm
exit $?
