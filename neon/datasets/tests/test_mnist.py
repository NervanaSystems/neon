# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
import os
from nose.plugins.attrib import attr
import shutil

from neon.datasets.mnist import MNIST
from neon.backends.cpu import CPU
from neon.backends.par import NoPar


class TestMNIST(object):

    tmp_repo = os.path.join(os.path.dirname(__file__), 'repo')

    def setup(self):
        os.makedirs(self.tmp_repo)

    def teardown(self):
        shutil.rmtree(self.tmp_repo, ignore_errors=True)

    @attr('slow')
    def test_get_inputs(self):
        d = MNIST(repo_path=self.tmp_repo)
        d.backend = CPU(rng_seed=0)
        d.backend.actual_batch_size = 128
        d.backend.par = NoPar(d.backend)
        inputs = d.get_inputs(train=True)
        # TODO: make this work (numpy import errors at the moment)
        assert inputs['train'] is not None
