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
Test the skip-thought container
"""
from __future__ import print_function
import numpy as np

from neon.backends import gen_backend
from neon.models import Model
from neon.initializers import Uniform, Orthonormal, Constant
from neon.transforms import Logistic, Tanh
from neon.layers import GRU
from neon.layers.container import SkipThought


def test_skip_thought(backend_default):
    be = backend_default
    be.bsz = 32

    vs = 2000
    es = 300
    init_embed = Uniform(low=-0.1, high=0.1)
    nh = 640

    skip = SkipThought(vs, es, init_embed, nh, rec_layer=GRU,
                       init_rec=Orthonormal(), activ_rec=Tanh(), activ_rec_gate=Logistic(),
                       init_ff=Uniform(low=-0.1, high=0.1), init_const=Constant(0.0))
    model = Model(skip)

    model.initialize(dataset=[(100, 32), (100, 32), (100, 32)])

    s_s = be.array(np.random.randint(100, size=(100, 32)), dtype=np.int32)
    s_p = be.array(np.random.randint(100, size=(100, 32)), dtype=np.int32)
    s_n = be.array(np.random.randint(100, size=(100, 32)), dtype=np.int32)

    out = model.fprop((s_s, s_p, s_n))

    e_p = be.array(np.random.randint(100, size=(2000, 3200)), dtype=np.int32)
    e_n = be.array(np.random.randint(100, size=(2000, 3200)), dtype=np.int32)

    error_out = model.bprop((e_p, e_n))

    print(out)
    print(error_out)

if __name__ == '__main__':
    be = gen_backend('gpu')
    test_skip_thought(be)
