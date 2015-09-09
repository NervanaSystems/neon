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
import logging
import numpy as np

from neon import NervanaObject

logging.basicConfig(level=20)
logger = logging.getLogger()


def test_dropout(backend):
    ng = NervanaObject.be
    ng.bsz = ng.batch_size = 15

    d_array2 = ng.array(np.random.randn(24).reshape((6, 4)), dtype=np.float32)
    d_error = ng.array(np.random.randn(24).reshape((6, 4)), dtype=np.float32)
    mask = ng.empty((6, 4))

    logger.info("FPROP")
    print d_array2.get()
    # d_array2[:] = ng.dropout(0.5) * d_array2
    ng.make_binary_mask(mask, keepthresh=0.5)
    d_array2[:] = mask * d_array2
    print d_array2.get()

    logger.info("BPROP")
    print d_error.get()
    d_error[:] = (d_array2 != 0) * d_error
    print d_error.get()
