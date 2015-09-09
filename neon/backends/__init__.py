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
"""
Defines gen_backend function
"""

import atexit
import logging
import os
import sys
import numpy as np

from neon import NervanaObject
from neon.backends.autodiff import Autodiff


def gen_backend(backend='cpu', rng_seed=None, default_dtype=np.float32,
                batch_size=0, stochastic_round=False, device_id=0):
    """
    Construct and return a backend instance of the appropriate type based on
    the arguments given. With no parameters, a single CPU core, float32
    backend is returned.

    Arguments:
        backend (string, optional): 'cpu' or 'gpu'.
        rng_seed (numeric, optional): Set this to a numeric value which can be
                                      used to seed the random number generator
                                      of the instantiated backend.  Defaults to
                                      None, which doesn't explicitly seed (so
                                      each run will be different)
        default_dtype (dtype): Default tensor data type. CPU backend supports
                               np.float64, np.float32 and np.float16; GPU
                               backend supports np.float32 and np.float16.
        batch_size (int): Set the size the data batches.
        stochastic_round (int/bool, optional): Set this to True or an integer
                                               to implent stochastic rounding.
                                               If this is False rounding will
                                               be to nearest.
                                               If True will perform stochastic
                                               rounding using default bit width.
                                               If set to an integer will round
                                               to that number of bits.
                                               Only affects the gpu backend.
        device_id (numeric, optional): Set this to a numeric value which can be
                                       used to select which device to run the
                                       process on

    Returns:
        Backend: newly constructed backend instance of the specifed type.

    Notes:
        * Attempts to construct a GPU instance without a CUDA capable card or
          without nervanagpu package installed will cause the
          program to display an error message and exit.
    """
    logger = logging.getLogger(__name__)

    if NervanaObject.be is not None:
        # backend was already generated
        # clean it up first
        cleanup_backend()
    else:
        # at exit from python force cleanup of backend
        # only register this function once, will use
        # NervanaObject.be instead of a global
        atexit.register(cleanup_backend)

    if backend == 'cpu' or backend is None:
        from neon.backends.nervanacpu import NervanaCPU
        be = NervanaCPU(rng_seed=rng_seed, default_dtype=default_dtype)
    elif backend == 'gpu':
        gpuflag = False
        # check nvcc
        from neon.backends.util import check_gpu
        gpuflag = (check_gpu.get_compute_capability() >= 5.0)
        if gpuflag is False:
            raise RuntimeError("Can't find GPU with CUDA compute capability " +
                               "5.0 or greater")
        from neon.backends.nervanagpu import NervanaGPU
        # init gpu
        be = NervanaGPU(rng_seed=rng_seed, default_dtype=default_dtype,
                        stochastic_round=stochastic_round, device_id=device_id)
    elif backend == 'mgpu':
        raise NotImplementedError("mgpu will be ready soon")
    else:
        raise ValueError("backend must be one of "
                         "('cpu', 'gpu', 'mgpu')")

    logger.info("Backend: {}, RNG seed: {}".format(backend, rng_seed))

    NervanaObject.be = be
    be.bsz = batch_size
    return be


def cleanup_backend():
    be = NervanaObject.be
    from neon.backends.nervanacpu import NervanaCPU
    if type(be) is not NervanaCPU:
        from neon.backends.nervanagpu import NervanaGPU
        assert type(be) is NervanaGPU
        be.ctx.pop()

    del(be)
    NervanaObject.be = None
