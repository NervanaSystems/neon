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
from math import ceil

from neon import NervanaObject
from neon.backends.autodiff import Autodiff
from neon.backends.util.check_gpu import get_device_count


def gen_backend(backend='cpu', rng_seed=None, datatype=np.float32,
                batch_size=0, stochastic_round=False, device_id=0,
                max_devices=get_device_count(), compat_mode=None):
    """
    Construct and return a backend instance of the appropriate type based on
    the arguments given. With no parameters, a single CPU core, float32
    backend is returned.

    Arguments:
        backend (string, optional): 'cpu' or 'gpu'.
        rng_seed (numeric, optional): Set this to a numeric value which can be used to seed the
                                      random number generator of the instantiated backend.
                                      Defaults to None, which doesn't explicitly seed (so each run
                                      will be different)
        dataype (dtype): Default tensor data type. CPU backend supports np.float64, np.float32 and
                         np.float16; GPU backend supports np.float32 and np.float16.
        batch_size (int): Set the size the data batches.
        stochastic_round (int/bool, optional): Set this to True or an integer to implent
                                               stochastic rounding. If this is False rounding will
                                               be to nearest. If True will perform stochastic
                                               rounding using default bit width. If set to an
                                               integer will round to that number of bits.
                                               Only affects the gpu backend.
        device_id (numeric, optional): Set this to a numeric value which can be used to select
                                       device on which to run the process
        max_devices (int, optional): For use with multi-GPU backend only.
                                      Controls the maximum number of GPUs to run
                                      on.
        compat_mode (str, optional): if this is set to 'caffe' then the conv and pooling
                                     layer output sizes will match that of caffe as will
                                     the dropout layer implementation

    Returns:
        Backend: newly constructed backend instance of the specifed type.

    Notes:
        * Attempts to construct a GPU instance without a CUDA capable card or without nervanagpu
          package installed will cause the program to display an error message and exit.
    """
    logger = logging.getLogger(__name__)

    if NervanaObject.be is not None:
        # backend was already generated clean it up first
        cleanup_backend()
    else:
        # at exit from python force cleanup of backend only register this function once, will use
        # NervanaObject.be instead of a global
        atexit.register(cleanup_backend)

    if backend == 'cpu' or backend is None:
        from neon.backends.nervanacpu import NervanaCPU
        be = NervanaCPU(rng_seed=rng_seed, default_dtype=datatype, compat_mode=compat_mode)
    elif backend == 'gpu' or backend == 'mgpu':
        gpuflag = False
        # check nvcc
        from neon.backends.util import check_gpu
        gpuflag = (check_gpu.get_compute_capability(device_id) >= 5.0)
        if gpuflag is False:
            raise RuntimeError("Device " + str(device_id) + " does not have CUDA compute " +
                               "capability 5.0 or greater")
        if backend == 'gpu':
            from neon.backends.nervanagpu import NervanaGPU
            # init gpu
            be = NervanaGPU(rng_seed=rng_seed, default_dtype=datatype,
                            stochastic_round=stochastic_round,
                            device_id=device_id,
                            compat_mode=compat_mode)
        else:
            try:
                from mgpu.nervanamgpu import NervanaMGPU
                # init multiple GPU
                be = NervanaMGPU(rng_seed=rng_seed,
                                 default_dtype=datatype,
                                 stochastic_round=stochastic_round,
                                 num_devices=max_devices)
            except ImportError:
                logger.error("Multi-GPU support is a premium feature "
                             "available exclusively through the Nervana cloud."
                             " Please contact info@nervanasys.com for details.")
                raise
    else:
        raise ValueError("backend must be one of ('cpu', 'gpu', 'mgpu')")

    logger.info("Backend: {}, RNG seed: {}".format(backend, rng_seed))

    NervanaObject.be = be
    be.bsz = batch_size
    return be


def cleanup_backend():
    if NervanaObject.be is None:
        return;
    be = NervanaObject.be
    from neon.backends.nervanacpu import NervanaCPU
    if type(be) is not NervanaCPU:
        from neon.backends.nervanagpu import NervanaGPU
        try:
            if type(be) is NervanaGPU:
                be.ctx.pop()
                be.ctx.detach()
            else:
                from mgpu.nervanamgpu import NervanaMGPU
                assert type(be) is NervanaMGPU
                for ctx in be.ctxs:
                    ctx.pop()
                    ctx.detach()
        except:
            pass
    del(be)
    NervanaObject.be = None
