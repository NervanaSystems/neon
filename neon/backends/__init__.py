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
Houses code for each of the core backend and associated Tensor data structures.
"""

import logging
import os
import sys

# import shortcuts
from neon.backends.cpu import CPU
from neon.backends.par import NoPar, ModelPar, DataPar


def gen_backend(model, gpu=None, nrv=False, datapar=False, modelpar=False,
                flexpoint=False, rng_seed=None, numerr_handling=None,
                half=False, stochastic_round=0, device_id=None):
    """
    Construct and return a backend instance of the appropriate type based on
    the arguments given.  With no parameters, a single CPU core, float32
    backend is returned.

    Arguments:
        model (neon.models.model.Model): The instantiated model upon which we
                                         will utilize this backend.
        gpu (string, optional): Attempt to utilize a CUDA capable GPU if
                                installed in the system. Defaults to None which
                                implies a CPU based backend.  If 'cudanet',
                                utilize a cuda-convnet2 based backed, which
                                supports Kepler and Maxwell GPUs with single
                                precision. If 'nervanagpu', attempt to utilize
                                the NervanaGPU Maxwell backend with float16 and
                                float32 support.
        nrv (bool, optional): If True, attempt to utilize the Nervana Engine
                              for computation (must be installed on the
                              system).  Defaults to False which implies a CPU
                              based backend.
        datapar (bool, optional): Set to True to ensure that data is
                                  partitioned and each chunk is processed in
                                  parallel on different compute cores. Requires
                                  mpi4py.  Defaults to False which implies that
                                  all data will be processed sequentially on a
                                  single compute core.
        modelpar (bool, optional): Set to True to ensure that the nodes in each
                                   model layer are partitioned and distributed
                                   across multiple compute cores.  Requires
                                   mpi4py.  Defaults to False which implies
                                   that all nodes in all model layers will be
                                   processed by the same single compute core.
        flexpoint (bool, optional): If True, attempt to use FlexPoint(TM)
                                    element typed data instead of the default
                                    float32 which is in place if set to False.
        rng_seed (numeric, optional): Set this to a numeric value which can be
                                      used to seed the random number generator
                                      of the instantiated backend.  Defaults to
                                      None, which doesn't explicitly seed (so
                                      each run will be different)
        stochastic_round (numeric, optional): Only affects the max backend. If
                                              1, perform stochastic rounding.
                                              If 0, round to nearest.
        numerr_handling (dict, optional): Dictate how numeric errors are
                                          displayed and handled.  The keys and
                                          values permissible for this dict
                                          match that seen in numpy.seterr.
                                          If set to None (the default),
                                          behavior is equivalent to
                                          {'all': 'warn'}
        device_id (numeric, optional): Set this to a numeric value which can be
                                       used to select which device to run the
                                       process on

    Returns:
        Backend: newly constructed backend instance of the specifed type.

    Notes:
        * Attempts to construct a GPU instance without a CUDA capable card or
          without cudanet or nervanagpu package installed will cause the
          program to display an error message and exit.
        * Attempts to construct a parallel instance without mpi4py installed
          will cause the program to display an error message and exit.
        * The returned backend will still need to call its par.init_model()
          at some point after the model has been linked, in order for parallel
          training to proceed.
    """
    logger = logging.getLogger(__name__)
    gpuflag = False

    if datapar and modelpar:
        raise NotImplementedError('Hybrid parallelization scheme not '
                                  'implemented yet.  Try with at most one of'
                                  'datapar or modelpar')
    if modelpar:
        par = ModelPar()
    elif datapar:
        par = DataPar()
    else:
        par = NoPar()

    if par.device_id is not None:
        if device_id is not None:
            logger.warn('Ignoring device id specified in command line.')
        device_id = par.device_id

    if gpu is not None:
        gpu = gpu.lower()
        if sys.platform.startswith("linux"):
            gpuflag = (os.system("nvidia-smi > /dev/null 2>&1") == 0)
        elif sys.platform.startswith("darwin"):
            gpuflag = (os.system("kextstat | grep -i cuda > /dev/null 2>&1") ==
                       0)
        if gpuflag and gpu == 'cudanet':
            try:
                import cudanet  # noqa
                from neon.backends.cc2 import GPU
                be_name = 'Cudanet'
                be = GPU(rng_seed=rng_seed, device_id=device_id)
            except ImportError:
                logger.warning("cudanet not found, can't run via GPU")
                gpuflag = False
        elif gpuflag and gpu == 'nervanagpu':
            try:
                import nervanagpu  # noqa
                try:
                    # import pycuda.autoinit
                    import pycuda.driver as drv
                    drv.init()
                    device_id = device_id if device_id is not None else 0
                    global ctx
                    ctx = drv.Device(device_id).make_context()
                    import atexit
                    atexit.register(ctx.pop)
                    from neon.backends.gpu import GPU
                    be_name = 'NervanaGPU'
                    be = GPU(rng_seed=rng_seed,
                             stochastic_round=stochastic_round,
                             device_id=device_id)
                except ImportError:
                    logger.warning("pycuda error, can't run via GPU")
                    gpuflag = False
            except ImportError:
                logger.warning("nervanagpu not found, can't run via GPU")
                gpuflag = False
        if gpuflag is False:
            raise RuntimeError("Can't find CUDA capable GPU")
    elif nrv:
        nrv = False
        try:
            from umd.nrv_backend import NRVBackend
            nrv = True
        except ImportError:
            logger.warning("Nervana Engine system software not found")

    if flexpoint:
        logger.warning("Flexpoint(TM) backend not currently available")

    if nrv:
        be_name = 'NRV'
        be = NRVBackend(rng_seed=rng_seed, seterr_handling=numerr_handling,
                        device_id=device_id)
    elif not gpuflag:
        be_name = 'CPU'
        be = CPU(rng_seed=rng_seed, seterr_handling=numerr_handling)
    logger.info("{} backend, RNG seed: {}, numerr: {}".format
                (be_name, rng_seed, numerr_handling))

    par.associate(be)
    return be
