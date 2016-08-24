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
"""
General functions for running the unit tests via pytest.
"""
import itertools
import numpy as np
import pytest
import re
import os
import sys

from neon.backends import gen_backend
from neon.backends.nervanacpu import NervanaCPU
from neon.layers.container import DeltasTree

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


def pytest_addoption(parser):
    '''
    Add a --all option to run the full range of parameters for tests generated using the
    pytest test generators
    '''
    parser.addoption("--all", action="store_true", help="run all tests")
    parser.addoption("--device_id", type=int, default=0, help="GPU device to use")
    return


@pytest.fixture
def device_id(request):
    return request.config.getoption("--device_id")


@pytest.fixture(scope='session')
def data():
    path_to_data = '~/nervana/data/'
    return path_to_data


def get_backend(request, datatype=np.float32):
    be = gen_backend(backend=request.param,
                     datatype=datatype,
                     device_id=request.config.getoption("--device_id"),
                     batch_size=128,
                     rng_seed=0)
    if request.param == 'gpu':
        be.enable_winograd = 2 if be.enable_winograd else be.enable_winograd
    return be


@pytest.fixture(scope='module', params=['gpu', 'cpu'])
def backend_default(request):
    '''
    Fixture to setup the backend before running a test.  Also registers the teardown function to
    clean up the backend after a test is done.  This has module scope, so this will be run once
    for each test in a given test file (module).

    This fixture is parameterized to run both the cpu and gpu backends for every test
    '''
    be = get_backend(request)

    # add a cleanup call - will run after all test in module are done
    def cleanup():
        be = request.getfuncargvalue('backend_default')
        del be
    request.addfinalizer(cleanup)

    # tests using this fixture can access the backend object from
    # backend or use the NervanaObject.be global
    return be


@pytest.fixture(scope='module', params=['gpu'])
def backend_gpu(request):
    '''
    Fixture to setup the backend before running a test.  Also registers the teardown function to
    clean up the backend after a test is done.  This has module scope, so this will be run once
    for each test in a given test file (module).

    This fixture is parameterized to run both the cpu and gpu backends for every test
    '''
    be = get_backend(request)

    # add a cleanup call - will run after all test in module are done
    def cleanup():
        be = request.getfuncargvalue('backend_gpu')
        del be
    request.addfinalizer(cleanup)

    # tests using this fixture can access the backend object from
    # backend or use the NervanaObject.be global
    return be


@pytest.fixture(scope='module', params=['cpu'])
def backend_cpu(request):
    '''
    Fixture that returns a cpu backend using 32 bit dtype.
    For use in tests like gradient checking whihch need higher
    precision
    '''
    be = get_backend(request, datatype=np.float32)

    # add a cleanup call - will run after all tests in module are done
    def cleanup():
        be = request.getfuncargvalue('backend_cpu')
        del be
    request.addfinalizer(cleanup)

    # tests using this fixture can access the backend object from
    # backend or use the NervanaObject.be global
    return be


@pytest.fixture(scope='module', params=['cpu'])
def backend_cpu64(request):
    '''
    Fixture that returns a cpu backend using 64 bit dtype.
    For use in tests like gradient checking whihch need higher
    precision
    '''
    be = get_backend(request, datatype=np.float64)

    # add a cleanup call - will run after all tests in module are done
    def cleanup():
        be = request.getfuncargvalue('backend_cpu64')
        del be
    request.addfinalizer(cleanup)

    # tests using this fixture can access the backend object from
    # backend or use the NervanaObject.be global
    return be


def idfunc(vals):
    '''
    Print out a human readable format for the parameterized tests
    '''
    dtype = re.compile('float\d\d').search('{}'.format(vals[1])).group()
    return '{}_{}'.format(vals[0], dtype)


gpu_cpu_32_16 = itertools.product(['gpu', 'cpu'], [np.float16, np.float32])


@pytest.fixture(scope='module', params=list(gpu_cpu_32_16), ids=idfunc)
def backend_tests(request):
    '''
    Fixture that returns cpu and gpu backends for 16 and 32 bit
    '''
    be = gen_backend(backend=request.param[0],
                     datatype=request.param[1],
                     batch_size=128,
                     device_id=request.config.getoption("--device_id"),
                     rng_seed=0)

    # add a cleanup call - will run after all tests in module are done
    def cleanup():
        be = request.getfuncargvalue('backend_tests')
        del be
    request.addfinalizer(cleanup)

    # tests using this fixture can access the backend object from
    # backend or use the NervanaObject.be global
    return be


def get_backend_pair(device_id, dtype=np.float32, bench=False):
    from neon.backends.nervanagpu import NervanaGPU
    ng = NervanaGPU(default_dtype=dtype, bench=bench, device_id=device_id)
    nc = NervanaCPU(default_dtype=dtype)
    return (ng, nc)


@pytest.fixture(scope='module')
def backend_pair(request):
    ng, nc = get_backend_pair(device_id=request.config.getoption("--device_id"))

    def cleanup():
        ng, nc = request.getfuncargvalue('backend_pair')
        del ng
        del nc
    request.addfinalizer(cleanup)

    return (ng, nc)


@pytest.fixture(scope='module')
def backend_pair_bench(request):
    ng, nc = get_backend_pair(device_id=request.config.getoption("--device_id"), bench=True)

    def cleanup():
        ng, nc = request.getfuncargvalue('backend_pair_bench')
        del ng
        del nc
    request.addfinalizer(cleanup)

    return (ng, nc)


@pytest.fixture(scope='module', params=[np.float16, np.float32])
def backend_pair_dtype(request):
    ng, nc = get_backend_pair(dtype=request.param,
                              device_id=request.config.getoption("--device_id"))

    def cleanup():
        ng, nc = request.getfuncargvalue('backend_pair_dtype')
        del ng
        del nc
    request.addfinalizer(cleanup)

    return (ng, nc)


@pytest.fixture
def deltas_buffer():
    # empty DeltasTree object for tests that need
    # to allocate shared deltas buffers
    return DeltasTree()


@pytest.fixture
def deltas_buffer_wref():
    # returns 2 empty DeltasTree object for
    # tests that need to allocate shared
    # deltas buffers for 2 models
    # (one test, one reference)
    return (DeltasTree(), DeltasTree())
