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
'''
General functions for running the unit tests via pytest.
'''

import itertools
import numpy as np
import pytest

from neon.backends import gen_backend


def pytest_addoption(parser):
    '''
    Add a --all option to run the full range of parameters for tests generated using the
    pytest test generators
    '''
    parser.addoption("--all", action="store_true",
                     help="run all tests")
    return

@pytest.fixture(scope='session')
def data():
   path_to_data = '~/nervana/data/'
   return path_to_data

@pytest.fixture(scope='module', params=['gpu', 'cpu'])
def backend_default(request):
    '''
    Fixture to setup the backend before running a test.  Also registers the teardown function to
    clean up the backend after a test is done.  This has module scope, so this will be run once
    for each test in a given test file (module).

    This fixture is parameterized to run both the cpu and gpu backends for every test
    '''
    be = gen_backend(backend=request.param,
                     datatype=np.float32,
                     batch_size=128,
                     rng_seed=0)

    # add a cleanup call - will run after all test in module are done
    def cleanup():
        be = request.getfuncargvalue('backend_default')
        del be
    request.addfinalizer(cleanup)

    # tests using this fixture can access the backend object from
    # backend or use the NervanaObject.be global
    return be


@pytest.fixture(scope='module')
def backend_cpu64(request):
    '''
    Fixture that returns a cpu backend using 64 bit dtype.
    For use in tests like gradient checking whihch need higher
    precision
    '''
    be = gen_backend(backend='cpu',
                     datatype=np.float64,
                     batch_size=128,
                     rng_seed=0)

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
    dtype = str(vals[1])
    dtype = dtype.split("numpy.")[1].strip("'>")
    return vals[0] + '_' + dtype

gpu_cpu_32_16 = itertools.product(['gpu','cpu'], [np.float16, np.float32])
@pytest.fixture(scope='module', params=list(gpu_cpu_32_16), ids=idfunc)
def backend_tests(request):
    '''
    Fixture that returns cpu and gpu backends for 16 and 32 bit
    '''
    be = gen_backend(backend=request.param[0],
                     datatype=request.param[1],
                     batch_size=128,
                     rng_seed=0)

    # add a cleanup call - will run after all tests in module are done
    def cleanup():
        be = request.getfuncargvalue('backend_tests')
        del be
    request.addfinalizer(cleanup)

    # tests using this fixture can access the backend object from
    # backend or use the NervanaObject.be global
    return be
