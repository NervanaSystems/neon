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
Utility functions for testing
"""
import numpy as np
import numpy.random as nprnd


def sparse_rand(shape, frac=0.05, round_up=False):
    # generate an input with sparse activation
    # in the input dimension for LSTM testing
    # frac is the fraction of the matrix elements
    # which will be nonzero. Set round_up to
    # True to get a binary matrix, i.e. elements
    # are either set to 0 or 1
    num_el = np.prod(shape)
    inds = nprnd.permutation(num_el)[0:int(frac*num_el)]

    # draw frac*num_el random numbers
    vals = nprnd.random(inds.size)

    if round_up:
        vals = np.ceil(vals)
    out = np.zeros(shape)
    out.flat[inds] = vals
    return (out, inds)


def allclose_with_out(x, y, atol=0.0, rtol=1.0e-5):
    # run the np.allclose on x and y
    # if it fails print some stats
    # before returning
    ac = np.allclose(x, y, rtol=rtol, atol=atol)
    if not ac:
        dd = np.abs(x-y)
        print 'abs errors: %e [%e, %e] Abs Thresh = %e' \
              % (np.median(dd), np.min(dd), np.max(dd), atol)
        amax = np.argmax(dd)
        print 'worst case: %e %e' % (x.flat[amax], y.flat[amax])
        dd = np.abs(dd - atol)/np.abs(y)
        print 'rel errors: %e [%e, %e] Rel Thresh = %e' \
              % (np.median(dd), np.min(dd), np.max(dd), rtol)
        amax = np.argmax(dd)
        print 'worst case: %e %e' % (x.flat[amax], y.flat[amax])
    return ac


def symallclose(x, y, rtol=1.0e-5):
    # symetric relative allclose function
    # checks abs(x-y)/(abs(x) + abs(y))
    dd = np.divide(np.abs(x-y), np.abs(x) + np.abs(y))
    return all(np.less_equal(dd, rtol))
