#!/usr/bin/env python
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

import numpy as np
from nose.plugins.attrib import attr
from neon.util.testing import assert_tensor_equal


def m_assert_tensor_equal(t1, t2):
    for _t1, _t2, ctx in zip(t1._tensorlist, t2._tensorlist, t1._ctxs):
        ctx.push()
        assert_tensor_equal(_t1, _t2)
        ctx.pop()


@attr('cuda')
class TestGPU(object):

    def setup(self):
        from neon.backends.mgpu import MGPU, MGPUTensor
        # this code gets called prior to each test
        try:
            self.be = MGPU(rng_seed=0, num_dev=2)
        except AssertionError:
            # likely that only one GPU device is available
            self.be = MGPU(rng_seed=0, num_dev=1)
        self.gpt = MGPUTensor

    def reduction_test(self):
        nr = self.be.num_dev
        if nr == 1:  # This shouldn't be supported
            return
        # create a numpy array as the test-bed
        asize = 9
        # round up to the nearest multiple of num_dev
        bsize = -(-asize // nr) * nr
        h_a = np.random.randn(asize * nr).reshape(
            (nr, asize)).astype(self.be.default_dtype)
        h_result = np.sum(h_a, axis=0, keepdims=True)

        d_a = self.be.empty((1, asize))
        u_a = self.be.empty((1, bsize))
        self.be.scatter(h_a, d_a)
        self.be.reduce(d_a, u_a)
        print(h_result)
        print(d_a.tlist[0].asnumpyarray())

        for i in range(nr):
            np.testing.assert_allclose(d_a.tlist[i].asnumpyarray(),
                                       h_result, atol=1e-6, rtol=0)

    def memset_test(self):
        # create a numpy array as the test-bed
        asize = 9

        h_result = np.zeros((1, asize))
        d_a = self.be.zeros((1, asize))

        for i in range(self.be.num_dev):
            np.testing.assert_allclose(d_a.tlist[i].asnumpyarray(),
                                       h_result, atol=1e-6, rtol=0)

    def frag2rep_test(self):
        nr = self.be.num_dev
        if nr == 1:  # This shouldn't be supported
            return
        np.random.seed(0)
        # create a numpy array as the test-bed
        (rows, cols) = (24, 128)
        indim = rows * cols
        odim = indim * nr

        # h_frags has the data in the order we expect on the device
        h_frags_t = np.random.randn(odim).reshape(
            (nr * cols, rows)).astype(self.be.default_dtype)
        h_frags = h_frags_t.transpose().astype(
            self.be.default_dtype, order='C')

        d_frags = self.be.empty((rows, cols))
        d_frags_t = self.be.empty((cols, rows))

        d_reps = self.be.empty((rows, cols * nr))
        d_reps_t = self.be.empty((cols * nr, rows))

        self.be.scatter(h_frags_t, d_frags_t)
        self.be.transpose(d_frags_t, d_frags)

        np.testing.assert_allclose(d_frags.asnumpyarray(),
                                   h_frags, atol=1e-5, rtol=0)

        self.be.fragment_to_replica(d_frags_t, d_reps_t)
        self.be.transpose(d_reps_t, d_reps)

        for i in range(nr):
            np.testing.assert_allclose(d_frags.asnumpyarray(),
                                       d_reps.tlist[i].asnumpyarray(),
                                       atol=1e-5, rtol=0)
        print("Frag2Rep OK")

        d_frags_t.fill(0)
        self.be.replica_to_fragment(d_reps_t, d_frags_t)
        self.be.transpose(d_frags_t, d_frags)
        for i in range(nr):
            np.testing.assert_allclose(d_frags.asnumpyarray(),
                                       d_reps.tlist[i].asnumpyarray(),
                                       atol=1e-5, rtol=0)
        print("Rep2Frag OK")
