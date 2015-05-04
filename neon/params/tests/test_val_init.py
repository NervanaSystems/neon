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

import math

from neon.backends.cpu import CPU
from neon.params.val_init import (UniformValGen, AutoUniformValGen,
                                  GaussianValGen, NormalValGen,
                                  SparseEigenValGen, NodeNormalizedValGen)


class TestValInit(object):

    def __init__(self):
        # this code gets called prior to each test
        self.be = CPU()

    def test_uni_basics(self):
        uni = UniformValGen(backend=self.be)
        assert str(uni) == ("UniformValGen utilizing CPU backend\n\t"
                            "low: 0.0, high: 1.0")

    def test_uni_gen(self):
        uni = UniformValGen(backend=self.be)
        res = uni.generate(shape=[1, 1])
        assert res.shape == (1, 1)
        out = self.be.empty((1, 1))
        self.be.min(res, axes=None, out=out)
        assert out.asnumpyarray() >= 0.0
        self.be.max(res, axes=None, out=out)
        assert out.asnumpyarray() < 1.0

    def test_uni_params(self):
        low = -5.5
        high = 10.2
        uni = UniformValGen(backend=self.be, low=low, high=high)
        assert str(uni) == ("UniformValGen utilizing CPU backend\n\t"
                            "low: {low}, high: {high}".format(low=low,
                                                              high=high))
        res = uni.generate(shape=[4, 4])
        assert res.shape == (4, 4)
        out = self.be.empty((1, 1))
        self.be.min(res, axes=None, out=out)
        assert out.asnumpyarray() >= low
        self.be.max(res, axes=None, out=out)
        assert out.asnumpyarray() < high

    def test_autouni_gen(self):
        autouni = AutoUniformValGen(backend=self.be, relu=True)
        assert autouni.relu is True
        assert str(autouni) == ("AutoUniformValGen utilizing CPU backend\n\t"
                                "low: nan, high: nan")
        res = autouni.generate([3, 3])
        assert res.shape == (3, 3)
        out = self.be.empty((1, 1))
        self.be.min(res, axes=None, out=out)
        expected_val = math.sqrt(2) * (1.0 / math.sqrt(3))
        assert out.asnumpyarray() >= - expected_val
        self.be.max(res, axes=None, out=out)
        assert out.asnumpyarray() < expected_val

    def test_gaussian_gen(self):
        loc = 5
        scale = 2.0
        gauss = GaussianValGen(backend=self.be, loc=loc, scale=scale)
        assert str(gauss) == ("GaussianValGen utilizing CPU backend\n\t"
                              "loc: {}, scale: {}".format(loc, scale))
        res = gauss.generate([5, 10])
        assert res.shape == (5, 10)
        # TODO: test distribution of vals to ensure ~gaussian dist

    def test_normal_gen(self):
        loc = -2.5
        scale = 3.0
        gauss = NormalValGen(backend=self.be, loc=loc, scale=scale)
        assert str(gauss) == ("GaussianValGen utilizing CPU backend\n\t"
                              "loc: {}, scale: {}".format(loc, scale))
        res = gauss.generate([9, 3])
        assert res.shape == (9, 3)
        # TODO: test distribution of vals to ensure ~gaussian dist

    def test_sparseeig_gen(self):
        sparseness = 10
        eigenvalue = 3.1
        eig = SparseEigenValGen(backend=self.be, sparseness=sparseness,
                                eigenvalue=eigenvalue)
        assert str(eig) == ("SparseEigenValGen utilizing CPU backend\n\t"
                            "sparseness: {}, eigenvalue: "
                            "{}".format(sparseness, eigenvalue))
        res = eig.generate([20, 20])
        assert res.shape == (20, 20)
        # TODO: test distribution of vals

    def test_nodenorm_gen(self):
        scale = 3.0
        nodenorm = NodeNormalizedValGen(backend=self.be, scale=scale)
        assert str(nodenorm) == ("NodeNormalizedValGen utilizing CPU backend"
                                 "\n\tscale: {}".format(scale))
        res = nodenorm.generate([8, 9])
        assert res.shape == (8, 9)
        out = self.be.empty((1, 1))
        self.be.min(res, axes=None, out=out)
        expected_val = scale * math.sqrt(6) / math.sqrt(8 + 9.)
        assert out.asnumpyarray() >= - expected_val
        self.be.max(res, axes=None, out=out)
        assert out.asnumpyarray() < expected_val
