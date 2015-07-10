#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from nose.plugins.attrib import attr

from neon.datasets.cifar100 import CIFAR100
from neon.backends.cpu import CPU


class TestCIFAR100(object):

    tmp_repo = os.path.join(os.path.dirname(__file__), 'repo')

    def setup(self):
        os.makedirs(self.tmp_repo)

    def teardown(self):
        shutil.rmtree(self.tmp_repo, ignore_errors=True)

    @attr('slow')
    def test_fine_labels(self):
        data = CIFAR100(coarse=False, repo_path=self.tmp_repo)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = 128
        data.load()
        assert len(data.inputs['train']) == 50000
        assert len(data.targets['train'][0]) == 100

    @attr('slow')
    def test_coarse_labels(self):
        data = CIFAR100(coarse=True, repo_path=self.tmp_repo)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = 128
        data.load()
        assert len(data.inputs['train']) == 50000
        assert len(data.targets['train'][0]) == 20
