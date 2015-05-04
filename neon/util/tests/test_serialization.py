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

import os

from neon.util.persist import ensure_dirs_exist


class TestSerialize(object):

    def test_dir_creation(self):
        test_dir = os.path.join('.', 'temp_dir')
        test_file = os.path.join(test_dir, 'temp_file.txt')
        assert not os.path.exists(test_file)
        assert not os.path.isdir(test_dir)
        ensure_dirs_exist(test_file)
        try:
            assert os.path.isdir(test_dir)
        finally:
            try:
                os.rmdir(test_dir)
            except OSError:
                pass

    def test_empty_dir_path(self):
        test_file = ('temp_file.txt')
        assert not os.path.exists(test_file)
        assert not os.path.isdir(test_file)
        ensure_dirs_exist(test_file)
        try:
            assert not os.path.isdir(test_file)
            assert not os.path.exists(test_file)
        finally:
            try:
                os.rmdir(test_file)
            except OSError:
                pass
