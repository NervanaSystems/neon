#!/usr/bin/env python
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
from neon import logger as neon_logger
import os
import sys


def get_mkl_lib(device_id=None, verbose=False):
    if sys.platform == 'win32':
        # find *.dll
        current_path = os.path.dirname(os.path.realpath(__file__))
        mkl_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'mklml.dll')
        if not os.path.isfile(mkl_engine_path):
            neon_logger.display("mklml.dll not found; falling back to cpu backend")
            return 0

        mkl_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'mklEngine.dll')
        if not os.path.isfile(mkl_engine_path):
            neon_logger.display("mklEngine.dll not found; falling back to cpu backend")
            return 0

        math_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'cmath.dll')
        if not os.path.isfile(math_engine_path):
            neon_logger.display("cmath.dll not found; falling back to cpu backend")
            return 0

        header_path = os.path.join(os.path.dirname(__file__), 'mklEngine',
                                   'src', 'math_cpu.header')
        if os.path.isfile(header_path):
            neon_logger.display("math_cpu.header not found; falling back to cpu backend")
            return 0
        return 1

    else:
        # find *.so
        current_path = os.path.dirname(os.path.realpath(__file__))
        mkl_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'mklEngine.so')
        if not os.path.isfile(mkl_engine_path):
            neon_logger.display("mklEngine.so not found; falling back to cpu backend")
            return 0

        math_engine_path = os.path.join(current_path, os.pardir, 'mklEngine', 'cmath.so')
        if not os.path.isfile(math_engine_path):
            neon_logger.display("cmath.so not found; falling back to cpu backend")
            return 0

        header_path = os.path.join(os.path.dirname(__file__), 'mklEngine',
                                   'src', 'math_cpu.header')
        if os.path.isfile(header_path):
            neon_logger.display("math_cpu.header not found; falling back to cpu backend")
            return 0
        return 1
