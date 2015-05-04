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
"""
Facility to optionally trace through function calls
"""
import inspect
import os
import types
import numpy
import sys
import re
import logging


class Tracer(object):
    file_filter = ".*"
    logger = None

    @classmethod
    def setup(cls, file_filter, output_file):
        # setup logger
        cls.logger = logging.getLogger('neon.trace')
        cls.logger.setLevel(logging.DEBUG)
        cls.logger.propagate = False
        fh = logging.FileHandler(output_file, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        cls.logger.addHandler(fh)

        # copy filetr
        cls.file_filter = file_filter

        # enable trace
        sys.settrace(tracefunc)


def get_class_name(frame):
    try:
        class_name = frame.f_locals['self'].__class__.__name__
    except KeyError:
        class_name = "NotAClass"

    return class_name


def get_function_args(frame):
    legal_types = (types.NoneType, types.BooleanType, types.IntType,
                   types.LongType, types.FloatType, types.StringType,
                   numpy.ndarray, numpy.generic)
    args, _, _, values = inspect.getargvalues(frame)
    str = '('
    num_printed = 0
    for i in args:
        if type(i) in legal_types and type(values[i]) in legal_types:
            str += "%s = %s, " % (i, values[i])
            num_printed += 1
    if num_printed != len(args):
        str += '..some omitted.., '
    if len(args):
        str = str[:-2]
    str += ')'
    return str


# experimental
def get_function_args1(frame):
    argvalues = inspect.getargvalues(frame)
    return inspect.formatargvalues(*argvalues)


def tracefunc(frame, event, arg, indent=[0]):
    frame_info = None
    try:
        frame_info = inspect.getframeinfo(frame)
        file_name = os.path.basename(frame_info.filename)

        # skip if not interested in the file
        if not re.search(Tracer.file_filter, file_name):
            return

        if event == "call":
            indent[0] += 2
            print_str = "-" * indent[0] + \
                        file_name + \
                        ": " + str(frame_info.lineno) +\
                        ": " + get_class_name(frame) + \
                        ": call > " + frame.f_code.co_name + \
                        get_function_args(frame)
        elif event == "return":
            print_str = "-" * indent[0] + \
                        file_name + \
                        ": " + str(frame_info.lineno) +\
                        ": " + get_class_name(frame) + \
                        ": ret < " + frame.f_code.co_name
            indent[0] -= 2
        else:
            return
        Tracer.logger.debug(print_str)  # print print_str
    finally:
        # safe delete of stack frame references
        del frame_info
        del frame
    return tracefunc
