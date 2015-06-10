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
Contains various functions for checking and setting required and optional
parameters.
"""

import numpy as np


def req_param(obj, paramlist):
    for param in paramlist:
        if not hasattr(obj, param):
            raise ValueError("req param %s missing for %s" % (param,
                             obj.__class__.__name__))


def opt_param(obj, paramlist, default_value=None):
    for param in paramlist:
        if not hasattr(obj, param):
            setattr(obj, param, default_value)


def ensure_dtype(dtype):
    """
    Check if the provided dtype is in string format, and if so, convert to
    actual dtype.
    """
    if dtype in ['float16', 'np.float16', 'numpy.float16']:
        dtype = np.float16
    elif dtype in ['float32', 'np.float32', 'numpy.float32']:
        dtype = np.float32
    elif dtype in ['float64', 'np.float64', 'numpy.float64']:
        dtype = np.float64
    elif dtype in [np.float32, np.float16, np.float64]:
        pass
    else:
        raise ValueError('Datatype not understood')
    return dtype
