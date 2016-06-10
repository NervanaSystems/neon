# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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
Contains various functions and wrappers to make code Python 2 and Python 3
compatible.
"""
from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below

import sys  # noqa
import logging  # noqa


logger = logging.getLogger(__name__)
PY3 = (sys.version_info[0] >= 3)

if PY3:
    xrange = range
else:
    xrange = xrange  # pylint: disable=xrange-builtin

if not PY3:
    import cPickle as the_pickle  # noqa
else:
    import pickle as the_pickle  # noqa
pickle = the_pickle


def pickle_load(filepath):
    """
    Py2Py3 compatible Pickle load

    Arguments:
        filepath (str): File containing pickle data stream to load

    Returns:
        Unpickled object
    """
    if PY3:
        return pickle.load(filepath, encoding='latin1')
    else:
        return pickle.load(filepath)
