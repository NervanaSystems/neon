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
Contains various functions and wrappers to make code python 2 and python 3
compatible
"""

import sys
import logging


logger = logging.getLogger(__name__)
PY3 = (sys.version_info[0] >= 3)

# keep range calls consistent between python 2 and 3
# note: if you need a list and not an iterator you can do list(range(x))
range = range
if not PY3:
    logger.info("using xrange as range")
    range = xrange

# keep cPickle, Queue, StringIO import consistent between python 2 and 3 (where
# each was renamed)
if not PY3:
    import cPickle as the_pickle
    import Queue as the_queue
    from StringIO import StringIO
else:
    import pickle as the_pickle
    import queue as the_queue
    from io import StringIO

pickle = the_pickle
queue = the_queue
StringIO = StringIO
