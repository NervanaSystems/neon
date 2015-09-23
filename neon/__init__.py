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
Nervana's deep learning library
"""

try:
    from neon.version import VERSION as __version__  # noqa
except ImportError:
    import sys
    print("ERROR: Version information not found.  Ensure you have built "
          "the software.\n    From the top level dir issue: 'make'")
    sys.exit(1)
import inspect
import logging

# setup a preliminary stream based logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def get_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)

    defaults = list(reversed(defaults))
    args = list(reversed(args))

    while len(defaults) != len(args):
        defaults+=(None,)

    return dict(zip(args, defaults))


class NervanaObject(object):
    """
    Base (global) object available to all other classes.

    Args:
        name (str, optional)

    Attributes:
        be (Backend): Hardware backend being used.  See `backends` dir
        name (str, optional): The name assigned to a given instance.
    """
    be = None

    def __init__(self, name=None):
        self.name = name
        self._desc = None

    def get_description(self):
        # shortcut that makes aliases work
        if self._desc:
            return self._desc

        desc = {}
        defaults = get_args(self.__init__)
        for arg in defaults:
            if arg == 'self':
                continue

            # all args need to go in the __dict__ so we can read
            # them out the way they were read in. alternatively,
            # you can override get_description to say how to
            # put them in to the description dictionary.
            if arg in self.__dict__:
                if self.__dict__[arg] != defaults[arg]:
                    if isinstance(self.__dict__[arg], NervanaObject):
                        desc[arg] = self.__dict__[arg].get_description()
                    else:
                        desc[arg] = self.__dict__[arg]
            else:
                logger.warning("can't describe argument '{}' to {}".format(arg, self))

        desc['type'] = self.__class__.__name__
        self._desc = desc
        return desc
