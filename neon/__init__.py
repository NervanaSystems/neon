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
Nervana's deep learning library
"""
from __future__ import print_function
from builtins import zip

try:
    from neon.version import VERSION as __version__  # noqa
except ImportError:
    import sys
    print("ERROR: Version information not found.  Ensure you have built "
          "the software.\n    From the top level dir issue: 'make'")
    sys.exit(1)
from copy import deepcopy
import inspect
import logging


DISPLAY_LEVEL_NUM = 41
logging.addLevelName(DISPLAY_LEVEL_NUM, "DISPLAY")


def display(self, message, *args, **kwargs):
    if self.isEnabledFor(DISPLAY_LEVEL_NUM):
        self._log(DISPLAY_LEVEL_NUM, message, args, **kwargs)

logging.Logger.display = display

# setup a preliminary stream based logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def get_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)

    defaults = list(reversed(defaults)) if defaults is not None else []
    args = list(reversed(args))

    while len(defaults) != len(args):
        defaults += (None,)

    return dict(list(zip(args, defaults)))


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
    __counter = 0

    def __init__(self, name=None):
        if name is None:
            name = '{}_{}'.format(self.classnm, self.__counter)
        self.name = name
        self._desc = None
        type(self).__counter += 1

    @classmethod
    def gen_class(cls, pdict):
        return cls(**pdict)

    def __del__(self):
        type(self).__counter -= 1

    @property
    def classnm(self):
        """
        Convenience method for getting the class name
        """
        return self.__class__.__name__

    @property
    def modulenm(self):
        """
        Convenience method for getting the full module path
        """
        return self.__class__.__module__ + '.' + self.__class__.__name__

    def get_description(self, skip=[], **kwargs):
        if type(skip) is not list:
            skip = list(skip)
        else:
            skip = deepcopy(skip)
        skip.append('self')

        config = {}
        defaults = get_args(self.__init__)
        for arg in defaults:
            if arg in skip:
                continue

            # all args need to go in the __dict__ so we can read
            # them out the way they were read in. alternatively,
            # you can override get_description to say how to
            # put them in to the description dictionary.
            if arg in self.__dict__:
                if self.__dict__[arg] != defaults[arg]:
                    if isinstance(self.__dict__[arg], NervanaObject):
                        config[arg] = self.__dict__[arg].get_description()
                    else:
                        config[arg] = self.__dict__[arg]
            else:
                logger.warning("can't describe argument '{}' to {}".format(arg, self))

        desc = {'type': self.modulenm, 'config': config}
        self._desc = desc
        return desc
