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
import logging
import os

from neon.util.compat import pickle

logger = logging.getLogger(__name__)


def ensure_dirs_exist(path):
    """
    Simple helper that ensures that any directories specified in the path are
    created prior to use.

    Arguments:
        path (str): the path (may be to a file or directory).  Any intermediate
                    directories will be created.

    Returns:
        str: The unmodified path value.
    """
    outdir = os.path.dirname(path)
    if outdir != '' and not os.path.isdir(outdir):
        os.makedirs(outdir)
    return path


def save_obj(obj, save_path):
    """
    Dumps a python data structure to a saved on-disk representation.  We
    currently support writing to the following file formats (expected filename
    extension in brackets):

        * python pickle (.pkl)

    Arguments:
        obj (object): the python object to be saved.
        save_path (str): Where to write the serialized object (full path and
                         file name)

    See Also:
        :py:func:`~neon.models.model.Model.serialize`
    """
    if save_path is None or len(save_path) == 0:
        return
    save_path = os.path.expandvars(os.path.expanduser(save_path))
    logger.debug("serializing object to: %s", save_path)
    ensure_dirs_exist(save_path)

    pickle.dump(obj, open(save_path, 'wb'), -1)


def load_obj(load_path):
    """
    Loads a saved on-disk representation to a python data structure. We
    currently support the following file formats:

        * python pickle (.pkl)

    Arguments:
        load_path (str): where to the load the serialized object (full path
                            and file name)

    """
    if isinstance(load_path, str):
        load_path = os.path.expandvars(os.path.expanduser(load_path))
        if load_path.endswith('.gz'):
            import gzip
            load_path = gzip.open(load_path)
        else:
            load_path = open(load_path)
    fname = load_path.name

    logger.debug("deserializing object from:  %s", fname)
    try:
        return pickle.load(load_path)
    except AttributeError:
        msg = ("Problems deserializing: %s.  Its possible the interface "
               "for this object has changed since being serialized.  You "
               "may need to remove and recreate it." % load_path)
        logger.error(msg)
        raise AttributeError(msg)


def load_class(ctype):
    """
    Helper function to take a string with the neon module and
    classname then import and return  the class object

    Arguments:
        ctype (str): string with the neon module and class
                     (e.g. 'neon.layers.layer.Linear')
    Returns:
        class
    """
    # extract class name and import neccessary module.
    class_path = ctype
    parts = class_path.split('.')
    module = '.'.join(parts[:-1])
    try:
        cls = __import__(module)
    except ImportError as err:
        # we allow a shortcut syntax that skips neon
        # from import path, try again with this prepended
        if parts[0] != "neon":
            parts.insert(0, "neon")
            module = '.'.join(parts[:-1])
            cls = __import__(module)
        else:
            raise err
    for comp in parts[1:]:
        cls = getattr(cls, comp)
    return cls


def serialize(model, callbacks=None, datasets=None, dump_weights=True, keep_states=True):
    pdict = model.serialize(fn=None, keep_states=keep_states)
    if callbacks is not None:
        pdict['callbacks'] = callbacks.serialize()

    if datasets is not None:
        pdict['datasets'] = datasets.serialize()

    return pdict
