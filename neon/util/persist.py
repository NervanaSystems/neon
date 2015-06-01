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
Utility functions for saving various types of objects state.
"""

import logging
import os
import yaml

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


def convert_scalar_node(val):
    """
    Helper to extract and return the appropriately types value of a ScalarNode
    object.

    Arguments:
        val: (yaml.nodes.ScalarNode): object to extract value from

    Returns:
        float, int, string: the actual value
    """
    if not isinstance(val, yaml.nodes.ScalarNode):
        return val
    if val.tag.endswith("int"):
        return int(val.value)
    elif val.tag.endswith("float"):
        return float(val.value)
    else:
        # assume a string
        return val.value


def extract_child_node_vals(node, keys):
    """
    Helper to iterate through the immediate children of the yaml node object
    passed, looking for the key values specified.

    Arguments:
        node (yaml.nodes.Node): the parent node upon which to begin the search
        keys (list): set of strings indicating the child keys we want to
                     extract corresponding values for.

    Returns:
        dict: with one item for each key.  value is value found in search for
              that key, or None if not found.
    """
    res = dict()
    for child in node.value:
        # child node values are two element tuples, where the first is a scalar
        # node, and the second can be other types of nodes.
        tag = child[0].value
        if isinstance(child[1], yaml.nodes.ScalarNode):
            val = convert_scalar_node(child[1])
        elif isinstance(child[1], yaml.nodes.SequenceNode):
            val = [convert_scalar_node(x) for x in child[1].value]
        elif isinstance(child[1], yaml.nodes.MappingNode):
            val = dict()
            for item in child[1].value:
                val[item[0].value] = convert_scalar_node(item[1])
        else:
            logger.warning("unknown node type: %s, ignoring tag %s",
                           str(type(child[1])), tag)
            val = None
        for key in keys:
            if tag == key:
                res[key] = val
    for key in keys:
        if key not in res:
            res[key] = None
    return res


def obj_multi_constructor(loader, tag_suffix, node):
    """
    Utility function used to actually import and generate a new class instance
    from its name and parameters

    Arguments:
        loader (yaml.loader.SafeLoader): carries out actual loading
        tag_suffix (str): The latter portion of the tag, representing the full
                          module and class name of the object being
                          instantiated.
        node (yaml.MappingNode): tag/value set specifying the parameters
                                 required for constructing new objects of this
                                 type
    """
    # extract class name and import neccessary module.
    parts = tag_suffix.split('.')
    module = '.'.join(parts[:-1])
    try:
        cls = __import__(module)
    except ImportError as err:
        # we allow a shortcut syntax that skips neon. from import path, try
        # again with this prepended
        if parts[0] != "neon":
            parts.insert(0, "neon")
            module = '.'.join(parts[:-1])
            cls = __import__(module)
            if 'datasets' in parts:
                # clear any previous datasets loaded with a different backend
                cls.datasets.dataset.Dataset.inputs = {
                    'train': None, 'test': None, 'validation': None}
                cls.datasets.dataset.Dataset.targets = {
                    'train': None, 'test': None, 'validation': None}
        else:
            raise err
    for comp in parts[1:]:
        cls = getattr(cls, comp)

    # need to create a new object
    try:
        res = cls(**loader.construct_mapping(node, deep=True))
    except TypeError as e:
        logger.warning("Unable to construct '%s' instance.  Error: %s",
                       cls.__name__, e.message)
        res = None
    return res


def initialize_yaml():
    yaml.add_multi_constructor('!obj:', obj_multi_constructor,
                               yaml.loader.SafeLoader)


def deserialize(load_path, verbose=True):
    """
    Converts a serialized object into a python data structure.  We currently
    support reading from the following file formats (expected filename
    extension in brackets):

        * python pickle (.pkl)
        * YAML (.yaml)

    Arguments:
        load_path (str, File): path and name of the serialized on-disk file to
                               load (or an already loaded file object).
                               The type to write is inferred based on filename
                               extension.  If no extension given, pickle format
                               is attempted.

    Returns:
        object: Converted in-memory python data structure.

    See Also:
        serialize
    """
    if isinstance(load_path, str):
        load_path = open(os.path.expandvars(os.path.expanduser(load_path)))
    fname = load_path.name

    if verbose:
        logger.warn("deserializing object from:  %s", fname)

    if (fname.lower().endswith('.yaml') or fname.lower().endswith('.yml')):
        initialize_yaml()
        return yaml.safe_load(load_path)
    else:
        try:
            return pickle.load(load_path)
        except AttributeError:
            msg = ("Problems deserializing: %s.  Its possible the interface "
                   "for this object has changed since being serialized.  You "
                   "may need to remove and recreate it." % load_path)
            logger.error(msg)
            raise AttributeError(msg)


def serialize(obj, save_path, verbose=True):
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
        deserialize
    """
    if save_path is None or len(save_path) == 0:
        return
    save_path = os.path.expandvars(os.path.expanduser(save_path))
    if verbose:
        logger.warn("serializing object to: %s", save_path)
    ensure_dirs_exist(save_path)

    pickle.dump(obj, open(save_path, 'wb'), -1)


class YAMLable(yaml.YAMLObject):

    """
    Base class for any objects we'd like to be able to safely parse from yaml
    configuration strems (or dump suitable representation back out to such a
    stream).
    """
    yaml_loader = yaml.SafeLoader
