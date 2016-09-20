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
import importlib
import logging
import os
import pkgutil
import sys
import appdirs

from neon.util.compat import pickle, pickle_load

logger = logging.getLogger(__name__)


def get_cache_dir(subdir=None):
    """
    Function for getting cache directory to store reused files like kernels, or scratch space
    for autotuning, etc.
    """
    cache_dir = os.environ.get("NEON_CACHE_DIR")

    if cache_dir is None:
        cache_dir = appdirs.user_cache_dir("neon", "neon")

    if subdir:
        subdir = subdir if isinstance(subdir, list) else [subdir]
        cache_dir = os.path.join(cache_dir, *subdir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir


def get_data_cache_dir(data_dir, subdir=None):
    """
    Function for getting cache directory to store data cache files.

    Since the data cache contains large files, it is ideal to control the
    location independently from the system cache, which defaults to
    the user homedir if not otherwise specified.

    This function will make the directory if it doesn't yet exist.

    Arguments:
        data_dir (str): the dir to use if NEON_DATA_CACHE_DIR is not
                        present in the environment.
        subdir (str): sub directory inside of the cache dir that should
                      be returned.
    """
    data_cache_dir = os.environ.get("NEON_DATA_CACHE_DIR")

    if data_cache_dir is None:
        data_cache_dir = data_dir

    if subdir:
        subdir = subdir if isinstance(subdir, list) else [subdir]
        data_cache_dir = os.path.join(data_cache_dir, *subdir)

    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    return data_cache_dir


def get_data_cache_or_nothing(subdir=None):
    cache_root = os.environ.get("NEON_DATA_CACHE_DIR")

    if cache_root is None:
        cache_root = ''
    else:
        cache_root = ensure_dirs_exist(os.path.join(cache_root, subdir if subdir else ""))

    return cache_root


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

    pickle.dump(obj, open(save_path, 'wb'), 2)


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
            load_path = gzip.open(load_path, 'rb')
        else:
            load_path = open(load_path, 'rb')
    fname = load_path.name

    logger.debug("deserializing object from:  %s", fname)
    try:
        return pickle_load(load_path)
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
        clss = __import__(module)
        for comp in parts[1:]:
            clss = getattr(clss, comp)
        return clss
    except (ValueError, ImportError) as err:
        if len(module) == 0:
            # try to find the module inside neon
            pkg = sys.modules['neon']
            prfx = pkg.__name__ + '.'
            for imptr, nm, _ in pkgutil.iter_modules(pkg.__path__, prefix=prfx):
                mod = importlib.import_module(nm)
                if hasattr(mod, ctype):
                    return getattr(mod, ctype)
        raise err


def serialize(model, callbacks=None, datasets=None, dump_weights=True, keep_states=True):
    """
    Serialize the model, callbacks and datasets.

    Arguments:
        model (Model): Model object
        callbacks (Callbacks, optional): Callbacks
        datasets (iterable, optional): Datasets
        dump_weights (bool, optional): Ignored
        keep_states (bool, optional): Whether to save optimizer states too.

    Returns:
        dict: Model data, callbacks and datasets

    """
    pdict = model.serialize(fn=None, keep_states=keep_states)
    if callbacks is not None:
        pdict['callbacks'] = callbacks.serialize()

    if datasets is not None:
        pdict['datasets'] = datasets.serialize()

    return pdict
