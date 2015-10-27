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
from neon import initializers
from neon import transforms
from neon import layers


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


def initialize_obj(yamldict, obj_type):
    '''
    Helper function for initializing object defined in YAML configuration
    file.  Given the configuration and the type of object being configured
    will create an instance of the 'obj_type' class using the parameters
    for that object in the config file.

    Arguments:
        yamldict (dict): dictionary represenation of the YAML model
                         configuration file
        obj_type (module): neon module which contains the class definitions
                           for the object being initialized (e.g.
                           neon.initializers.initializer)

    Returns:
        object instance : a newly created instance of an object of
                          type specified in by the 'type' in yamldict

    '''
    # get classname and pop off config since class
    # not used as parameter for initialization
    classname = yamldict.pop('type')

    # get the reference to the class in the module
    obj_class = getattr(obj_type, classname)

    # create an instance of the class with the
    # remaining parameters from the config
    ret_obj = obj_class(**yamldict)

    # put the classname back into the config
    # for later use
    yamldict['type'] = classname
    return ret_obj


def initialize_layer(layerdict):
    '''
    Helper function that instantiates a layer from the configuration
    stored in the YAML configuration files.  This function takes the
    configuration dictionary from a single layer.

    Arguments:
        layerdict (dict): dictionary with the layer configuration
                          parmaters
        lastlayer (bool): set True for final layer in model - used
                          to infer whether shortcut can be used in
                          nonlinear transform bprop calculation

    Returns:
        object instance: a newly created instance of a layer object
    '''
    # type of layer being created
    layer_class = getattr(layers, layerdict.pop('type'))

    # config the layer initializer if specified
    if 'init' in layerdict:
        yaml_init = layerdict.pop('init')
        layerdict['init'] = initialize_obj(yaml_init, initializers)

    # if we have a merge layer, we need to create the layers in its
    # layer container before we create it.
    if 'layer_container' in layerdict:
        lc = layerdict['layer_container']
        new_lc = []
        for obj in lc:
            if isinstance(obj, list):
                new_lc.append([initialize_layer(l) for l in obj])
            else:
                new_lc.append(initialize_layer(l))
        layerdict['layer_container'] = new_lc

    # get_description isn't smart enough to figure
    # out our macros yet, so writing out yaml gives
    # Activation layers, which have transform as an attr.
    if 'bias' in layerdict:
        init = initialize_obj(layerdict['bias'], initializers)
        layerdict['bias'] = init

    if 'activation' in layerdict or 'transform' in layerdict:
        trsf_name = None
        key = None
        if 'activation' in layerdict:
            trsf_name = layerdict.pop('activation')
            key = 'activation'
        else:
            trsf_name = layerdict.pop('transform')['type']
            key = 'transform'

        trsf_class = getattr(transforms, trsf_name)
        layerdict[key] = trsf_class()

    # safe_load / safe_dump prints lists, not tuples
    if 'fshape' in layerdict and isinstance(layerdict['fshape'], list):
        layerdict['fshape'] = tuple(layerdict['fshape'])

    return layer_class(**layerdict)
