# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
Tools for parsing neon model definition files (YAML formatted) and
generating neon model objects from the definition.
"""
from copy import deepcopy
import numpy as np
import yaml

from neon import NervanaObject
from neon.layers import GeneralizedCost
from neon.models import Model
import neon.optimizers
from neon.layers.container import Sequential


def format_yaml_dict(yamldict, type_prefix):
    """
    Helper function for format the YAML model config into
    the proper format for object and layer initialization

    Arguments:
        yamldict (dict): dictionary with model parameters

        type_prefix (str): module path for this object

    Returns:
        dict : formatted dict
    """
    yamldict['type'] = type_prefix + yamldict['type']
    return yamldict


def create_objects(root_yaml,
                   be_type='gpu',
                   batch_size=128,
                   rng_seed=None,
                   device_id=0,
                   default_dtype=np.float32,
                   stochastic_rounding=False):
    """
    Instantiate objects as per the given specifications.

    Arguments:
        root_yaml (dict): Model definition dictionary parse from YAML file

        be_type (str): backend either 'gpu', 'mgpu' or 'cpu'

        batch_size (int): Batch size.
        rng_seed (None or int): random number generator seed

        device_id (int): for GPU backends id of device to use

        default_dtype (type): numpy data format for default data types,

        stochastic_rounding (bool or int): number of bits for stochastic rounding
                                           use False for no rounding

    Returns:
        tuple: Contains model, cost and optimizer objects.
    """

    assert NervanaObject.be is not None, 'Must generate a backend before running this function'

    # can give filename or parse dictionary
    if type(root_yaml) is str:
        with open(root_yaml, 'r') as fid:
            root_yaml = yaml.safe_load(fid.read())

    # in case references were used
    root_yaml = deepcopy(root_yaml)

    # initialize layers
    yaml_layers = root_yaml['layers']

    # currently only support sequential in yaml
    layer_dict = {'layers': yaml_layers}
    layers = Sequential.gen_class(layer_dict)

    # initialize model
    model = Model(layers=layers)

    # cost (before layers for shortcut derivs)
    cost_name = root_yaml['cost']
    cost = GeneralizedCost.gen_class({'costfunc': {'type': cost_name}})

    # create optimizer
    opt = None
    if 'optimizer' in root_yaml:
        yaml_opt = root_yaml['optimizer']
        typ = yaml_opt['type']
        opt = getattr(neon.optimizers, typ).gen_class(yaml_opt['config'])

    return model, cost, opt
