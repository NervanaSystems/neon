# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
'''
Tools for parsing NEON model definition files (YAML formatted) and
generating NEON model objects from the definition.
'''

import numpy as np
import yaml

from neon import NervanaObject
from neon.layers import GeneralizedCost
from neon.models import Model
from neon.optimizers import optimizer
import neon.transforms as transforms
from neon.util.persist import initialize_layer, initialize_obj


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

        rng_seed (None or int): random number generator seed

        device_id (int): for GPOU backends id of device to use

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

    # cost (before layers for shortcut derivs)
    cost_name = root_yaml['cost']
    cost = GeneralizedCost(costfunc=getattr(transforms, cost_name)())

    # initialize layers
    yaml_layers = root_yaml['layers']
    layers = []
    for i in range(len(yaml_layers)):
        ld = yaml_layers[i]
        l = initialize_layer(ld)
        layers.append(l)

    # initialize model
    model = Model(layers=layers)

    # create optimizer
    optim = None
    if 'optimizer' in root_yaml:
        yaml_opt = root_yaml['optimizer']
        if yaml_opt['type'] == 'MultiOptimizer':
            # multioptimizer init
            for ltype in yaml_opt:
                opt = yaml_opt[ltype]
                if isinstance(opt, dict):
                    if 'schedule' in opt:
                        opt['schedule'] = optimizer.Schedule(opt['schedule'])
                    yaml_opt[ltype] = initialize_obj(opt, optimizer)
            optim = optimizer.MultiOptimizer(yaml_opt)
        else:
            optim = initialize_obj(yaml_opt, optimizer)
    return model, cost, optim
