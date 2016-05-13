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
import logging

from neon.models.model import Model
from neon.callbacks.callbacks import Callbacks
from neon.util.persist import load_class, load_obj

logger = logging.getLogger(__name__)


def deserialize(fn, datasets=None, inference=False):
    """
    Helper function to load all objects from a serialized file,
    this includes callbacks and datasets as well as the model, layers,
    etc.

    Arguments:
        datasets (DataSet, optional): If the dataset is not serialized
                                      in the file it can be passed in
                                      as an argument.  This will also
                                      override any dataset in the serialized
                                      file
        inference (bool, optional): if true only the weights will be loaded, not
                                    the states
    Returns:
        Model: the model object
        Dataset: the data set object
        Callback: the callbacks
    """
    config_dict = load_obj(fn)

    if datasets is not None:
        logger.warn('Ignoring datasets serialized in archive file %s' % fn)
    elif 'datasets' in config_dict:
        ds_cls = load_class(config_dict['datasets']['type'])
        dataset = ds_cls.gen_class(config_dict['datasets']['config'])
        datasets = dataset.gen_iterators()

    if 'train' in datasets:
        data_iter = datasets['train']
    else:
        key = list(datasets.keys())[0]
        data_iter = datasets[key]
        logger.warn('Could not find training set iterator'
                    'using %s instead' % key)

    model = Model(config_dict, data_iter)

    callbacks = None
    if 'callbacks' in config_dict:
        # run through the callbacks looking for dataset objects
        # replace them with the corresponding data set above
        cbs = config_dict['callbacks']['callbacks']
        for cb in cbs:
            if 'config' not in cb:
                cb['config'] = {}
            for arg in cb['config']:
                if type(cb['config'][arg]) is dict and 'type' in cb['config'][arg]:
                    if cb['config'][arg]['type'] == 'Data':
                        key = cb['config'][arg]['name']
                        if key in datasets:
                            cb['config'][arg] = datasets[key]
                        else:
                            cb['config'][arg] = None
        # now we can generate the callbacks
        callbacks = Callbacks.load_callbacks(config_dict['callbacks'], model)
    return (model, dataset, callbacks)
