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
from builtins import zip
import numpy as np
import re
from neon.util.persist import load_obj
from neon import logger as neon_logger


class ModelDescription(dict):
    """
    Container class for the model serialization dictionary.  Provides
    helper methods for searching and manipulating the dictionary.

    Arguments:
        pdict (dict or str): the configuration dictionary generated
                             by Model.serialize() or the name of a
                             pickle file containing that dictionary
    """
    def __init__(self, pdict):
        if type(pdict) is str:
            pdict = load_obj(pdict)
        super(ModelDescription, self).__init__(pdict)

    @property
    def version(self):
        """
        Print neon version.

        Returns:
            str: version string

        """
        return self['neon_version']

    def layers(self, field='name', regex=None):
        """
        Print out the layer names in the model with some
        options for filtering the results.

        Arguments:
            field (str, optional): the configuration field to file against
                                   (e.g. layer 'name')
            regex (str, optional): regular expression to apply to field
                                   to file the results (e.g. "conv")

        Example:
            layers(field='name', regex='conv') will return all layers
            with the name containing "conv"
        """
        if regex is not None:
            regex = re.compile(regex)
        return self.find_layers(self['model']['config'], field, regex=regex)

    @staticmethod
    def find_layers(layers, field, regex=None):
        """
        Print out the layer names in the model with some
        options for filtering the results.

        Arguments:
            layers (dict): model configuration dictionary
            field (str, optional): the configuration field to file against
                                   (e.g. layer 'name')
            regex (str, optional): regular expression to apply to field
                                   to file the results (e.g. "conv")

        Returns:
            list of dict: Layer config dictionary
        """
        matches = []
        for l in layers['layers']:
            if field in l['config']:
                value = l['config'][field]
                if regex is None or regex.match(value):
                    matches.append(value)
            if type(l) is dict and 'layers' in l['config']:
                matches.extend(ModelDescription.find_layers(l['config'], field, regex=regex))
        return matches

    def getlayer(self, layer_name):
        """
        Find a layer by its name.

        Arguments:
            name (str): name of the layer

        Returns:
            dict: Layer config dictionary
        """
        return self.find_by_name(self['model']['config'], layer_name)

    @staticmethod
    def find_by_name(layers, layer_name):
        """
        Find a layer by its name.

        Arguments:
            layers (dict): model configuration dictionary
            layer_name (str) name of the layer

        Returns:
            dict: Layer config dictionary
        """
        for l in layers['layers']:
            if 'name' in l['config'] and l['config']['name'] == layer_name:
                    return l
            if type(l) is dict and 'config' in l and 'layers' in l['config']:
                val = ModelDescription.find_by_name(l['config'], layer_name)
                if val is not None:
                    return val

    @staticmethod
    def match(o1, o2):
        """
        Compare two ModelDescription object instances

        Arguments:
            o1 (ModelDescription, dict): object to compare
            o2 (ModelDescription, dict): object to compare

        Returns:
            bool: true if objects match
        """
        type_o1 = type(o1)
        if type_o1 is not type(o2):
            return False

        if type_o1 is dict:
            if set(o1.keys()) != set(o2.keys()):
                neon_logger.display('Missing keys')
                return False
            for key in o1:
                if key == 'name':
                    # ignore layer names
                    return True
                if not ModelDescription.match(o1[key], o2[key]):
                    return False
        elif any([type_o1 is x for x in [list, tuple]]):
            if len(o1) != len(o2):
                return False
            for val1, val2 in zip(o1, o2):
                if not ModelDescription.match(val1, val2):
                    return False
        elif type_o1 is np.ndarray:
            match = np.array_equal(o1, o2)
            return match
        else:
            return o1 == o2
        return True

    def __eq__(self, other):
        # check the model params for a match
        if 'model' in self and 'model' in other:
            return self.match(self['model'], other['model'])
        else:
            return False
