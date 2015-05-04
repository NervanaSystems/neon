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
Generic Model interface.  Defines the operations and parameters any model
must support.
"""

import logging

logger = logging.getLogger(__name__)


class Model(object):
    """
    Abstract base model class.  Identifies operations to be implemented.
    """

    def fit(self, datasets):
        """
        Utilize the passed dataset(s) to train a model (learn model
        parameters).

        :param datasets: collection of datasets stored in an approporiate
                         backend.
        :type datasets: tuple of neon.datasets.Dataset objects
        """
        raise NotImplementedError()

    def predict(self, datasets):
        """
        Utilize a fit model to generate predictions against the datasets
        provided.

        :param datasets: collection of datasets stored in an approporiate
                         backend.
        :type datasets: tuple of neon.datasets.Dataset objects
        """
        raise NotImplementedError()

    def get_params(self):
        np_params = dict()
        for i, ll in enumerate(self.layers):
            if ll.has_params:
                lkey = ll.name + '_' + str(i)
                np_params[lkey] = ll.get_params()
        np_params['epochs_complete'] = self.epochs_complete
        return np_params

    def set_params(self, params_dict):
        for i, ll in enumerate(self.layers):
            if ll.has_params:
                lkey = ll.name + '_' + str(i)
                ll.set_params(params_dict[lkey])
        self.epochs_complete = params_dict['epochs_complete']
