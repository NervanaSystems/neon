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

import os
import shutil
import logging

from neon.util.persist import serialize

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

    def save_snapshot(self):
        """
        Save snapshots of the model at specified epochs.
        If serialize_schedule is a list of ints, it will serialize at those
        epochs.
        If serialize_schedule is a single int, it will serialize when
        epochs_complete is a multiple of serialize_schedule
        """
        if self.serialize_schedule is None:
            return

        if not hasattr(self, 'serialized_path'):
            logger.error('Serialize schedule specified, but no serialize '
                         'path provided, not saving')
            return

        if not isinstance(self.serialize_schedule, (list, int)):
            logger.error('Serialize schedule must be a list of epochs or a '
                         'single int indicating interval between save epochs')
            return

        if isinstance(self.serialize_schedule, list):
            dosave = self.epochs_complete in self.serialize_schedule
            if dosave:
                # add 1 to match periodic schedule
                check_point = \
                    self.serialize_schedule.index(self.epochs_complete) + 1
            else:
                check_point = None
        else:
            dosave = self.epochs_complete % self.serialize_schedule == 0
            check_point = self.epochs_complete/self.serialize_schedule

        if dosave:
            serialize(self.get_params(), self.serialized_path)

            if hasattr(self, 'save_checkpoints'):
                if self.save_checkpoints > 0:
                    # save_checkpoints is the number of previous
                    # checkpoints to save
                    file_parts = os.path.splitext(self.serialized_path)
                    cp_fname_str = file_parts[0] + '_cp%d' + file_parts[1]
                    if os.path.exists(cp_fname_str % check_point):
                        logger.warning(
                            'Checkpoint file exists, overwriting it.  ' +
                            'Check for stale check point files in ' +
                            'serialization directory')

                    # will have a duplicate copy of the current file
                    shutil.copy(
                        self.serialized_path,
                        cp_fname_str % check_point)

                    # only keep last "save_checkpoints" files
                    cp_rng = range(check_point-self.save_checkpoints, -1, -1)
                    for cp_ind in cp_rng:
                        # will not run here until at least
                        # min checkspints saved
                        cp_fname = cp_fname_str % cp_ind
                        if os.path.exists(cp_fname):
                            os.remove(cp_fname)
                        else:
                            # all older files should already be deleted
                            # don't need to continue
                            break
