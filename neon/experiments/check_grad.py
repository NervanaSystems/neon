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
Numerical gradient checking to validate backprop code.
"""

import logging
import numpy as np

from neon.datasets.synthetic import UniformRandom
from neon.experiments.experiment import Experiment
from neon.models.mlp import MLP
from neon.util.compat import range


logger = logging.getLogger(__name__)


class GradientChecker(Experiment):
    """
    In this `Experiment`, a model is trained on a fake training dataset to
    validate the backprop code within the given model.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def transfer(self, experiment):
        self.model = experiment.model
        self.dataset = experiment.dataset

    def save_state(self):
        for ind in range(len(self.trainable_layers)):
            layer = self.model.layers[self.trainable_layers[ind]]
            self.weights[ind][:] = layer.weights

    def load_state(self):
        for ind in range(len(self.trainable_layers)):
            layer = self.model.layers[self.trainable_layers[ind]]
            layer.weights[:] = self.weights[ind]

    def check_layer(self, layer, inputs, targets):
        # Check up to this many weights.
        nmax = 30
        if type(layer.updates) == list:
            updates = layer.updates[0].asnumpyarray().ravel()
        else:
            updates = layer.updates.asnumpyarray().ravel()
        weights = layer.weights.asnumpyarray().ravel()
        grads = np.zeros(weights.shape)
        inds = np.random.choice(np.arange(weights.shape[0]),
                                min(weights.shape[0], nmax),
                                replace=False)
        for ind in inds:
            saved = weights[ind]
            weights[ind] += self.eps
            self.model.fprop(inputs)
            cost1 = self.model.cost.apply_function(targets).asnumpyarray()

            weights[ind] -= 2 * self.eps
            self.model.fprop(inputs)
            cost2 = self.model.cost.apply_function(targets).asnumpyarray()

            grads[ind] = ((cost1 - cost2) / self.model.layers[-1].batch_size *
                          layer.learning_rule.learning_rate / (2 * self.eps))
            weights[ind] = saved

        grads -= updates
        diff = np.linalg.norm(grads[inds]) / nmax
        if diff < 0.0002:
            logger.info('diff %g. layer %s OK.', diff, layer.name)
            return True

        logger.error('diff %g. gradient check failed on layer %s.',
                     diff, layer.name)
        return False

    def check_layerb(self, layer):
        # Check up to this many weights.
        nmax = 30
        if type(layer.updates) == list:
            updates = layer.updates[0].asnumpyarray().ravel()
        else:
            updates = layer.updates.asnumpyarray().ravel()
        weights = layer.weights.asnumpyarray().ravel()
        grads = np.zeros(weights.shape)
        inds = np.random.choice(np.arange(weights.shape[0]),
                                min(weights.shape[0], nmax),
                                replace=False)
        for ind in inds:
            saved = weights[ind]
            weights[ind] += self.eps
            self.model.data_layer.reset_counter()
            self.model.fprop()
            cost1 = self.model.cost_layer.get_cost().asnumpyarray()

            weights[ind] -= 2 * self.eps
            self.model.data_layer.reset_counter()
            self.model.fprop()
            cost2 = self.model.cost_layer.get_cost().asnumpyarray()

            grads[ind] = ((cost1 - cost2) / self.model.batch_size *
                          layer.learning_rule.learning_rate / (2 * self.eps))
            weights[ind] = saved

        grads -= updates
        diff = np.linalg.norm(grads[inds]) / nmax
        if diff < 0.0002:
            logger.info('diff %g. layer %s OK.', diff, layer.name)
            return True

        logger.error('diff %g. gradient check failed on layer %s.',
                     diff, layer.name)
        return False

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """
        if not (hasattr(self.model, 'fprop') and hasattr(self.model, 'bprop')):
            logger.error('Config file not compatible.')
            return

        self.eps = 1e-4
        self.weights = []
        self.trainable_layers = []
        for ind in range(len(self.model.layers)):
            layer = self.model.layers[ind]
            if not (hasattr(layer, 'weights') and hasattr(layer, 'updates')):
                continue
            self.weights.append(layer.backend.copy(layer.weights))
            self.trainable_layers.append(ind)

        if not hasattr(layer, 'dataset'):
            if isinstance(self.model, MLP):
                datashape = (self.model.data_layer.nout,
                             self.model.cost_layer.nin)
            else:
                datashape = (self.model.layers[0].nin,
                             self.model.layers[-1].nout)
            self.dataset = UniformRandom(self.model.batch_size,
                                         self.model.batch_size,
                                         datashape[0], datashape[1])
            self.dataset.set_batch_size(self.model.batch_size)
            self.dataset.backend = self.model.backend
            self.dataset.load()
        ds = self.dataset

        if isinstance(self.model, MLP):
            self.model.data_layer.dataset = ds
            self.model.data_layer.use_set('train')
            self.model.fprop()
            self.model.bprop()
            self.model.update(0)

            self.save_state()
            self.model.data_layer.reset_counter()
            self.model.fprop()
            self.model.bprop()
            self.model.update(0)
            self.load_state()
        else:
            inputs = ds.get_batch(ds.get_inputs(train=True)['train'], 0)
            targets = ds.get_batch(ds.get_targets(train=True)['train'], 0)

            self.model.fprop(inputs)
            self.model.bprop(targets, inputs)
            self.model.update(0)

            self.save_state()
            self.model.fprop(inputs)
            self.model.bprop(targets, inputs)
            self.model.update(0)
            self.load_state()

        for ind in self.trainable_layers[::-1]:
            layer = self.model.layers[ind]
            if isinstance(self.model, MLP):
                result = self.check_layerb(layer)
            else:
                result = self.check_layer(layer, inputs, targets)
            if result is False:
                break
