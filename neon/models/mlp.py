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
Simple multi-layer perceptron model.
"""

import logging

from neon.backends.backend import Block
from neon.models.model import Model
from neon.util.param import opt_param, req_param

logger = logging.getLogger(__name__)


class MLP(Model):

    """
    Fully connected, feed-forward, multi-layer perceptron model
    """

    def __init__(self, **kwargs):
        self.initialized = False
        self.__dict__.update(kwargs)
        req_param(self, ['layers', 'batch_size'])
        opt_param(self, ['step_print'], -1)
        opt_param(self, ['accumulate'], False)
        opt_param(self, ['reuse_deltas'], True)
        opt_param(self, ['timing_plots'], False)
        opt_param(self, ['serialize_schedule'])

    def link(self, initlayer=None):
        for ll, pl in zip(self.layers, [initlayer] + self.layers[:-1]):
            ll.set_previous_layer(pl)
        self.print_layers()

    def initialize(self, backend, initlayer=None):
        self.data_layer = self.layers[0]
        self.cost_layer = self.layers[-1]
        self.class_layer = self.layers[-2]
        if not hasattr(self.cost_layer, 'ref_layer'):
            self.cost_layer.ref_layer = self.data_layer
        if self.initialized:
            return
        self.backend = backend
        kwargs = {"backend": self.backend, "batch_size": self.batch_size,
                  "accumulate": self.accumulate}
        for ll, pl in zip(self.layers, [initlayer] + self.layers[:-1]):
            ll.initialize(kwargs)

        self.nin_max = max(map(lambda x: x.nin, self.layers[1:-1]))
        self.global_deltas = None
        if self.reuse_deltas:
            self.global_deltas = backend.zeros(
                (2 * self.nin_max, self.batch_size),
                dtype=self.layers[1].deltas_dtype)
            self.global_deltas.name = "delta_pool"

        for idx, ll in enumerate(self.layers[1:-1]):
            ll.set_deltas_buf(self.global_deltas,
                              offset=((idx % 2) * self.nin_max))

        self.initialized = True

        # Make some scratch space for NervanaGPU backend:
        if self.backend.__module__ == 'neon.backends.gpu':
            self.backend.init_mempool((1, self.batch_size),
                                      dtype=self.layers[1].deltas_dtype)

    def fprop(self):
        for ll, pl in zip(self.layers, [None] + self.layers[:-1]):
            y = None if pl is None else pl.output
            ll.fprop(y)

    def bprop(self):
        for ll, nl in zip(reversed(self.layers),
                          reversed(self.layers[1:] + [None])):
            error = None if nl is None else nl.deltas
            ll.bprop(error)

    def print_layers(self, debug=False):
        printfunc = logger.debug if debug else logger.info
        netdesc = 'Layers:\n'
        for layer in self.layers:
            netdesc += '\t' + str(layer) + '\n'
        printfunc("%s", netdesc)

    def update(self, epoch):
        for layer in self.layers:
            layer.update(epoch)

    def get_classifier_output(self):
        return self.class_layer.output

    def print_training_error(self, error, num_batches, partial=False):
        rederr = self.backend.reduce_tensor(error)
        if self.backend.rank() != 0:
            return
        if partial is True:
            assert self.step_print != 0
            logger.info('%d:%d training error: %0.5f', self.epochs_complete,
                        num_batches / self.step_print,
                        rederr)
        else:
            errorval = rederr / num_batches
            logger.info('epoch: %d, training error: %0.5f',
                        self.epochs_complete,
                        errorval)

    def print_test_error(self, setname, misclass, nrecs):
        redmisclass = self.backend.reduce_tensor(misclass)
        if self.backend.rank() != 0:
            return

        misclassval = redmisclass / nrecs
        logging.info("%s set misclass rate: %0.5f%%",
                     setname, 100. * misclassval)

    def fit(self, dataset):
        """
        Learn model weights on the given datasets.
        """
        error = self.backend.zeros((1, 1), dtype=self.cost_layer.weight_dtype)
        self.data_layer.init_dataset(dataset)
        self.data_layer.use_set('train')
        logger.info('commencing model fitting')
        while self.epochs_complete < self.num_epochs:
            self.backend.begin(Block.epoch, self.epochs_complete)
            error.fill(0.0)
            mb_id = 1
            self.data_layer.reset_counter()
            while self.data_layer.has_more_data():
                self.backend.begin(Block.minibatch, mb_id)
                self.backend.begin(Block.fprop, mb_id)
                self.fprop()
                self.backend.end(Block.fprop, mb_id)
                self.backend.begin(Block.bprop, mb_id)
                self.bprop()
                self.backend.end(Block.bprop, mb_id)
                self.backend.begin(Block.update, mb_id)
                self.update(self.epochs_complete)
                self.backend.end(Block.update, mb_id)
                if self.step_print > 0 and mb_id % self.step_print == 0:
                    self.print_training_error(self.cost_layer.get_cost(),
                                              mb_id, partial=True)
                self.backend.add(error, self.cost_layer.get_cost(), error)
                self.backend.end(Block.minibatch, mb_id)
                mb_id += 1
            self.epochs_complete += 1
            self.print_training_error(error, self.data_layer.num_batches)
            self.print_layers(debug=True)
            self.backend.end(Block.epoch, self.epochs_complete - 1)
            self.save_snapshot()
        self.data_layer.cleanup()

    def set_train_mode(self, mode):
        for ll in self.layers:
            ll.set_train_mode(mode)

    def predict_generator(self, dataset, setname):
        """
        Generate predicitons and true labels for the given dataset, one
        mini-batch at a time.

        Agruments:
            dataset: A neon dataset instance
            setname: Which set to compute predictions for (test, train, val)

        Returns:
            tuple: on each call will yield a 2-tuple of outputs and references.
                   The first item is the model probabilities for each class,
                   and the second item is either the one-hot or raw labels with
                   ground truth.

        See Also:
            predict_fullset
        """
        self.data_layer.init_dataset(dataset)
        assert self.data_layer.has_set(setname)
        self.data_layer.use_set(setname, predict=True)
        self.data_layer.reset_counter()
        nrecs = self.batch_size * 1
        outputs = self.backend.empty((self.class_layer.nout, nrecs))
        if self.data_layer.has_labels:
            reference = self.backend.empty((1, nrecs))
        else:
            reference = self.backend.empty(outputs.shape)

        while self.data_layer.has_more_data():
            self.fprop()
            outputs = self.get_classifier_output()
            reference = self.cost_layer.get_reference()
            yield (outputs, reference)

        self.data_layer.cleanup()

    def predict_fullset(self, dataset, setname):
        """
        Generate predicitons and true labels for the given dataset.
        Note that this requires enough memory to house the predictions and
        labels for the entire dataset at one time (not recommended for large
        datasets, see predict_generator instead).

        Agruments:
            dataset: A neon dataset instance
            setname: Which set to compute predictions for (test, train, val)

        Returns:
            tuple: on each call will yield a 2-tuple of outputs and references.
                   The first item is the model probabilities for each class,
                   and the second item is either the one-hot or raw labels with
                   ground truth.

        See Also:
            predict_generator
        """
        self.data_layer.init_dataset(dataset)
        assert self.data_layer.has_set(setname)
        self.data_layer.use_set(setname, predict=True)
        nrecs = self.batch_size * self.data_layer.num_batches
        outputs = self.backend.empty((self.class_layer.nout, nrecs))
        if self.data_layer.has_labels:
            reference = self.backend.empty((1, nrecs))
        else:
            reference = self.backend.empty(outputs.shape)

        batch = 0
        for batch_preds, batch_refs in self.predict_generator(dataset,
                                                              setname):
            start = batch * self.batch_size
            end = start + self.batch_size
            outputs[:, start:end] = batch_preds
            reference[:, start:end] = batch_refs
            batch += 1

        return outputs, reference

    def predict_live_init(self, dataset):
        self.data_layer.init_dataset(dataset)
        for ll in self.layers:
            ll.set_train_mode(False)

    def predict_live(self):
        self.fprop()
        return self.get_classifier_output()
