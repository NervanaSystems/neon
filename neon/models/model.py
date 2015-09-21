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

from neon import NervanaObject
from neon.transforms import CrossEntropyBinary, Logistic
from neon.util.persist import load_obj
from neon.layers import Merge, Activation
import numpy as np


class Model(NervanaObject):
    """
    Basic model class which stores a list of layers describing the model. Can train the layer
    weights on a dataset, evaluate on a test set and serialize the mode.
    Additional functionality can be added to fit through callback functions.

    Arguments:
        layers (list): List of layers that compose a model.
        name (str): Model name.  Defaults to "model"
        optimizer (Optimizer): Optimizer object which defines the learning rule
                               for updating model parameters (ie DescentMomentum, AdaDelta)
    """

    def __init__(self, layers=[], name="model", optimizer=None):
        super(Model, self).__init__(name)
        self.optimizer = optimizer
        self.params = None
        self.states = None
        self.epoch_index = 0
        self.finished = False

        self.layers = []
        self.layers_to_optimize = []

        for layer in layers:
            if isinstance(layer, list):
                self.layers.extend(layer)
            else:
                self.layers.append(layer)

        for layer in self.layers:
            if layer.has_params:
                self.layers_to_optimize.append(layer)

            elif isinstance(layer, Merge):
                self.layers_to_optimize += layer.layers_to_optimize

    def set_shortcut(self):
        # infer whether bprop shortcut can be used on final activation
        # self.cost should be set to run this otherwise do nothing
        lastlayer = self.layers[-1]
        try:
            if self.cost.costfunc.__class__ is CrossEntropyBinary:
                if (lastlayer.__class__ is Activation and
                   lastlayer.transform.__class__ is Logistic):
                    lastlayer.transform.set_shortcut(True)
        except:
            # if any attributes are not set or any other exception
            # is thrown leave transform.shortcut as is (do nothing)
            pass

    def load_weights(self, weight_path):
        """
        Loads the layer weights saved in weight_path from serialize().

        Arguments:
            weight_path (str): File containing serialized python dict with layer
                               weights and states.
        """
        pdict = load_obj(weight_path)
        self.epoch_index = pdict['epoch_index']

        param_layers = [l for l in self.layers_to_optimize]
        param_dict_list = pdict['layer_params_states']
        for l, ps in zip(param_layers, param_dict_list):
            l.set_params(ps['params'])
            if 'states' in ps:
                l.set_states(ps['states'])

    def fit(self, dataset, cost, optimizer, num_epochs, callbacks):
        """
        Trains the model parameters on a dataset by minimizing the cost function through
        gradient descent and updates the layer weights according to a learning rule
        defined in optimizer.

        Arguments:
            dataset (iterator): An iterable of minibatches where each
                element is a (x, y) tuple where x is the input data and y are the labels.
                x is of dimension (feature_size, batch_size)
                y is of dimension (label_size, batch_size)
                Length of the iterator is num_batches which is num_data / batch_size
            cost (Cost): Defines the function which the model is minimizing based
                on the output of the last layer and the input labels
            optimizer (Optimizer): Defines the learning rule for updating the model parameters
            num_epochs: Number of times to iterate over the dataset.
        """

        self.cost = cost
        self.set_shortcut()  # infer if bprop shortcut can be used
        self.optimizer = optimizer
        self.total_cost = self.be.empty((1, 1))

        callbacks.on_train_begin(num_epochs)

        while self.epoch_index < num_epochs and not self.finished:

            callbacks.on_epoch_begin(self.epoch_index)

            self._epoch_fit(dataset, callbacks)

            callbacks.on_epoch_end(self.epoch_index)

            self.epoch_index += 1

        callbacks.on_train_end()

    def _epoch_fit(self, dataset, callbacks):
        """
        Helper function for fit which performs training on a dataset for one epoch.

        Arguments:
            dataset (iterable): Dataset iterator to perform fit on
        """
        epoch = self.epoch_index
        self.total_cost[:] = 0
        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):

            callbacks.on_minibatch_begin(epoch, mb_idx)

            x = self.fprop(x)

            self.total_cost[:] = self.total_cost + self.cost.get_cost(x, t)

            # deltas back propagate through layers
            # for every layer in reverse except the 0th one
            delta = self.cost.get_errors(x, t)
            self.bprop(delta)

            self.optimizer.optimize(self.layers_to_optimize, epoch=epoch)

            callbacks.on_minibatch_end(epoch, mb_idx)

        # now we divide total cost by the number of batches,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on
        self.total_cost[:] = self.total_cost / dataset.nbatches

    def fprop(self, x, inference=False):
        """
        Forward propagates a minibatch x through the model.

        Arguments:
            x (Tensor): Input minibatch data
            inference (bool): Flag for performing training or inference
                Only affects batch norm and dropout layers.

        Returns:
            Tensor: the output of the final layer in the model
        """
        for l in self.layers:
            x = l.fprop(x, inference)
        return x

    def bprop(self, delta, do_acts=True):
        """
        Back propagates the error of a minibatch through the model.

        Arguments:
            delta (Tensor): Derivative of cost with respect to the last layer's output
            do_acts (bool): Whether to compute the output deltas of layer. The first layer
                does not need to compute output deltas and so do_acts is set to False.
        """
        for l in reversed(self.layers[1:]):
            delta = l.bprop(delta)
        return self.layers[0].bprop(delta, do_acts=False)

    def eval(self, dataset, metric):
        """
        Evaluates a model on a dataset according to an input metric.

        Arguments:
            datasets (iterable): dataset to evaluate on.
            metric (Cost): what function to evaluate dataset on.
        """
        running_error = np.zeros((len(metric.metric_names)), dtype=np.float32)
        nprocessed = 0
        dataset.reset()
        for x, t in dataset:
            x = self.fprop(x, inference=True)

            # This logic is for handling partial batch sizes at the end of the dataset
            bsz = min(dataset.ndata - nprocessed, self.be.bsz)
            metric(x, t)
            running_error += metric.outputs.get()[:, :bsz].sum(axis=1)
            nprocessed += bsz
        running_error /= nprocessed
        return running_error

    def get_outputs(self, dataset):
        """
        Get the activation outputs of the final model layer for the dataset

        Arguments:
            dataset (iterable): Dataset iterator to perform fit on

        Returns:
            Host numpy array: the output of the final layer for the entire Dataset
        """
        Ypred = None
        dataset.reset()  # Move "pointer" back to beginning of dataset
        n = dataset.nbatches

        for idx, (x, t) in enumerate(dataset):
            x = self.fprop(x, inference=True)
            if Ypred is None:
                Ypred = np.empty((n * x.shape[1], x.shape[0]), dtype=x.dtype)
                nsteps = x.shape[1] / self.be.bsz
            cur_batch = slice(idx * x.shape[1], (idx + 1) * x.shape[1])
            Ypred[cur_batch] = x.get().T

        # Handle the recurrent case
        if nsteps != 1:
            b, s = (self.be.bsz, nsteps)
            Ypred = Ypred.reshape((n, b, s, -1)).transpose(1, 0, 2, 3).copy().reshape(n*b, s, -1)

        return Ypred[:dataset.ndata]

    def get_description(self):
        """
        Gets a description of the model required to reconstruct the model with
        no weights like from a yaml file.

        Returns:
            dict: Description of each component of the model.
        """
        pdict = dict()
        pdict['backend'] = 'gpu'
        pdict['cost'] = self.cost.costfunc.__class__.__name__
        pdict['layers'] = [l.get_description() for l in self.layers]
        if self.optimizer:
            pdict['optimizer'] = self.optimizer.get_description()
        return pdict

    # serialize tells how to write out the parameters we've learned so
    # far and associate them with layers. it can ignore layers with no
    # learned parameters. the model stores states to pass to the
    # optimizers.  if we're saving the model out for inference, we
    # don't need to remember states.

    def serialize(self, keep_states=True):
        """
        Creates a dictionary storing the layer parameters and epochs complete.

        Arguments:
            keep_states (bool): Whether to save optimizer states.

        Returns:
            dict: Model data including layer parameters and epochs complete.
        """

        pdict = dict()
        params_states = [l.get_params_serialize(keep_states) for l in self.layers_to_optimize]
        pdict['layer_params_states'] = params_states
        # start training again on the next epoch
        pdict['epoch_index'] = self.epoch_index + 1
        return pdict
