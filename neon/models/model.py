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
from __future__ import division, print_function
from builtins import str, zip
from collections import OrderedDict
import logging

from neon import __version__ as __neon_version__
from neon import NervanaObject, logger as neon_logger
from neon.backends.backend import Block
from neon.transforms import CrossEntropyBinary, Logistic
from neon.util.persist import load_obj, save_obj, load_class
from neon.util.modeldesc import ModelDescription
from neon.layers import Sequential, Activation
from neon.layers.container import DeltasTree, SkipThought
from neon.util.beamsearch import BeamSearch
from neon.optimizers.optimizer import get_param_list
import numpy as np

logger = logging.getLogger(__name__)


class Model(NervanaObject):
    """
    Class which stores a list of layers describing the model. Can train the layer
    weights on a dataset, evaluate on a test set and serialize the model.
    Additional functionality can be added to the fit method through callback functions.

    Arguments:
        layers: layer container, a list of layers (that will be containerized),
                or a serialized model description
        dataset (NervanaDataIterator): Data set (ignored, will be removed)
        weights_only (bool): set to True if you do not want to recreate layers
                             and states during deserialization from a serialized model
                             description.  Defaults to False.
        name (str): Model name.  Defaults to "model"
        optimizer (Optimizer): Optimizer object which defines the learning rule for updating
                               model parameters (i.e., GradientDescentMomentum, Adadelta)
    """

    def __init__(self, layers, dataset=None, weights_only=False, name="model", optimizer=None):
        super(Model, self).__init__(name)
        self.optimizer = optimizer
        self.params = None  # should be able to remove
        self.states = None  # should be able to remove
        self.epoch_index = 0
        self.finished = False
        self.initialized = False
        self.cost = None
        self.nbatches = 0
        self.ndata = 0

        if dataset is not None:
            logger.warning('dataset is a deprecated argument and will be ignored')

        if type(layers) in (ModelDescription, dict):
            # load up the model from a serialized file (dataset could be None here)
            self.deserialize(layers, load_states=(not weights_only))
        elif isinstance(layers, (str, bytes)):
            self.load_params(layers, load_states=(not weights_only))
        else:
            # Wrap the list of layers in a Sequential container if a raw list of layers
            if isinstance(layers, list):
                self.layers = Sequential(layers)
            else:
                self.layers = layers

                if isinstance(layers, SkipThought) and hasattr(layers, 'layer_dict'):
                    self.layer_dict = layers.layer_dict

        self.layers.propagate_parallelism("Data")

    @property
    def layers_to_optimize(self):
        """
        Helper function to return the layers which will be optimized.
        """
        return self.layers.layers_to_optimize

    def set_shortcut(self):
        # infer whether bprop shortcut can be used on final activation
        # self.cost should be set to run this otherwise do nothing
        lastlayer = self.layers[-1]
        try:
            if self.cost.costfunc.__class__ is CrossEntropyBinary:
                if (lastlayer.__class__ is Activation and
                   lastlayer.transform.__class__ is Logistic):
                    lastlayer.transform.set_shortcut(True)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception:
            # if any attributes are not set or any other exception
            # is thrown leave transform.shortcut as is (do nothing)
            pass

    def initialize(self, dataset, cost=None):
        """
        Propagate shapes through the layers to configure, then allocate space.

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform initialization on
            cost (Cost): Defines the function which the model is minimizing based
                         on the output of the last layer and the input labels.
        """
        if self.initialized:
            return

        # Propagate shapes through the layers to configure
        prev_input = dataset
        prev_input = self.layers.configure(prev_input)

        if cost is not None:
            cost.initialize(prev_input)
            self.cost = cost

        # Now allocate space
        self.layers.allocate()
        self.layers.allocate_deltas()
        self.initialized = True

    def allocate_deltas(self):
        if getattr(self, 'global_deltas', None) is None:
            self.global_deltas = DeltasTree()
            self.layers.allocate_deltas(self.global_deltas)

            # allocate the buffers now that all the
            # nesting and max sizes have been determined
            self.global_deltas.allocate_buffers(self.be)

        # set the deltas
        self.layers.set_deltas(self.global_deltas)

    def __str__(self):
        """
        String representation of model's layers
        """
        config_string = "Network Layers:\n" + self.layers.nested_str()
        return config_string

    def fit(self, dataset, cost, optimizer, num_epochs, callbacks):
        """
        Trains the model parameters on a dataset by minimizing the cost function through
        gradient descent and updates the layer weights according to a learning rule
        defined in optimizer.

        Arguments:
            dataset (NervanaDataIterator): An iterable of minibatches where each
                element is a (x, y) tuple where x is the input data and y are the labels.
                x is of dimension (feature_size, batch_size)
                y is of dimension (label_size, batch_size)
                Length of the iterator is num_batches which is num_data / batch_size.
            cost (Cost): Defines the function which the model is minimizing based
                         on the output of the last layer and the input labels.
            optimizer (Optimizer): Defines the learning rule for updating the model parameters.
            num_epochs: Number of times to iterate over the dataset.
            callbacks (Callbacks): Defines callbacks to run at the end of each mini-batch / epoch.
        """
        self.nbatches = dataset.nbatches
        self.ndata = dataset.ndata
        # self.set_shortcut()  # infer if bprop shortcut can be used
        self.total_cost = np.empty([1, 1], dtype=np.float32)
        self.optimizer = optimizer
        self.initialize(dataset, cost)

        callbacks.on_train_begin(num_epochs)
        while self.epoch_index < num_epochs and not self.finished:
            self.nbatches = dataset.nbatches

            callbacks.on_epoch_begin(self.epoch_index)

            self._epoch_fit(dataset, callbacks)

            callbacks.on_epoch_end(self.epoch_index)

            self.epoch_index += 1

        callbacks.on_train_end()

    def _epoch_fit(self, dataset, callbacks):
        """
        Helper function for fit which performs training on a dataset for one epoch.

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform fit on
        """
        epoch = self.epoch_index
        self.total_cost[:] = 0
        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):
            callbacks.on_minibatch_begin(epoch, mb_idx)
            self.be.begin(Block.minibatch, mb_idx)

            x = self.fprop(x)

            self.total_cost[:] = self.total_cost + self.cost.get_cost(x, t)

            # deltas back propagate through layers
            # for every layer in reverse except the 0th one
            delta = self.cost.get_errors(x, t)

            self.bprop(delta)
            self.optimizer.optimize(self.layers_to_optimize, epoch=epoch)

            self.be.end(Block.minibatch, mb_idx)
            callbacks.on_minibatch_end(epoch, mb_idx)

        # now we divide total cost by the number of batches,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on
        self.total_cost[:] = self.total_cost / dataset.nbatches

    def fprop(self, x, inference=False):
        """
        Forward propagates a minibatch x through the model.

        Arguments:
            x (Tensor): Input minibatch data.
            inference (bool): Flag for performing training or inference
                Only affects batch norm and dropout layers.

        Returns:
            Tensor: the output of the final layer in the model
        """
        res = self.layers.fprop(x, inference)
        self.be.convert_data(res, False)
        return res

    def bprop(self, delta):
        """
        Back propagates the error of a minibatch through the model.

        Arguments:
            delta (Tensor): Derivative of cost with respect to the last layer's output

        Returns:
            Tensor: Deltas to propagate to the next layer
        """
        res = self.layers.bprop(delta)
        self.be.convert_data(res, False)
        return res

    def eval(self, dataset, metric):
        """
        Evaluates a model on a dataset according to an input metric.

        Arguments:
            datasets (NervanaDataIterator): dataset to evaluate on.
            metric (Cost): what function to evaluate dataset on.

        Returns:
            Host numpy array: the error of the final layer for the evaluation dataset
        """
        self.initialize(dataset)
        running_error = np.zeros((len(metric.metric_names)), dtype=np.float32)
        nprocessed = 0
        dataset.reset()
        if hasattr(dataset, 'seq_length'):
            ndata = dataset.ndata*dataset.seq_length
        else:
            ndata = dataset.ndata
        for x, t in dataset:
            x = self.fprop(x, inference=True)

            # This logic is for handling partial batch sizes at the end of the dataset
            nsteps = x.shape[1] // self.be.bsz if not isinstance(x, list) else \
                x[0].shape[1] // self.be.bsz

            bsz = min(ndata - nprocessed, self.be.bsz)
            running_error += metric(x, t, calcrange=slice(0, nsteps * bsz)) * nsteps * bsz
            nprocessed += bsz * nsteps
        running_error /= nprocessed
        return running_error

    def get_outputs(self, dataset):
        """
        Get the activation outputs of the final model layer for the dataset

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform fit on

        Returns:
            Host numpy array: the output of the final layer for the entire Dataset
        """
        self.initialize(dataset)
        dataset.reset()  # Move "pointer" back to beginning of dataset
        n = dataset.nbatches
        x = self.layers.layers[-1].outputs
        assert not isinstance(x, list), "Can not get_outputs with Branch terminal"
        Ypred = None
        for idx, input_data in enumerate(dataset):
            x = self.fprop(input_data[0], inference=True)
            if Ypred is None:
                (dim0, dim1) = x.shape
                Ypred = np.empty((n * dim1, dim0), dtype=x.dtype)
                nsteps = dim1 // self.be.bsz
            cur_batch = slice(idx * dim1, (idx + 1) * dim1)
            Ypred[cur_batch] = x.get().T

        # Handle the recurrent case.
        if nsteps != 1:
            b, s = (self.be.bsz, nsteps)
            Ypred = Ypred.reshape((n, s, b, -1)).transpose(0, 2, 1, 3).copy().reshape(n * b, s, -1)

        return Ypred[:dataset.ndata]

    def get_outputs_beam(self, dataset, num_beams=0, steps=None):
        """
        Get the activation outputs of the final model layer for the dataset

        Arguments:
            dataset (NervanaDataIterator) Dataset iterator to perform fit on
            num_beams (int, optional) Nonzero to use beamsearch for sequence to sequence models
            steps (Int): Length of desired output in number of time steps

        Returns:
            Host numpy array: the output of the final layer for the entire Dataset
        """
        self.initialize(dataset)
        dataset.reset()  # Move "pointer" back to beginning of dataset
        n = dataset.nbatches
        x = self.layers.layers[-1].outputs
        assert not isinstance(x, list), "Can not get_outputs with Branch terminal"
        if num_beams > 0:
            beamsearch = BeamSearch(self.layers)
        Ypred = None
        logger.info('Performing beam search with ' + str(num_beams) + ' beams')
        for idx, (x, t) in enumerate(dataset):
            if num_beams > 0:
                x = beamsearch.beamsearch(x, num_beams, steps=steps)
            else:
                x = self.fprop(x, inference=True)
            if Ypred is None:
                (dim0, dim1) = x.shape
                Ypred = np.empty((n * dim1, dim0), dtype=x.dtype)
                nsteps = dim1 // self.be.bsz
            cur_batch = slice(idx * dim1, (idx + 1) * dim1)
            Ypred[cur_batch] = x.get().T

        # Handle the beam search case.
        if dim0 == getattr(dataset, 'seq_length', None):
            b = self.be.bsz
            s = dataset.seq_length
            Ypred = Ypred.reshape((n*b, s, -1))

        # Handle the recurrent case.
        elif nsteps != 1:
            b, s = (self.be.bsz, nsteps)
            Ypred = Ypred.reshape((n, s, b, -1)).transpose(0, 2, 1, 3).copy().reshape(n * b, s, -1)

        return Ypred[:dataset.ndata]

    def get_description(self, get_weights=False, keep_states=False):
        """
        Gets a description of the model required to reconstruct the model with
        no weights like from a yaml file.

        Arguments:
            get_weights:  (Default value = False)
            keep_states:  (Default value = False)

        Returns:
            dict: Description of each component of the model.
        """
        pdict = dict()
        pdict['neon_version'] = __neon_version__
        compat_mode = self.be.compat_mode if self.be.compat_mode is not None else 'neon'
        pdict['backend'] = {'type': self.be.__class__.__name__,
                            'compat_mode': compat_mode,
                            'rng_seed': self.be.rng_seed,
                            'rng_state': self.be.rng_get_state()}

        if self.cost:
            pdict['cost'] = self.cost.get_description()
        if self.optimizer:
            pdict['optimizer'] = self.optimizer.get_description()

        pdict['model'] = self.layers.get_description(get_weights=get_weights,
                                                     keep_states=keep_states)
        return pdict

    def save_params(self, param_path, keep_states=True):
        """
        Serializes and saves model parameters to the path specified.

        Arguments:
            param_path (str): File to write serialized parameter dict to.
            keep_states (bool): Whether to save optimizer states too.
                                Defaults to True.
        """
        self.serialize(keep_states=keep_states, fn=param_path)

    def load_params(self, param_path, load_states=True):
        """
        Loads the model parameters (per layer weights, epochs run, optimizer
        states) saved in param_path from serialize().

        Arguments:
            param_path (str): File containing serialized python dict with layer
                              weights and states.
            load_states (bool):  if False, then only the weights will be loaded
                                 into a model in which the layers have already been
                                 created, otherwise will (re)create the layers from
                                 the serialized parameters and set the learning
                                 states as well
        """
        self.deserialize(load_obj(param_path), load_states=load_states)
        logger.info('Model weights loaded from %s', param_path)

    def load_weights(self, weight_path):
        """
        .. deprecated:: 1.1.4
           Use :func:`load_params` instead

        Arguments:
            weight_path:
        """
        logger.warning('Calling deprecated load_weights function.  Use '
                       'load_params instead')
        self.load_params(weight_path)

    def deserialize(self, model_dict, data=None, load_states=True):
        """
        Loads per layer (weights, states) and other model parameters from the
        dictionary passed.

        Arguments:
            model_dict (dict): dictionary describing the model including layers,
                               cost, optimizers, backend settings, etc.
                               generated by the serialize function
            data (NervanaDataIterator):   Data set (ignored, will be removed)
            load_states (bool):  if False, then only the weights will be loaded
                                 into a model in which the layers have already been
                                 created, otherwise will (re)create the layers from
                                 the serialized parameters and set the learning
                                 states as well
        """

        if data is not None:
            logger.warning('data is a deprecated argument and will be ignored')

        if 'epoch_index' in model_dict:
            self.epoch_index = model_dict['epoch_index']
        if 'model' not in model_dict:
            logger.error('Using old model serialization format. '
                         'Serialized the model into new format')

            param_layers = [l for l in self.layers_to_optimize]
            param_dict_list = model_dict['layer_params_states']
            for l, ps in zip(param_layers, param_dict_list):
                l.set_params(ps)
                if 'states' in ps and load_states:
                    l.set_states(ps)
            return

        if 'backend' in model_dict:
            if 'compat_mode' in model_dict['backend']:
                self.be.compat_mode = model_dict['backend']['compat_mode']
        else:
            model_dict['backend'] = {}

        typ = model_dict['model']['type']
        main_container = load_class(typ)

        if not hasattr(self, 'layers'):
            self.layers = main_container.gen_class(model_dict['model']['config'])

        self.layers.load_weights(model_dict['model'], load_states)

        if load_states and 'rng_state' in model_dict['backend']:
            try:
                self.be.rng_set_state(model_dict['backend']['rng_state'])
            except ValueError as e:
                # could come about when switching backend types (ex GPU to CPU)
                logger.warning("Problems restoring existing RNG state: %s", str(e))

    # serialize tells how to write out the parameters we've learned so
    # far and associate them with layers. it can ignore layers with no
    # learned parameters. the model stores states to pass to the
    # optimizers.  if we're saving the model out for inference, we
    # don't need to remember states.
    def serialize(self, fn=None, keep_states=True):
        """
        Creates a dictionary storing the layer parameters and epochs complete.

        Arguments:
            fn (str): file to save pkl formatted model dictionary
            keep_states (bool): Whether to save optimizer states.

        Returns:
            dict: Model data including layer parameters and epochs complete.
        """

        # get the model dict with the weights
        pdict = self.get_description(get_weights=True, keep_states=keep_states)
        pdict['epoch_index'] = self.epoch_index + 1
        if self.initialized:
            if not hasattr(self.layers, 'decoder'):
                pdict['train_input_shape'] = self.layers.in_shape
            else:
                # serialize shapes both for encoder and decoder
                pdict['train_input_shape'] = (self.layers.encoder.in_shape +
                                              self.layers.decoder.in_shape)
        if fn is not None:
            save_obj(pdict, fn)
            return
        return pdict

    def set_batch_size(self, N):
        """
        Set the actual minibatch size, so even though the buffers are allocated considering
        excessive padding, the processing for some layers may be shortened.
        Currently most of the neon layers don't use that to control the processing. The
        interface is here only for when someone wants to set that information and experiment.

        Arguments:
            N:

        Returns:

        """
        return self.layers.set_batch_size(N)

    def set_seq_len(self, S):
        """
        Set the actual minibatch sequence length, so even though the buffers are allocated
        considering excessive padding, the processing for some layers may be shortened.
        Currently most of the neon layers don't use that to control the processing. The
        interface is here only for when someone wants to set that information and experiment.

        Arguments:
            S:

        Returns:

        """
        return self.layers.set_seq_len(S)

    def benchmark(self, dataset, inference=False, cost=None, optimizer=None,
                  niterations=20, nskip=2):
        """
        Measure runtime for computing fprop and bprop separately, as well as
        full minibatch run times. For inference case, only the fprop is measured.

        Arguments:
             dataset (NervanaDataIterator) Dataset iterator to perform fit on

             cost (Cost): Defines the function which the model is minimizing based
                          on the output of the last layer and the input labels

             niterations (optional, int): Number of minibatches to average over

             nskip (optional, int): Number of iterations at the beginning to skip
                                    when calculating the runtime statistics
             inference (bool, optional): Is inference use case
             optimizer (Optimizer): Defines the learning rule for updating the model parameters.
        Returns:
            dictionary with fprop, bprop run times
        """
        # initialize model
        if inference is False and (cost is None or optimizer is None):
            raise RuntimeError("Need cost and optimizer to benchmark bprop")

        self.cost = cost
        self.initialize(dataset, cost)
        self.optimizer = optimizer
        self.total_cost = np.empty((1, 1))
        self.total_cost[:] = 0

        # iterate through minibatches of the dataset
        times = OrderedDict()
        time_keys = ['fprop'] if inference else ['fprop', 'bprop', 'iteration']
        for ky in time_keys:
            times[ky] = np.full(niterations + nskip, -1.0)
        count = 0

        fprop_start = self.be.init_mark()
        fprop_end = self.be.init_mark()
        bprop_end = self.be.init_mark()

        while count < niterations + nskip:
            dataset.reset()
            for mb_idx, (x, t) in enumerate(dataset):

                self.be.record_mark(fprop_start)  # mark start of fprop

                x = self.fprop(x, inference)

                if inference is False:
                    self.total_cost[:] = self.total_cost + self.cost.get_cost(x, t)

                self.be.record_mark(fprop_end)  # mark end of fprop and start of bprop

                if inference is False:
                    delta = self.cost.get_errors(x, t)
                    self.bprop(delta)
                    self.optimizer.optimize(self.layers_to_optimize, epoch=0)

                    self.be.record_mark(bprop_end)  # mark end of bprop
                    self.be.synchronize_mark(bprop_end)
                else:
                    self.be.synchronize_mark(fprop_end)

                times['fprop'][count] = self.be.get_time(fprop_start, fprop_end)
                if inference is False:
                    times['bprop'][count] = self.be.get_time(fprop_end, bprop_end)
                    times['iteration'][count] = times['fprop'][count] + times['bprop'][count]

                count += 1
                if count >= niterations + nskip:
                    break

        # print results
        header = ('Func', 'Mean', 'Median', 'Min', 'Max', 'Units')
        stats = tuple(stat.lower() for stat in header[1:-1])

        fmt_titles = '| {:^11} ' * len(header) + '|'
        fmt_nums = '| {func:<11} ' + '|  {%s:<10.5g} ' * len(stats) % (stats) + '| {units:^11} |'

        head_str = fmt_titles.format(*header)
        sep = '-' * len(head_str)
        neon_logger.display(sep)
        neon_logger.display(head_str)
        neon_logger.display(sep)
        out_stats = {}
        for step in times:
            timesu = np.array(times[step][nskip:])  # in ms
            out_stats[step] = {}
            for stat in stats:
                out_stats[step][stat] = getattr(np, stat)(timesu)
            neon_logger.display(fmt_nums.format(units='msec', func=step, **out_stats[step]))
        neon_logger.display(sep)
        return out_stats


class GAN(Model):
    """
    Model for Generative Adversarial Networks.

    Arguments:
        layers: Generative Adversarial layer container
        noise_dim (Tuple): Dimensionality of the noise feeding the generator
        noise_type (Str): Noise distribution, 'normal' (default) or 'uniform'
        weights_only (bool): set to True if you do not want to recreate layers
                             and states during deserialization from a serialized model
                             description.  Defaults to False.
        name (str): Model name.  Defaults to "model"
        optimizer (Optimizer): Optimizer object which defines the learning rule for updating
                               model parameters (i.e., GradientDescentMomentum, Adadelta)
        k (int): Number of data batches per noise batch
        wgan_param_clamp (float or None): In case of WGAN weight clamp value, None for others
        wgan_train_sched (bool): Whether to use the FAIR WGAN training schedule of critics
    """
    def __init__(self, layers, noise_dim, noise_type='normal', weights_only=False,
                 name="model", optimizer=None, k=1,
                 wgan_param_clamp=None, wgan_train_sched=False):
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.k = k
        self.wgan_param_clamp = wgan_param_clamp
        self.wgan_train_sched = wgan_train_sched
        super(GAN, self).__init__(layers, weights_only=weights_only, name=name,
                                  optimizer=optimizer)

    @staticmethod
    def clip_param_in_layers(layer_list, abs_bound=None):
        """
        Element-wise clip all parameter tensors to between
        ``-abs_bound`` and ``+abs_bound`` in a list of layers.

        Arguments:
            layer_list (list): List of layers
            be (Backend object): Backend in which the tensor resides
            abs_bound (float, optional): Value to element-wise clip gradients
                                         or parameters. Defaults to None.
        """
        param_list = get_param_list(layer_list)
        for (param, grad), states in param_list:
            if abs_bound:
                param[:] = param.backend.clip(param, -abs(abs_bound), abs(abs_bound))

    def fill_noise(self, z, normal=True):
        """
        Fill z with either uniform or normally distributed random numbers
        """
        if normal:
            # Note fill_normal is not deterministic
            self.be.fill_normal(z)
        else:
            z[:] = 2 * self.be.rand() - 1.

    def initialize(self, dataset, cost=None):
        """
        Propagate shapes through the layers to configure, then allocate space.

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform initialization on
            cost (Cost): Defines the function which the model is minimizing based
                         on the output of the last layer and the input labels.
        """
        if self.initialized:
            return

        # Propagate shapes through the layers to configure
        prev_input = self.layers.configure(self.noise_dim)

        if cost is not None:
            cost.initialize(prev_input)
            self.cost = cost

        # Now allocate space
        self.layers.generator.allocate(accumulate_updates=False)
        self.layers.discriminator.allocate(accumulate_updates=True)
        self.layers.allocate_deltas()
        self.initialized = True

        self.zbuf = self.be.iobuf(self.noise_dim)
        self.ybuf = self.be.iobuf((1,))
        self.z0 = self.be.iobuf(self.noise_dim)  # a fixed noise buffer for generating images
        self.fill_noise(self.z0, normal=(self.noise_type == 'normal'))
        self.cost_dis = np.empty((1,), dtype=np.float32)
        self.current_batch = self.gen_iter = self.last_gen_batch = 0

    def get_k(self, giter):
        """
        WGAN training schedule for generator following Arjovsky et al. 2017

        Arguments:
            giter (int): Counter for generator iterations
        """
        if self.wgan_train_sched and (giter < 25 or giter % 500 == 0):
            return 100
        else:
            return self.k

    def _epoch_fit(self, dataset, callbacks):
        """
        Helper function for fit which performs training on a dataset for one epoch.

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform fit on
        """
        epoch = self.epoch_index
        self.total_cost[:] = 0
        last_gen_iter = self.gen_iter
        z, y_temp = self.zbuf, self.ybuf

        # iterate through minibatches of the dataset
        for mb_idx, (x, _) in enumerate(dataset):
            callbacks.on_minibatch_begin(epoch, mb_idx)
            self.be.begin(Block.minibatch, mb_idx)

            # clip all discriminator parameters to a cube in case of WGAN
            if self.wgan_param_clamp:
                self.clip_param_in_layers(self.layers.discriminator.layers_to_optimize,
                                          self.wgan_param_clamp)

            # train discriminator on noise
            self.fill_noise(z, normal=(self.noise_type == 'normal'))
            Gz = self.fprop_gen(z)
            y_noise = self.fprop_dis(Gz)
            y_temp[:] = y_noise
            delta_noise = self.cost.costfunc.bprop_noise(y_noise)
            self.bprop_dis(delta_noise)
            self.layers.discriminator.set_acc_on(True)

            # train discriminator on data
            y_data = self.fprop_dis(x)
            delta_data = self.cost.costfunc.bprop_data(y_data)
            self.bprop_dis(delta_data)
            self.optimizer.optimize(self.layers.discriminator.layers_to_optimize, epoch=epoch)
            self.layers.discriminator.set_acc_on(False)

            # keep GAN cost values for the current minibatch
            # abuses get_cost(y,t) using y_noise as the "target"
            self.cost_dis[:] = self.cost.get_cost(y_data, y_temp, cost_type='dis')

            # train generator
            if self.current_batch == self.last_gen_batch + self.get_k(self.gen_iter):
                self.fill_noise(z, normal=(self.noise_type == 'normal'))
                Gz = self.fprop_gen(z)
                y_temp[:] = y_data
                y_noise = self.fprop_dis(Gz)
                delta_noise = self.cost.costfunc.bprop_generator(y_noise)
                delta_dis = self.bprop_dis(delta_noise)
                self.bprop_gen(delta_dis)
                self.optimizer.optimize(self.layers.generator.layers_to_optimize, epoch=epoch)
                # keep GAN cost values for the current minibatch
                self.cost_dis[:] = self.cost.get_cost(y_temp, y_noise, cost_type='dis')
                # accumulate total cost.
                self.total_cost[:] = self.total_cost + self.cost_dis
                self.last_gen_batch = self.current_batch
                self.gen_iter += 1

            self.be.end(Block.minibatch, mb_idx)
            callbacks.on_minibatch_end(epoch, mb_idx)
            self.current_batch += 1

        # now we divide total cost by the number of generator iterations,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on the generator
        assert self.gen_iter > last_gen_iter, \
            "at least one generator iteration is required for total cost estimation in this epoch"
        self.total_cost[:] = self.total_cost / (self.gen_iter - last_gen_iter)

        # Package a batch of data for plotting
        self.data_batch, self.noise_batch = x, self.fprop_gen(self.z0)

    def fprop_gen(self, x, inference=False):
        """
        fprop the generator layer stack
        """
        return self.layers.generator.fprop(x, inference)

    def fprop_dis(self, x, inference=False):
        """
        fprop the discriminator layer stack
        """
        return self.layers.discriminator.fprop(x, inference)

    def bprop_dis(self, delta):
        """
        bprop the discriminator layer stack
        """
        return self.layers.discriminator.bprop(delta)

    def bprop_gen(self, delta):
        """
        bprop the generator layer stack
        """
        return self.layers.generator.bprop(delta)
