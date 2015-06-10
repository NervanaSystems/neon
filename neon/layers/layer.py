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
Generic single neural network layer built to handle data from a particular
backend.  We introduce several basic variants here to handle things like
dataset inputs (DataLayer), objective function being optimized (CostLayer),
and internal hidden WeightLayer and ActivationLayer
"""

import logging
import numpy as np
from neon.backends.cpu import CPU
from neon.optimizers.gradient_descent import (GradientDescent,
                                              GradientDescentPretrain,
                                              GradientDescentMomentum,
    GradientDescentMomentumWeightDecay)  # noqa
from neon.optimizers.adadelta import AdaDelta
from neon.optimizers.rmsprop import RMSProp
from neon.util.compat import range
from neon.util.param import req_param, opt_param, ensure_dtype
from neon.util.defaults import default_weight_init, default_lrule_init
from neon.util.persist import YAMLable
from neon.transforms.batch_norm import BatchNorm
from neon.transforms.linear import Linear

logger = logging.getLogger(__name__)


class Layer(YAMLable):
    """
    Top-level generic neural network layer class from which all other layer
    types inherit.

    Attributes:
        name (string): Name identifying this layer (in logs, etc.)
    """
    def __init__(self, **kwargs):
        self.initialized = False
        self.__dict__.update(kwargs)

        opt_param(self, ['name'], 'layer')

        opt_param(self, ['pre_act_dtype', 'output_dtype', 'deltas_dtype',
                         'weight_dtype', 'updates_dtype'], np.float32)
        opt_param(self, ['prev_layer'])
        opt_param(self, ['activation'], Linear())

        opt_param(self, ['is_local', 'is_data', 'is_cost'], False)
        opt_param(self, ['is_random'], False)

        opt_param(self, ['skip_act', 'has_params'], False)
        opt_param(self, ['prev_names'], [])

        opt_param(self, ['backend_type'], 'np.float32')
        self.backend_type = ensure_dtype(self.backend_type)  # string to dtype
        logger.info("Setting layer dtype to" + str(self.backend_type))
        for some_type in ['pre_act_dtype', 'output_dtype', 'deltas_dtype',
                          'weight_dtype', 'updates_dtype']:
            setattr(self, some_type, self.backend_type)

    def set_previous_layer(self, pl):
        if pl.is_local:
            if self.is_local:
                self.ifmshape = pl.ofmshape
                self.nifm = pl.nofm
            self.nin = pl.nofm * np.prod(pl.ofmshape)
        else:
            if self.is_local:
                if not hasattr(self, 'ifmshape'):
                    sqdim = np.int(np.sqrt(pl.nout))
                    self.ifmshape = (sqdim, sqdim)
                self.nifm = 1
            self.nin = pl.nout
        self.prev_layer = pl
        if self.is_local:
            self.link_local()
        self.set_weight_shape()

    def initialize(self, kwargs):
        if self.initialized:
            return
        self.__dict__.update(kwargs)
        req_param(self, ['backend', 'batch_size'])

        self.output = None
        self.deltas = None
        self.initialized = True

    def set_weight_shape(self):
        pass

    def link_local(self):
        req_param(self, ['nifm', 'ifmshape', 'fshape'])

        opt_param(self, ['ofmlocs', 'links'])
        opt_param(self, ['deltasbuf', 'outputbuf'])

        opt_param(self, ['nofm'], self.nifm)
        opt_param(self, ['pooling'], False)
        opt_param(self, ['stride'], 1)
        opt_param(self, ['pad'], 0)

        assert len(self.ifmshape) == len(self.fshape)
        ofmshape = []
        for dim in range(len(self.ifmshape)):
            assert self.ifmshape[dim] >= self.fshape[dim]
            num = self.ifmshape[dim] - self.fshape[dim] + 2 * self.pad
            ofmshape.extend([num // self.stride + 1])
        self.ofmshape = tuple(ofmshape)
        self.negpad = -self.pad
        self.ifmsize = np.prod(self.ifmshape)
        self.ofmsize = np.prod(self.ofmshape)
        self.fpsize = np.prod(self.fshape)
        self.fsize = self.nifm * self.fpsize
        self.nout = self.nofm * self.ofmsize
        logger.debug('name=%s, nifm=%d, ifmshape=%s, ofmshape=%s',
                     self.name, self.nifm, self.ifmshape, self.ofmshape)

    def initialize_local(self):
        if isinstance(self.backend, CPU):
            self.make_aux_buffers(self.nifm, self.ifmshape, self.nofm,
                                  self.ofmshape, self.fshape, self.stride)

    def __str__(self):
        if self.is_local:
            ionumstr = '{} x {} inputs, {} x {} nodes'.format(
                self.nifm, self.format_tuple(self.ifmshape),
                self.nofm, self.format_tuple(self.ofmshape))
        else:
            ionumstr = '{} inputs, {} nodes'.format(self.nin, self.nout)

        ret = '{} {}: {}'.format(self.__class__.__name__, self.name, ionumstr)
        ret += ', {} act_fn'.format(self.activation.__class__.__name__)
        return ret

    def format_tuple(self, tup):
        result = '(' + str(tup[0])
        for dim in range(1, len(tup)):
            result += ' x ' + str(tup[dim])
        return result + ')'

    def allocate_output_bufs(self):
        make_zbuf = self.backend.zeros
        opt_param(self, ['out_shape'], (self.nout, self.batch_size))
        opt_param(self, ['delta_shape'], (self.nin, self.batch_size))

        self.output = make_zbuf(self.out_shape, self.output_dtype)

        self.pre_act = self.activation.pre_act_buffer(self.backend,
                                                      self.output,
                                                      self.pre_act_dtype)

    def set_deltas_buf(self, delta_pool, offset):
        self.deltas = None
        if self.prev_layer is None:
            return
        if self.prev_layer.is_data:
            return

        if delta_pool is None:
            self.deltas = self.backend.zeros(self.delta_shape,
                                             self.deltas_dtype)
        else:
            self.deltas = delta_pool[offset:(offset + self.delta_shape[0])]

    def make_links(self, nifm, ifmsize, ifmshape, ofmshape, fshape, stride):
        # Figure out local connections to the previous layer.
        # This function works for any number of dimensions.
        ndims = len(ifmshape)
        dimsizes = np.empty(ndims, dtype='int32')
        for dim in range(ndims):
            dimsizes[dim] = np.prod(ifmshape[dim:])
        links = []
        for ofmdim in np.ndindex(ofmshape):
            # This variable tracks the top left corner of
            # the receptive field.
            src = ofmdim[-1]
            for dim in range(-1, -ndims, -1):
                src += dimsizes[dim] * ofmdim[dim - 1]
            src *= stride
            indlist = list(range(src, src + fshape[-1]))
            for dim in range(-1, -ndims, -1):
                indarray = np.array(indlist)
                for dimind in range(1, fshape[dim - 1]):
                    indlist.extend(list(indarray + dimind * dimsizes[dim]))
            if self.pooling is False:
                indarray = np.array(indlist)
                for ifm in range(1, nifm):
                    indlist.extend(list(indarray + ifm * ifmsize))
            links.append(indlist)
        self.links = np.array(links, dtype='int32')

    def make_aux_buffers(self, nifm, ifmshape, nofm, ofmshape, fshape, stride):
        buf_size = self.batch_size * nifm
        if (self.prev_layer is not None and not self.prev_layer.is_data):
            self.deltasbuf = self.backend.empty((self.ifmsize, buf_size))

        assert self.ofmsize is not 0
        ofmstarts = np.arange(0, (self.ofmsize * nofm), self.ofmsize)
        self.ofmlocs = np.empty((self.ofmsize, nofm), dtype='int32')
        for dst in range(self.ofmsize):
            self.ofmlocs[dst] = ofmstarts + dst
        self.make_links(nifm, self.ifmsize, ifmshape, ofmshape, fshape, stride)

        if self.pooling is True:
            self.outputbuf = self.backend.empty((self.ofmsize, buf_size))
            if self.op == 'max':
                self.tempbuf = np.empty(
                    (self.ofmsize, self.batch_size * nifm), dtype='int32')
            elif self.op == 'l2':
                self.tempbuf = self.backend.empty(
                    (self.fpsize, self.batch_size * nifm))

    def fprop(self, inputs):
        raise NotImplementedError('This class should not be instantiated.')

    def bprop(self, error):
        raise NotImplementedError('This class should not be instantiated.')

    def update(self, epoch):
        pass

    def set_train_mode(self, mode):
        pass


class CostLayer(Layer):
    """
    Pseudo-layer that should sit in the last level of the network defining the
    objective function to be optimized.
    """
    def __init__(self, **kwargs):
        self.is_cost = True
        self.nout = 1
        super(CostLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        super(CostLayer, self).initialize(kwargs)
        req_param(self, ['cost'])
        opt_param(self, ['ref_label'], 'targets')
        opt_param(self, ['raw_label'], False)
        opt_param(self, ['category_label'], 'l_id')
        self.reference = None
        self.cost.olayer = self.prev_layer
        kwargs['raw_label'] = self.raw_label
        self.cost.initialize(kwargs)
        self.deltas = self.cost.get_deltabuf()

    def __str__(self):
        return ('{} {}: {} nodes, {} cost_fn'. format(
                self.__class__.__name__, self.name, self.nin,
                self.cost.__class__.__name__))

    def set_reference(self):
        if self.ref_layer is not None:
            refs = getattr(self.ref_layer, self.ref_label)
            if isinstance(refs, dict):
                self.reference = refs[self.category_label]
            else:
                self.reference = refs

    def fprop(self, inputs):
        pass

    def bprop(self, error):
        # Since self.deltas already pointing to destination of act gradient
        # we just have to scale by mini-batch size
        self.set_reference()
        self.cost.apply_derivative(self.reference)
        self.backend.divide(self.deltas, self.backend.actual_batch_size,
                            out=self.deltas)

    def get_cost(self):
        self.set_reference()
        scale_cost = (True if self.backend.__module__ == 'neon.backends.gpu'
                      else False)
        result = self.cost.apply_function(self.reference,
                                          scale_by_batchsize=scale_cost)
        if not scale_cost:  # Check for fp16 backend and use scaling
            self.backend.divide(result, self.batch_size, result)
        return result

    def get_reference(self):
        self.set_reference()
        return self.reference


class DataLayer(Layer):
    """
    Typically the first layer of a neural network.  Connects a Dataset to the
    network.
    """
    def __init__(self, **kwargs):
        self.is_data = True
        opt_param(self, ['has_labels'], False)
        super(DataLayer, self).__init__(**kwargs)
        # req_param(self, ['dataset'])

    def initialize(self, kwargs):
        super(DataLayer, self).initialize(kwargs)
        self.reset_counter()
        if self.is_local is True:
            req_param(self, ['nofm', 'ofmshape'])
            self.nout = self.nofm * np.prod(self.ofmshape)
        else:
            req_param(self, ['nout'])

    def init_dataset(self, dataset):
        """
        Must be called prior to consuming data.
        Allows us to switch to a new dataset (useful for changing sets after
        training).  No checking is done for input size, so should match the
        dimensions of datasets between changes
        """
        self.dataset = dataset

    def __str__(self):
        if self.is_local:
            ionumstr = '{} x {} nodes'.format(self.nofm,
                                              self.format_tuple(self.ofmshape))
        else:
            ionumstr = "{} nodes".format(self.nout)

        return ("{} {}: {}".format(self.__class__.__name__,
                                   self.name, ionumstr))

    def set_previous_layer(self, pl):
        pass

    def has_more_data(self):
        return True if (self.batch_idx < self.num_batches) else False

    def reset_counter(self):
        self.batch_idx = 0

    def fprop(self, inputs):
        self.output, self.targets = self.dataset.get_mini_batch(self.batch_idx)
        self.batch_idx += 1

    def bprop(self, error):
        pass

    def has_set(self, setname):
        return self.dataset.has_set(setname)

    def use_set(self, setname, predict=False):
        self.num_batches = self.dataset.init_mini_batch_producer(
            batch_size=self.batch_size,
            setname=setname,
            predict=predict)
        self.reset_counter()

    def cleanup(self):
        # delete helper queues if any
        self.dataset.del_mini_batch_producer()


class ImageDataLayer(DataLayer):

    def __init__(self, **kwargs):
        super(ImageDataLayer, self).__init__(**kwargs)

    def fprop(self, inputs):
        self.output, self.targets, self.labels = self.dataset.get_mini_batch(
            self.batch_idx)
        self.batch_idx += 1


class ActivationLayer(Layer):
    """
    Just applies an activation to the inputs.
    """
    def set_previous_layer(self, pl):
        if pl.is_local:
            self.is_local = True
            self.ifmshape = pl.ofmshape
            self.nifm = pl.nofm
            self.nin = pl.nofm * np.prod(pl.ofmshape)
        else:
            self.nin = pl.nout
        self.prev_layer = pl

    def initialize(self, kwargs):
        super(ActivationLayer, self).initialize(kwargs)
        req_param(self, ['activation'])
        self.nout = self.nin
        self.allocate_output_bufs()

    def fprop(self, inputs):
        self.pre_act[:] = inputs
        self.activation.fprop_func(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        self.activation.bprop_func(self.backend, self.pre_act, error,
                                   self.skip_act)
        if self.deltas is not None:
            self.deltas[:] = error


class SliceLayer(ActivationLayer):
    """
    Just takes a portion of the inputs and passes it on
    Useful for limitations of the GPU convolutional layer
    for a local layer, takes 0:end_idx feature maps
    for a flat layer, takes 0:end_idx inputs
    """
    def __init__(self, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        req_param(self, ['end_idx'])

    def set_previous_layer(self, pl):
        if pl.is_local:
            self.is_local = True
            self.ifmshape = pl.ofmshape
            self.ofmshape = pl.ofmshape
            self.nifm = pl.nofm
            self.nin = pl.nofm * np.prod(pl.ofmshape)
            self.nofm = self.end_idx
        else:
            self.nin = pl.nout
        self.prev_layer = pl

    def initialize(self, kwargs):
        self.__dict__.update(kwargs)
        req_param(self, ['backend', 'batch_size'])
        self.output = None
        self.deltas = None
        if self.is_local:
            self.nofm = self.end_idx
            self.end_idx = np.prod(self.ifmshape) * self.end_idx
        self.nout = self.end_idx
        self.allocate_output_bufs()

    def fprop(self, inputs):
        self.output[:] = inputs[:self.end_idx]

    def bprop(self, error):
        self.deltas.fill(0.0)
        self.deltas[:self.end_idx] = error


class WeightLayer(Layer):
    """
    Typical hidden layer with weight parameters to be learned.
    """
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)
        self.distributable = True
        self.has_params = True
        self.params_initialized = False

    def initialize(self, kwargs):
        super(WeightLayer, self).initialize(kwargs)
        req_param(self, ['nin', 'nout'])
        opt_param(self, ['weight_init'], default_weight_init())
        opt_param(self, ['lrule_init'], default_lrule_init())
        opt_param(self, ['accumulate'], False)
        opt_param(self, ['batch_norm'], False)

        self.weight_init.initialize(self.backend)
        self.params = []
        self.updates = []

        if self.batch_norm:
            self.bn = BatchNorm()
            kwargs['layer'] = self
            self.bn.initialize(kwargs)

    def get_params(self):
        np_params = dict()
        for p in ['weights', 'biases']:
            if hasattr(self, p):
                p_tensor = getattr(self, p)
                np_params[p] = np.array(p_tensor.asnumpyarray(),
                                        dtype=p_tensor.dtype).reshape(
                                            p_tensor.shape)

        if self.batch_norm:
            np_params.update(self.bn.get_params())

        np_params.update(self.learning_rule.get_params())
        return np_params

    def set_params(self, params_dict):
        for p in ['weights', 'biases']:
            if p in params_dict:
                getattr(self, p)[:] = params_dict[p]

        if self.batch_norm:
            self.bn.set_params(params_dict)

        self.learning_rule.set_params(params_dict)

    def allocate_param_bufs(self):
        if self.params_initialized:
            return
        make_ebuf = self.backend.empty
        self.weights = self.weight_init.generate(self.weight_shape,
                                                 self.weight_dtype)
        self.weights.name = self.name  # naming weights for timing diagnostics
        self.weight_updates = make_ebuf(self.weight_shape, self.updates_dtype)

        self.use_biases = 'bias_init' in self.weight_init.__dict__
        opt_param(self, ['brule_init'], None)
        if self.use_biases is True:
            self.biases = make_ebuf(self.bias_shape, self.weight_dtype)
            self.biases.fill(self.weight_init.bias_init)
            self.bias_updates = make_ebuf(self.bias_shape, self.updates_dtype)
            self.params.extend([self.weights, self.biases])
            self.updates.extend([self.weight_updates, self.bias_updates])
        else:
            self.params.extend([self.weights])
            self.updates.extend([self.weight_updates])

        if self.accumulate:
            self.utemp = map(lambda x: make_ebuf(x.shape, self.updates_dtype),
                             self.updates)
        for upm in self.updates:
            upm.fill(0.0)
        self.learning_rule = self.init_learning_rule(self.lrule_init)
        self.bias_rule = None
        if self.brule_init is not None and self.use_biases:
            self.bias_rule = self.init_learning_rule(self.brule_init)
            self.bias_rule.allocate_state([self.updates[-1]])
            self.learning_rule.allocate_state(self.updates[:-1])
        else:
            self.learning_rule.allocate_state(self.updates)
        self.params_initialized = True

    def update(self, epoch):
        if self.bias_rule is None:
            self.learning_rule.apply_rule(self.params, self.updates, epoch)
        else:
            self.learning_rule.apply_rule(self.params[:-1],
                                          self.updates[:-1], epoch)
            self.bias_rule.apply_rule([self.params[-1]],
                                      [self.updates[-1]], epoch)

        if self.accumulate:
            for upm in self.updates:
                upm.fill(0.0)

    def normalize_weights(self, wts):
        norms = self.backend.norm(wts, order=2, axis=1)
        self.backend.divide(wts, norms.reshape((norms.shape[0], 1)), out=wts)

    def set_train_mode(self, mode):
        if self.batch_norm and mode is False:
            self.bn.set_inference_mode()

    def init_learning_rule(self, lrule_init):
        dtype = self.weight_dtype  # TODO: Cool to reuse this here?
        lrname = self.name + '_lr'
        if lrule_init['type'] == 'gradient_descent':
            lr = GradientDescent(name=lrname,
                                 lr_params=lrule_init['lr_params'])
        elif lrule_init['type'] == 'gradient_descent_pretrain':
            lr = GradientDescentPretrain(
                name=lrname, lr_params=lrule_init['lr_params'])
        elif lrule_init['type'] == 'gradient_descent_momentum':
            lr = GradientDescentMomentum(
                name=lrname, lr_params=lrule_init['lr_params'],
                param_dtype=dtype, gradient_dtype=dtype)
        elif lrule_init['type'] == 'gradient_descent_momentum_weight_decay':
            lr = GradientDescentMomentumWeightDecay(
                name=lrname, lr_params=lrule_init['lr_params'],
                param_dtype=dtype, gradient_dtype=dtype)
        elif lrule_init['type'] == 'adadelta':
            lr = AdaDelta(name=lrname, lr_params=lrule_init['lr_params'])
        elif lrule_init['type'] == 'rmsprop':
            lr = RMSProp(name=lrname, lr_params=lrule_init['lr_params'],
                         param_dtype=dtype, gradient_dtype=dtype)
        else:
            raise AttributeError("invalid learning rule params specified")
        lr.initialize(self.backend)
        return lr
