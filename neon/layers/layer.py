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
from __future__ import division
from builtins import str, zip
from functools import wraps
import logging
import numpy as np
import copy

from neon import NervanaObject
from neon.backends import Autodiff
from neon.backends.backend import Tensor
from neon.transforms.activation import Rectlin


logger = logging.getLogger(__name__)


def interpret_in_shape(xshape):
    """
    Helper function to interpret the tensor layout of preceding layer to handle non-recurrent,
    recurrent, and local layers.
    """
    if isinstance(xshape, (int, np.integer)):
        return (xshape, 1)
    else:
        if len(xshape) == 2:
            return xshape
        else:
            return (np.prod(xshape), 1)


class Layer(NervanaObject):

    """
    Top level generic neural network layer class from which all other layer
    types inherit.

    Arguments:
        name (string): Name identifying this layer (in logs, etc.)
        parallelism (int): Type of parallelism preferred by this layer. Possible
            values are "Unknown", "Disabled", and "Data". Only applicable to
            distributed backends (see gen_backend for details).
    """

    def __init__(self, name=None, parallelism="Unknown"):
        super(Layer, self).__init__(name)
        self.outputs = None
        self.has_params = False
        self.inputs = None
        self.owns_output = True
        self.owns_delta = False
        self.deltas = None
        self.parallelism = parallelism
        self.revert_list = []
        self.next_layer = None
        self.actual_bsz = None
        self.actual_seq_len = None
        self.acc_on = False
        self.is_mklop = False

    def __str__(self):
        """
        Format the layer as a printable string.
        """
        ret = '{} {}'.format(self.classnm, self.name)
        return ret

    def nested_str(self, level=0):
        """
        Utility function for displaying layer info with a given indentation level.

        Arguments:
            level (int, optional): indentation level

        Returns:
            str: layer info at the given indentation level
        """

        return "  " * level + str(self)

    def configure(self, in_obj):
        """
        Set shape based parameters of this layer given an input tuple, int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer, Tensor or dataset): object that provides shape
                                                           information for layer
        """
        if isinstance(in_obj, Layer):
            self.prev_layer = in_obj
            self.in_shape = in_obj.out_shape
            if self.parallelism == "Unknown":
                self.parallelism = in_obj.parallelism
        else:
            self.prev_layer = None
            if isinstance(in_obj, (tuple, int, list)):
                self.in_shape = in_obj  # input is a shape tuple or int directly
            elif isinstance(in_obj, Tensor):
                self.in_shape = (in_obj.shape[0], in_obj.shape[1] // self.be.bsz)
            else:
                self.in_shape = in_obj.shape  # This is a dataset

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.
        Don't reallocate if it already exists.
        Only allocate space if layer owns its own output (e.g., bias and activation work
        in place, so do not own their output).
        Outputs can be allocated from a pre-allocated pool if shared_outputs is provided.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into

        """
        if self.outputs:
            return
        if self.owns_output:
            self.outputs = self.be.iobuf(self.out_shape, shared=shared_outputs,
                                         parallelism=self.parallelism)

    def allocate_deltas(self, global_deltas):
        global_deltas.proc_layer(self)

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (DeltasTree): list of pre-allocated tensors (provided by layer container)
        """
        if self.next_layer is not None and self.next_layer.parallelism != self.parallelism:
            self.owns_delta = True

        if self.owns_delta and self.prev_layer:
            if type(self.prev_layer) in (BranchNode,):
                self.deltas = self.prev_layer.deltas
            else:
                self.deltas = self.be.iobuf(self.in_shape, shared=delta_buffers.buffers[0],
                                            parallelism=self.parallelism)
                delta_buffers.buffers.reverse()
        else:
            self.deltas = None

    def set_next(self, layer):
        """
        Set next_layer to provided layer.

        Arguments:
            layer (layer): Next layer
        """
        self.next_layer = layer

    def get_is_mklop(self):
        """
        is_mklop true means this op is on mkl backend
        and may require convert when from non-mkl op
        """
        return self.is_mklop

    def set_is_mklop(self):
        self.is_mklop = True

    def set_not_mklop(self):
        self.is_mklop = False

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        raise NotImplementedError

    def _fprop_inference(self, inputs):
        """
        Apply the forward pass transformation to the input data.

        May skip any computation not needed for doing inference only.

        Calling bprop subsequently is not valid.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        raise NotImplementedError

    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        raise NotImplementedError

    def get_terminal(self):
        """
        Used for recursively getting final nodes from layer containers.
        """
        return self

    def serialize(self):
        """
        Get state parameters for this layer.

        Returns:
            varies: whatever data this model wants to receive in order to restore state
        """
        if self.has_params:
            return self.get_params()

    def load_weights(self, pdict, load_states=True):
        """
        Load weights.

        Arguments:
            pdict:
            load_states:  (Default value = True)
        """
        self.set_params(pdict)
        if load_states:
            self.set_states(pdict)

    def get_param_attrs(self):
        return dict(parallel=(self.parallelism in ("Data", "Model")),
                    distributed=(self.parallelism == "Model"))

    def set_params(self, pdict):
        pass

    def set_states(self, pdict):
        pass

    def set_batch_size(self, N):
        """
        Set minibatch size.

        Arguments:
            N (int): minibatch size
        """
        self.actual_bsz = N

    def set_seq_len(self, S):
        """
        Set sequence length.

        Arguments:
            S (int): sequence length
        """
        self.actual_seq_len = S

    def get_description(self, **kwargs):
        return super(Layer, self).get_description(**kwargs)

    def set_acc_on(self, acc_on):
        """
        Set the acc_on flag according to bool argument. If set to true, the
        layer will accumulate some (preset) parameters on calls to functions
        that are decorated with the accumulates decorator. In order to use this
        feature, accumulate_updates=True must have been passed to the layer's
        allocate function

        This currently only works for a few hard coded parameters in select layers

        Arguments:
           acc_on (bool): Value to set the acc_on flag to.
        """
        if (not hasattr(self, "accumulate_updates")):
            raise BufferError("accumulate_updates not set")
        if self.accumulate_updates:
            self.acc_on = acc_on

    @staticmethod
    def accumulates(f):
        """
        Higher order decorator function that enables accumulation functionality for that function.
        Object that use this decorator are required to have an acc_param attribute. This attribute
        tuple declares the names for existing temp parameter and real parameter buffers. The
        temp parameter buffer copies the value of the parameter buffer before f is called, and
        after f is called the temp and normal buffers are summed. This decorator could be used
        to wrap any function that may want to accumulate parameters instead of overwriting.
        """
        def accum_pre(self):
            """
            copy the real acc params to the temp buffer
            """
            if self.acc_on:
                for (acc_p, p) in self.acc_params:
                    acc_p[:] = p

        def accum_post(self):
            """
            element wise sum of temp buffer and updated param buffers
            """
            if self.acc_on:
                for (acc_p, p) in self.acc_params:
                    p[:] = p + acc_p

        @wraps(f)
        def accum_wrapper(self, *args, **kwargs):
            """
            Wraps the function f with calls to accum_pre and accum_post
            """
            if not isinstance(self, Layer):
                raise (TypeError("accumulates can only by subclasses of Layer"))
            accum_pre(self)
            out = f(self, *args, **kwargs)
            accum_post(self)
            return out
        return accum_wrapper


class BranchNode(Layer):
    """
    Layer that allows branching.  Used to send outputs to multiple layer pathways.
    Each pathway will get the entire output of the layer preceding the branch node.
    """
    instances = {}

    def __new__(cls, name=None):
        """
        Branch nodes need to have a unique name,
        which identifies them.  This method checks
        to see if the branch node being made has already
        been created using its name and if so it returns
        that instance.
        """
        if name in cls.instances:
            return cls.instances[name]
        else:
            return Layer.__new__(cls)

    def __init__(self, name=None):
        # don't init if this is not a new instance
        # see __new__ above
        if name not in BranchNode.instances:
            BranchNode.instances[name] = self
            super(BranchNode, self).__init__(name)
            self.owns_output = False

    def fprop(self, inputs=None, inference=False):
        """
        Passes output from preceding layer on without modification.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        if self.outputs is None and inputs is not None:
            self.outputs = inputs
        return self.outputs

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        if hasattr(self, 'in_shape') and in_obj is None:
            return self  # previously configured, so just return
        super(BranchNode, self).configure(in_obj)
        self.out_shape = self.in_shape
        return self

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (DeltasTree): list of pre-allocated tensors (provided by layer container)
        """
        if self.deltas is None:
            self.deltas = self.be.iobuf(self.in_shape, shared=delta_buffers.buffers[0])
            delta_buffers.buffers.reverse()

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Branch nodes should be skipped in bprop, since their deltas are shared

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        pass


class SkipNode(Layer):
    """
    Layer that allows pass-through as in [He2015]_.

    Notes:

    .. [He2015] http://arxiv.org/abs/1502.03167
    """

    def __init__(self, name=None):
        super(SkipNode, self).__init__(name)
        self.owns_delta = True
        self.is_mklop = True

    def fprop(self, inputs=None, inference=False, beta=0):
        """
        Passes output from preceding layer on without modification

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): Scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.be.fprop_skipnode(inputs, self.outputs, beta)
        return self.outputs

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(SkipNode, self).configure(in_obj)
        self.out_shape = self.in_shape
        return self

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Skip nodes just pass back what they got.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        # for better performance, mkl do nothing
        # otherwise, convert back and deal with beta and alpha.
        self.be.bprop_skipnode(error, self.deltas, alpha, beta)
        return self.deltas


class Pooling(Layer):

    """
    Pooling layer implementation.

    Arguments:
        fshape (int, tuple(int, int)): one or two dimensional shape
            of pooling window
        op (str, optional): pooling operation in [max, avg]. Defaults to "max"
        strides (int, dict, optional): strides to apply pooling window
            over. An int applies to both dimensions, or a dict with str_h
            and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        name (str, optional): layer name. Defaults to "PoolingLayer"
    """

    def __init__(self, fshape, op="max", strides={}, padding={},
                 name=None):
        super(Pooling, self).__init__(name)
        self.poolparams = {'str_h': None, 'str_w': None, 'str_d': None, 'str_c': None,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0, 'pad_c': 0,
                           'J': 1, 'T': 1, 'D': 1, 'op': op}  # 3D paramaters

        # keep args around in __dict__ for get_description
        self.op = op
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.owns_delta = True
        self.is_mklop = True
        if isinstance(fshape, int):
            fshape = {'R': fshape, 'S': fshape}
        elif isinstance(fshape, tuple):
            fkeys = ('R', 'S') if len(fshape) == 2 else ('T', 'R', 'S')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        elif fshape == 'all':
            fshape = dict(R=None, S=None)
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.poolparams.update(d)
        self.nglayer = None

    def __str__(self):
        return "Pooling Layer '%s': %d x (%dx%d) inputs, %d x (%dx%d) outputs" % (
               self.name,
               self.in_shape[0], self.in_shape[1], self.in_shape[2],
               self.out_shape[0], self.out_shape[1], self.out_shape[2])

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Pooling, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.poolparams.update(shapedict)
            if self.poolparams['R'] is None:
                self.poolparams['R'] = shapedict['H']
                self.poolparams['S'] = shapedict['W']
            self.nglayer = self.be.pool_layer(self.be.default_dtype, **self.poolparams)
            (K, M, P, Q, N) = self.nglayer.dimO
            self.out_shape = (K, M, P, Q) if len(self.in_shape) == 4 else (K, P, Q)
        return self

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        super(Pooling, self).set_deltas(delta_buffers)
        if self.op == "max":
            self.argmax = self.be.empty(self.outputs.shape, dtype=np.uint8)
        else:
            self.argmax = None

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.fprop_pool(self.nglayer, inputs, self.outputs, self.argmax, beta=beta)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """

        self.be.bprop_pool(self.nglayer, error, self.deltas, self.argmax, alpha, beta)
        return self.deltas


class ParameterLayer(Layer):

    """
    Intermediate class used for common functionality for any layer with weights.

    Not intended to be used directly.

    Arguments:
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "ParameterLayer"
    """

    def __init__(self, init=None, name=None,
                 parallelism="Unknown"):
        super(ParameterLayer, self).__init__(name, parallelism)
        self.has_params = True
        self.init = init
        self.W = None
        self.dW = None
        self.weight_shape = None
        self.batch_sum = None
        self.batch_sum_shape = None
        self.states = []
        self.owns_delta = True

    def allocate(self, shared_outputs=None, accumulate_updates=False):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
            accumulate_updates (bool): allocate additional scratch accumulation
                buffers.
        """
        super(ParameterLayer, self).allocate(shared_outputs)
        self.accumulate_updates = accumulate_updates
        if self.W is None:
            self.init_params(self.weight_shape)
        if self.batch_sum_shape is not None:
            self.batch_sum = self.be.empty(self.batch_sum_shape, dtype=np.float32,
                                           **self.get_param_attrs())

    def init_params(self, shape):
        """
        Allocate layer parameter buffers and initialize them with the
            supplied initializer.

        Arguments:
            shape (int, tuple): shape to allocate for layer parameter
                buffers.
            accumulate_updates (bool): allocate additional scratch accumulation
                buffers.
        """
        self.W = self.be.empty(shape, **self.get_param_attrs())
        self.dW = self.be.empty_like(self.W)
        self.states = []

        if isinstance(self.init, Tensor) or isinstance(self.init, np.ndarray):
            assert self.init.shape == self.W.shape, "Initial weights shape does not match"
            self.W[:] = self.init
        else:
            self.init.fill(self.W)
        if self.accumulate_updates:
            self.acc_dW = self.be.empty_like(self.dW)
            self.acc_params = [(self.acc_dW, self.dW)]

    def get_params(self):
        """
        Get layer parameters, gradients, and states for optimization.
        """
        return ((self.W, self.dW), self.states)

    def get_params_serialize(self, keep_states=True):
        return self.get_description(get_weights=True, keep_states=keep_states)

    def get_description(self, get_weights=False, keep_states=True):
        """
        Get layer parameters. All parameters are needed for optimization, but
        only weights are serialized.

        Arguments:
            get_weights (bool, optional): Control whether all parameters are returned or
                                          just weights for serialization.
            keep_states (bool, optional): Control whether all parameters are returned
                or just weights for serialization.
        """
        serial_dict = super(ParameterLayer, self).get_description()
        if get_weights:
            serial_dict['params'] = {'W': self.W.get()}
            if keep_states:
                serial_dict['states'] = [s.get() for s in self.states]
        return serial_dict

    def set_params(self, pdict):
        """
        Set layer parameters (weights). Allocate space for other parameters but do not initialize
        them.

        Arguments:
            pdict (dict, ndarray): dictionary or ndarray with layer parameters
                                   [support for ndarray is DEPRECATED and will be removed]
        """
        assert type(pdict) is dict
        for key in pdict['params']:
            if not hasattr(self, key):
                setattr(self, key, None)

            attr = getattr(self, key)
            if isinstance(attr, Tensor):
                # this attr has already been allocated
                # get set the values
                attr.set(pdict['params'][key])
            elif type(pdict['params'][key]) is np.ndarray:
                setattr(self, key, self.be.array(pdict['params'][key], **self.get_param_attrs()))
            else:
                setattr(self, key, pdict['params'][key])

        if self.dW is None:
            self.dW = self.be.empty_like(self.W)

    def set_states(self, pdict):
        if 'states' not in pdict:
            # if states was not serialized then leave
            # this empty, the optimizer will initialize it
            self.states = []
        else:
            # this needs to be done in two steps for MGPU backend
            if self.states is None or len(self.states) == 0:
                self.states = [self.be.zeros_like(self.dW)
                               for i in range(len(pdict['states']))]

            for ind in range(len(pdict['states'])):
                self.states[ind].set(pdict['states'][ind])


class Convolution(ParameterLayer):

    """
    Convolutional layer implementation.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window.
            Order of the axis should be height, width, channels.  For 4d
            convolution, the order should be depth, height, width, channels.
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to all dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = 1
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to all dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = 0
        dilation (int, dict, optional): dilation to apply to dimensions of
            the filter. An int applies to all dimensions, or a dict with dil_h
            and dil_w applies to h and w dimensions distinctly.  Defaults
            to dil_w = dil_h = 1
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "ConvolutionLayer"
    """

    def __init__(self, fshape, strides={}, padding={}, dilation={}, init=None,
                 bsum=False, name=None, parallelism="Data"):
        super(Convolution, self).__init__(init, name, parallelism)
        self.is_mklop = True
        self.nglayer = None
        self.bsum = bsum
        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'dil_h': 1, 'dil_w': 1, 'dil_d': 1,
                           'T': 1, 'D': 1}  # 3D paramaters

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.dilation = dilation

        if isinstance(fshape, tuple) or isinstance(fshape, list):
            fkeys = ('R', 'S', 'K') if len(fshape) == 3 else ('T', 'R', 'S', 'K')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        if isinstance(dilation, int):
            dilation = {'dil_h': dilation, 'dil_w': dilation}
        for d in [fshape, strides, padding, dilation]:
            self.convparams.update(d)

    def __str__(self):
        input_spatial_dim = len(self.in_shape) - 1
        output_spatial_dim = len(self.out_shape) - 1
        input_spatial_str = "%d x (" + "x".join(("%d",) * input_spatial_dim) + ")"
        output_spatial_str = "%d x (" + "x".join(("%d",) * output_spatial_dim) + ")"
        padstr_str = ",".join(("%d",) * input_spatial_dim)
        padstr_dim = ([] if input_spatial_dim == 2 else ['d']) + ['h', 'w']

        pad_tuple = tuple(self.convparams[k] for k in ['pad_' + d for d in padstr_dim])
        str_tuple = tuple(self.convparams[k] for k in ['str_' + d for d in padstr_dim])
        dil_tuple = tuple(self.convparams[k] for k in ['dil_' + d for d in padstr_dim])

        fmt_tuple = (self.name,) + self.in_shape + self.out_shape + (
                     pad_tuple + str_tuple + dil_tuple)
        fmt_string = "Convolution Layer '%s': " + \
                     input_spatial_str + " inputs, " + output_spatial_str + " outputs, " + \
                     padstr_str + " padding, " + padstr_str + " stride, " + \
                     padstr_str + " dilation"

        return ((fmt_string % fmt_tuple))

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Convolution, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.convparams.update(shapedict)
            self.nglayer = self.be.conv_layer(self.be.default_dtype, **self.convparams)
            (K, M, P, Q, N) = self.nglayer.dimO
            self.out_shape = (K, P, Q) if M == 1 else (K, M, P, Q)
        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)
        if self.bsum:
            self.batch_sum_shape = (self.nglayer.K, 1)
        return self

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.fprop_conv(self.nglayer, inputs, self.W, self.outputs, beta=beta,
                           bsum=self.batch_sum, layer_op=self)
        return self.outputs

    @Layer.accumulates
    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.bprop_conv(self.nglayer, self.W, error, self.deltas,
                               alpha=alpha, beta=beta, layer_op=self)
        self.be.update_conv(self.nglayer, self.inputs, error, self.dW, layer_op=self)
        return self.deltas


class Convolution_bias(Convolution):
    """
    Convolutional layer with bias.
    Contains weight and bias as parameters which are updated during training.

    Arguments:
        init: init method for weight
        bias: init method for bias

    """
    def __init__(self, fshape, strides={}, padding={}, dilation={}, init=None,
                 bsum=False, name=None, bias=None, parallelism="Data"):

        super(Convolution_bias, self).__init__(fshape, strides=strides,
                                               padding=padding, dilation=dilation,
                                               init=init,
                                               bsum=bsum, name=name,
                                               parallelism=parallelism)
        self.init_bias = bias
        self.weight_bias = None
        self.grad_bias = None
        self.bias = bias
        self.states = [[] for i in range(2)]

    def init_params(self, shape):
        # init weight
        super(Convolution_bias, self).init_params(shape)
        self.states = [[] for i in range(2)]

        # init bias, using channel number
        dim0 = self.out_shape[0]
        self.weight_bias = self.be.zeros((dim0, 1), **self.get_param_attrs())
        self.grad_bias = self.be.empty_like(self.weight_bias)
        if isinstance(self.init_bias, Tensor) or isinstance(self.init_bias, np.ndarray):
            assert self.init.shape == self.bias.shape, "Initial weights shape does not match"
            self.weight_bias[:] = self.init
        else:
            self.init_bias.fill(self.weight_bias)
        if self.accumulate_updates:
            self.acc_grad_bias = self.be.empty_like(self.grad_bias)
            self.acc_params += [(self.acc_grad_bias, self.grad_bias)]

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Convolution_bias, self).configure(in_obj)
        return self

    def get_params(self):
        return [((self.W, self.dW), self.states[0]),
                ((self.weight_bias, self.grad_bias), self.states[1])]

    def get_description(self, get_weights=False, keep_states=True):
        serial_dict = super(ParameterLayer, self).get_description()
        if get_weights:
            serial_dict['params'] = {}
            for key in ['W', 'weight_bias']:
                serial_dict['params'][key] = getattr(self, key).get()
            if keep_states:
                serial_dict['states'] = [[s.get() for s in slist] for slist in self.states]
        return serial_dict

    def convert_format(self, pdict):
        assert isinstance(pdict, list) and len(pdict) == 2

        (p1, p2) = pdict
        p = copy.deepcopy(p1)
        p['config']['bias'] = p2['config']['init']
        # Assume p1 and p2 will both have 'states' key or both will not have 'states' key
        if 'states' in p1 and 'states' in p2:
            p['states'] = [p1['states'], p2['states']]
        p['params']['weight_bias'] = p2['params']['W']
        p['type'] = 'neon.layers.layer.Convolution_bias'

        return p

    def set_params(self, pdict):
        """
        Set layer parameters (weights). Allocate space for other parameters but do not initialize
        them.
        Arguments:
            pdict (dict, ndarray): dictionary or ndarray with layer parameters
                                   [support for ndarray is DEPRECATED and will be removed]
        """

        if type(pdict) is list:
            logger.error('Using old serialization format with unfused bias. This will be'
                         ' deprecated in future release. Resave serialized file'
                         ' using current format.')
            pdict = self.convert_format(pdict)

        for key, val in pdict['params'].items():
            if not hasattr(self, key):
                setattr(self, key, None)

            if isinstance(getattr(self, key), Tensor):
                getattr(self, key).set(val)
            elif isinstance(val, np.ndarray):
                setattr(self, key, self.be.array(val, **self.get_param_attrs()))
            else:
                setattr(self, key, val)

        if self.dW is None:
            assert self.W is not None
            self.dW = self.be.empty_like(self.W)

        if self.grad_bias is None:
            assert self.weight_bias is not None
            self.grad_bias = self.be.empty_like(self.weight_bias)

    def set_states(self, pdict):
        if type(pdict) is list:
            pdict = self.convert_format(pdict)
        assert 'states' in pdict.keys() \
               and any(pdict['states']), "weights " \
                                         "do not contain states information, " \
                                         "consider setting load_states=False " \
                                         "in load_params"

        if not any(self.states):
            self.states = [[self.be.array(x, **self.get_param_attrs()) for x in slist]
                           for slist in pdict['states']]
        else:
            for dlist, slist in zip(self.states, pdict['states']):
                for dst, src in zip(dlist, slist):
                    dst.set(src)

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.fprop_conv(self.nglayer, inputs, self.W, self.outputs, beta=beta,
                           bias=self.weight_bias, bsum=self.batch_sum, layer_op=self)
        return self.outputs

    @Layer.accumulates
    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.bprop_conv(self.nglayer, self.W, error, self.deltas,
                               alpha=alpha, beta=beta, layer_op=self)
        self.be.update_conv(self.nglayer, self.inputs, error, self.dW,
                            grad_bias=self.grad_bias, layer_op=self)
        return self.deltas


class Deconvolution(ParameterLayer):

    """
    Deconvolutional layer implementation.

    Arguments:
        fshape (tuple): three dimensional shape of convolution window
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to all dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = 1
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to all dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = 0
        dilation (int, dict, optional): dilation to apply to dimensions of
            the filter. An int applies to all dimensions, or a dict with dil_h
            and dil_w applies to h and w dimensions distinctly.  Defaults
            to dil_w = dil_h = 1
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "DeconvolutionLayer"
    """

    def __init__(self, fshape, strides={}, padding={}, dilation={}, init=None, bsum=False,
                 name=None):
        super(Deconvolution, self).__init__(init, name)
        self.nglayer = None
        self.bsum = bsum
        self.deconvparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                             'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                             'dil_h': 1, 'dil_w': 1, 'dil_d': 1,
                             'T': 1, 'M': 1}  # 3D paramaters

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.dilation = dilation

        if isinstance(fshape, tuple) or isinstance(fshape, list):
            fkeys = ('R', 'S', 'C') if len(fshape) == 3 else ('T', 'R', 'S', 'C')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        if isinstance(dilation, int):
            dilation = {'dil_h': dilation, 'dil_w': dilation}
        for d in [fshape, strides, padding, dilation]:
            self.deconvparams.update(d)

    def __str__(self):
        input_spatial_dim = len(self.in_shape) - 1
        output_spatial_dim = len(self.out_shape) - 1
        input_spatial_str = "%d x (" + "x".join(("%d",) * input_spatial_dim) + ")"
        output_spatial_str = "%d x (" + "x".join(("%d",) * output_spatial_dim) + ")"
        padstr_str = ",".join(("%d",) * input_spatial_dim)
        padstr_dim = ([] if input_spatial_dim == 2 else ['d']) + ['h', 'w']

        pad_tuple = tuple(self.deconvparams[k] for k in ['pad_' + d for d in padstr_dim])
        str_tuple = tuple(self.deconvparams[k] for k in ['str_' + d for d in padstr_dim])
        dil_tuple = tuple(self.deconvparams[k] for k in ['dil_' + d for d in padstr_dim])

        fmt_tuple = (self.name,) + self.in_shape + self.out_shape + (
                     pad_tuple + str_tuple + dil_tuple)
        fmt_string = "Deconvolution Layer '%s': " + \
                     input_spatial_str + " inputs, " + output_spatial_str + " outputs, " + \
                     padstr_str + " padding, " + padstr_str + " stride, " + \
                     padstr_str + " dilation"

        return ((fmt_string % fmt_tuple))

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Deconvolution, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('K', 'P', 'Q') if len(self.in_shape) == 3 else ('K', 'M', 'P', 'Q')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.deconvparams.update(shapedict)
            self.nglayer = self.be.deconv_layer(self.be.default_dtype, **self.deconvparams)
            (C, D, H, W, N) = self.nglayer.dimI
            self.out_shape = (C, H, W) if D == 1 else (C, D, H, W)
        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)
        if self.bsum:
            self.batch_sum_shape = (self.nglayer.C, 1)
        return self

    def fprop(self, inputs, inference=False):
        """
        fprop for deconv is equivalent to bprop for conv.
        bprop_conv takes in error and deltas as "E" and "grad_I"
        for deconv, bprop_conv will take in input as "E" and output as "grad_I"

        Arguments:
            inference (bool): is inference only
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.bprop_conv(layer=self.nglayer, F=self.W, E=inputs, grad_I=self.outputs,
                           bsum=self.batch_sum)
        return self.outputs

    @Layer.accumulates
    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        bprop for deconv is equivalent to fprop for conv.
        fprop_conv takes input and output as "I" and "O".
        for deconv, fprop_conv will take error as input and delta as output

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.fprop_conv(self.nglayer, error, self.W, self.deltas, alpha=alpha, beta=beta)
        self.be.update_conv(self.nglayer, error, self.inputs, self.dW)
        return self.deltas


class Linear(ParameterLayer):

    """
    A fully connected layer implemented as the dot product of inputs and
    weights.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): Layer name. Defaults to "LinearLayer"
    """

    def __init__(self, nout, init, bsum=False, name=None, parallelism="Disabled"):
        super(Linear, self).__init__(init, name, parallelism)
        self.nout = nout
        self.inputs = None
        self.bsum = bsum

    def __str__(self):
        return "Linear Layer '%s': %d inputs, %d outputs" % (
               self.name, self.nin, self.nout)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Linear, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.out_shape = (self.nout, self.nsteps)
        if self.weight_shape is None:
            self.weight_shape = (self.nout, self.nin)
        if self.bsum:
            self.batch_sum_shape = (self.nout, 1)
        return self

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        if self.actual_bsz is None and self.actual_seq_len is None:
            self.be.compound_dot(A=self.W, B=self.inputs, C=self.outputs, beta=beta,
                                 bsum=self.batch_sum)
        else:
            bsz = self.be.bsz if self.actual_bsz is None else self.actual_bsz
            steps = self.nsteps if self.actual_seq_len is None else self.actual_seq_len
            self.be.compound_dot(A=self.W,
                                 B=self.inputs[:, :bsz * steps],
                                 C=self.outputs[:, :bsz * steps],
                                 beta=beta,
                                 bsum=self.batch_sum)

        return self.outputs

    @Layer.accumulates
    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.compound_dot(A=self.W.T, B=error, C=self.deltas, alpha=alpha, beta=beta)
        self.be.compound_dot(A=error, B=self.inputs.T, C=self.dW)
        return self.deltas


class BinaryLinear(Linear):

    """
    A binary fully connected layer implemented as the dot product of inputs
    and binarized weights.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): Layer name. Defaults to "BinaryLinearLayer"
    """

    def __str__(self):
        return "BinaryLinear Layer '%s': %d inputs, %d outputs" % (
               self.name, self.nin, self.nout)

    def allocate(self, shared_outputs=None):
        super(BinaryLinear, self).allocate(shared_outputs)
        self.Wb = self.be.empty_like(self.W)

    def fprop(self, inputs, inference=False, beta=0.0):
        self.inputs = inputs
        self.be.binarize(self.W, self.Wb, stochastic=False)

        not_binarized = self.be.zeros(self.inputs.shape)
        not_binarized[:] = self.be.not_equal(self.be.absolute(self.inputs), 1)
        if np.any(not_binarized.get()):
            gemm = self.be.compound_dot
        else:
            gemm = self.be.xnor_compound_dot

        if self.actual_bsz is None and self.actual_seq_len is None:
            gemm(A=self.Wb, B=self.inputs, C=self.outputs, beta=beta,
                 bsum=self.batch_sum)
        else:
            bsz = self.be.bsz if self.actual_bsz is None else self.actual_bsz
            steps = self.nsteps if self.actual_seq_len is None else self.actual_seq_len

            gemm(A=self.Wb,
                 B=self.inputs[:, :bsz * steps],
                 C=self.outputs[:, :bsz * steps],
                 beta=beta,
                 bsum=self.batch_sum)

        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        if self.deltas:
            self.be.compound_dot(A=self.Wb.T, B=error, C=self.deltas, alpha=alpha, beta=beta)
        self.be.compound_dot(A=error, B=self.inputs.T, C=self.dW)
        return self.deltas


class Bias(ParameterLayer):

    """
    A bias layer implemented that adds a learned bias to inputs and produces
    outputs of the same shape.

    Arguments:
        init (Initializer, optional): Initializer object to use for
            initializing layer bias
        name (str, optional): Layer name. Defaults to "BiasLayer"
    """

    def __init__(self, init, name=None):
        super(Bias, self).__init__(init, name)
        self.y = None
        self.owns_output = False
        self.owns_delta = False

    def __str__(self):
        if len(self.in_shape) == 3:
            layer_string = "Bias Layer '%s': size %d x (%dx%d)" % (
                self.name, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        else:
            layer_string = "Bias Layer '%s': size %d" % (self.name, self.bias_size)
        return layer_string

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Bias, self).configure(in_obj)
        self.out_shape = self.in_shape
        self.bias_size = self.in_shape[0]
        if self.weight_shape is None:
            self.weight_shape = (self.bias_size, 1)
        return self

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inference (bool): is inference only
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        self.outputs = self.inputs = inputs
        if self.y is None or self.y.base is not self.outputs:
            self.y = self.outputs.reshape((self.bias_size, -1))
        self.y[:] = self.y + self.W
        return self.outputs

    @Layer.accumulates
    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas is None:
            self.deltas = error.reshape(self.y.shape)
        self.be.sum(self.deltas, axis=1, out=self.dW)
        return error


class Activation(Layer):

    """
    A layer that applies a specified transform to the inputs and
    produces outputs of the same shape.

    Generally used to implemenent nonlinearities for layer post activations.

    Arguments:
        transform (Transform): a transform object with fprop and bprop
            functions to apply
        name (str, optional): Layer name. Defaults to "ActivationLayer"
    """

    def __init__(self, transform, name=None):
        super(Activation, self).__init__(name)
        self.transform = transform
        self.owns_output = False
        self.owns_delta = True

    def __str__(self):
        return "Activation Layer '%s': %s" % (
               self.name, self.transform.__class__.__name__)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Activation, self).configure(in_obj)
        self.nglayer = self.be.relu_layer()
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def get_is_mklop(self):
        if self.transform is None:
            return False
        return self.transform.is_mklop

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.outputs = self.inputs = inputs
        self.be.fprop_transform(self.nglayer, self.transform, self.inputs,
                                self.outputs, type(self.transform) == Rectlin)
        return self.outputs

    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        self.be.bprop_transform(self.nglayer, self.transform, self.outputs,
                                error, self.deltas, type(self.transform) == Rectlin)
        return self.deltas


class Reshape(Layer):

    """
    A layer that reshape the input

    Arguments:
        reshape: (tuple(int)): multi-dimensional shape of how to reshape the input.

        It can contain 0, which will be replaced by the size on that dimension
        from inputs.

        It can contain -1, which will be configured to match the total
        size of the tensor.

        The length of the reshape can be smaller or bigger than the input shape.

        The batch size dimension is implicit.The shape interpretation is consistent
        with rest of neon. If reshape to 2D, it will assume the 2nd dimension is
        time and combine it with backend batch size. If reshape to 3D, it will
        assume to be (C, H, W) dimensions and add batch size dimension in the end.
    """

    def __init__(self, reshape, name=None):
        super(Reshape, self).__init__(name)
        if isinstance(reshape, int):
            reshape = (reshape,)
        self.reshape = reshape
        self.owns_output = False

    def __str__(self):
        return "Reshape Layer '%s' input shape %s to %s" % (self.name, self.in_shape, self.reshape)

    def configure(self, in_obj):
        """
        Configure the output shape based on input shape and reshape shape.
        The function replaces 0 and -1 and add the batch size dimension.
        """
        super(Reshape, self).configure(in_obj)
        if isinstance(self.in_shape, tuple):
            if len(self.in_shape) == 2:
                self.in_shape_t = (
                    self.in_shape[0], self.in_shape[1] * self.be.bsz)
            else:
                self.in_shape_t = (int(np.prod(self.in_shape)), self.be.bsz)
        else:
            self.in_shape_t = (self.in_shape, self.be.bsz)

        self.out_shape = list(self.reshape)

        if 0 in self.reshape:
            dim_to_keep = np.where(np.array(self.reshape) == 0)[0][0]
            self.out_shape[dim_to_keep] = list(self.in_shape)[dim_to_keep]

        if -1 in self.reshape:
            missing_dim = -int(np.prod(self.in_shape)) // int(np.prod(self.out_shape))
            self.out_shape = [missing_dim if x == -1 else x for x in self.out_shape]

        self.out_shape = tuple(self.out_shape)

        if len(self.out_shape) == 2:
            self.out_shape_t = (
                self.out_shape[0], self.out_shape[1] * self.be.bsz)
        else:
            self.out_shape_t = (int(np.prod(self.out_shape)), self.be.bsz)

        assert np.prod(self.out_shape) == np.prod(self.in_shape)
        return self

    def fprop(self, inputs, inference=False):
        """
        In cases that inputs from previous layer are contiguous tensor, the layer
        creates a reshaped view.
        In cases that they are non-contiguous, the layer does a copy of the data
        and then creates a reshaped view.
        """
        if inputs.is_contiguous is False:
            if self.inputs is None:
                self.inputs = self.be.empty_like(inputs)
                self.outputs = self.inputs.reshape(self.out_shape_t)
            self.inputs.copy(inputs)
        else:
            if self.inputs is None or self.inputs is not inputs:
                self.inputs = inputs
                self.outputs = self.inputs.reshape(self.out_shape_t)

        return self.outputs

    def bprop(self, error):
        """
        Backward propagation reshapes the error inputs for previous layer.
        """
        if self.deltas is None:
            self.deltas = error.reshape(self.in_shape_t)
        return self.deltas


class DataTransform(Layer):

    """
    A layer that applies a specified transform to input data in fprop only.

    Only supported as the first layer in the network.

    Arguments:
        transform (Transform): a transform object with fprop function to apply
        name (str, optional): Layer name. Defaults to "DataTransformLayer"
    """

    def __init__(self, transform, name=None):
        super(DataTransform, self).__init__(name)
        self.transform = transform
        self.owns_output = False

    def __str__(self):
        return "DataTransform Layer '%s': %s" % (
               self.name, self.transform.__class__.__name__)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(DataTransform, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.outputs = self.inputs = inputs
        self.outputs[:] = self.transform(self.inputs)
        return self.outputs

    def bprop(self, *args):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            *args (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        return None


class CompoundLayer(list):
    """
    Base class for macro layers.
    """
    def __init__(self, bias=None, batch_norm=False, activation=None, name=None):
        super(CompoundLayer, self).__init__()
        if batch_norm and (bias is not None):
            raise AttributeError('Batchnorm and bias cannot be combined')
        self.activation = activation
        self.batch_norm = batch_norm
        self.bias = bias
        self.base_name = name

    def init_base_name(self):
        if self.base_name is None:
            self.base_name = self[-1].name

    def add_postfilter_layers(self):
        self.init_base_name()
        if self.bias is not None:
            name = self.base_name + '_bias'
            self.append(Bias(init=self.bias, name=name))
        if self.batch_norm:
            name = self.base_name + '_bnorm'
            self.append(BatchNorm(name=name))
        if self.activation is not None:
            name = self.base_name + '_' + self.activation.classnm
            self.append(Activation(transform=self.activation, name=name))


class Affine(CompoundLayer):

    """
    A linear layer with a learned bias and activation, implemented as a list
    composing separate linear, bias/batchnorm and activation layers.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        name (str): the root name for the layer, suffixes are automatically
            generated for the component layers

    """

    def __init__(self, nout, init, bias=None,
                 batch_norm=False, activation=None, name=None,
                 parallelism="Disabled"):
        super(Affine, self).__init__(bias=bias, batch_norm=batch_norm,
                                     activation=activation, name=name)
        self.append(Linear(nout, init, bsum=batch_norm, name=name,
                           parallelism=parallelism))
        self.add_postfilter_layers()


class BinaryAffine(CompoundLayer):

    """
    A binary linear layer with a learned bias and activation, implemented
    as a list composing separate linear, bias/batchnorm and activation layers.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        name (str): the root name for the layer, suffixes are automatically
            generated for the component layers

    """

    def __init__(self, nout, init, bias=None,
                 batch_norm=False, activation=None, name=None):
        super(BinaryAffine, self).__init__(bias=bias, batch_norm=batch_norm,
                                           activation=activation, name=name)
        self.append(BinaryLinear(nout, init, bsum=batch_norm, name=name))
        self.add_postfilter_layers()

    def add_postfilter_layers(self):
        self.init_base_name()
        if self.bias is not None:
            name = self.base_name+'_bias'
            self.append(Bias(init=self.bias, name=name))
        if self.batch_norm:
            name = self.base_name+'_sbnorm'
            self.append(ShiftBatchNorm(name=name))
        if self.activation is not None:
            name = self.base_name + '_' + self.activation.classnm
            self.append(Activation(transform=self.activation, name=name))


class Conv(CompoundLayer):

    """
    A convolutional layer with a learned bias and activation, implemented as a
    list composing separate Convolution, Bias and Activation layers.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to all dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = 1
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to all dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = 0
        dilation (int, dict, optional): dilation to apply to dimensions of
            the filter. An int applies to all dimensions, or a dict with dil_h
            and dil_w applies to h and w dimensions distinctly.  Defaults
            to dil_w = dil_h = 1
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        name (str): the root name for the layer, suffixes are automatically
            generated for the component layers

    """

    def __init__(self, fshape, init, strides={}, padding={}, dilation={},
                 bias=None,
                 batch_norm=False,
                 activation=None,
                 name=None):
        super(Conv, self).__init__(bias=bias, batch_norm=batch_norm,
                                   activation=activation, name=name)

        if bias is not None:
            self.append(Convolution_bias(fshape=fshape, strides=strides, padding=padding,
                                         dilation=dilation, init=init, bsum=batch_norm, bias=bias,
                                         name=name))
        else:
            self.append(Convolution(fshape=fshape, strides=strides, padding=padding,
                                    dilation=dilation, init=init, bsum=batch_norm,
                                    name=name))
        self.add_postfilter_layers()

    def add_postfilter_layers(self):
        self.init_base_name()

        if self.batch_norm:
            name = self.base_name + '_bnorm'
            self.append(BatchNorm(name=name))
        if self.activation is not None:
            name = self.base_name + '_' + self.activation.classnm
            self.append(Activation(transform=self.activation, name=name))


class Deconv(CompoundLayer):

    """
    Same as Conv layer, but implements a composite deconvolution layer.
    """

    def __init__(self, fshape, init, strides={}, padding={}, dilation={},
                 bias=None, batch_norm=False, activation=None, name=None):
        super(Deconv, self).__init__(bias=bias, batch_norm=batch_norm,
                                     activation=activation, name=name)
        self.append(Deconvolution(fshape=fshape, strides=strides, padding=padding,
                                  dilation=dilation, init=init, bsum=batch_norm))
        self.add_postfilter_layers()


class LRN(Layer):

    """
    Local Response Normalization layer.  This layer normalizes the output
    of each pixel/element across channels using the formula:

    .. math::

        output(h,w)_j = \\frac{output(h,w)_j}{(1 + (ascale/N) \sum{x(h,w)_i^2})^{bpower}}

    :math:`x(h,w)_i` is the input element at coordinate :math:`(h,w)` of the i-th feature map,
    :math:`output(h,w)_j` is the corresponding normalized output and the
    sum is taken over :math:`i` in the range :math:`[j - (depth-1)/2, j + (depth-1)/2]`

    Arguments:
        depth (int): the number of neighboring feature maps to include in
                     the normalization, depth must be odd and (depth-1)/2
                     neighbors are included from each side, zeros are added
                     as needed
        ascale (float): the normalization scaling factor (see equation above)
        bpower (float): the normalization exponent (see equation above)
        name (str): layer name

    """

    def __init__(self, depth, alpha=1., beta=0., ascale=1., bpower=1., name=None):
        super(LRN, self).__init__(name=name)
        self.J = depth
        self.depth = depth  # needed for serialization
        self.alpha = alpha
        self.beta = beta
        self.ascale = ascale
        self.bpower = bpower
        self.owns_delta = True
        self.lrnparams = {'J': self.J}
        self.nglayer = None

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(LRN, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.lrnparams.update(shapedict)
            self.nglayer = self.be.lrn_layer(self.be.default_dtype, **self.lrnparams)
            self.out_shape = self.in_shape
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(LRN, self).allocate(shared_outputs)
        self.denom = self.be.iobuf(self.in_shape)

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.fprop_lrn(self.nglayer,
                          inputs, self.outputs, self.denom,
                          self.alpha, self.beta, self.ascale, self.bpower)
        return self.outputs

    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.bprop_lrn(self.nglayer,
                              self.inputs, self.outputs, error, self.deltas, self.denom,
                              self.alpha, self.beta, self.ascale, self.bpower)
        return self.deltas


class Dropout(Layer):

    """
    A dropout layer.

    Applies an element-wise multiplication of inputs with a keep mask.

    A keep mask is a tensor of ones and zeros of the same shape as the input.

    Each fprop call generates an new keep mask stochastically where there
    distribution of ones in the mask is controlled by the keep param.

    Arguments:
       keep (float): fraction of the inputs that should be stochastically kept.
    """

    def __init__(self, keep=0.5, name=None):
        super(Dropout, self).__init__(name)
        self.keep = keep
        self.keep_mask = None
        self.caffe_mode = self.be.check_caffe_compat()
        if self.caffe_mode:
            self._train_scaling = 1.0 / keep  # scaling factor during training
        else:
            self._train_scaling = 1.0  # override scaling factor to retain binary mask
        self.owns_output = False

    def __str__(self):
        return "Dropout Layer '%s': %d inputs and outputs, keep %d%% (caffe_compat %s)" % (
               self.name, self.nout, 100 * self.keep, self.caffe_mode)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Dropout, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(Dropout, self).allocate(shared_outputs)
        self.keep_mask = self.be.iobuf(self.out_shape, parallelism=self.parallelism)

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.outputs = self.inputs = inputs
        if inference:
            return self._fprop_inference(inputs)

        self.be.make_binary_mask(self.keep_mask, self.keep)
        self.outputs[:] = self.keep_mask * inputs * self._train_scaling

        return self.outputs

    def _fprop_inference(self, inputs):
        """
        Apply the forward pass transformation to the input data.

        May skip any computation not needed for doing inference only.

        Calling bprop subsequently is not valid.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        if not self.caffe_mode:
            self.outputs[:] = inputs * self.keep
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if not self.deltas:
            self.deltas = error
        self.deltas[:] = self.keep_mask * error * alpha * self._train_scaling + beta * error
        return self.deltas


class LookupTable(ParameterLayer):

    """
    A lookup table layer or a word embedding layer.

    The layer converts a word into a dense representation. When given a sentence,
    which is a vector of words (as integers), a matrix of vectors/embeddings for
    each word in the sentence is returned.

    LookupTable of dimensions embedding_dim by vocab_size is learnt.

    input shape - (nin, batch_size)

    output shape - (embedding_dim, nin * batch_size)

    weight shape - (embedding_dim, vocab_size)

    Arguments:
        vocab_size (int) : Number of words in the vocabulary
        embedding_dim (int) : Desired size of the word embedding
        init (Initializer): Initializer object to use for initializing layer weights
        name (str, optional): Layer name. Defaults to "LookupTableLayer"
    """

    def __init__(self, vocab_size, embedding_dim, init, update=True,
                 pad_idx=None, name=None):
        super(LookupTable, self).__init__(init, name)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.update = update
        self.pad_idx = pad_idx
        self.outputs_t = None

    def __str__(self):
        return "LookupTable Layer : %d inputs, (%d, %d) outputs size" % (
            self.nin, self.embedding_dim, self.nin)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(LookupTable, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.out_shape = (self.embedding_dim, self.nin)
        if self.weight_shape is None:
            self.weight_shape = (self.vocab_size, self.embedding_dim)
        return self

    def allocate(self, shared_outputs=None):
        super(LookupTable, self).allocate(shared_outputs=shared_outputs)
        if self.inputs is None:
            self.inputs = self.be.zeros((1, self.nin * self.be.bsz),
                                        dtype=np.int32)  # inputs is np.float32
        self.dW[:] = 0
        if self.pad_idx is not None:
            self.W[self.pad_idx] = 0
        if self.outputs_t is None:
            self.outputs_t = self.be.empty_like(self.outputs.T)

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.inputs[:] = inputs.reshape(self.inputs.shape)
        self.outputs_t[:] = self.W.take(self.inputs, axis=0)
        self.outputs[:] = self.outputs_t.T
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.update:
            self.dW[:] = 0
            self.be.compound_bprop_lut(self.nin, self.inputs, error, self.outputs_t,
                                       self.dW, self.pad_idx, alpha, beta)
        return self.deltas


class GeneralizedCost(NervanaObject):

    """
    A cost layer that applies the provided cost function and computes errors
    with respect to inputs and targets.

    Arguments:
       costfunc (Cost): class with costfunc that computes errors
    """

    def __init__(self, costfunc, name=None):
        super(GeneralizedCost, self).__init__(name)
        self.costfunc = costfunc
        self.outputs = None
        self.deltas = None
        self.cost_buffer = self.be.empty((1, 1))

    def initialize(self, in_obj):
        """
        Determine dimensions of cost and error buffers and allocate space from the input layer

        Arguments:
            in_obj (Layer): input layer from which to calculate costs
        """

        assert isinstance(in_obj, Layer)
        self.prev_layer = in_obj
        self.parallelism = self.prev_layer.parallelism
        self.costfunc.parallelism = self.parallelism
        (_, self.nstep) = interpret_in_shape(in_obj.out_shape)
        self.outputs = self.be.iobuf((1, self.nstep),
                                     parallelism=self.parallelism,
                                     persist_values=False)
        self.deltas = self.be.iobuf(in_obj.out_shape,
                                    parallelism=self.parallelism,
                                    persist_values=False)
        self.cost = np.empty([1, 1], dtype=np.float32)

    def get_cost(self, inputs, targets):
        """
        Compute the cost function over the inputs and targets.

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets (Tensor): Tensor containing target values.

        Returns:
            Tensor containing cost

        """
        self.outputs[:] = self.costfunc(inputs, targets)
        self.be.mean(self.outputs, axis=1, out=self.cost_buffer)
        self.cost = self.cost_buffer.get()
        self.be.clean_data(self.cost, True)
        return self.cost

    def get_errors(self, inputs, targets):
        """
        Compute the derivative of the cost function

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets (Tensor): Tensor containing target values.

        Returns:
            Tensor of same shape as the inputs containing their respective
            deltas.
        """
        self.deltas[:] = self.costfunc.bprop(inputs, targets)
        self.be.clean_data(self.deltas, True)
        return self.deltas


class GeneralizedGANCost(GeneralizedCost):

    """
    A cost layer that applies the provided cost function and computes errors
    with respect to inputs and targets. Supports different typer of cost for
    GAN models

    Arguments:
       costfunc (Cost): class with costfunc that computes errors
    """

    def __init__(self, costfunc, name=None):
        super(GeneralizedGANCost, self).__init__(costfunc, name)

    def get_cost(self, inputs, targets, cost_type=None):
        """
        Compute the cost function over the inputs and targets.

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets (Tensor): Tensor containing target values.
            cost_type (String, default None): Cost type for GAN models

        Returns:
            Tensor containing cost

        """
        self.outputs[:] = self.costfunc(inputs, targets, cost_type)
        self.be.mean(self.outputs, axis=1, out=self.cost_buffer)
        self.cost = self.cost_buffer.get()
        return self.cost


class GeneralizedCostMask(GeneralizedCost):

    """
    A cost layer that applies the provided cost function and computes errors
    with respect to inputs and targets. Applies mask to deltas.

    Arguments:
       costfunc (Cost): class with costfunc that computes errors
    """
    def __init__(self, costfunc, weights=1.0, name=None):
        super(GeneralizedCostMask, self).__init__(costfunc, name)
        self.weights = weights

    def get_cost(self, inputs, targets_mask):
        """
        Compute the cost function over the inputs and targets.

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets_mask ((Tensor, Tensor)): Tuple with Tensor target values and Tensor mask

        Returns:
            Tensor containing cost
        """
        targets, mask = targets_mask
        masked_input = inputs * mask
        masked_targets = targets * mask
        self.outputs[:] = self.costfunc(masked_input, masked_targets)
        self.cost_buffer[:] = self.be.mean(self.outputs, axis=1) * self.weights
        self.cost[:] = self.cost_buffer.get()
        return self.cost

    def get_errors(self, inputs, targets_mask):
        """
        Compute the derivative of the cost function

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                             targets
            targets_mask ((Tensor, Tensor)): Tuple with Tensor target values
                                             and Tensor mask

        Returns:
            Tensor of same shape as the inputs containing their respective
            deltas.
        """
        targets, mask = targets_mask
        self.deltas[:] = self.costfunc.bprop(inputs, targets) * mask * self.weights
        return self.deltas


class BatchNorm(Layer):

    """
    A batch normalization layer as described in [Ioffe2015]_.

    Normalizes a batch worth of inputs by subtracting batch mean and
    dividing by batch variance.  Then scales by learned factor gamma and
    shifts by learned bias beta.

    Uses the inputs to fprop to infer if a precomputed batch-sum is
    supplied from previous layer (input is tuple), or if the sum still
    needs to be computed.

    Notes:

    .. [Ioffe2015] http://arxiv.org/abs/1502.03167
    """

    def __init__(self, rho=0.9, eps=1e-3, name=None, binary=False):
        super(BatchNorm, self).__init__(name)
        self.allparams = None
        self.is_mklop = True
        self.x = None  # used to point to reshaped view of inputs
        self.xhat = None
        self.has_params = True
        self.owns_delta = True
        self.error_view = None
        self.rho = rho
        self.eps = eps
        self.states = [[] for i in range(2)]
        self.relu = False
        self.beta = None
        self.gamma = None
        self.gmean = None
        self.gvar = None
        self.stats_dtype = np.float64 if self.be.default_dtype is np.float64 else np.float32
        self.binary = binary

    def __str__(self):
        return "BatchNorm Layer '%s': %d inputs, %d steps, %d feature maps" % (
               self.name, self.nin, self.nsteps, self.nfm)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(BatchNorm, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.nfm = self.in_shape[0] if isinstance(self.in_shape, tuple) else self.nin
        self.nglayer = self.be.batchnorm_layer(self.in_shape)
        return self

    def allocate(self, shared_outputs=None, accumulate_updates=False):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
            accumulate_updates bool: allocate additional scratch accumulation
                                               buffers.
        """
        super(BatchNorm, self).allocate(shared_outputs)
        self.y = self.outputs.reshape((self.nfm, -1))
        self.xvar = self.be.zeros((self.nfm, 1), dtype=self.stats_dtype)
        self.accumulate_updates = accumulate_updates
        if self.allparams is None:
            self.init_params(self.nfm)
        if self.prev_layer in (None, True) or not getattr(self.prev_layer, 'batch_sum', None):
            self.xsum = self.be.zeros((self.nfm, 1),
                                      dtype=self.stats_dtype, **self.get_param_attrs())
            self.xsum.auto_reduce = False
            self.compute_batch_sum = True
        else:
            self.xsum = self.prev_layer.batch_sum
            self.compute_batch_sum = False

    def init_params(self, dim0):
        self.beta = self.be.zeros((dim0, 1), dtype=self.stats_dtype, **self.get_param_attrs())
        self.gamma = self.be.ones((dim0, 1), dtype=self.stats_dtype, **self.get_param_attrs())
        self.params = [self.beta, self.gamma]

        self.grad_params = [self.be.zeros_like(p) for p in self.params]
        self.inf_params = [self.be.zeros_like(p) for p in self.params]

        (self.grad_beta, self.grad_gamma) = self.grad_params
        (self.gmean, self.gvar) = self.inf_params

        self.allparams = self.params + self.inf_params

        # Scratch buffers used for accumulation
        if self.accumulate_updates:
            self.acc_grad_beta = self.be.empty_like(self.grad_beta)
            self.acc_grad_gamma = self.be.empty_like(self.grad_gamma)
            self.acc_params = [(self.acc_grad_beta, self.grad_beta),
                               (self.acc_grad_gamma, self.grad_gamma)]

    @property
    def plist(self):
        return [((p, g), s) for p, g, s in zip(self.params, self.grad_params, self.states)]

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Normalize inputs (x) over batch mean and variance.
        xhat = (x - xmean) / xvar

        Scale and shift normalized inputs (xhat) by learned parameters gamma and beta.
        y = xhat * gamma + beta

        Accumulate partial results to global mean and variance buffers used for inference.

        Arguments:
            inputs:
            inference:  (Default value = False)
            beta:  (Default value = 0.0)

        Returns:
            Tensor: output data
        """
        if self.inputs is None or self.inputs.base is not inputs:
            self.inputs = inputs.reshape((self.nfm, -1))

        self.be.compound_fprop_bn(
            self.inputs, self.xsum, self.xvar, self.gmean, self.gvar, self.gamma,
            self.beta, self.y, self.eps, self.rho, self.compute_batch_sum, beta, self.relu,
            binary=self.binary, inference=inference, outputs=self.outputs, layer=self.nglayer)

        return self.outputs

    def _fprop_inference(self, inputs, beta=0.0):
        """
        Apply one linear transformation that captures normalization, gamma scaling and beta shift.
        """
        xhat = (inputs - self.gmean) / self.be.sqrt(self.gvar + self.eps)  # Op-tree only
        self.y[:] = self.y * beta + xhat * self.gamma + self.beta
        return self.outputs

    @Layer.accumulates
    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Compute gradients for learning gamma and beta as well as layer weights.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:

        """
        assert alpha == 1.0 and beta == 0.0
        if not self.error_view:
            self.error_view = error.reshape((self.nfm, -1))

        self.be.compound_bprop_bn(self.deltas, self.grad_gamma, self.grad_beta,
                                  self.error_view,
                                  self.inputs, self.xsum, self.xvar, self.gamma,
                                  self.eps, binary=self.binary, layer=self.nglayer)
        return self.deltas

    def get_params(self):
        return self.plist

    def get_params_serialize(self, keep_states=True):
        return self.get_description(get_weights=True, keep_states=keep_states)

    def get_description(self, get_weights=False, keep_states=True):
        """
        Get layer parameters.

        Arguments:
            get_weights (bool, optional): Control whether all parameters are returned or
                                          just weights for serialization.
            keep_states (bool, optional): Controls whether the states should be returned
        """
        serial_dict = super(BatchNorm, self).get_description()
        if get_weights:
            serial_dict['params'] = {}
            for key in ['beta', 'gamma', 'gmean', 'gvar']:
                serial_dict['params'][key] = getattr(self, key).get()

            if keep_states:
                serial_dict['states'] = [[s.get() for s in slist] for slist in self.states]
        return serial_dict

    def set_params(self, pdict):
        if type(pdict['params']) is dict:
            for key, val in pdict['params'].items():
                if isinstance(getattr(self, key), Tensor):
                    getattr(self, key).set(val)
                else:
                    setattr(self, key, self.be.array(val, **self.get_param_attrs()))

            self.params = [self.beta, self.gamma]
            self.inf_params = [self.gmean, self.gvar]
            self.allparams = self.params + self.inf_params
        else:
            logger.error('Using old serialization format.  This will be'
                         ' deprecated in future release. Resave serialized file'
                         ' using current format')

            self.allparams = [self.be.array(x, **self.get_param_attrs()) for x in pdict['params']]
            self.params = self.allparams[:2]
            self.inf_params = self.allparams[2:]
            (self.beta, self.gamma) = self.params
            (self.gmean, self.gvar) = self.inf_params

        self.grad_params = [self.be.zeros_like(p) for p in self.params]
        (self.grad_beta, self.grad_gamma) = self.grad_params

    def set_states(self, pdict):
        if not any(self.states):
            self.states = [[self.be.array(x, **self.get_param_attrs()) for x in slist]
                           for slist in pdict['states']]
        else:
            for dlist, slist in zip(self.states, pdict['states']):
                for dst, src in zip(dlist, slist):
                    dst.set(src)


class BatchNormAutodiff(BatchNorm):

    """
    An example to use autodiff in batchnorm.
    """

    def __init__(self, rho=0.99, eps=1e-6, name=None):
        super(BatchNormAutodiff, self).__init__(rho, eps, name)

    def get_forward_optree(self):
        """
        Initialize the fprop optree for batchnorm.
        """
        # get fprop op-tree
        xvar = self.be.var(self.x, axis=1)
        xmean = self.be.mean(self.x, axis=1)
        xhat = (self.x - xmean) / self.be.sqrt(xvar + self.eps)
        return xhat * self.gamma + self.beta

    def fprop(self, inputs, inference=False):
        """
        Compute the actual fprop from op-tree, update the global estimations

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        if inference:
            return self._fprop_inference(inputs)
        self.init_buffers(inputs)
        if self.allparams is None:
            self.init_params(self.nfm)
            self.fprop_op_tree = self.get_forward_optree()

        # the actual f-prop
        self.y[:] = self.fprop_op_tree

        # for inference
        self.gmean[:] = (self.gmean * self.rho + (1.0 - self.rho) * self.be.mean(self.x, axis=1))
        self.gvar[:] = (self.gvar * self.rho + (1.0 - self.rho) * self.be.var(self.x, axis=1))

        return self.outputs

    def bprop(self, error):
        """
        Use Autodiff.back_prop_grad to back propagate gradients for the
        corresponding tensors.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if not self.deltas:
            self.deltas = error.reshape((self.nfm, -1))

        # autodiff will automatically cache and reuse the object
        # if we know the `error` buffer at init, we can also create the autodiff
        # object at layer's init
        ad = Autodiff(self.fprop_op_tree, self.be, next_error=self.deltas)

        # back propagate
        ad.back_prop_grad([self.x, self.gamma, self.beta],
                          [self.deltas, self.grad_gamma, self.grad_beta])

        return error


class ShiftBatchNorm(BatchNorm):

    """
    Shift based batch normalization.

    Reference:
        http://arxiv.org/pdf/1602.02830v3.pdf
    """
    def __init__(self, rho=0.99, eps=1e-6, name=None):
        super(ShiftBatchNorm, self).__init__(rho, eps, name, binary=True)

    def _fprop_inference(self, inputs, beta=0.0):
        """
        Apply one linear transformation that captures normalization, gamma scaling and beta shift.
        """
        input_ms = self.be.empty_like(inputs)
        input_ms[:] = inputs - self.gmean
        inv_v = self.be.empty_like(self.gvar)
        inv_v[:] = 1.0 / self.be.sqrt(self.gvar + self.eps)
        xhat = self.be.shift(input_ms, inv_v)
        self.y[:] = self.y * beta + self.be.shift(xhat, self.gamma) + self.beta
        return self.outputs


class RoiPooling(Layer):
    """
    RoiPooling uses max pooling to convert the features inside any ROI into a small
    feature map with a fixed spatial extend of H x W, where H and W are layer
    parameters independent of any particular ROI.
    Each ROI is defined as a 4-tuple as (xmin, ymin, xmax, ymax)

    ROIPooling is applied independently to each feature map channel, as in standard
    max pooling.

    ROIPooling takes as input a tuple (img_fm, rois) where:
    (1) img_fm: output from the convolutional layers (e.g. for VGG-16, 62x62)
    (2) rois: proposed ROIs, in the form (rois_per_img, 5). The first index is the
        image_id within the minibatch. Since faster-rcnn uses batch size 1, this is always 0.

    The output shape (out_shape) is a tuple - (batch_size, rois_per_img), then
    the following layers will allocate buffers accordingly.
    """

    def __init__(self, HW=(7, 7), bprop_enabled=True, spatial_scale=0.0625, name=None):
        super(RoiPooling, self).__init__(name)

        self.HW = HW
        self.roi_H, self.roi_W = self.HW
        self.spatial_scale = spatial_scale  # 0.0625 is 1/16

        # it has its own output buffer besides being a container
        self.owns_output = True
        self.owns_delta = True

        self.img = None
        self.rois = None
        self.rois_per_img = None
        self.fm_channel = None
        self.fm_height = None
        self.fm_width = None

        self.bprop_enabled = bprop_enabled

    def configure(self, in_obj):
        """
        Must receive a list of shapes for configurations
        Need both the layer container and roi dataset to configure shapes
        'in_obj' will include be [image_shape, roi_shape] (e.g [(3, 600, 1000), 5])

        Arguments:
            in_obj:

        Returns:

        """
        # configure to get the shape of feature map
        assert len(in_obj) == 2, "Input to ROIpooling must be a 2-tuple"
        self.prev_layer = in_obj
        img_fm, rois = in_obj

        # configure number of rois
        assert rois.out_shape[0] == 5, "Input ROIs must be a 5-tuple"
        self.rois_per_img = rois.out_shape[1]

        # configure input image feature map shapes
        self.in_shape = img_fm.out_shape
        (self.fm_channel, self.fm_height, self.fm_width) = self.in_shape
        self.error_in_reshape = (self.fm_channel, -1)
        self.fm_reshape_shape = (
            self.fm_channel, self.fm_height * self.fm_width, self.be.bsz)

        # make the out_shape as a tuple, as if the roi_per_image a
        # time_step dimension
        self.out_shape = (self.fm_channel * self.roi_H * self.roi_W, self.rois_per_img)
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(RoiPooling, self).allocate(shared_outputs)
        self.owns_output = True
        self.error = self.be.iobuf(self.in_shape)
        self.outputs = self.be.iobuf(self.out_shape, shared=shared_outputs)
        self.max_idx = self.be.iobuf(self.out_shape, dtype=np.int32)

    def init_buffers(self, inputs):
        """
        Initialize buffers for images and ROIs

        Arguments:
            inputs:

        Returns:

        """
        assert len(inputs) == 2, "inputs must contain both images and ROIs"
        self.img = inputs[0]
        self.rois = inputs[1].transpose()
        assert self.rois.shape[1] == 5, "ROI entry must be 5-value tuple"

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        self.init_buffers(inputs)

        self.outputs.fill(0)
        self.max_idx.fill(0)

        # fprop through the roipooling layer
        self.be.roipooling_fprop(self.img, self.rois, self.outputs, self.max_idx,
                                 self.rois_per_img, self.fm_channel, self.fm_height,
                                 self.fm_width, self.roi_H, self.roi_W, self.spatial_scale)

        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """

        self.error.fill(0)

        if self.bprop_enabled:
            # # bprop through the roipooling layer
            self.be.roipooling_bprop(error, self.rois, self.error, self.max_idx,
                                     self.rois_per_img, self.fm_channel, self.fm_height,
                                     self.fm_width, self.roi_H, self.roi_W, self.spatial_scale)

        # bprop back through the imagenet layer container
        return self.error
