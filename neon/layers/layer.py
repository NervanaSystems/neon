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
import logging

from neon import NervanaObject
from neon.backends import Autodiff
from neon.backends.backend import Tensor
import numpy as np

logger = logging.getLogger(__name__)


def interpret_in_shape(xshape):
    """
    Helper function to interpret the tensor layout of preceding layer to handle non-recurrent,
    recurrent, and local layers
    """
    if isinstance(xshape, int):
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

    def __init__(self, name="layer", parallelism="Unknown"):
        super(Layer, self).__init__(name)
        self.outputs = None
        self.has_params = False
        self.inputs = None
        self.owns_output = True
        self.owns_delta = False
        self.deltas = None
        self.parallelism = parallelism
        self.next_layer = None

    def __str__(self):
        """
        Format the layer as a printable string.
        """
        ret = '{} {}'.format(self.__class__.__name__, self.name)
        return ret

    def nested_str(self, level=0):
        """
        Utility function for displaying layer info with a given indentation level

        Arguments:
            level (int, optional): indentation level
        """

        return "  " * level + str(self)

    def configure(self, in_obj):
        """
        sets shape based parameters of this layer given an input tuple or int
        or input layer

        Arguments:
            in_obj (int, tuple, Layer or Tensor or dataset): object that provides shape
                                                             information for layer

        Returns:
            (tuple): shape of output data
        """
        if isinstance(in_obj, Layer):
            self.prev_layer = in_obj
            self.in_shape = in_obj.out_shape
            if self.parallelism == "Unknown":
                self.parallelism = in_obj.parallelism
        else:
            self.prev_layer = None
            if isinstance(in_obj, tuple) or isinstance(in_obj, int):
                self.in_shape = in_obj  # input is a shape tuple or int directly
            elif isinstance(in_obj, Tensor):
                self.in_shape = (in_obj.shape[0], in_obj.shape[1] / self.be.bsz)
            else:
                self.in_shape = in_obj.shape  # This is a dataset

    def allocate(self, shared_outputs=None):
        """
        Allocates output buffer to store activations from fprop.
        Don't reallocate if it already exists.
        Only allocate space if layer owns its own output (i.e. bias, activation work in-place,
        so do not own their output).
        outputs can be allocated from a pre-allocated pool if shared_outputs is provided

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into

        """
        if self.outputs:
            return
        if self.owns_output:
            self.outputs = self.be.iobuf(self.out_shape, shared=shared_outputs,
                                         parallelism=self.parallelism)

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (i.e. bias, activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)

        """
        if self.next_layer is not None and self.next_layer.parallelism != self.parallelism:
            self.owns_delta = True

        if self.owns_delta and self.prev_layer:
            if type(self.prev_layer) is BranchNode:
                self.deltas = self.prev_layer.deltas
            else:
                self.deltas = self.be.iobuf(self.in_shape, shared=delta_buffers[0],
                                            parallelism=self.parallelism)
                delta_buffers.reverse()
        else:
            self.deltas = None

    @property
    def classnm(self):
        """
        Convenience method for getting the class name
        """
        return self.__class__.__name__

    def set_next(self, layer):
        self.next_layer = layer

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
        Used for recursively getting final nodes from layer containers
        """
        return self

    def serialize(self):
        """
        Get state parameters for this layer

        Returns:
            ?: whatever data this model wants to receive in order to restore state
        """
        if self.has_params:
            return self.get_params()


class BranchNode(Layer):

    """
    Layer that allows branching.  Used to send outputs to multiple layer pathways.
    Each pathway will get the entire output of the layer preceding the branch node.
    """

    def __init__(self, name='branch'):
        super(BranchNode, self).__init__(name)
        self.owns_output = False

    def fprop(self, inputs=None, inference=False):

        """
        Passes output from preceding layer on without modification
        """
        if self.outputs is None and inputs is not None:
            self.outputs = inputs
        return self.outputs

    def configure(self, in_obj):
        """
        sets shape based parameters of this layer given an input tuple or int
        or input layer

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
        if self.deltas is None:
            self.deltas = self.be.iobuf(self.in_shape, shared=delta_buffers[0])
            delta_buffers.reverse()

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Branch nodes should be skipped in bprop, since their deltas are shared
        """
        pass


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
                 name="PoolingLayer"):
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
        if isinstance(fshape, int):
            fshape = {'R': fshape, 'S': fshape}
        elif isinstance(fshape, tuple):
            fkeys = ('R', 'S') if len(fshape) == 2 else ('T', 'R', 'S')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
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
        super(Pooling, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.poolparams.update(shapedict)
            self.nglayer = self.be.pool_layer(self.be.default_dtype, **self.poolparams)
            (K, M, P, Q, N) = self.nglayer.dimO
            self.out_shape = (K, M, P, Q) if len(self.in_shape) == 4 else (K, P, Q)
        return self

    def set_deltas(self, delta_buffers):
        super(Pooling, self).set_deltas(delta_buffers)
        if self.op == "max":
            self.argmax = self.be.empty(self.outputs.shape, dtype=np.uint8)
        else:
            self.argmax = None

    def fprop(self, inputs, inference=False):
        self.inputs = inputs
        self.be.fprop_pool(self.nglayer, inputs, self.outputs, self.argmax)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
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

    def __init__(self, init=None, name="ParameterLayer",
                 parallelism="Unknown"):
        super(ParameterLayer, self).__init__(name, parallelism)
        self.has_params = True
        self.init = init
        self.W = None
        self.weight_shape = None
        self.batch_sum = None
        self.batch_sum_shape = None
        self.states = []
        self.owns_delta = True

    def allocate(self, shared_outputs=None):
        super(ParameterLayer, self).allocate(shared_outputs)
        if self.W is None:
            self.init_params(self.weight_shape)
        if self.batch_sum_shape is not None:
            parallel, distributed = self.get_param_attrs()
            self.batch_sum = self.be.empty(self.batch_sum_shape, dtype=np.float32,
                                           parallel=parallel, distributed=distributed)

    def init_params(self, shape):
        """
        Allocate layer parameter buffers and initialize them with the
            supplied initializer.

        Arguments:
            shape (int, tuple): shape to allocate for layer parameter
                buffers.
        """
        parallel, distributed = self.get_param_attrs()
        self.W = self.be.empty(shape, parallel=parallel, distributed=distributed)
        self.dW = self.be.empty_like(self.W)
        if isinstance(self.init, Tensor) or isinstance(self.init, np.ndarray):
            self.W[:] = self.init
        else:
            self.init.fill(self.W)

    def get_param_attrs(self):
        if self.parallelism == "Data":
            parallel = True
            distributed = False
        elif self.parallelism == "Model":
            parallel = True
            distributed = True
        else:
            parallel = False
            distributed = False

        return (parallel, distributed)

    def get_params(self):
        """
        Get layer parameters, gradients, and states for optimization
        """
        return ((self.W, self.dW), self.states)

    def get_params_serialize(self, keep_states=True):
        """
        Get layer parameters. All parameters are needed for optimization, but
        only Weights are serialized.

        Arguments:
            keep_states (bool): Control whether all parameters are returned
                or just weights for serialization. Defaults to True.
        """
        serial_dict = {'params': {'W': self.W.asnumpyarray(),
                                  'name': self.name}}
        if keep_states:
            serial_dict['states'] = [s.asnumpyarray() for s in self.states]
        return serial_dict

    def set_params(self, pdict):
        """
        Set layer parameters (weights). Allocate space for other parameters but do not initialize
        them.

        Arguments:
            pdict (dict, ndarray): dictionary or ndarray with layer parameters
                                   [support for ndarray is DEPRECATED and will be removed]
        """
        if type(pdict) is dict:
            for key in pdict:
                setattr(self, key, pdict[key])
        else:
            # for backward compatibility will be deprecated
            logger.warn('Using old serialization file type, will be deprecated.'
                        '  Save model into new format')
            self.W = pdict
        self.W = self.be.array(self.W)
        self.dW = self.be.empty_like(self.W)

    def set_states(self, states):
        self.states = [self.be.array(x) for x in states]


class Convolution(ParameterLayer):

    """
    Convolutional layer implementation.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "ConvolutionLayer"
    """

    def __init__(self, fshape, strides={}, padding={}, init=None, bsum=False,
                 name="ConvolutionLayer", parallelism="Data",
                 deterministic_update=False):
        super(Convolution, self).__init__(init, name, parallelism)
        self.nglayer = None
        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'T': 1, 'D': 1, 'bsum': bsum,
                           'deterministic_update': deterministic_update}  # 3D paramaters

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding

        if isinstance(fshape, tuple):
            fkeys = ('R', 'S', 'K') if len(fshape) == 3 else ('T', 'R', 'S', 'K')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.convparams.update(d)

    def __str__(self):
        return ("Convolution Layer '%s': %d x (%dx%d) inputs, %d x (%dx%d) "
                "outputs, padding %d, stride %d" %
                (self.name,
                 self.in_shape[0], self.in_shape[1], self.in_shape[2],
                 self.out_shape[0], self.out_shape[1], self.out_shape[2],
                 self.convparams['pad_h'], self.convparams['str_h']))

    def configure(self, in_obj):
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
        if self.convparams['bsum']:
            self.batch_sum_shape = (self.nglayer.K, 1)
        return self

    def fprop(self, inputs, inference=False):
        self.inputs = inputs
        self.be.fprop_conv(self.nglayer, inputs, self.W, self.outputs, bsum=self.batch_sum)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        if self.deltas:
            self.be.bprop_conv(self.nglayer, self.W, error, self.deltas, alpha=alpha, beta=beta)
        self.be.update_conv(self.nglayer, self.inputs, error, self.dW)
        return self.deltas


class Deconvolution(ParameterLayer):

    """
    Deconvolutional layer implementation.

    Arguments:
        fshape (tuple): three dimensional shape of convolution window
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "DeconvolutionLayer"
    """

    def __init__(self, fshape, strides={}, padding={}, init=None, bsum=False,
                 name="DeconvolutionLayer"):
        super(Deconvolution, self).__init__(init, name)
        self.nglayer = None
        self.deconvparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                             'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                             'bsum': bsum}

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding

        if isinstance(fshape, tuple):
            # fshape[2] should now map to C (nifm)
            fshape = {'R': fshape[0], 'S': fshape[1], 'C': fshape[2]}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.deconvparams.update(d)

    def __str__(self):
        return "Deconvolution Layer '%s': %d x (%dx%d) inputs, %d x (%dx%d) outputs" % (
               self.name,
               self.in_shape[0], self.in_shape[1], self.in_shape[2],
               self.out_shape[0], self.out_shape[1], self.out_shape[2])

    def configure(self, in_obj):
        super(Deconvolution, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            shapedict = {'K': self.in_shape[0],
                         'P': self.in_shape[1],
                         'Q': self.in_shape[2],
                         'N': self.be.bsz}
            self.deconvparams.update(shapedict)
            self.nglayer = self.be.deconv_layer(self.be.default_dtype, **self.deconvparams)
            self.out_shape = (self.nglayer.C, self.nglayer.H, self.nglayer.W)
        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)
        if self.deconvparams['bsum']:
            self.batch_sum_shape = (self.nglayer.C, 1)
        return self

    def fprop(self, inputs, inference=False):
        """
        fprop for deconv is equivalent to bprop for conv.
        bprop_conv takes in error and deltas as "E" and "grad_I"
        for deconv, bprop_conv will take in input as "E" and output as "grad_I"
        """
        self.inputs = inputs
        self.be.bprop_conv(layer=self.nglayer, F=self.W, E=inputs, grad_I=self.outputs,
                           bsum=self.batch_sum)
        return self.outputs

    def bprop(self, error, beta=0.0):
        """
        bprop for deconv is equivalent to fprop for conv.
        fprop_conv takes input and output as "I" and "O".
        for deconv, fprop_conv will take error as input and delta as output
        """
        assert beta == 0., "beta parameter not supported for deconvolution yet"
        if self.deltas:
            self.be.fprop_conv(self.nglayer, error, self.W, self.deltas)
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

    def __init__(self, nout, init, bsum=False, name="LinearLayer"):
        super(Linear, self).__init__(init, name, "Disabled")
        self.nout = nout
        self.inputs = None
        self.bsum = bsum

    def __str__(self):
        return "Linear Layer '%s': %d inputs, %d outputs" % (
               self.name, self.nin, self.nout)

    def configure(self, in_obj):
        super(Linear, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.out_shape = (self.nout, self.nsteps)
        if self.weight_shape is None:
            self.weight_shape = (self.nout, self.nin)
        if self.bsum:
            self.batch_sum_shape = (self.nout, 1)
        return self

    def fprop(self, inputs, inference=False):
        self.inputs = inputs
        self.be.compound_dot(A=self.W, B=self.inputs, C=self.outputs, bsum=self.batch_sum)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        if self.deltas:
            self.be.compound_dot(A=self.W.T, B=error, C=self.deltas, alpha=alpha, beta=beta)
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

    def __init__(self, init, name="BiasLayer"):
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
        super(Bias, self).configure(in_obj)
        self.out_shape = self.in_shape
        self.bias_size = self.in_shape[0]
        if self.weight_shape is None:
            self.weight_shape = (self.bias_size, 1)
        return self

    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        if self.y is None or self.y.base is not self.outputs:
            self.y = self.outputs.reshape((self.bias_size, -1))
        self.y[:] = self.y + self.W
        return self.outputs

    def bprop(self, error):
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

    def __init__(self, transform, name="ActivationLayer"):
        super(Activation, self).__init__(name)
        self.transform = transform
        self.owns_output = False

    def __str__(self):
        return "Activation Layer '%s': %s" % (
               self.name, self.transform.__class__.__name__)

    def configure(self, in_obj):
        super(Activation, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        self.outputs[:] = self.transform(self.inputs)
        return self.outputs

    def bprop(self, error):
        if not self.deltas:
            self.deltas = error
        error[:] = self.transform.bprop(self.outputs) * error
        return error


class DataTransform(Layer):

    """
    A layer that applies a specified transform to input data in fprop only.

    Only supported as the first layer in the network.

    Arguments:
        transform (Transform): a transform object with fprop function to apply
        name (str, optional): Layer name. Defaults to "DataTransformLayer"
    """

    def __init__(self, transform, name="DataTransformLayer"):
        super(DataTransform, self).__init__(name)
        self.transform = transform
        self.owns_output = False

    def __str__(self):
        return "DataTransform Layer '%s': %s" % (
               self.name, self.transform.__class__.__name__)

    def configure(self, in_obj):
        super(DataTransform, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        self.outputs[:] = self.transform(self.inputs)
        return self.outputs

    def bprop(self, *args):
        return None


class Affine(list):

    """
    A linear layer with a learned bias and activation, implemented as a list
    composing separate linear, bias/batchnorm and activation layers.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights and bias
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        linear_name (str): the name to call the Linear layer. Defaults to 'LinearLayer'.
        bias_name (str): the name to call the Bias layer. Defautls to 'BiasLayer'.
        act_name (str): the name to call the Activation layer. Defaults to 'ActivationLayer'.

    """

    def __init__(self, nout, init, bias=None, batch_norm=False, activation=None,
                 linear_name='LinearLayer', bias_name='BiasLayer',
                 act_name='ActivationLayer'):
        list.__init__(self)
        self.append(Linear(nout, init, bsum=batch_norm, name=linear_name))
        self.add_postfilter_layers(bias, batch_norm, activation, bias_name, act_name)

    def add_postfilter_layers(self, bias=None, batch_norm=False, activation=None,
                              bias_name='BiasLayer', act_name='ActivationLayer'):
        if batch_norm and (bias is not None):
            raise AttributeError('Batchnorm and bias cannot be combined')
        if bias is not None:
            self.append(Bias(init=bias, name=bias_name))
        if batch_norm:
            self.append(BatchNorm())
        if activation is not None:
            self.append(Activation(transform=activation, name=act_name))


class Conv(Affine):

    """
    A convolutional layer with a learned bias and activation, implemented as a
    list composing separate Convolution, Bias and Activation layers.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window
        init (Initializer, optional): Initializer object to use for
            initializing layer weights and bias
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        pad (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        conv_name (str): the name to call the Convolutional layer. Defaults to 'ConvolutionLayer'
        bias_name (str): the name to call the Bias layer. Defaults to 'BiasLayer'
        act_name (str): the name to call the Activation layer. Defaults to ActivationLayer.

    """

    def __init__(self, fshape, init, strides={}, padding={}, bias=None, batch_norm=False,
                 activation=None, conv_name='ConvolutionLayer',
                 bias_name='BiasLayer', act_name='ActivationLayer',
                 deterministic_update=False):
        list.__init__(self)
        self.append(Convolution(fshape=fshape, strides=strides, padding=padding,
                                init=init, bsum=batch_norm, name=conv_name,
                                deterministic_update=deterministic_update))
        self.add_postfilter_layers(bias, batch_norm, activation, bias_name, act_name)


class Deconv(Affine):

    """
    Same as Conv layer, but implements a composite deconvolution layer
    """

    def __init__(self, fshape, init, strides={}, padding={}, bias=None, batch_norm=False,
                 activation=None, conv_name='DeconvolutionLayer',
                 bias_name='BiasLayer', act_name='ActivationLayer'):
        list.__init__(self)
        self.append(Deconvolution(fshape=fshape, strides=strides, padding=padding,
                                  init=init, bsum=batch_norm))
        self.add_postfilter_layers(bias, batch_norm, activation, bias_name, act_name)


class LRN(Layer):
    def __init__(self, depth, alpha=1., beta=0., ascale=1., bpower=1.):
        super(LRN, self).__init__(name="LRNLayer")
        self.J = depth
        self.alpha = alpha
        self.beta = beta
        self.ascale = ascale
        self.bpower = bpower
        self.owns_delta = True
        self.lrnparams = {'J': self.J}
        self.nglayer = None

    def configure(self, in_obj):
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
        super(LRN, self).allocate(shared_outputs)
        self.denom = self.be.iobuf(self.in_shape)

    def fprop(self, inputs, inference=False):
        self.inputs = inputs
        self.be.fprop_lrn(self.nglayer,
                          inputs, self.outputs, self.denom,
                          self.alpha, self.beta, self.ascale, self.bpower)
        return self.outputs

    def bprop(self, error):
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

    def __init__(self, keep=0.5, name="droplayer"):
        super(Dropout, self).__init__(name)
        self.keep = keep
        self.keep_mask = None
        self.caffe_mode = self.be.check_caffe_compat()
        if self.caffe_mode:
            self._train_scaling = 1.0/keep  # scaling factor during training
        else:
            self._train_scaling = 1.0  # override scaling factor to retain binary mask
        self.owns_output = False

    def __str__(self):
        return "Dropout Layer '%s': %d inputs and outputs, keep %d%% (caffe_compat %s)" % (
               self.name, self.nout, 100*self.keep, self.caffe_mode)

    def configure(self, in_obj):
        super(Dropout, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def allocate(self, shared_outputs=None):
        super(Dropout, self).allocate(shared_outputs)
        self.keep_mask = self.be.iobuf(self.out_shape, parallelism=self.parallelism)

    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        if inference:
            return self._fprop_inference(inputs)

        self.be.make_binary_mask(self.keep_mask, self.keep)
        self.outputs[:] = self.keep_mask * inputs * self._train_scaling

        return self.outputs

    def _fprop_inference(self, inputs):
        if not self.caffe_mode:
            self.outputs[:] = inputs * self.keep
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        if not self.deltas:
            self.deltas = error
        self.deltas[:] = self.keep_mask * error * alpha * self._train_scaling + beta * error
        return self.deltas


class DropoutBinary(Dropout):
    """
    A dropout layer that does no scaling by keep ratio during training

    Arguments:
       keep (float): fraction of the inputs that should be stochastically kept.

    NOTE: this class will be deprecated, the Dropout class executes this behavior
    by default now
    """

    def __init__(self, keep=0.5, name="dropbinarylayer"):
        super(DropoutBinary, self).__init__(keep, name)
        self._train_scaling = 1.0  # override scaling factor to retain binary mask
        logger.warning('DropoutBinary class will be deprecated, please use '
                       'Dropout layer instead')

    def __str__(self):
        return "Dropout Binary Layer '%s': %d inputs and outputs, keep %d%%" % (
               self.name, self.nout, 100 * self.keep)

    def _fprop_inference(self, inputs):
        self.outputs[:] = inputs * self.keep
        return self.outputs


class LookupTable(ParameterLayer):

    """
    A lookup table layer or a word embedding layer

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
                 pad_idx=None, name="LookupTableLayer"):
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
        super(LookupTable, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.out_shape = (self.embedding_dim, self.nin)
        if self.weight_shape is None:
            self.weight_shape = (self.vocab_size, self.embedding_dim)
        return self

    def allocate(self):
        super(LookupTable, self).allocate()
        if self.inputs is None:
            self.inputs = self.be.zeros(
                (1, self.nin * self.be.bsz), dtype=np.int32)  # inputs is np.float32
        self.dW[:] = 0
        if self.pad_idx is not None:
            self.W[:, self.pad_idx] = 0
        if self.outputs_t is None:
            self.outputs_t = self.be.empty_like(self.outputs.T)

    def fprop(self, inputs, inference=False):
        self.inputs[:] = inputs.reshape(self.inputs.shape)
        self.outputs_t[:] = self.W.take(self.inputs, axis=0)
        self.outputs[:] = self.outputs_t.T
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0):
        self.dW[:] = 0
        self.be.compound_bprop_lut(self.nin, self.inputs, error, self.outputs_t, self.dW,
                                   self.pad_idx, alpha, beta)

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

    def initialize(self, in_obj):
        """
        Determine dimensions of cost and error buffers and allocate space from the input layer

        Arguments:
            in_obj (Layer): input layer from which to calculate costs
        """

        assert isinstance(in_obj, Layer)
        self.prev_layer = in_obj
        (_, self.nstep) = interpret_in_shape(in_obj.out_shape)
        self.outputs = self.be.iobuf((1, self.nstep))
        self.deltas = self.be.iobuf(in_obj.out_shape, parallelism="Disabled")
        self.cost = self.be.empty((1, 1))

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
        self.cost[:] = self.be.mean(self.outputs, axis=1)
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
        return self.deltas


class GeneralizedCostMask(GeneralizedCost):

    """
    A cost layer that applies the provided cost function and computes errors
    with respect to inputs and targets. Applies mask to deltas.

    Arguments:
       costfunc (Cost): class with costfunc that computes errors
    """

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
        self.outputs[:] = self.costfunc(masked_input, targets)
        self.cost[:] = self.be.mean(self.outputs, axis=1)
        return self.cost

    def get_errors(self, inputs, targets_mask):
        """
        Compute the derivative of the cost function

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets_mask ((Tensor, Tensor)): Tuple with Tensor target values and Tensor mask

        Returns:
            Tensor of same shape as the inputs containing their respective
            deltas.
        """
        targets, mask = targets_mask
        self.deltas[:] = self.costfunc.bprop(inputs, targets) * mask
        return self.deltas


class BatchNorm(Layer):

    """
    A batch normalization layer as described in [Ioffe2015]_

    Normalizes a batch worth of inputs by subtracting batch mean and
    dividing by batch variance.  Then scales by learned factor gamma and
    shifts by learned bias beta.

    Uses the inputs to fprop to infer if a precomputed batch-sum is
    supplied from previous layer (input is tuple), or if the sum still
    needs to be computed.

    Notes:

    .. [Ioffe2015] http://arxiv.org/abs/1502.03167
    """

    def __init__(self, rho=0.99, eps=1e-6, name="BatchNormLayer"):
        super(BatchNorm, self).__init__(name)
        self.allparams = None
        self.x = None  # used to point to reshaped view of inputs
        self.xhat = None
        self.has_params = True
        self.rho = rho
        self.eps = eps
        self.states = [[] for i in range(2)]
        self.relu = False

    def __str__(self):
        return "BatchNorm Layer '%s': %d inputs, %d steps, %d feature maps" % (
               self.name, self.nin, self.nsteps, self.nfm)

    def configure(self, in_obj):
        super(BatchNorm, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.nfm = self.in_shape[0] if isinstance(self.in_shape, tuple) else self.nin
        return self

    def allocate(self, shared_outputs=None):
        super(BatchNorm, self).allocate(shared_outputs)
        self.y = self.outputs.reshape((self.nfm, -1))
        self.xvar = self.be.zeros((self.nfm, 1))
        if self.allparams is None:
            self.init_params(self.nfm)
        if self.prev_layer is None or self.prev_layer.batch_sum is None:
            self.xsum = self.be.zeros((self.nfm, 1))
            self.compute_batch_sum = True
        else:
            self.xsum = self.prev_layer.batch_sum
            self.compute_batch_sum = False

    def init_params(self, dim0):
        if self.parallelism == "Data":
            parallel = True
            distributed = False
        elif self.parallelism == "Model":
            parallel = True
            distributed = True
        else:
            parallel = False
            distributed = False

        self.beta = self.be.zeros((dim0, 1), parallel=parallel, distributed=distributed)
        self.gamma = self.be.ones((dim0, 1), parallel=parallel, distributed=distributed)
        self.params = [self.beta, self.gamma]

        self.grad_params = [self.be.zeros_like(p) for p in self.params]
        self.inf_params = [self.be.zeros_like(p) for p in self.params]

        (self.grad_beta, self.grad_gamma) = self.grad_params
        (self.gmean, self.gvar) = self.inf_params

        self.allparams = self.params + self.inf_params
        self.plist = [((p, g), s) for p, g, s in zip(self.params, self.grad_params, self.states)]

    def fprop(self, inputs, inference=False):
        """
        Normalize inputs (x) over batch mean and variance.
        xhat = (x - xmean) / xvar

        Scale and shift normalized inputs (xhat) by learned parameters gamma and beta.
        y = xhat * gamma + beta

        Accumulate partial results to global mean and variance buffers used for inference.
        """
        if self.inputs is None or self.inputs.base is not inputs:
            self.inputs = inputs.reshape((self.nfm, -1))

        if inference:
            return self._fprop_inference(self.inputs)

        if self.compute_batch_sum:
            self.xsum[:] = self.be.sum(self.inputs, axis=1)

        self.be.compound_fprop_bn(
            self.inputs, self.xsum, self.xvar, self.gmean, self.gvar,
            self.gamma, self.beta, self.outputs, self.eps, self.rho, self.relu)

        return self.outputs

    def _fprop_inference(self, inputs):
        """
        Apply one linear transformation that captures normalization, gamma scaling and beta shift.
        """
        xhat = (inputs - self.gmean) / self.be.sqrt(self.gvar + self.eps)  # Op-tree only
        self.y[:] = xhat * self.gamma + self.beta
        return self.outputs

    def bprop(self, error):
        """
        Compute gradients for learning gamma and beta as well as layer weights.
        """
        if not self.deltas:
            self.deltas = error.reshape((self.nfm, -1))

        self.be.compound_bprop_bn(
            self.deltas, self.grad_gamma, self.grad_beta, self.inputs,
            self.xsum, self.xvar, self.gamma, self.eps)

        return error

    def get_params(self):
        return self.plist

    def get_params_serialize(self, keep_states=True):
        serial_dict = {'params': [p.asnumpyarray() for p in self.allparams]}
        if keep_states:
            serial_dict['states'] = [[s.asnumpyarray() for s in slist] for slist in self.states]
        return serial_dict

    def set_params(self, allparams):
        self.allparams = [self.be.array(x) for x in allparams]
        self.params = self.allparams[:2]
        self.inf_params = self.allparams[2:]
        self.grad_params = [self.be.zeros_like(p) for p in self.params]

        (self.beta, self.gamma) = self.params
        (self.grad_beta, self.grad_gamma) = self.grad_params
        (self.gmean, self.gvar) = self.inf_params
        self.plist = [((p, g), s) for p, g, s in zip(self.params, self.grad_params, self.states)]

    def set_states(self, states):
        self.states = [[self.be.array(x) for x in slist] for slist in states]


class BatchNormAutodiff(BatchNorm):

    """
    An example to use autodiff in batchnorm.
    """

    def __init__(self, rho=0.99, eps=1e-6, name="BatchNormAutodiffLayer"):
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
