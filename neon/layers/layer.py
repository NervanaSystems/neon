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
    """
    def __init__(self, name="layer"):
        super(Layer, self).__init__(name)
        self.outputs = None
        self.deltas = None
        self.has_params = False
        self.inputs = None

    def __str__(self):
        """
        Format the layer as a printable string.
        """
        ret = '{} {}'.format(self.__class__.__name__, self.name)
        return ret

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
        else:
            self.prev_layer = None
            if isinstance(in_obj, tuple) or isinstance(in_obj, int):
                self.in_shape = in_obj  # input is a shape tuple or int directly
            elif isinstance(in_obj, Tensor):
                self.in_shape = (in_obj.shape[0], in_obj.shape[1] / self.be.bsz)

            else:
                self.in_shape = in_obj.shape  # This is a dataset

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

    def serialize(self):
        """
        Get state parameters for this layer

        Returns:
            ?: whatever data this model wants to receive in order to restore state
        """
        if self.has_params:
            return self.get_params()


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

        if isinstance(fshape, int):
            fshape = {'R': fshape, 'S': fshape}
        elif isinstance(fshape, tuple):
            fshape = {'R': fshape[0], 'S': fshape[1]}
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
            shapedict = {'C': self.in_shape[0],
                         'H': self.in_shape[1],
                         'W': self.in_shape[2],
                         'N': self.be.bsz}
            self.poolparams.update(shapedict)
            self.nglayer = self.be.pool_layer(self.be.default_dtype, **self.poolparams)
            self.out_shape = (self.nglayer.K, self.nglayer.P, self.nglayer.Q)
        return self

    def allocate(self, shared_outputs=None, shared_deltas=None):
        self.outputs = self.be.iobuf(self.out_shape) if shared_outputs is None else shared_outputs
        self.deltas = self.be.iobuf(self.in_shape) if shared_deltas is None else shared_deltas

    def fprop(self, inputs, inference=False):
        self.inputs = inputs
        self.be.fprop_pool(self.nglayer, inputs, self.outputs)
        return self.outputs

    def bprop(self, error):
        self.be.bprop_pool(self.nglayer, self.inputs, error, self.deltas)
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
    def __init__(self, init=None, name="ParameterLayer"):
        super(ParameterLayer, self).__init__(name)
        self.has_params = True
        self.init = init
        self.W = None
        self.weight_shape = None
        self.states = []

    def allocate(self, shared_outputs=None, shared_deltas=None):
        self.outputs = self.be.iobuf(self.out_shape) if shared_outputs is None else shared_outputs
        self.deltas = self.be.iobuf(self.in_shape) if shared_deltas is None else shared_deltas

        if self.W is None:
            self.init_params(self.weight_shape)

    def init_params(self, shape):
        """
        Allocate layer parameter buffers and initialize them with the
            supplied initializer.

        Arguments:
            shape (int, tuple): shape to allocate for layer paremeter
                buffers.
        """
        self.W = self.be.empty(shape)
        self.dW = self.be.empty_like(self.W)
        self.init.fill(self.W)

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
        serial_dict = {'params': self.W.asnumpyarray()}
        if keep_states:
            serial_dict['states'] = [s.asnumpyarray() for s in self.states]
        return serial_dict

    def set_params(self, W):
        """
        Set layer parameters (weights). Allocate space for other parameters but
            do not initialize them.

        Arguments:
            W (Tensor): Tensor containing weights to use.
        """
        self.W = self.be.array(W)
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
                 name="ConvolutionLayer"):
        super(Convolution, self).__init__(init, name)
        self.nglayer = None
        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'T': 1, 'D': 1, 'bsum': bsum}  # 3D paramaters
        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding

        if isinstance(fshape, tuple):
            fshape = {'R': fshape[0], 'S': fshape[1], 'K': fshape[2]}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.convparams.update(d)

    def __str__(self):
        return ("Conv Layer '%s': %d x (%dx%d) inputs, %d x (%dx%d) "
                "outputs, padding %d, stride %d" %
                (self.name,
                 self.in_shape[0], self.in_shape[1], self.in_shape[2],
                 self.out_shape[0], self.out_shape[1], self.out_shape[2],
                 self.convparams['pad_h'], self.convparams['str_h']))

    def configure(self, in_obj):
        super(Convolution, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            shapedict = {'C': self.in_shape[0],
                         'H': self.in_shape[1],
                         'W': self.in_shape[2],
                         'N': self.be.bsz}
            self.convparams.update(shapedict)
            self.nglayer = self.be.conv_layer(self.be.default_dtype, **self.convparams)
            self.out_shape = (self.nglayer.K, self.nglayer.P, self.nglayer.Q)
        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)
        if self.convparams['bsum']:
            self.batch_sum = self.be.zeros((self.nglayer.K, 1), dtype=np.float32)
        else:
            self.batch_sum = None
        return self

    def fprop(self, inputs, inference=False):
        self.inputs = inputs
        self.be.fprop_conv(self.nglayer, inputs, self.W, self.outputs, bsum=self.batch_sum)
        if self.convparams['bsum']:
            return (self.outputs, self.batch_sum)
        else:
            return self.outputs

    def bprop(self, error, do_acts=True):
        if do_acts:
            self.be.bprop_conv(self.nglayer, self.W, error, self.deltas)
        self.be.update_conv(self.nglayer, self.inputs, error, self.dW)
        return self.deltas


class Deconv(ParameterLayer):
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
    def __init__(self, fshape, strides={}, padding={}, init=None, name="DeconvolutionLayer"):
        super(Deconv, self).__init__(init, name)
        self.nglayer = None
        self.deconvparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                             'pad_h': 0, 'pad_w': 0, 'pad_d': 0}

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
        return "Deconv Layer '%s': %d x (%dx%d) inputs, %d x (%dx%d) outputs" % (
               self.name,
               self.in_shape[0], self.in_shape[1], self.in_shape[2],
               self.out_shape[0], self.out_shape[1], self.out_shape[2])

    def configure(self, in_obj):
        super(Deconv, self).configure(in_obj)
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
        return self

    def fprop(self, inputs, inference=False):
        """
        fprop for deconv is equivalent to bprop for conv.
        bprop_conv takes in error and deltas as "E" and "grad_I"
        for deconv, bprop_conv will take in input as "E" and output as "grad_I"
        """
        self.inputs = inputs
        self.be.bprop_conv(layer=self.nglayer, F=self.W, E=inputs, grad_I=self.outputs)
        return self.outputs

    def bprop(self, error, do_acts=True):
        """
        bprop for deconv is equivalent to fprop for conv.
        fprop_conv takes input and output as "I" and "O".
        for deconv, fprop_conv will take error as input and delta as output
        """
        if do_acts:
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
        super(Linear, self).__init__(init, name)
        self.nout = nout
        self.inputs = None
        self.bsum = False  # TODO: bsum for linear is not implemented

    def __str__(self):
        return "Linear Layer '%s': %d inputs, %d outputs" % (
               self.name, self.nin, self.nout)

    def configure(self, in_obj):
        super(Linear, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.out_shape = (self.nout, self.nsteps)
        if self.weight_shape is None:
            self.weight_shape = (self.nout, self.nin)
        # TODO: Set up self.batch_sum if self.bsum
        return self

    def fprop(self, inputs, inference=False):
        self.inputs = inputs
        self.be.compound_dot(A=self.W, B=self.inputs, C=self.outputs)  # , bsum=self.batch_sum
        if self.bsum:
            return (self.outputs, self.batch_sum)
        else:
            return self.outputs

    def bprop(self, error, do_acts=True):
        if do_acts:
            self.be.compound_dot(A=self.W.T, B=error, C=self.deltas)
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

    def allocate(self, shared_outputs=None, shared_deltas=None):
        # Bias layers should share output with their associated weight layer
        if self.prev_layer is not None:
            self.outputs = self.prev_layer.outputs
        if self.W is None:
            self.init_params(self.weight_shape)

    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        if self.y is None or self.y.base is not self.outputs:
            self.y = self.outputs.reshape((self.bias_size, -1))
        self.y[:] = self.y + self.W
        return self.outputs

    def bprop(self, error):
        if not self.deltas:
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

    def __str__(self):
        return "Activation Layer '%s': %s" % (
               self.name, self.transform.__class__.__name__)

    def configure(self, in_obj):
        super(Activation, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def allocate(self, shared_outputs=None, shared_deltas=None):
        self.outputs = self.prev_layer.outputs

    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        self.outputs[:] = self.transform(self.inputs)
        return self.outputs

    def bprop(self, error):
        if not self.deltas:
            self.deltas = error
        error[:] = self.transform.bprop(self.outputs) * self.deltas
        return error


class Affine(list):
    """
    A linear layer with a learned bias and activation, implemented as a list
    composing separate linear, bias and activation layers.

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
        if bias is not None:
            self.append(Bias(init=bias, name=bias_name))
        if batch_norm:
            self.append(BatchNorm())
        if activation is not None:
            self.append(Activation(transform=activation, name=act_name))


class Conv(list):
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
    def __init__(self, fshape, init, strides={}, pad={}, bias=None, batch_norm=False,
                 activation=None, conv_name='ConvolutionLayer',
                 bias_name='BiasLayer', act_name='ActivationLayer'):
        list.__init__(self)
        self.append(Convolution(fshape=fshape, strides=strides, padding=pad,
                                init=init, bsum=batch_norm, name=conv_name))
        if bias is not None:
            self.append(Bias(init=bias, name=bias_name))
        if batch_norm:
            self.append(BatchNorm())
        if activation is not None:
            self.append(Activation(transform=activation, name=act_name))


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

    def __str__(self):
        return "Linear Layer '%s': %d inputs and outputs, keep %d%%" % (
               self.name, self.nout, 100*self.keep)

    def configure(self, in_obj):
        super(Dropout, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def allocate(self, shared_outputs=None, shared_deltas=None):
        self.keep_mask = self.be.iobuf(self.out_shape)

    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        if inference:
            return self._fprop_inference(inputs)
        self.be.make_binary_mask(self.keep_mask, self.keep)
        self.outputs[:] = self.keep_mask * inputs
        return self.outputs

    def _fprop_inference(self, inputs):
        self.outputs[:] = inputs * self.keep
        return self.outputs

    def bprop(self, error, do_acts=False):
        if self.deltas is None:
            self.deltas = error
        self.deltas[:] = self.keep_mask * error
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
        assert isinstance(in_obj, Layer)
        self.prev_layer = in_obj
        (_, self.nstep) = interpret_in_shape(in_obj.out_shape)
        self.outputs = self.be.iobuf((1, self.nstep))
        self.deltas = self.be.iobuf(in_obj.out_shape)
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
        if self.deltas is None:
            self.deltas = self.be.empty_like(inputs)
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
        if self.deltas is None:
            self.deltas = self.be.empty_like(inputs)
        self.deltas[:] = self.costfunc.bprop(inputs, targets) * mask
        return self.deltas


class BatchNorm(Layer):
    """
    A batch normalization layer as described in [Ioffe]_

    Normalizes a batch worth of inputs by subtracting batch mean and dividing by
    batch variance.  Then scales by learned factor gamma and shifts by learned bias beta.

    Notes:

    .. [Ioffe] arXiv:1502.03167
    """
    def __init__(self, rho=0.99, eps=1e-6, name="BatchNormLayer"):
        super(BatchNorm, self).__init__(name)
        self.allparams = None
        self.x = None  # used to point to reshaped view of inputs
        self.xhat = None
        self.has_params = True
        self.outputs = None
        self.rho = rho
        self.eps = eps
        self.states = [[] for i in range(2)]
        self.relu = False
        self.bn_compound = True if hasattr(self.be, 'compound_bprop_bn') else False

    def __str__(self):
        return "BatchNorm Layer '%s': %d inputs, %d steps, %d feature maps" % (
               self.name, self.nin, self.nsteps, self.nfm)

    def configure(self, in_obj):
        super(BatchNorm, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.nfm = self.in_shape[0] if isinstance(self.in_shape, tuple) else self.nin

        return self

    def allocate(self, shared_outputs=None, shared_deltas=None):
        self.outputs = self.be.iobuf(self.out_shape)
        self.y = self.outputs.reshape((self.nfm, -1))
        self.xvar = self.be.zeros((self.nfm, 1))
        if self.allparams is None:
            self.init_params(self.nfm)
        if self.bn_compound:
            self.xsum = self.be.zeros((self.nfm, 1))
        else:
            self.xmean = self.be.zeros((self.nfm, 1))

    def init_params(self, dim0):
        self.beta = self.be.zeros((dim0, 1))
        self.gamma = self.be.ones((dim0, 1))
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
        if type(inputs) is tuple:
            inputs, bsum = inputs
        else:
            bsum = None

        if inference:
            if self.x is None or self.x.base is not self.inputs:
                self.x = self.inputs.reshape((self.nfm, -1))
            return self._fprop_inference(inputs)

        if self.bn_compound:
            # custom batch norm kernel
            if self.inputs is None:
                self.inputs = inputs.reshape((self.nfm, -1))

            if bsum is None:
                self.xsum[:] = self.be.sum(self.inputs, axis=1)
            else:
                self.xsum = bsum

            self.be.compound_fprop_bn(
                self.inputs, self.xsum, self.xvar, self.gmean, self.gvar,
                self.gamma, self.beta, self.outputs, self.eps, self.rho, self.relu)

        else:
            self.inputs = inputs
            if self.x is None or self.x.base is not self.inputs:
                self.x = self.inputs.reshape((self.nfm, -1))

            # These are cached op-trees
            self.xvar[:] = self.be.var(self.x, axis=1)
            self.xmean[:] = self.be.mean(self.x, axis=1)
            self.xhat = (self.x - self.xmean) / self.be.sqrt(self.xvar + self.eps)

            self.gmean[:] = self.gmean * self.rho + (1.0 - self.rho) * self.xmean
            self.gvar[:] = self.gvar * self.rho + (1.0 - self.rho) * self.xvar
            self.y[:] = self.xhat * self.gamma + self.beta
        return self.outputs

    def _fprop_inference(self, inputs):
        """
        Apply one linear transformation that captures normalization, gamma scaling and beta shift.
        """
        xhat = (self.x - self.gmean) / self.be.sqrt(self.gvar + self.eps)  # Op-tree only
        self.y[:] = xhat * self.gamma + self.beta
        return self.outputs

    def bprop(self, error):
        """
        Compute gradients for learning gamma and beta as well as layer weights.
        """
        if not self.deltas:
            self.deltas = error.reshape((self.nfm, -1))

        if self.bn_compound:
            # custom batch norm kernel
            self.be.compound_bprop_bn(
                self.deltas, self.grad_gamma, self.grad_beta, self.inputs,
                self.xsum, self.xvar, self.gamma, self.eps)
        else:
            self.grad_gamma[:] = self.be.sum(self.xhat * self.deltas, axis=1)
            self.grad_beta[:] = self.be.sum(self.deltas, axis=1)
            xtmp = (self.xhat * self.grad_gamma + self.grad_beta) / float(self.x.shape[1])
            self.deltas[:] = self.gamma * (self.deltas - xtmp) / self.be.sqrt(self.xvar + self.eps)
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
