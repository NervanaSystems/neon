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

logger = logging.getLogger(__name__)


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

    def __str__(self):
        """
        Format the layer as a printable string.
        """
        ret = '{} {}'.format(self.__class__.__name__, self.name)
        return ret

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
        fshape (Union[int, Tuple[int, int]]): one or two dimensional shape
            of pooling window
        op (Optional[str]): pooling operation in [max, avg]. Defaults to "max"
        strides (Optional[Union[int, dict]]): strides to apply pooling window
            over. An int applies to both dimensions, or a dict with str_h
            and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (Optional[Union[int, dict]]): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        name (Optional[str]): layer name. Defaults to "PoolingLayer"
    """
    def __init__(self, fshape, op="max", strides={}, padding={},
                 name="PoolingLayer"):
        super(Pooling, self).__init__(name)
        self.poolparams = {'str_h': None, 'str_w': None, 'str_d': None, 'str_j': None,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0, 'pad_j': 0,
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

    def init_buffers(self, inputs):
        self.inputs = inputs
        if self.nglayer is None:
            assert hasattr(self.inputs, 'lshape')
            self.poolparams['C'] = self.inputs.lshape[0]
            self.poolparams['H'] = self.inputs.lshape[1]
            self.poolparams['W'] = self.inputs.lshape[2]
            self.poolparams['N'] = self.be.bsz
            self.nglayer = self.be.pool_layer(self.be.default_dtype, **self.poolparams)
            self.outputs = self.be.iobuf(self.nglayer.nOut, self.outputs)
            self.outputs.lshape = (self.nglayer.K, self.nglayer.P, self.nglayer.Q)
            self.deltas = self.be.iobuf(self.inputs.shape[0], self.deltas)

    def fprop(self, inputs, inference=False):
        self.init_buffers(inputs)
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
        init (Optional[Initializer]): Initializer object to use for
            initializing layer weights
        name (Optional[str]): layer name. Defaults to "ParameterLayer"
    """
    def __init__(self, init=None, name="ParameterLayer"):
        super(ParameterLayer, self).__init__(name)
        self.has_params = True
        self.init = init
        self.W = None
        self.weight_shape = None
        self.states = []

    def fprop(self, inputs, inference=False):
        self.init_buffers(inputs)
        if self.W is None:
            self.init_params(self.weight_shape)

    def init_params(self, shape):
        """
        Allocate layer parameter buffers and initialize them with the
            supplied initializer.

        Arguments:
            shape (Union[int, tuple]): shape to allocate for layer paremeter
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
        strides (Optional[Union[int, dict]]): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (Optional[Union[int, dict]]): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        init (Optional[Initializer]): Initializer object to use for
            initializing layer weights
        name (Optional[str]): layer name. Defaults to "ConvolutionLayer"
    """

    def __init__(self, fshape, strides={}, padding={}, init=None, name="ConvolutionLayer"):
        super(Convolution, self).__init__(init, name)
        self.nglayer = None
        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'T': 1, 'D': 1}  # 3D paramaters

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

    def init_buffers(self, inputs):
        """
        Helper for allocating output and delta buffers (but not initializing
        them)

        Arguments:
            inputs (Tensor): tensor used for frop inputs, used to determine
                shape of buffers being allocated.
        """
        self.inputs = inputs
        if not self.nglayer:
            assert hasattr(self.inputs, 'lshape')
            self.convparams['C'] = self.inputs.lshape[0]
            self.convparams['H'] = self.inputs.lshape[1]
            self.convparams['W'] = self.inputs.lshape[2]
            self.convparams['N'] = self.be.bsz
            self.nglayer = self.be.conv_layer(self.be.default_dtype, **self.convparams)
            self.outputs = self.be.iobuf(self.nglayer.nOut, self.outputs)
            self.outputs.lshape = (self.nglayer.K, self.nglayer.P, self.nglayer.Q)
            self.deltas = self.be.iobuf(self.inputs.shape[0], self.deltas)

        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)

    def fprop(self, inputs, inference=False):
        super(Convolution, self).fprop(inputs)
        self.be.fprop_conv(self.nglayer, inputs, self.W, self.outputs)
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
        strides (Optional[Union[int, dict]]): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (Optional[Union[int, dict]]): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        init (Optional[Initializer]): Initializer object to use for
            initializing layer weights
        name (Optional[str]): layer name. Defaults to "DeconvolutionLayer"
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

    def init_buffers(self, inputs):
        self.inputs = inputs
        if not self.nglayer:
            assert hasattr(self.inputs, 'lshape')
            # We switch H, W and C with P, Q and K
            # so that in the GPU, we can reverse calculate
            # H and W
            self.deconvparams['K'] = self.inputs.lshape[0]
            self.deconvparams['P'] = self.inputs.lshape[1]
            self.deconvparams['Q'] = self.inputs.lshape[2]
            self.deconvparams['N'] = self.be.bsz
            self.nglayer = self.be.deconv_layer(self.be.default_dtype, **self.deconvparams)
            self.outputs = self.be.iobuf(self.nglayer.dimI2[0], self.outputs)
            self.outputs.lshape = (self.nglayer.C, self.nglayer.H, self.nglayer.W)
            self.deltas = self.be.iobuf(self.inputs.shape[0], self.deltas)

        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)

    def fprop(self, inputs, inference=False):
        """
        fprop for deconv is equivalent to bprop for conv.
        bprop_conv takes in error and deltas as "E" and "grad_I"
        for deconv, bprop_conv will take in input as "E" and output as "grad_I"
        """
        super(Deconv, self).fprop(inputs)
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
        nout (Union[int, tuple]): Desired size or shape of layer output
        init (Optional[Initializer]): Initializer object to use for
            initializing layer weights
        name (Optional[str]): Layer name. Defaults to "LinearLayer"
    """
    def __init__(self, nout, init, name="LinearLayer"):
        super(Linear, self).__init__(init, name)
        self.nout = nout

    def init_buffers(self, inputs):
        self.inputs = inputs
        if self.outputs is None:
            self.nin = inputs.shape[0]
            # non recurrent case:
            if inputs.shape[1] == self.be.bsz:
                self.outputs = self.be.iobuf(self.nout)
                self.deltas = self.be.iobuf(self.nin)
            else:
                self.nsteps = inputs.shape[1] / self.be.bsz
                self.outputs = self.be.iobuf((self.nout, self.nsteps))
                self.deltas = self.be.iobuf((self.nin, self.nsteps))

        if self.weight_shape is None:
            self.weight_shape = (self.nout, inputs.shape[0])

    def fprop(self, inputs, inference=False):
        super(Linear, self).fprop(inputs)
        self.be.compound_dot(A=self.W, B=inputs, C=self.outputs)
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
        init (Optional[Initializer]): Initializer object to use for
            initializing layer bias
        name (Optional[str]): Layer name. Defaults to "BiasLayer"
    """
    def __init__(self, init, name="BiasLayer"):
        super(Bias, self).__init__(init, name)
        self.reshaped_outputs = None

    def init_buffers(self, inputs):
        self.inputs = inputs
        self.outputs = inputs
        if self.reshaped_outputs is None:
            if hasattr(inputs, 'lshape'):
                self.bsize = inputs.lshape[0]
            else:
                self.bsize = inputs.shape[0]
            self.reshaped_outputs = self.outputs.reshape((self.bsize,
                                                          self.outputs.size /
                                                          self.bsize))
        if self.weight_shape is None:
            self.weight_shape = (self.bsize, 1)

    def fprop(self, inputs, inference=False):
        super(Bias, self).fprop(inputs)

        # reshaped_outputs is a different view of outputs, which is
        # the same as inputs (we call it outputs for naming reasons)
        self.reshaped_outputs[:] = self.reshaped_outputs + self.W
        return self.outputs

    def bprop(self, error):
        if not self.deltas:
            self.deltas = error.reshape(self.reshaped_outputs.shape)
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
        name (Optional[str]): Layer name. Defaults to "ActivationLayer"
    """
    def __init__(self, transform, name="ActivationLayer"):
        super(Activation, self).__init__(name)
        self.transform = transform

    def init_buffers(self, inputs):
        self.inputs = inputs
        if self.outputs is None:
            self.outputs = self.inputs

    def fprop(self, inputs, inference=False):
        self.init_buffers(inputs)
        self.outputs[:] = self.transform(self.inputs)
        return self.outputs

    def bprop(self, error):
        error[:] = self.transform.bprop(self.inputs) * error
        return error


class Affine(list):
    """
    A linear layer with a learned bias and activation, implemented as a list
    composing separate linear, bias and activation layers.

    Arguments:
        nout (Union[int, tuple]): Desired size or shape of layer output
        init (Optional[Initializer]): Initializer object to use for
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
        self.append(Linear(nout, init, name=linear_name))
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
        init (Optional[Initializer]): Initializer object to use for
            initializing layer weights and bias
        strides (Optional[Union[int, dict]]): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        pad (Optional[Union[int, dict]]): padding to apply to edges of
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
        self.append(Convolution(fshape=fshape, strides=strides, padding=pad, init=init))
        if bias is not None:
            self.append(Bias(init=bias))
        if batch_norm:
            self.append(BatchNorm())
        if activation is not None:
            self.append(Activation(transform=activation))


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

    def init_buffers(self, inputs, inference):
        self.inputs = inputs
        if self.outputs is None:
            self.outputs = self.be.zeros(inputs.shape)
            if hasattr(self.inputs, 'lshape'):
                self.outputs.lshape = self.inputs.lshape
            if not inference:
                self.keep_mask = self.be.zeros(inputs.shape)

    def fprop(self, inputs, inference=False):
        self.init_buffers(inputs, inference)
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
        self.cost = self.be.empty((1, 1))
        self.outputs = None
        self.deltas = None

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
        if self.outputs is None:
            self.nstep = 1  # Non-recurrent case
            if inputs.shape[1] != self.be.bsz:
                self.nstep = inputs.shape[1] / self.be.bsz
            # For non recurrent case, this is the same as be.iobuf(1)
            self.outputs = self.be.iobuf((1, self.nstep))

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
            self.nstep = 1  # Non-recurrent case
            if inputs.shape[1] != self.be.bsz:
                self.nstep = inputs.shape[1] / self.be.bsz
            self.deltas = self.be.empty_like(inputs)
        # Cost function divides by (bsz * nsteps), so we multiply by nsteps here
        # to get the average gradient over the batch (recurrent case only)
        self.deltas[:] = self.costfunc.bprop(inputs, targets) * self.nstep
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
        if self.outputs is None:
            self.nstep = 1  # Non-recurrent case
            if inputs.shape[1] != self.be.bsz:
                self.nstep = inputs.shape[1] / self.be.bsz
            # For non recurrent case, this is the same as be.iobuf(1)
            self.outputs = self.be.iobuf((1, self.nstep))
        masked_input = inputs * mask
        self.outputs[:] = self.costfunc(masked_input, targets)
        self.cost[:] = self.be.mean(self.outputs, axis=1) * self.nstep
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
            self.nstep = 1  # Non-recurrent case
            if inputs.shape[1] != self.be.bsz:
                self.nstep = inputs.shape[1] / self.be.bsz
            self.deltas = self.be.empty_like(inputs)
        # Cost function divides by (bsz * nsteps), so we multiply by nsteps here
        # to get the average gradient over the batch (recurrent case only)
        self.deltas[:] = self.costfunc.bprop(inputs, targets) * self.nstep * mask

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

    def set_bn_shape(self, inputs):
        self.nfm = inputs.shape[0] if not hasattr(inputs, 'lshape') else inputs.lshape[0]
        self.bn_shape = (self.nfm, inputs.size / self.nfm)

    def init_buffers(self, inputs):
        self.inputs = inputs
        if self.x is None:
            self.nout = self.inputs.shape[0]
            self.outputs = self.be.iobuf(self.nout)

            if hasattr(self.inputs, 'lshape'):
                self.outputs.lshape = self.inputs.lshape
            # This is for local layers -- the first dimension should be number of featuremaps
            self.set_bn_shape(self.inputs)

            self.x = self.inputs.reshape(self.bn_shape)
            self.y = self.outputs.reshape(self.bn_shape)

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
        if inference:
            return self._fprop_inference(inputs)
        self.init_buffers(inputs)
        if self.allparams is None:
            self.init_params(self.nfm)

        # These are cached op-trees
        self.xvar = self.be.var(self.x, axis=1)
        self.xmean = self.be.mean(self.x, axis=1)
        self.xhat = (self.x - self.xmean) / self.be.sqrt(self.xvar + self.eps)

        self.gmean[:] = self.gmean * self.rho + (1.0 - self.rho) * self.xmean
        self.gvar[:] = self.gvar * self.rho + (1.0 - self.rho) * self.xvar
        self.y[:] = self.xhat * self.gamma + self.beta
        return self.outputs

    def _fprop_inference(self, inputs):
        """
        Apply one linear transformation that captures normalization, gamma scaling and beta shift.
        """
        self.init_buffers(inputs)
        xhat = (self.x - self.gmean) / self.be.sqrt(self.gvar + self.eps)  # Op-tree only
        self.y[:] = xhat * self.gamma + self.beta
        return self.outputs

    def bprop(self, error):
        """
        Compute gradients for learning gamma and beta as well as layer weights.
        """
        if not self.deltas:
            self.deltas = error.reshape(self.bn_shape)

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
            self.deltas = error.reshape(self.bn_shape)

        # autodiff will automatically cache and reuse the object
        # if we know the `error` buffer at init, we can also create the autodiff
        # object at layer's init
        ad = Autodiff(self.fprop_op_tree, self.be, next_error=self.deltas)

        # back propagate
        ad.back_prop_grad([self.x, self.gamma, self.beta],
                          [self.deltas, self.grad_gamma, self.grad_beta])

        return error
