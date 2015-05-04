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
Batch normalization transform functions and classes.
"""

import logging
from neon.transforms.activation import Activation
from neon.util.param import req_param, opt_param
import numpy as np


logger = logging.getLogger(__name__)


class BatchNorm(Activation):

    """
    Embodiment of a Batch Normalization Transform.  Can be used in a
    WeightLayer after the linear transform but before the typical activation
    (ReLu, sigmoid, tanh, etc.)

    The transform does z-norm along the batch dimension then adjusts using
    an affine transform with learned parameters gamma and beta.  gamma and beta
    gradients (updates) are computed via bprop, and the updates are applied
    at the same time that the associated WeightLayer is updated.  Likewise,
    the same learning rule is used for gamma and beta as the WeightLayer

    Forward pass: (gamma/beta are scalar parameters for each unit)

        x' = (x - mean) / sqrt(var + eps)
        y  = gamma * x' + beta

    Backward pass:

        dy/dx = dy/dx' * dx'/dx
        = gamma * [1*(var+eps)^-1/2 + (x-mean) * (var+eps)^-3/2 * (2x)^-1/2]

    """

    def initialize(self, kwargs):
        """
        Initialize the Batch Normalization transform. This function will be
        called from WeightLayer.initialize with a reference to the layer.

        Arguments:
            _eps (numeric, optional): value used for numerical stability when
                                      normalizing by variance
            _iscale (numeric, optional): explicitly set an affine scale value
                                         to be used in inference instead of
                                         calculated scale from training
            _ishift (numeric, optional): explicitly set an affine shift value
                                         to be used in inference instead of
                                         calculated shift from training
        """
        self.__dict__.update(kwargs)
        self.dtype = self.layer.weight_dtype
        self.bigtype = np.float32 if self.dtype is np.float16 else self.dtype
        opt_param(self, ['_iscale', '_ishift'])
        opt_param(self, ['_eps'], 1e-6)
        req_param(self, ['layer'])

        self.backend = self.layer.backend
        self.is_local = self.layer.is_local
        self.batch_size = self.layer.batch_size
        if self.is_local:
            self.in1d = (self.layer.nofm, 1)
            self.ofmsize = self.layer.ofmsize
            self.orig_shape = (self.layer.nofm * self.ofmsize, self.batch_size)
            self.in_shape = (self.layer.nofm, self.ofmsize * self.batch_size)
        else:
            self.in_shape = (self.layer.nout, self.batch_size)
            self.in1d = (self.layer.nout, 1)

        self.train_mode = True
        logger.info("BatchNormalization set to train mode")
        self.nbatches = 0

        self._xhat = self.backend.zeros(self.in_shape, dtype=self.dtype)

        self._mean = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self._vars = self.backend.zeros(self.in1d, dtype=self.bigtype)

        # Global mean and var to be used during inference
        self._gmean = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self._gvars = self.backend.zeros(self.in1d, dtype=self.bigtype)

        # learned params and their update buffers
        self._beta = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self._gamma = self.backend.ones(self.in1d, dtype=self.bigtype)
        self.layer.params.extend([self._beta, self._gamma])

        self._beta_updates = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self._gamma_updates = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self.layer.updates.extend([self._beta_updates, self._gamma_updates])

    def get_params(self):
        np_params = dict()
        for p in ['_gamma', '_beta']:
            if hasattr(self, p):
                p_tensor = getattr(self, p)
                np_params[p] = np.array(p_tensor.asnumpyarray(),
                                        dtype=p_tensor.dtype).reshape(
                    p_tensor.shape)
        return np_params

    def set_params(self, params_dict):
        for p in ['_gamma', '_beta']:
            if p in params_dict:
                getattr(self, p)[:] = params_dict[p]

    def set_inference_mode(self):
        """
        If implemented following Ioffe et al. 2015, there appears to be a bug
        with using inference mode. As more data is accumulated, the prediction
        gets worse and worse. As a workaround, stay in train mode where the
        variance and mean statistics are computed , which seems
        to perform quite well.
        """
        logger.warning("Batch Normalization inference mode not supported. "
                       "Using training mode.")
        self.train_mode = True  # Set to 'False' to force inference mode
        if self.train_mode is False:
            if self._iscale is None:
                # normalize global variance -- inference scaling factor
                self.backend.divide(self._gvars, self.nbatches, self._gvars)
                m = self.batch_size
                if self.is_local:
                    m *= self.ofmsize
                unbiaser = float(m / (m - 1.))
                self.backend.multiply(self._gvars, unbiaser, self._gvars)
                self.backend.add(self._gvars, self._eps, self._gvars)
                self.backend.sqrt(self._gvars, out=self._gvars)
                self.backend.divide(self._gamma, self._gvars, self._gvars)
                self._iscale = self._gvars

                # normalize global mean -- inference shifting factor
                self.backend.divide(self._gmean, self.nbatches, self._gmean)
                self.backend.multiply(self._gmean, self._gvars, self._gmean)
                self.backend.subtract(self._beta, self._gmean, self._gmean)
                self._ishift = self._gmean

    def apply_function(self, backend, inputs, outputs):
        """
        Though this function is necessary for Activation conformance, no action
        is required for batch norm here.
        """
        pass

    def apply_derivative(self, backend, inputs, outputs):
        """
        Though this function is necessary for Activation conformance, no action
        is required for batch norm here.
        """
        pass

    def fprop_func(self, backend, inputs, outputs):
        """
        Applies BatchNorm function and its derivative to the dataset passed.

        For a fully connected layer, this is done by computing the mean and
        variance of the `inputs` over the mini-batch dimension,

        For a convolutional (local) layer, the `inputs` are reshaped so that
        the statistics are collected within each feature map as well as
        over the mini-batch dimension.

        The means and variances are also accumulated into a global estimate
        that is used for inference.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also
                                 acts as storage for the output of the
                                 derivative function.
            outputs (array_like): Storage for the transformed output.
        """
        if self.is_local:
            inputs = inputs.reshape(self.in_shape)
            outputs = outputs.reshape(self.in_shape)

        if self.train_mode:
            # Calc batch statistics
            backend.mean(inputs, axes=1, out=self._mean)
            backend.variance(inputs, axes=1, out=self._vars, mean=self._mean)
            # increment the global estimates (TODO: stop after an epoch)
            backend.add(self._gvars, self._vars, self._gvars)
            backend.add(self._gmean, self._mean, self._gmean)
            self.nbatches += 1

            # Just store sqrt(vars + eps) since it's used as a unit
            backend.add(self._vars, self._eps, self._vars)
            backend.sqrt(self._vars, out=self._vars)

            # Every operation below uses broadcasting over minibatch dim
            backend.subtract(inputs, self._mean, out=self._xhat)
            backend.divide(self._xhat, self._vars, out=self._xhat)
            backend.multiply(self._xhat, self._gamma, out=outputs)
            backend.add(outputs, self._beta, out=outputs)
        else:
            # Inference mode: Using accumulated scale and shift
            backend.multiply(inputs, self._iscale, out=outputs)
            backend.add(outputs, self._ishift, out=outputs)

        if self.is_local:
            inputs = inputs.reshape(self.orig_shape)
            outputs = outputs.reshape(self.orig_shape)

    def bprop_func(self, backend, pre_act, error, skip_act=False):
        """
        Calculates the backpropagated error and gradients for gamma and beta
        parameters.

        Updates to gamma and beta are accumulated into `self._gamma_updates`
        and `self._beta_updates` respectively, which are applied to
        `self._gamma` and `self._beta` during the call to `update` of the
        associated WeightLayer

        Arguments:
            backend (Backend): The backend class to use for computation.
            pre_act (array_like): Storage allocated by associated WeightLayer
                                  that is used for computing updates.
            error (array_like): gradient of error with respect to input
                                activations.
            skip_act (boolean): Not used
        """
        if self.is_local:
            pre_act = pre_act.reshape(self.in_shape)
            error = error.reshape(self.in_shape)

        backend.multiply(self._xhat, error, out=pre_act)
        backend.sum(pre_act, axes=1, out=self._gamma_updates)
        backend.sum(error, axes=1, out=self._beta_updates)

        # Compute the backpropagated error into error
        backend.multiply(self._xhat, self._gamma_updates, out=self._xhat)
        backend.add(self._xhat, self._beta_updates, out=self._xhat)
        backend.divide(self._xhat, float(self._xhat.shape[1]), out=self._xhat)
        backend.subtract(error, self._xhat, out=error)
        backend.multiply(error, self._gamma, out=error)
        backend.divide(error, self._vars, out=error)

        if self.is_local:
            pre_act = pre_act.reshape(self.orig_shape)
            error = error.reshape(self.orig_shape)
