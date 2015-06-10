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
Cross entropy transform functions and classes.
"""

from neon.transforms.cost import Cost
from neon.transforms.logistic import Logistic
from neon.transforms.softmax import Softmax
from neon.util.param import opt_param


def cross_entropy(backend, outputs, targets, temp, epsilon=2**-23,
                  scale_by_batchsize=False):
    """
    Evaluates cross entropy on pairwise elements from outputs and targets.

    Given that this is undefined for predicted outputs equal to exactly 0 or
    1.0, we first add epsilon prior to taking log

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (Tensor): predicted output values to be compared.
        targets (Tensor): known outcome values to be compared against.
        temp (list): temporary buffers.
        epsilon (numeric): unit roundoff error.  Defaults to 2^-23, which
                           matches python float32 machine epsilon.
        scale_by_batchsize: Prescale the cross_entropy, useful for

    Returns:
        Tensor: Calculated cross entropy values for each element.
    """
    # Compute (t-1)*log(1-y).
    backend.add(targets, -1.0, out=temp[0])
    backend.subtract(1.0, outputs, out=temp[1])
    backend.add(temp[1], epsilon, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(temp[0], temp[1], out=temp[0])

    # Compute t*log(y).
    backend.add(outputs, epsilon, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(targets, temp[1], out=temp[1])

    # Compute t*log(y) - (t-1)*log(1-y)
    backend.subtract(temp[0], temp[1], out=temp[0])
    result = backend.empty((1, 1), dtype=outputs.dtype)
    if scale_by_batchsize:
        backend.divide(temp[0], temp[0].shape[1], temp[0])
    return backend.sum(temp[0], axes=None, out=result)


def cross_entropy_multi(backend, outputs, targets, temp, epsilon=2**-23,
                        scale_by_batchsize=False):
    """
    Evaluates cross entropy on elements from outputs and targets.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (Tensor): predicted output values to be compared.
        targets (Tensor): known outcome values to be compared against.
        temp (Tensor): temporary buffers.
        epsilon (numeric): unit roundoff error.  Defaults to 2^-23, which
                           matches python float32 machine epsilon.

    Returns:
        Tensor: Calculated cross entropy values for each element.
    """

    # Compute (t*log(y)).
    backend.add(outputs, epsilon, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(targets, temp[1], out=temp[1])
    backend.multiply(temp[1], -1.0, out=temp[0])
    result = backend.empty((1, 1), dtype=outputs.dtype)
    if scale_by_batchsize:
        backend.divide(temp[0], temp[0].shape[1], temp[0])
    return backend.sum(temp[0], axes=None, out=result)


def cross_entropy_derivative(backend, outputs, targets, temp, scale=1.0,
                             epsilon=2**-23):
    """
    Applies derivative of the cross entropy to the pairwise elements from
    outputs and targets.

    Note that this is undefined for predicted outputs equal to exactly 0 or
    1.0, so we clip these to epsilon (backend machine precision) and 1.0 -
    epsilon respectively.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (Tenor): predicted output values to be compared.
        targets (Tensor): known outcome values to be compared against.
        temp (Tensor): temporary buffers.
        epsilon (numeric): unit roundoff error.  Defaults to 2^-23, which
                           matches python float32 machine epsilon.

    Returns:
        Tensor: Calculated cross entropy values for each element.
    """
    backend.subtract(outputs, targets, out=temp[0])
    backend.subtract(1.0, outputs, out=temp[1])
    backend.multiply(temp[1], outputs, out=temp[1])
    backend.clip(temp[1], epsilon, 1 - epsilon, out=temp[1])
    backend.divide(temp[0], temp[1], out=temp[0])
    return temp[0]


def cross_entropy_multi_derivative(backend, outputs, targets, temp, scale=1.0):
    """
    Applies derivative of the cross entropy to the pairwise elements from
    outputs and targets.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (Tensor): predicted output values to be compared.
        targets (Tensor): known outcome values to be compared against.
        temp (Tensor): temporary buffers.

    Returns:
        Tensor: Calculated cross entropy values for each element.
    """
    backend.divide(targets, outputs, out=temp[0])
    backend.multiply(temp[0], -scale, out=temp[0])
    return temp[0]


def shortcut_derivative(backend, outputs, targets, temp, scale=1.0):
    """
    For use when combining cost with matched activation
    i.e. cross_entropy_binary with logistic or
         cross_entropy_multi  with softmax
    Derivative has simpler form and removes numerical errors
    """
    backend.subtract(outputs, targets, out=temp[0])
    backend.multiply(temp[0], scale, out=temp[0])
    return temp[0]


class CrossEntropy(Cost):

    """
    Embodiment of a cross entropy cost function.
    """

    def __init__(self, **kwargs):
        opt_param(self, ['epsilon'], 2**-23)  # default float32 machine epsilon
        super(CrossEntropy, self).__init__(**kwargs)

    def initialize(self, kwargs):
        opt_param(self, ['shortcut_deriv'], True)
        # raw label indicates whether the reference labels are indexes (raw)
        # or one-hot (default)
        super(CrossEntropy, self).initialize(kwargs)
        if isinstance(self.olayer.activation, Softmax):
            self.ce_function = cross_entropy_multi
            if self.shortcut_deriv:
                self.cd_function = shortcut_derivative
                self.olayer.skip_act = True
            else:
                self.cd_function = cross_entropy_multi_derivative
        elif isinstance(self.olayer.activation, Logistic):
            self.ce_function = cross_entropy
            if self.shortcut_deriv:
                self.cd_function = shortcut_derivative
                self.olayer.skip_act = True
            else:
                self.cd_function = cross_entropy_derivative
        else:
            self.ce_function = cross_entropy
            self.cd_function = cross_entropy_derivative

    def __str__(self):
        return ("Cost Function: {shrtct} {rl}\n".format(
                shrtct=self.shortcut_deriv, rl=self.raw_label))

    def set_outputbuf(self, databuf):
        temp_dtype = self.temp_dtype
        if not self.outputbuf or self.outputbuf.shape != databuf.shape:
            tempbuf1 = self.backend.zeros(databuf.shape, temp_dtype)
            tempbuf2 = self.backend.zeros(databuf.shape, temp_dtype)
            tempbuf3 = self.backend.zeros((1, databuf.shape[1]), temp_dtype)
            tempbuf4 = self.backend.zeros(databuf.shape, temp_dtype)
            self.temp = [tempbuf1, tempbuf2, tempbuf3, tempbuf4]
        self.outputbuf = databuf

    def get_deltabuf(self):
        # used by layer2 only.
        return self.temp[0]

    def raw_to_onehot(self, labels):
        self.temp[3].fill(0.0)

        for row in range(self.outputbuf.shape[0]):
            self.backend.equal(labels, row, self.temp[3][row:(row+1)])

        return self.temp[3]

    def apply_logloss(self, targets, eps=1e-15):
        """
        Logloss function -- does normalization prior to computing multiclass
        log loss function if the output layer is not softmax
        """
        if self.raw_label:
            targets = self.raw_to_onehot(targets)
        if isinstance(self.olayer.activation, Softmax):
            return self.ce_function(self.backend, self.outputbuf, targets,
                                    self.temp)
        self.backend.add(self.outputbuf, eps, out=self.temp[0])
        self.backend.sum(self.temp[0], axes=0, out=self.temp[2])
        self.backend.divide(self.temp[0], self.temp[2], out=self.temp[0])

        return cross_entropy_multi(self.backend, self.temp[0], targets,
                                   self.temp)

    def apply_function(self, targets, scale_by_batchsize=False):
        """
        Apply the cross entropy cost function to the datasets passed.
        """
        if self.raw_label:
            targets = self.raw_to_onehot(targets)
        result = self.ce_function(self.backend, self.outputbuf, targets,
                                  self.temp, epsilon=self.epsilon,
                                  scale_by_batchsize=scale_by_batchsize)
        self.backend.multiply(result, self.scale, out=result)
        return result

    def apply_derivative(self, targets):
        """
        Apply the derivative of the cross entropy cost function to the datasets
        passed.
        """
        if self.raw_label:
            targets = self.raw_to_onehot(targets)
        return self.cd_function(self.backend, self.outputbuf,
                                targets, self.temp, self.scale)
