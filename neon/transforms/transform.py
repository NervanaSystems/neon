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
from neon import NervanaObject


class Transform(NervanaObject):

    """
    Base class for activation or cost functions and their derivatives. Child classes
    can either implement the below ``__call__`` and ``bprop`` methods, or alternatively
    define ``self.func`` and ``self.funcgrad``. The latter is typically used for code
    compactness when the operations can be fit into a lambda function.
    """
    def __init__(self, name=None):
        """
        Class constructor.
        """
        super(Transform, self).__init__(name)
        self.is_mklop = False

    def __call__(self, x):
        """
        Compute f(x)

        Args:
            x (Tensor or OpTree): input

        Returns:
            func (OpTree): computes the output func(x)
        """
        return self.func(x)

    def bprop(self, x):
        """
        Returns the derivative of f(x).

        Args:
            x (Tensor or OpTree): input

        Returns:
            funcgrad (OpTree): computes the derivative of the func(x)
        """
        return self.funcgrad(x)
