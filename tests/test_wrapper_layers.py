# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
from neon.initializers.initializer import Uniform
from neon.transforms.activation import Rectlin
from neon.layers.layer import Linear, Convolution, Convolution_bias, Conv, Bias, Activation, Affine


def test_conv_wrapper(backend_default):
    """
    Verify that the Conv wrapper constructs the right layer objects.
    """
    conv = Conv((4, 4, 3), Uniform())
    assert isinstance(conv, list)
    assert len(conv) == 1
    assert isinstance(conv[0], Convolution)

    conv = Conv((4, 4, 3), Uniform(), bias=Uniform())
    assert isinstance(conv, list)
    assert len(conv) == 1
    assert isinstance(conv[0], Convolution_bias)

    conv = Conv((4, 4, 3), Uniform(), activation=Rectlin())
    assert isinstance(conv, list)
    assert len(conv) == 2
    assert isinstance(conv[0], Convolution)
    assert isinstance(conv[1], Activation)

    conv = Conv((4, 4, 3), Uniform(), bias=Uniform(), activation=Rectlin())
    assert isinstance(conv, list)
    assert isinstance(conv[0], Convolution_bias)
    assert isinstance(conv[1], Activation)
    assert len(conv) == 2


def test_affine_wrapper(backend_default):
    """
    Verify that the Affine wrapper constructs the right layer objects.
    """
    nout = 11
    aff = Affine(nout, Uniform())
    assert isinstance(aff, list)
    assert len(aff) == 1
    assert isinstance(aff[0], Linear)
    assert aff[0].nout == nout

    aff = Affine(nout, Uniform(), bias=Uniform())
    assert isinstance(aff, list)
    assert len(aff) == 2
    assert isinstance(aff[0], Linear)
    assert isinstance(aff[1], Bias)

    aff = Affine(nout, Uniform(), activation=Rectlin())
    assert isinstance(aff, list)
    assert len(aff) == 2
    assert isinstance(aff[0], Linear)
    assert isinstance(aff[1], Activation)

    aff = Affine(nout, Uniform(), bias=Uniform(), activation=Rectlin())
    assert isinstance(aff, list)
    assert len(aff) == 3
    assert isinstance(aff[0], Linear)
    assert isinstance(aff[1], Bias)
    assert isinstance(aff[2], Activation)
