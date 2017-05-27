# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
Generative Adversarial Network models for image data
"""
from neon.initializers import Gaussian
from neon.layers import (Sequential, Reshape, Convolution, BatchNorm, Activation,
                         Deconvolution, Linear, GeneralizedGANCost, Bias)
from neon.transforms import Rectlin, Tanh, Logistic, Identity
from neon.layers.container import GenerativeAdversarial
from neon.models.model import GAN
from neon.transforms.cost import GANCost

# common configuration for all layers
init_w = Gaussian(scale=0.02)
relu = Rectlin(slope=0)
lrelu = Rectlin(slope=0.2)
bn_prm = dict(rho=0.1, eps=1e-5)


def conv_layer(name, n_feature, ker_size=4, strides=2, padding=1,
               activation=lrelu, batch_norm=True, bias=None):
    """
    Layer configuration for deep-convolutional (DC) discriminator

    Arguments:
        name (string): Layer name
        n_feature (int): Number of output feature maps
        ker_size (int): Size of convolutional kernel (defaults to 4)
        strides (int): Stride of convolution (defaults to 2)
        padding (int): Padding of convolution (defaults to 1)
        activation (object): Activation function (defaults to leaky ReLu)
        batch_norm(bool): Enable batch normalization (defaults to True)
    """
    layers = []
    layers.append(Convolution(fshape=(ker_size, ker_size, n_feature), strides=strides,
                              padding=padding, dilation={}, init=init_w, bsum=batch_norm,
                              name=name))
    if batch_norm:
        layers.append(BatchNorm(name=name + '_bnorm', **bn_prm))
    if bias is not None:
        layers.append(Bias(init=None, name=name + '_bias'))
    layers.append(Activation(transform=activation, name=name + '_rectlin'))
    return layers


def deconv_layer(name, n_feature, ker_size=4, strides=2, padding=1,
                 activation=lrelu, batch_norm=True, bias=None):
    """
    Layer configuration for deep-convolutional (DC) discriminator

    Arguments:
        name (string): Layer name'
        n_feature (int): Number of output feature maps
        ker_size (int): Size of convolutional kernel (defaults to 4)
        strides (int): Stride of convolution (defaults to 2)
        padding (int): Padding of convolution (defaults to 1)
        activation (object): Activation function (defaults to leaky ReLu)
        batch_norm(bool): Enable batch normalization (defaults to True)
    """
    layers = []
    layers.append(Deconvolution(fshape=(ker_size, ker_size, n_feature), strides=strides,
                                padding=padding, dilation={}, init=init_w, bsum=batch_norm,
                                name=name))
    if batch_norm:
        layers.append(BatchNorm(name=name + '_bnorm', **bn_prm))
    if bias is not None:
        layers.append(Bias(init=None, name=name + '_bias'))
    layers.append(Activation(transform=activation, name=name + '_rectlin'))
    return layers


def mlp_layer(name, nout, activation=relu, batch_norm=False, bias=None):
    """
    Layer configuration for MLP generator/discriminator

    Arguments:
        name (string): Layer name
        nout (int): Number of output feature maps
        activation (object): Activation function (defaults to ReLu)
        batch_norm(bool): Enable batch normalization (defaults to False)
    """
    layers = []
    layers.append(Linear(nout=nout, init=init_w, bsum=batch_norm, name=name))
    if batch_norm:
        layers.append(BatchNorm(name=name + '_bnorm', **bn_prm))
    if bias is not None:
        layers.append(Bias(init=None, name=name + '_bias'))
    layers.append(Activation(transform=activation, name=name + '_rectlin'))
    return layers


def create_mlp_discriminator(im_size, n_feature, depth, batch_norm, finact):
    """
    Create MLP discriminator network

    Arguments:
        im_size (int): Image size
        n_feature (int): Base number of features
        depth (int): Depth of network
        batch_norm(bool): Enable batch normalization
        finact (object): Final activation function
    """
    assert depth > 1, "depth of the MLP has to be at least 2"
    layers = []
    for i in range(depth-1):
        lname = 'dis-{0}.{1}'.format(i, n_feature)
        layers.append(mlp_layer(lname, n_feature, batch_norm=batch_norm))
    lname = 'dis-{0}.{1}-{2}'.format(depth-1, n_feature, 1)
    layers.append(mlp_layer(lname, 1, activation=finact, batch_norm=batch_norm))
    return layers


def create_mlp_generator(im_size, n_chan, n_feature, depth, batch_norm, finact):
    """
    Create MLP generator network

    Arguments:
        im_size (int): Image size
        n_chan (int): Number of color channels
        n_feature (int): Base number of features
        depth (int): Depth of network
        batch_norm(bool): Enable batch normalization
        finact (object): Final activation function
    """
    assert depth > 1, "depth of the MLP has to be at least 2"
    layers = []
    for i in range(depth-1):
        lname = 'gen-{0}.{1}'.format(i, n_feature)
        layers.append(mlp_layer(lname, n_feature, batch_norm=batch_norm))
    lname = 'gen-{0}.{1}-{2}'.format(depth-1, n_feature, n_chan*im_size*im_size)
    layers.append(mlp_layer(lname, n_chan*im_size*im_size, activation=finact,
                            batch_norm=batch_norm))
    lname = 'gen-final'
    layers.append(Reshape(name=lname, reshape=(n_chan, im_size, im_size)))
    return layers


def create_dc_discriminator(im_size, n_chan, n_feature, n_extra_layers,
                            batch_norm, finact):
    """
    Create DC-GAN discriminator network

    Arguments:
        im_size (int): Image size
        n_chan (int): Number of color channels
        n_feature (int): Base number of features
        n_extra_layers (int): Number of extra convolution layers
        batch_norm(bool): Enable batch normalization
        finact (object): Final activation function
    """
    assert im_size % 16 == 0, "im_size has to be a multiple of 16"

    _is = im_size // 2
    _nf = n_feature  # image size and number of features

    layers = []
    lname = 'dis.initial.{0}-{1}'.format(n_chan, _nf)
    layers.append(conv_layer(lname, _nf, batch_norm=False))
    for _t in range(n_extra_layers):
        lname = 'dis.extra-layer-{0}.{1}'.format(_t, _nf)
        layers.append(conv_layer(lname, _nf, ker_size=3, strides=1, batch_norm=batch_norm))
    while _is > 4:
        lname = 'dis.pyramid.{0}-{1}'.format(_nf, _nf*2)
        layers.append(conv_layer(lname, _nf*2, batch_norm=batch_norm))
        _is, _nf = _is//2, _nf*2
    lname = 'dis.final.{0}-{1}'.format(_nf, 1)
    layers.append(conv_layer(lname, 1, strides=1, padding=0, activation=finact, batch_norm=False))
    return layers


def create_dc_generator(im_size, n_chan, n_noise, n_feature, n_extra_layers,
                        batch_norm, finact):
    """
    Create DC-GAN generator network

    Arguments:
        im_size (int): Image size
        n_chan (int): Number of color channels
        n_noise (int): Dimension of noise
        n_feature (int): Base number of features
        n_extra_layers (int): Number of extra convolution layers
        batch_norm(bool): Enable batch normalization
        finact (object): Final activation function
    """
    assert im_size % 16 == 0, "im_size has to be a multiple of 16"

    _nf = n_feature // 2
    _is = 4  # image size and number of features
    while _is != im_size:
        _nf *= 2
        _is *= 2

    layers = []
    lname = 'gen.initial.{0}-{1}'.format(n_noise, _nf)
    layers.append(deconv_layer(lname, _nf, strides=1, padding=0, batch_norm=batch_norm))
    _is, _nf = 4, _nf
    while _is < im_size//2:
        lname = 'gen.pyramid.{0}-{1}'.format(_nf, _nf//2)
        layers.append(deconv_layer(lname, _nf//2, batch_norm=batch_norm))
        _nf, _is = _nf//2, _is*2  # decrease image size and increase number of features
    for _t in range(n_extra_layers):
        lname = 'gen.extra-layer-{0}.{1}'.format(_t, _nf)
        layers.append(conv_layer(lname, _nf, ker_size=3, strides=1, batch_norm=batch_norm))
    lname = 'gen.final.{0}-{1}'.format(_nf, n_chan)

    layers.append(deconv_layer(lname, n_chan, activation=finact, batch_norm=False))
    return layers


def create_model(dis_model='dc', gen_model='dc',
                 cost_type='wasserstein', noise_type='normal',
                 im_size=64, n_chan=3, n_noise=100, n_gen_ftr=64, n_dis_ftr=64,
                 depth=4, n_extra_layers=0, batch_norm=True,
                 gen_squash=None, dis_squash=None, dis_iters=5,
                 wgan_param_clamp=None, wgan_train_sched=False):
    """
    Create a GAN model and associated GAN cost function for image generation

    Arguments:
        dis_model (str): Discriminator type, can be 'mlp' for a simple MLP or
                         'dc' for a DC-GAN style model. (defaults to 'dc')
        gen_model (str): Generator type, can be 'mlp' for a simple MLP or
                         'dc' for a DC-GAN style model. (defaults to 'dc')
        cost_type (str): Cost type, can be 'original', 'modified' following
                         Goodfellow2014 or 'wasserstein' following Arjovsky2017
                         (defaults to 'wasserstein')
        noise_type (str): Noise distribution, can be 'uniform or' 'normal'
                          (defaults to 'normal')
        im_size (int): Image size (defaults to 64)
        n_chan (int): Number of image channels (defaults to 3)
        n_noise (int): Number of noise dimensions (defaults to 100)
        n_gen_ftr (int): Number of generator feature maps (defaults to 64)
        n_dis_ftr (int): Number of discriminator feature maps (defaults to 64)
        depth (int): Depth of layers in case of MLP (defaults to 4)
        n_extra_layers (int): Number of extra conv layers in case of DC (defaults to 0)
        batch_norm (bool): Enable batch normalization (defaults to True)
        gen_squash (str or None): Squashing function at the end of generator (defaults to None)
        dis_squash (str or None): Squashing function at the end of discriminator (defaults to None)
        dis_iters (int): Number of critics for discriminator (defaults to 5)
        wgan_param_clamp (float or None): In case of WGAN weight clamp value, None for others
        wgan_train_sched (bool): Enable training schedule of number of critics (defaults to False)
    """
    assert dis_model in ['mlp', 'dc'], \
        "Unsupported model type for discriminator net, supported: 'mlp' and 'dc'"
    assert gen_model in ['mlp', 'dc'], \
        "Unsupported model type for generator net, supported: 'mlp' and 'dc'"
    assert cost_type in ['original', 'modified', 'wasserstein'], \
        "Unsupported GAN cost function type, supported: 'original', 'modified' and 'wasserstein'"

    # types of final squashing functions
    squash_func = dict(nosquash=Identity(), sym=Tanh(), asym=Logistic())
    if cost_type == 'wasserstein':
        if gen_model == 'mlp':
            gen_squash = gen_squash or 'nosquash'
        elif gen_model == 'dc':
            gen_squash = gen_squash or 'sym'
        dis_squash = dis_squash or 'nosquash'
    else:  # for all GAN costs other than Wasserstein
        gen_squash = gen_squash or 'sym'
        dis_squash = dis_squash or 'asym'

    assert gen_squash in ['nosquash', 'sym', 'asym'], \
        "Unsupported final squashing function for generator," \
        " supported: 'nosquash', 'sym' and 'asym'"
    assert dis_squash in ['nosquash', 'sym', 'asym'], \
        "Unsupported final squashing function for discriminator," \
        " supported: 'nosquash', 'sym' and 'asym'"

    gfa = squash_func[gen_squash]
    dfa = squash_func[dis_squash]

    # create model layers
    if gen_model == 'mlp':
        gen = create_mlp_generator(im_size, n_chan, n_gen_ftr, depth,
                                   batch_norm=False, finact=gfa)
        noise_dim = (n_noise,)
    elif gen_model == 'dc':
        gen = create_dc_generator(im_size, n_chan, n_noise, n_gen_ftr, n_extra_layers,
                                  batch_norm, finact=gfa)
        noise_dim = (n_noise, 1, 1)

    if dis_model == 'mlp':
        dis = create_mlp_discriminator(im_size, n_dis_ftr, depth, batch_norm=False, finact=dfa)
    elif dis_model == 'dc':
        dis = create_dc_discriminator(im_size, n_chan, n_dis_ftr, n_extra_layers, batch_norm,
                                      finact=dfa)
    layers = GenerativeAdversarial(generator=Sequential(gen, name="Generator"),
                                   discriminator=Sequential(dis, name="Discriminator"))

    return GAN(layers=layers, noise_dim=noise_dim, noise_type=noise_type, k=dis_iters,
               wgan_param_clamp=wgan_param_clamp, wgan_train_sched=wgan_train_sched), \
        GeneralizedGANCost(costfunc=GANCost(func=cost_type))
