from neon.initializers import Gaussian, Constant
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Dropout
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti
from neon.models import Model


def create_network():
    layers = [
        Conv((11, 11, 64), init=Gaussian(scale=0.01), bias=Constant(0),
             activation=Rectlin(), padding=3, strides=4),
        Pooling(3, strides=2),
        Conv((5, 5, 192), init=Gaussian(scale=0.01), bias=Constant(1),
             activation=Rectlin(), padding=2),
        Pooling(3, strides=2),
        Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0),
             activation=Rectlin(), padding=1),
        Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1),
             activation=Rectlin(), padding=1),
        Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1),
             activation=Rectlin(), padding=1),
        Pooling(3, strides=2),
        Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=1000, init=Gaussian(scale=0.01), bias=Constant(-7), activation=Softmax()),
    ]

    return Model(layers=layers), GeneralizedCost(costfunc=CrossEntropyMulti())
