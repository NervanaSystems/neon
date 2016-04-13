from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, Affine
from neon.models import Model
from neon.transforms import Rectlin, Softmax


def create_network():
    # weight initialization
    g1 = Gaussian(scale=0.01)
    g5 = Gaussian(scale=0.005)
    c0 = Constant(0)
    c1 = Constant(1)

    # model initialization
    padding = {'pad_d': 1, 'pad_h': 1, 'pad_w': 1}
    strides = {'str_d': 2, 'str_h': 2, 'str_w': 2}
    layers = [
        Conv((3, 3, 3, 64), padding=padding, init=g1, bias=c0, activation=Rectlin()),
        Pooling((1, 2, 2), strides={'str_d': 1, 'str_h': 2, 'str_w': 2}),
        Conv((3, 3, 3, 128), padding=padding, init=g1, bias=c1, activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Conv((3, 3, 3, 256), padding=padding, init=g1, bias=c1, activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Conv((3, 3, 3, 256), padding=padding, init=g1, bias=c1, activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Conv((3, 3, 3, 256), padding=padding, init=g1, bias=c1, activation=Rectlin()),
        Pooling((2, 2, 2), strides=strides),
        Affine(nout=2048, init=g5, bias=c1, activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=2048, init=g5, bias=c1, activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=101, init=g1, bias=c0, activation=Softmax())
    ]
    return Model(layers=layers)
