from neon.initializers import Gaussian, Constant
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Dropout, LRN
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


def create_network_lrn():
    init1 = Gaussian(scale=0.01)
    init2 = Gaussian(scale=0.005)

    layers = [
        Conv((11, 11, 96), padding=0, strides=4, init=init1, bias=Constant(0),
             activation=Rectlin(), name='conv1'),
        Pooling(3, strides=2, name='pool1'),
        LRN(5, ascale=0.0001, bpower=0.75, name='norm1'),
        Conv((5, 5, 256), padding=2, init=init1, bias=Constant(1.0), activation=Rectlin(),
             name='conv2'),
        Pooling(3, strides=2, name='pool2'),
        LRN(5, ascale=0.0001, bpower=0.75, name='norm2'),
        Conv((3, 3, 384), padding=1, init=init1, bias=Constant(0),
             activation=Rectlin(), name='conv3'),
        Conv((3, 3, 384), padding=1, init=init1, bias=Constant(1.0),
             activation=Rectlin(), name='conv4'),
        Conv((3, 3, 256), padding=1, init=init1, bias=Constant(1.0),
             activation=Rectlin(), name='conv5'),
        Pooling(3, strides=2, name='pool5'),
        Affine(nout=4096, init=init2, bias=Constant(1.0), activation=Rectlin(), name='fc6'),
        Dropout(keep=0.5, name='drop6'),
        Affine(nout=4096, init=init2, bias=Constant(1.0), activation=Rectlin(), name='fc7'),
        Dropout(keep=0.5, name='drop7'),
        Affine(nout=1000, init=init1, bias=Constant(0.0), activation=Softmax(), name='fc8')
    ]

    return Model(layers=layers), GeneralizedCost(costfunc=CrossEntropyMulti())
