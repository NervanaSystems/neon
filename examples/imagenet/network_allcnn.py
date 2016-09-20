from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, GeneralizedCost, Dropout, DataTransform, Activation
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Normalizer
from neon.models import Model


def create_network():
    layers = [
        DataTransform(transform=Normalizer(divisor=128.)),

        Conv((11, 11, 96), init=Kaiming(), activation=Rectlin(), strides=4, padding=1),
        Conv((1, 1, 96), init=Kaiming(), activation=Rectlin(), strides=1),
        Conv((3, 3, 96), init=Kaiming(), activation=Rectlin(), strides=2, padding=1),   # 54->2,

        Conv((5, 5, 256), init=Kaiming(), activation=Rectlin(), strides=1),             # 27->2,
        Conv((1, 1, 256), init=Kaiming(), activation=Rectlin(), strides=1),
        Conv((3, 3, 256), init=Kaiming(), activation=Rectlin(), strides=2, padding=1),  # 23->1,

        Conv((3, 3, 384), init=Kaiming(), activation=Rectlin(), strides=1, padding=1),
        Conv((1, 1, 384), init=Kaiming(), activation=Rectlin(), strides=1),
        Conv((3, 3, 384), init=Kaiming(), activation=Rectlin(), strides=2, padding=1),  # 12->,

        Dropout(keep=0.5),
        Conv((3, 3, 1024), init=Kaiming(), activation=Rectlin(), strides=1, padding=1),
        Conv((1, 1, 1024), init=Kaiming(), activation=Rectlin(), strides=1),
        Conv((1, 1, 1000), init=Kaiming(), activation=Rectlin(), strides=1),
        Pooling(6, op='avg'),
        Activation(Softmax())
    ]

    return Model(layers=layers), GeneralizedCost(costfunc=CrossEntropyMulti())
