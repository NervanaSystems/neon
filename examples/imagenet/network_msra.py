from builtins import zip
import itertools as itt
from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Activation
from neon.layers import MergeSum, SkipNode
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti
from neon.models import Model


def conv_params(fsize, nfm, strides=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm),
                strides=strides,
                activation=(Rectlin() if relu else None),
                padding=(fsize // 2),
                batch_norm=batch_norm,
                init=Kaiming(local=True))


def module_factory(nfm, bottleneck=True, stride=1):
    nfm_out = nfm * 4 if bottleneck else nfm
    use_skip = True if stride == 1 else False
    stride = abs(stride)
    sidepath = [SkipNode() if use_skip else Conv(
        **conv_params(1, nfm_out, stride, False))]

    if bottleneck:
        mainpath = [Conv(**conv_params(1, nfm, stride)),
                    Conv(**conv_params(3, nfm)),
                    Conv(**conv_params(1, nfm_out, relu=False))]
    else:
        mainpath = [Conv(**conv_params(3, nfm, stride)),
                    Conv(**conv_params(3, nfm, relu=False))]
    return [MergeSum([mainpath, sidepath]),
            Activation(Rectlin())]


def create_network(stage_depth):
    if stage_depth in (18, 18):
        stages = (2, 2, 2, 2)
    elif stage_depth in (34, 50):
        stages = (3, 4, 6, 3)
    elif stage_depth in (68, 101):
        stages = (3, 4, 23, 3)
    elif stage_depth in (102, 152):
        stages = (3, 8, 36, 3)
    else:
        raise ValueError('Invalid stage_depth value'.format(stage_depth))

    bottleneck = False
    if stage_depth in (50, 101, 152):
        bottleneck = True

    layers = [Conv(**conv_params(7, 64, strides=2)),
              Pooling(3, strides=2)]

    # Structure of the deep residual part of the network:
    # stage_depth modules of 2 convolutional layers each at feature map depths
    # of 64, 128, 256, 512
    nfms = list(itt.chain.from_iterable(
        [itt.repeat(2**(x + 6), r) for x, r in enumerate(stages)]))
    strides = [-1] + [1 if cur == prev else 2 for cur,
                      prev in zip(nfms[1:], nfms[:-1])]

    for nfm, stride in zip(nfms, strides):
        layers.append(module_factory(nfm, bottleneck, stride))

    layers.append(Pooling('all', op='avg'))
    layers.append(Affine(nout=1000, init=Kaiming(local=False), activation=Softmax()))
    return Model(layers=layers), GeneralizedCost(costfunc=CrossEntropyMulti())
