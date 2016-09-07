# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# ----------------------------------------------------------------------------
"""
Test the skip-thought container
"""
import itertools as itt
import numpy as np

from neon import NervanaObject
from neon.backends import gen_backend
from neon.models import Model
from neon.initializers import Uniform
from neon.layers import GRU

from skip_thought import SkipThought

def allclose_with_out(x, y, atol=0.0, rtol=1.0e-5):
    # run the np.allclose on x and y
    # if it fails print some stats
    # before returning
    ac = np.allclose(x, y, rtol=rtol, atol=atol)
    if not ac:
        dd = np.abs(x-y)
        print 'abs errors: %e [%e, %e] Abs Thresh = %e' \
              % (np.median(dd), np.min(dd), np.max(dd), atol)
        amax = np.argmax(dd)
        print 'worst case: %e %e' % (x.flat[amax], y.flat[amax])
        dd = np.abs(dd - atol)/np.abs(y)
        print 'rel errors: %e [%e, %e] Rel Thresh = %e' \
              % (np.median(dd), np.min(dd), np.max(dd), rtol)
        amax = np.argmax(dd)
        print 'worst case: %e %e' % (x.flat[amax], y.flat[amax])
    return ac


def test_skip_thought(backend_default):

    be = backend_default
    be.bsz = 32

    vs = 2000
    es = 300
    init_embed = Uniform(low=-0.1, high=0.1)
    nh = 640

    skip = SkipThought(vs, es, init_embed, nh)

    la = skip.layers_to_optimize
    model = Model(skip)
    model.initialize(dataset=[(100, 32), (100, 32), (100, 32)])

    s_s = be.array(np.random.randint(100, size=(100, 32)), dtype=np.int32)
    s_p = be.array(np.random.randint(100, size=(100, 32)), dtype=np.int32)
    s_n = be.array(np.random.randint(100, size=(100, 32)), dtype=np.int32)

    out = model.fprop((s_s, s_p, s_n))

    e_p = be.array(np.random.randint(100, size=(2000, 3200)), dtype=np.int32)
    e_n = be.array(np.random.randint(100, size=(2000, 3200)), dtype=np.int32)

    error_out = model.bprop((e_p, e_n))


if __name__ == '__main__':

    be = gen_backend('gpu')
    test_skip_thought(be)


