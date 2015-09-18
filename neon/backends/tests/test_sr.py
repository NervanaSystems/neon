import numpy as np
from neon.backends import gen_backend


def test_sr():
    """
    Performs stochastic rounding with 1 bit mantissa for an addition operation
    and checks that the resulting array is rounded correctly
    """
    gpu = gen_backend(backend='gpu', stochastic_round=False)
    n = 10
    A = gpu.ones((n, n), dtype=np.float16)
    B = gpu.ones((n, n), dtype=np.float16)
    gpu.multiply(B, 0.1, out=B)
    C = gpu.ones((n, n), dtype=np.float16)
    C.rounding = 1
    C[:] = A + B
    C_host = C.get()
    # make sure everything is either 1. (rounded down 1 bit) or 1.5 (rounded up 1 bit)
    print C_host
    assert sum([C_host.flatten()[i] in [1., 1.5] for i in range(n**2)]) == n**2
    assert sum([C_host.flatten()[i] in [1.5] for i in range(n**2)]) > .1 * n**2
    assert sum([C_host.flatten()[i] in [1.] for i in range(n**2)]) > .7 * n**2

if __name__ == '__main__':
    test_sr()
