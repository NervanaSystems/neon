import numpy as np
import pytest
from neon import NervanaObject, logger


@pytest.mark.unsupported
@pytest.mark.skip(reason="not implemented for backend_cpu")
def test_sr_cpu(backend_cpu):
    """
    Performs stochastic rounding with 1 bit mantissa for an addition operation
    and checks that the resulting array is rounded correctly
    """
    be = NervanaObject.be
    n = 10
    A = be.ones((n, n), dtype=np.float16)
    B = be.ones((n, n), dtype=np.float16)
    be.multiply(B, 0.1, out=B)
    C = be.ones((n, n), dtype=np.float16)
    C.rounding = 1
    C[:] = A + B
    C_host = C.get()
    # make sure everything is either 1. (rounded down 1 bit) or 1.5 (rounded up 1 bit)
    logger.display("Rounded Buf: {}".format(C_host))
    assert sum([C_host.flatten()[i] in [1., 1.5] for i in range(n**2)]) == n**2
    assert sum([C_host.flatten()[i] in [1.5] for i in range(n**2)]) > .1 * n**2
    assert sum([C_host.flatten()[i] in [1.] for i in range(n**2)]) > .7 * n**2


@pytest.mark.unsupported
@pytest.mark.skip(reason="float16 not supported for backend_mkl")
def test_sr_mkl(backend_mkl):
    """
    Performs stochastic rounding with 1 bit mantissa for an addition operation
    and checks that the resulting array is rounded correctly
    """
    be = NervanaObject.be
    n = 10
    A = be.ones((n, n), dtype=np.float16)
    B = be.ones((n, n), dtype=np.float16)
    be.multiply(B, 0.1, out=B)
    C = be.ones((n, n), dtype=np.float16)
    C.rounding = 1
    C[:] = A + B
    C_host = C.get()
    # make sure everything is either 1. (rounded down 1 bit) or 1.5 (rounded up 1 bit)
    logger.display("Rounded Buf: {}".format(C_host))
    assert sum([C_host.flatten()[i] in [1., 1.5] for i in range(n**2)]) == n**2
    assert sum([C_host.flatten()[i] in [1.5] for i in range(n**2)]) > .1 * n**2
    assert sum([C_host.flatten()[i] in [1.] for i in range(n**2)]) > .7 * n**2


@pytest.mark.hasgpu
def test_sr(backend_gpu):
    """
    Performs stochastic rounding with 1 bit mantissa for an addition operation
    and checks that the resulting array is rounded correctly
    """
    be = NervanaObject.be
    n = 10
    A = be.ones((n, n), dtype=np.float16)
    B = be.ones((n, n), dtype=np.float16)
    be.multiply(B, 0.1, out=B)
    C = be.ones((n, n), dtype=np.float16)
    C.rounding = 1
    C[:] = A + B
    C_host = C.get()
    # make sure everything is either 1. (rounded down 1 bit) or 1.5 (rounded up 1 bit)
    logger.display("Rounded Buf: {}".format(C_host))
    assert sum([C_host.flatten()[i] in [1., 1.5] for i in range(n**2)]) == n**2
    assert sum([C_host.flatten()[i] in [1.5] for i in range(n**2)]) > .1 * n**2
    assert sum([C_host.flatten()[i] in [1.] for i in range(n**2)]) > .7 * n**2

if __name__ == '__main__':
    test_sr()
