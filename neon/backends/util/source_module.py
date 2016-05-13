from __future__ import print_function
from neon import logger as neon_logger

try:
    from pycuda.compiler import SourceModule as pycSourceModule
except ImportError:
    neon_logger.display("PyCUDA module not found")
from neon.util.persist import get_cache_dir


class SourceModule(pycSourceModule):
    """
    Just provides a wrapper that sets the default cache dir to a neon-specific location
    """
    def __init__(self, source, nvcc="nvcc", options=None, keep=False, no_extern_c=False,
                 arch=None, code=None, cache_dir=None, include_dirs=[]):

        if cache_dir is None:
            cache_dir = get_cache_dir(['kernels', 'pycuda'])

        return super(SourceModule, self).__init__(source, nvcc, options, keep, no_extern_c,
                                                  arch, code, cache_dir, include_dirs)
