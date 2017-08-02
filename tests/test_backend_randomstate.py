# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
# pylint: skip-file
import numpy as np
import pytest
from neon.backends import gen_backend
from utils import tensors_allclose


@pytest.mark.hasgpu
def test_gpu_randomstate(device_id):
    # run 1
    be = gen_backend(backend='gpu', rng_seed=100, device_id=device_id)
    a = be.empty((3, 3))

    a[:] = be.rand()  # gpu rand
    x0 = a.get()
    x1 = be.rng.rand(3, 3)  # host rand
    a[:] = be.rand()  # gpu rand
    x2 = a.get()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    x3 = a.get()

    assert len(be.context_rand_state_map) == 1 and len(be.context_rand_state_alive) == 1
    for ctx in be.context_rand_state_alive:
        assert be.context_rand_state_alive[ctx] is True

    # run 2, using reset
    be.rng_reset()

    a[:] = be.rand()
    y0 = a.get()
    y1 = be.rng.rand(3, 3)
    a[:] = be.rand()
    y2 = a.get()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    y3 = a.get()

    assert len(be.context_rand_state_map) == 1 and len(be.context_rand_state_alive) == 1
    for ctx in be.context_rand_state_alive:
        assert be.context_rand_state_alive[ctx] is True

    del(be)

    # run 3, using a new backend
    be = gen_backend(backend='gpu', rng_seed=100, device_id=device_id)
    a = be.empty((3, 3))

    a[:] = be.rand()  # gpu rand
    z0 = a.get()
    z1 = be.rng.rand(3, 3)  # host rand
    a[:] = be.rand()  # gpu rand
    z2 = a.get()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    z3 = a.get()

    # check equality
    assert tensors_allclose([x0, x1, x2, x3], [y0, y1, y2, y3], rtol=0., atol=0.)
    assert tensors_allclose([x0, x1, x2, x3], [z0, z1, z2, z3], rtol=0., atol=0.)

    del(be)


def test_cpu_randomstate():
    # run 1
    be = gen_backend(backend='cpu', rng_seed=100)

    a = be.empty((3, 3))
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    x0 = a.get()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    x1 = a.get()

    # run 2, using reset
    be.rng_reset()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    y0 = a.get()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    y1 = a.get()

    del(be)

    # run 3, using a new backend
    be = gen_backend(backend='cpu', rng_seed=100)

    a = be.empty((3, 3))
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    z0 = a.get()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    z1 = a.get()

    # check equality
    assert tensors_allclose([x0, x1], [y0, y1], rtol=0., atol=0.)
    assert tensors_allclose([x0, x1], [z0, z1], rtol=0., atol=0.)
    del(be)


def test_mkl_randomstate():
    # run 1
    be = gen_backend(backend='mkl', rng_seed=100)

    a = be.empty((3, 3))
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    x0 = a.get()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    x1 = a.get()

    # run 2, using reset
    be.rng_reset()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    y0 = a.get()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    y1 = a.get()

    del(be)

    # run 3, using a new backend
    be = gen_backend(backend='mkl', rng_seed=100)

    a = be.empty((3, 3))
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    z0 = a.get()
    be.make_binary_mask(a, keepthresh=be.rng.rand())
    z1 = a.get()

    # check equality
    assert tensors_allclose([x0, x1], [y0, y1], rtol=0., atol=0.)
    assert tensors_allclose([x0, x1], [z0, z1], rtol=0., atol=0.)
    del(be)


@pytest.mark.hasgpu
def test_rng_funcs(backend_default):
    # like the tests above but also tests the rng_get_state and rng_set_state funcs
    be = backend_default

    # keep sz*sz*batch_size < 3*2048*32 (NervanaGPU._RNG_POOL_SIZE
    sz = 32
    x = be.zeros((sz, sz))

    # use binary mask to test rng resetting

    be.make_binary_mask(out=x)
    x1 = x.get().copy()
    r1 = be.rng_get_state()

    be.make_binary_mask(out=x)
    x2 = x.get().copy()

    # there should be infinitesmial chance for x1 to equal x2
    # this assert is to make sure the rng is doing something
    assert np.max(np.abs(x2 - x1)) > 0

    be.rng_reset()
    be.make_binary_mask(out=x)
    x1_2 = x.get().copy()

    assert np.max(np.abs(x1 - x1_2)) == 0.0

    be.rng_reset()
    be.rng_set_state(r1)
    be.make_binary_mask(out=x)
    x2_2 = x.get().copy()

    assert np.max(np.abs(x2 - x2_2)) == 0.0
