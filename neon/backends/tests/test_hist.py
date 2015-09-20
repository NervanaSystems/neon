# Copyright 2014 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import itertools as itt
from neon.backends.nervanacpu import NervanaCPU
from neon.backends.nervanagpu import NervanaGPU
from neon.backends.tests.utils import assert_tensors_allclose
from neon.backends.util import check_gpu


def ref_hist(inp, nbins=64, offset=-48):
    """
    Implement a log2 histogram geared towards visualizing neural net parameters.

    Bins are computed as the log2 of the magnitude of a tensor value.  Bins are
    rounded to the nearest int.

    Smallest value bin extends to -Inf to enable visualization of zeros.

    Log2 computation is always done in fp32 regardless of input dtype to give
    rounding a consistent behavior.
    """
    bins = np.arange(nbins + 1) + float(offset)
    bins[0] = -float('Inf')
    np_inp_log_abs = np.rint(np.log2(np.abs(inp.astype(np.float32))))
    np_hist, edges = np.histogram(np_inp_log_abs, density=False, bins=bins)
    if (np_hist.ndim < 2):
        np_hist = np_hist.reshape(1, np_hist.size)
    return np_hist


def test_edge_cases():
    """
    Test several edge cases related to min/max bin, and rounding.

    Also test backend dump_hist_data functionality.
    """
    gpuflag = (check_gpu.get_compute_capability(0) >= 5.0)
    if gpuflag is False:
        raise RuntimeError("Device does not have CUDA compute capability 5.0 or greater")
    ng = NervanaGPU()
    nc = NervanaCPU()
    # edge case test
    np_ref = dict()
    inputs = [
        ("edges", np.array([2 ** -48, 2 ** 15], dtype=np.float32)),
        ("rounding", np.array([2 ** 5, 63.99998856, 2 ** 6, 2 ** -3, 2 ** -4,
                               0.11262291, 92.22483826], dtype=np.float32)),
        ("fp16 rounding", np.array([45.21875], dtype=np.float16))
    ]
    for tag, inp in inputs:
        np_ref[tag] = ref_hist(inp)
        for be in [ng, nc]:
            be_inp = be.array(inp)
            be_hist = be_inp.hist(tag)
            assert_tensors_allclose(np_ref[tag], be_hist, err_msg=tag + str(be))

    # dump_hist_data test
    for be in [ng, nc]:
        be_hist_data, be_hist_map = be.dump_hist_data()
        for tag, inp in inputs:
            be_data = be_hist_data[be_hist_map[tag]]
            assert_tensors_allclose(np_ref[tag], be_data, err_msg=tag + str(be))

    del(ng)
    del(nc)


def test_hist(nbin_offset_dim_dtype_inp):
    """
    Compare the nervanagpu and nervanacpu hist implementation to the reference
    implementation above.

    Parameterized test case, uses pytest_generate_test to enumerate dim_dtype_inp
    tuples that drive the test.
    """

    (nbins, offset), dim, dtype, (name, inp_gen) = nbin_offset_dim_dtype_inp

    gpuflag = (check_gpu.get_compute_capability(0) >= 5.0)
    if gpuflag is False:
        raise RuntimeError("Device does not have CUDA compute capability 5.0 or greater")

    ng = NervanaGPU(hist_bins=nbins, hist_offset=offset)
    nc = NervanaCPU(hist_bins=nbins, hist_offset=offset)

    np_inp = inp_gen(dim).astype(dtype)
    np_hist = ref_hist(np_inp, nbins=nbins, offset=offset)
    for be in [ng, nc]:
        be_inp = be.array(np_inp, dtype=dtype)
        be_hist = be_inp.hist(name)
        assert_tensors_allclose(np_hist, be_hist)
    del(ng)
    del(nc)


def pytest_generate_tests(metafunc):
    """
    Build a list of test arguments for test_hist.

    Run a full but slow set if --all is specified as a py.test arg, or just
    run sanity tests otherwise.
    """
    # sanity tests
    bin_offs = [(64, -48),
                (32, 0)]
    dims = [(64, 32768),
            (64, 1)]
    dtypes = [np.float32, np.uint8]
    inputs = [
        ("normal dist", lambda dim: np.random.normal(64, 4, dim[0] * dim[1]).reshape(dim)),
    ]

    # more thorough tests for CI coverage
    if metafunc.config.option.all:

        bin_offs.extend([(64, -32),
                         (32, -16)])

        dims.extend([
            (64, 387200),
            (128, 128),
            (2, 32),
            (1, 1),
        ])

        dtypes.extend([
            np.float16,
            np.int8,
        ])

        # disabled this just so there are less tests.
        # there is some cuda driver init/destroy issue preventing memory being freed
        # between runs, so having more than a certain number of tests run fails
        # inputs.extend([
        #      ("ones", lambda dim: np.ones(dim)),
        # ])

    if 'nbin_offset_dim_dtype_inp' in metafunc.fixturenames:
        fargs = itt.product(bin_offs, dims, dtypes, inputs)
        metafunc.parametrize("nbin_offset_dim_dtype_inp", fargs)
