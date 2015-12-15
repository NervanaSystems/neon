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

import re
import os.path
import pycuda.driver as drv
from pycuda.tools import context_dependent_memoize
from math import ceil

base_dir  = os.path.dirname(__file__)
ptx_dir   = os.path.join(base_dir, "kernels", "ptx")
sass_dir  = os.path.join(base_dir, "kernels", "sass")
pre_dir   = os.path.join(base_dir, "kernels", "pre")
cubin_dir = os.path.join(base_dir, "kernels", "cubin")
dump_dir  = os.path.join(base_dir, "kernels", "dump")

kernels = {
    "hconv_bprop_C1_N64":    {"threads":  32, "sass": "hconv_bprop_C1_N64",    "params": "bprop1", "share": " 32*8*2 +  64*8*2"},
    "hconv_bprop_C128_N128": {"threads": 256, "sass": "hconv_xprop_X128_N128", "params": "bprop",  "share": "128*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "hconv_bprop_C128_N64":  {"threads": 128, "sass": "hconv_xprop_X128_N64",  "params": "bprop",  "share": "128*8*2 +  64*8*2 + 8", "args": {"prop": "b"}},
    "hconv_bprop_C32_N128":  {"threads":  64, "sass": "hconv_xprop_X32_N128",  "params": "bprop",  "share": " 32*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "hconv_bprop_C64_N128":  {"threads": 128, "sass": "hconv_xprop_X64_N128",  "params": "bprop",  "share": " 64*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "hconv_bprop_C64_N64":   {"threads":  64, "sass": "hconv_xprop_X64_N64",   "params": "bprop",  "share": " 64*8*2 +  64*8*2 + 8", "args": {"prop": "b"}},
    "hconv_fprop_K128_N128": {"threads": 256, "sass": "hconv_xprop_X128_N128", "params": "fprop",  "share": "128*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "hconv_fprop_K128_N64":  {"threads": 128, "sass": "hconv_xprop_X128_N64",  "params": "fprop",  "share": "128*8*2 +  64*8*2 + 8", "args": {"prop": "f"}},
    "hconv_fprop_K32_N128":  {"threads":  64, "sass": "hconv_xprop_X32_N128",  "params": "fprop",  "share": " 32*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "hconv_fprop_K64_N128":  {"threads": 128, "sass": "hconv_xprop_X64_N128",  "params": "fprop",  "share": " 64*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "hconv_fprop_K64_N64":   {"threads":  64, "sass": "hconv_xprop_X64_N64",   "params": "fprop",  "share": " 64*8*2 +  64*8*2 + 8", "args": {"prop": "f"}},
    "hconv_updat_C128_K128": {"threads": 256, "sass": "hconv_updat_C128_K128", "params": "updat",  "share": "(128*16 + 32)*2 + (128*16 + 32)*2 + 8", "occupancy": 4.0},
    "hconv_updat_C128_K64":  {"threads": 128, "sass": "hconv_updat_C128_K64",  "params": "updat",  "share": "(128*16 + 32)*2 + ( 64*16 + 32)*2 + 8", "occupancy": 3.0},
    "hgemm_nn_128x128":      {"threads": 256, "sass": "hgemm_nn_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4"},
    "hgemm_nt_128x128":      {"threads": 256, "sass": "hgemm_nt_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4"},
    "hgemm_tn_128x128":      {"threads": 256, "sass": "hgemm_tn_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4"},
    "hgemm_nn_vec_128x128":  {"threads": 256, "sass": "hgemm_nn_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4", "args": {"vec": "1"}},
    "hgemm_nt_vec_128x128":  {"threads": 256, "sass": "hgemm_nt_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4", "args": {"vec": "1"}},
    "hgemm_tn_vec_128x128":  {"threads": 256, "sass": "hgemm_tn_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4", "args": {"vec": "1"}},
    "hgemm_nn_128x64":       {"threads": 128, "sass": "hgemm_nn_128x64",       "params": "gemm",   "share": "128*8*2 +  64*8*2 + 4"},
    "hgemm_tn_128x64":       {"threads": 128, "sass": "hgemm_tn_128x64",       "params": "gemm",   "share": "128*8*2 +  64*8*2 + 4"},
    "hgemm_nn_vec_128x64":   {"threads": 128, "sass": "hgemm_nn_128x64",       "params": "gemm",   "share": "128*8*2 +  64*8*2 + 4", "args": {"vec": "1"}},
    "hgemm_tn_vec_128x64":   {"threads": 128, "sass": "hgemm_tn_128x64",       "params": "gemm",   "share": "128*8*2 +  64*8*2 + 4", "args": {"vec": "1"}},
    "hgemm_nn_128x32":       {"threads": 128, "sass": "hgemm_nn_128x32",       "params": "gemm",   "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "hgemm_tn_128x32":       {"threads": 128, "sass": "hgemm_tn_128x32",       "params": "gemm",   "share": "(128*16 +  0)*2 + 32*16*2 + 4"},
    "hgemm_nn_vec_128x32":   {"threads": 128, "sass": "hgemm_nn_128x32",       "params": "gemm",   "share": "(128*16 + 32)*2 + 32*16*2 + 4", "args": {"vec": "1"}},
    "hgemm_tn_vec_128x32":   {"threads": 128, "sass": "hgemm_tn_128x32",       "params": "gemm",   "share": "(128*16 +  0)*2 + 32*16*2 + 4", "args": {"vec": "1"}},
    "hgemm_nn_32x128":       {"threads": 128, "sass": "hgemm_nn_32x128",       "params": "gemm",   "share": "(32*16 + 32)*2 + (128*16 +  0)*2 + 4"},
    "hgemm_nt_32x128":       {"threads": 128, "sass": "hgemm_nt_32x128",       "params": "gemm",   "share": "(32*16 + 32)*2 + (128*16 + 32)*2 + 4"},
    "hgemm_nn_vec_32x128":   {"threads": 128, "sass": "hgemm_nn_32x128",       "params": "gemm",   "share": "(32*16 + 32)*2 + (128*16 +  0)*2 + 4", "args": {"vec": "1"}},
    "hgemm_nt_vec_32x128":   {"threads": 128, "sass": "hgemm_nt_32x128",       "params": "gemm",   "share": "(32*16 + 32)*2 + (128*16 + 32)*2 + 4", "args": {"vec": "1"}},

    "sconv_bprop_C1_N64":    {"threads":  32, "sass": "sconv_bprop_C1_N64",    "params": "bprop1", "share": " 32*8*2 +  64*8*2"},
    "sconv_bprop_C128_N128": {"threads": 256, "sass": "sconv_xprop_X128_N128", "params": "bprop",  "share": "128*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "sconv_bprop_C128_N64":  {"threads": 128, "sass": "sconv_xprop_X128_N64",  "params": "bprop",  "share": "128*8*2 +  64*8*2 + 8", "args": {"prop": "b"}},
    "sconv_bprop_C32_N128":  {"threads":  64, "sass": "sconv_xprop_X32_N128",  "params": "bprop",  "share": " 32*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "sconv_bprop_C64_N128":  {"threads": 128, "sass": "sconv_xprop_X64_N128",  "params": "bprop",  "share": " 64*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "sconv_bprop_C64_N64":   {"threads":  64, "sass": "sconv_xprop_X64_N64",   "params": "bprop",  "share": " 64*8*2 +  64*8*2 + 8", "args": {"prop": "b"}},
    "sconv_fprop_K128_N128": {"threads": 256, "sass": "sconv_xprop_X128_N128", "params": "fprop",  "share": "128*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "sconv_fprop_K128_N64":  {"threads": 128, "sass": "sconv_xprop_X128_N64",  "params": "fprop",  "share": "128*8*2 +  64*8*2 + 8", "args": {"prop": "f"}},
    "sconv_fprop_K32_N128":  {"threads":  64, "sass": "sconv_xprop_X32_N128",  "params": "fprop",  "share": " 32*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "sconv_fprop_K64_N128":  {"threads": 128, "sass": "sconv_xprop_X64_N128",  "params": "fprop",  "share": " 64*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "sconv_fprop_K64_N64":   {"threads":  64, "sass": "sconv_xprop_X64_N64",   "params": "fprop",  "share": " 64*8*2 +  64*8*2 + 8", "args": {"prop": "f"}},
    "sconv_updat_C128_K128": {"threads": 256, "sass": "sconv_updat_C128_K128", "params": "updat",  "share": "(128*16 + 32)*2 + (128*16 + 32)*2 + 8", "occupancy": 4.0},
    "sconv_updat_C128_K64":  {"threads": 128, "sass": "sconv_updat_C128_K64",  "params": "updat",  "share": "(128*16 + 32)*2 + ( 64*16 + 32)*2 + 8", "occupancy": 3.0},
    "sgemm_nn_128x128":      {"threads": 256, "sass": "sgemm_nn_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4"},
    "sgemm_nt_128x128":      {"threads": 256, "sass": "sgemm_nt_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4"},
    "sgemm_tn_128x128":      {"threads": 256, "sass": "sgemm_tn_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4"},
    "sgemm_nn_vec_128x128":  {"threads": 256, "sass": "sgemm_nn_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4", "args": {"vec": "1"}},
    "sgemm_nt_vec_128x128":  {"threads": 256, "sass": "sgemm_nt_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4", "args": {"vec": "1"}},
    "sgemm_tn_vec_128x128":  {"threads": 256, "sass": "sgemm_tn_128x128",      "params": "gemm",   "share": "128*8*2 + 128*8*2 + 4", "args": {"vec": "1"}},
    "sgemm_nn_128x64":       {"threads": 128, "sass": "sgemm_nn_128x64",       "params": "gemm",   "share": "128*8*2 +  64*8*2 + 4"},
    "sgemm_tn_128x64":       {"threads": 128, "sass": "sgemm_tn_128x64",       "params": "gemm",   "share": "128*8*2 +  64*8*2 + 4"},
    "sgemm_nn_vec_128x64":   {"threads": 128, "sass": "sgemm_nn_128x64",       "params": "gemm",   "share": "128*8*2 +  64*8*2 + 4", "args": {"vec": "1"}},
    "sgemm_tn_vec_128x64":   {"threads": 128, "sass": "sgemm_tn_128x64",       "params": "gemm",   "share": "128*8*2 +  64*8*2 + 4", "args": {"vec": "1"}},
    "sgemm_nn_128x32":       {"threads": 128, "sass": "sgemm_nn_128x32",       "params": "gemm",   "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "sgemm_tn_128x32":       {"threads": 128, "sass": "sgemm_tn_128x32",       "params": "gemm",   "share": "(128*16 +  0)*2 + 32*16*2 + 4"},
    "sgemm_nn_vec_128x32":   {"threads": 128, "sass": "sgemm_nn_128x32",       "params": "gemm",   "share": "(128*16 + 32)*2 + 32*16*2 + 4", "args": {"vec": "1"}},
    "sgemm_tn_vec_128x32":   {"threads": 128, "sass": "sgemm_tn_128x32",       "params": "gemm",   "share": "(128*16 +  0)*2 + 32*16*2 + 4", "args": {"vec": "1"}},
    "sgemm_nn_32x128":       {"threads": 128, "sass": "sgemm_nn_32x128",       "params": "gemm",   "share": "(32*16 + 32)*2 + (128*16 +  0)*2 + 4"},
    "sgemm_nt_32x128":       {"threads": 128, "sass": "sgemm_nt_32x128",       "params": "gemm",   "share": "(32*16 + 32)*2 + (128*16 + 32)*2 + 4"},
    "sgemm_nn_vec_32x128":   {"threads": 128, "sass": "sgemm_nn_32x128",       "params": "gemm",   "share": "(32*16 + 32)*2 + (128*16 +  0)*2 + 4", "args": {"vec": "1"}},
    "sgemm_nt_vec_32x128":   {"threads": 128, "sass": "sgemm_nt_32x128",       "params": "gemm",   "share": "(32*16 + 32)*2 + (128*16 + 32)*2 + 4", "args": {"vec": "1"}},
}

_params = {
    "fprop": [
        "float* param_Sum",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "int param_flags",
        "int param_offset_K",
        "int param_N",
        "int param_K",
        "int param_D",
        "int param_H",
        "int param_W",
        "int param_WN",
        "int param_HWN",
        "int param_DHWN",
        "int param_C",
        "int param_CRST",
        "int param_RST",
        "int param_RS",
        "int param_magic_RS",
        "int param_shift_RS",
        "int param_S",
        "int param_magic_S",
        "int param_shift_S",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "int param_str_d",
        "int param_str_h",
        "int param_str_w",
        "int param_Q",
        "int param_PQ",
        "int param_QN",
        "int param_PQN",
        "int param_MPQN",
        "int param_magic_Q",
        "int param_shift_Q",
        "int param_magic_PQ",
        "int param_shift_PQ",
    ],
    "fpropw": [
        "float* param_Sum",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "int param_flags",
        "int param_offset_K",
        "int param_K",
        "int param_C",
        "int param_Y",
        "int param_X",
        "int param_N",
        "int param_P",
        "int param_Q",
        "int param_XN",
        "int param_YXN",
        "int param_RSK",
        "int param_QN",
        "int param_PQN",
        "int param_shiftY",
        "int param_shiftX",
        "int param_shiftN",
        "int param_superY",
        "int param_superX",
        "int param_gridX",
        "int param_gridK",
        "int param_Y2",
        "int param_YXGK",
        "int param_X2GK",
        "int param_groupK",
        "int param_magic_YXGK",
        "int param_shift_YXGK",
        "int param_magic_X2GK",
        "int param_shift_X2GK",
        "int param_magic_groupK",
        "int param_shift_groupK",
        "int param_2XNp",
        "int param_XNp",
        "int param_4YXN_n3XNp",
        "int param_2SKp",
        "int param_SKp",
        "int param_4RSK_nSKp",
        "int param_pad_y",
        "int param_pad_x",
    ],
    "bprop1": [
        "float* param_Sum",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float para_alpha",
        "float param_beta",
        "int param_flags",
        "int param_offset_C",
        "int param_N",
        "int param_K",
        "int param_D",
        "int param_H",
        "int param_W",
        "int param_WN",
        "int param_HWN",
        "int param_DHWN",
        "int param_C",
        "int param_CRST",
        "int param_RST",
        "int param_magic_RST",
        "int param_shift_RST",
        "int param_RS",
        "int param_magic_RS",
        "int param_shift_RS",
        "int param_S",
        "int param_magic_S",
        "int param_shift_S",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "int param_str_d",
        "int param_str_h",
        "int param_str_w",
        "int param_Q",
        "int param_PQ",
        "int param_QN",
        "int param_PQN",
        "int param_MPQN",
        "int param_magic_Q",
        "int param_shift_Q",
        "int param_magic_PQ",
        "int param_shift_PQ",
        "int param_CRST8",
        "int param_MPQN8",
    ],
    "updat": [
        "float* param_Sum",
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "float param_beta",
        "int param_flags",
        "int param_offset_K",
        "int param_N",
        "int param_K",
        "int param_D",
        "int param_H",
        "int param_W",
        "int param_WN",
        "int param_HWN",
        "int param_DHWN",
        "int param_C",
        "int param_CRST",
        "int param_RST",
        "int param_magic_RST",
        "int param_shift_RST",
        "int param_RS",
        "int param_magic_RS",
        "int param_shift_RS",
        "int param_S",
        "int param_magic_S",
        "int param_shift_S",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "int param_str_d",
        "int param_str_h",
        "int param_str_w",
        "int param_P",
        "int param_Q",
        "int param_PQ",
        "int param_QN",
        "int param_PQN",
        "int param_MPQN",
        "int param_magic_Q",
        "int param_shift_Q",
        "int param_magic_PQ",
        "int param_shift_PQ",
        "int param_part_P",
        "int param_part_Q",
        "int param_part_PQ",
    ],
    "pool": [
        "float* param_O",
        "float* param_B",
        "float* param_I",
        "float param_alpha",
        "float param_beta",
        "int param_mode",
        "int param_N",
        "int param_W",
        "int param_H",
        "int param_D",
        "int param_C",
        "int param_WN",
        "int param_HWN",
        "int param_DHWN",
        "int param_P",
        "int param_Q",
        "int param_magic_P",
        "int param_shift_P",
        "int param_QN",
        "int param_PQN",
        "int param_MPQN",
        "int param_pad_j",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "int param_str_j",
        "int param_str_d",
        "int param_str_h",
        "int param_str_w",
        "int param_S",
        "int param_RS",
        "int param_RST",
        "int param_JRST",
        "int param_magic_S",
        "int param_shift_S",
        "int param_magic_RS",
        "int param_shift_RS",
        "int param_magic_RST",
        "int param_shift_RST",
        "int param_overlap",
    ],
    "pool2": [
        "float* param_E",
        "float* param_B",
        "float* param_I",
        "float param_alpha",
        "float param_beta",
        "int param_mode",
        "int param_N",
        "int param_W",
        "int param_H",
        "int param_D",
        "int param_C",
        "int param_WN",
        "int param_HWN",
        "int param_DHWN",
        "int param_magic_H",
        "int param_shift_H",
        "int param_pad_w",
        "int param_pad_h",
        "int param_pad_d",
        "int param_pad_c",
        "int param_str_w",
        "int param_str_h",
        "int param_str_d",
        "int param_str_c",
        "int param_magic_str_w",
        "int param_shift_str_w",
        "int param_magic_str_h",
        "int param_shift_str_h",
        "int param_magic_str_d",
        "int param_shift_str_d",
        "int param_magic_str_c",
        "int param_shift_str_c",
        "int param_S",
        "int param_R",
        "int param_T",
        "int param_J",
        "int param_RS",
        "int param_RST",
        "int param_JRST",
        "int param_magic_S",
        "int param_shift_S",
        "int param_magic_RS",
        "int param_shift_RS",
        "int param_magic_RST",
        "int param_shift_RST",
        "int param_Q",
        "int param_P",
        "int param_M",
        "int param_K",
        "int param_QN",
        "int param_PQN",
        "int param_MPQN",
    ],
    "gemm": [
        "float* param_C",
        "float* param_A",
        "float* param_B",
        "float param_alpha",
        "float param_beta",
        "int   param_flags",
        "int   param_lda",
        "int   param_ldb",
        "int   param_ldc",
        "int   param_m",
        "int   param_n",
        "int   param_k",
        "int   param_ldaz",
        "int   param_ldbz",
        "int   param_ldcz",
        "int   param_batch_loops",
    ],
}

_params["bprop"] = _params["fprop"] + [
        "int param_R",
        "int param_T",
        "int param_magic_str_w",
        "int param_shift_str_w",
        "int param_magic_str_h",
        "int param_shift_str_h",
        "int param_magic_str_d",
        "int param_shift_str_d",
    ]

_space_re = re.compile(r"\s+")

_share_template = r"""
    .shared .align 4 .b32 share[{0}];
"""

_kernel_template = r"""
.version 4.2
.target {0}
.address_size 64

.visible .entry  {1}(
{2}
)
.reqntid {3}
{{
{4}
    ret;
}}
"""

def get_ptx_file(kernel_name, arch):

    kernel_spec = kernels[kernel_name]
    thread_spec = kernel_spec["threads"]
    param_spec  = _params[kernel_spec["params"]]

    kernel_params = []
    for p in param_spec:
        ptype, pname = _space_re.split(p)

        if ptype[-1] == '*':
            ptype = '.u64'
        elif ptype == 'float':
            ptype = '.f32'
        else:
            ptype = '.u32'

        kernel_params.append("    .param %s %s" % (ptype, pname))

    kernel_params = ",\n".join(kernel_params)

    if "share" in kernel_spec:
        share = _share_template.format(eval(kernel_spec["share"]))
    else:
        share = ""

    kernel_text = _kernel_template.format(arch, kernel_name, kernel_params, thread_spec, share)
    kernel_ptx  = os.path.join(ptx_dir, kernel_name + ".ptx")

    current_text = ""
    if os.path.exists(kernel_ptx):
        f = open(kernel_ptx, "r")
        current_text = f.read()
        f.close()
    # only write out the kernel if text has changed.
    if kernel_text != current_text:
        f = open(kernel_ptx, "w")
        f.write(kernel_text)
        f.close()

    return kernel_ptx


@context_dependent_memoize
def get_kernel(kernel_name):

    #import ipdb; ipdb.set_trace()

    kernel_spec = kernels[kernel_name]
    params  = _params[kernel_spec["params"]]
    sig = ""
    for p in params:
        ptype, pname = _space_re.split(p)
        if ptype[-1] == '*':
            sig += "Q"
        elif ptype == 'float':
            sig += "f"
        else:
            sig += "I"

    module = drv.module_from_file(os.path.join(cubin_dir, kernel_name + ".cubin"))
    func   = module.get_function(kernel_name)
    func.prepare(sig)
    func.threads = kernel_spec["threads"]
    # print("Loaded: " + kernel_name)
    return func


def K_partitions(K, tiles):
    k = K
    partitions = []
    for tile_K in tiles:
        grid_K = (k + tiles[-1] - 1) // tile_K
        if grid_K > 0:
            partitions.append([tile_K, grid_K, K-k])
            k -= grid_K * tile_K
        if k <= 0:
            break
    return partitions


def xprop_conv_kernels(clss, op, tile_dim, tile_N, grid_N, K, tiles, PQM, RST, args):

    kernel_list = []
    for tile_K, grid_K, offset_K in K_partitions(K, tiles):

        kernel_name = "%s_%s_%s%d_N%d" % (clss, op, tile_dim, tile_K, tile_N)

        block = (kernels[kernel_name]["threads"], 1, 1)
        if RST > 1:
            grid = (PQM, grid_K, grid_N)
        else:
            grid = (grid_K, grid_N, PQM)

        kernel_list.append([kernel_name, grid, block, offset_K, args])

    return kernel_list


def update_grid(kernel_name, base_blocks, P, Q, SM_count):

    threads   = kernels[kernel_name]["threads"]
    occupancy = kernels[kernel_name]["occupancy"]

    # warps per scheduler for one block
    occ_per_block = threads / (32.0 * 4.0 * SM_count)

    grid = []
    for p in range(1, P+1):
        for q in range(1, Q+1):

            occup  = p*q*base_blocks * occ_per_block
            groups = occup / occupancy
            slots  = ceil(groups)

            # This is a heuristic that keeps the balance of work accross the SMs
            # while also maximizing the work that each block does
            heuristic = min(abs(x - slots) for x in range(4, 8)) + (slots - groups) / 100.0

            grid.append((p, q, heuristic))

    grid.sort(key=lambda x: x[-1])

    return (grid[0][0], grid[0][1], threads)
