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
cu_dir    = os.path.join(base_dir, "kernels", "cu")
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
        "float*     param_Sum",
        "{0}*       param_O",
        "const {0}* param_I",
        "const {0}* param_F",
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
    "bprop1": [
        "float*     param_Sum",
        "{0}*       param_O",
        "const {0}* param_I",
        "const {0}* param_F",
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
        "float*     param_Sum",
        "float*     param_F",
        "const {0}* param_I",
        "const {0}* param_E",
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
        "{0}*       param_O",
        "{0}*       param_B",
        "const {0}* param_I",
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
        "{0}*       param_E",
        "{0}*       param_B",
        "const {0}* param_I",
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
        "{0}*       param_C",
        "const {0}* param_A",
        "const {0}* param_B",
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

_pspace = re.compile(r"\s+")
_dtypes = {"s": "float", "h": "unsigned short"}

_share_template = r"""
    __shared__ float share[{0}];
    *{1} = share[0];
"""

_no_share_template = r"""
    *{0} = 0;
"""

_kernel_template = r"""
extern "C" __global__ void {0}(
{1}
) {{
{2}
}}
"""


def get_cu_file(kernel_name):

    kernel_spec = kernels[kernel_name]
    kernel_type = _dtypes[kernel_name[0]]
    param_spec  = _params[kernel_spec["params"]]

    kernel_params = []
    for p in param_spec:
        kernel_params.append("    " + p.format(kernel_type))
    kernel_params = ",\n".join(kernel_params)

    # the first param is always an output pointer
    out_param = _pspace.split(param_spec[0])[1]

    if "share" in kernel_spec:
        body = _share_template.format(kernel_spec["share"], out_param)
    else:
        body = _no_share_template.format(out_param)

    kernel_text = _kernel_template.format(kernel_name, kernel_params, body)
    kernel_cu   = os.path.join(cu_dir, kernel_name + ".cu")

    current_text = ""
    if os.path.exists(kernel_cu):
        f = open(kernel_cu, "r")
        current_text = f.read()
        f.close()
    # only write out the kernel if text has changed.
    if kernel_text != current_text:
        f = open(kernel_cu, "w")
        f.write(kernel_text)
        f.close()

    return kernel_cu


@context_dependent_memoize
def get_kernel(kernel_name):

    kernel_spec = kernels[kernel_name]
    params = _params[kernel_spec["params"]]
    sig = ""
    for p in params:
        if p[0:4] == "int ":
            sig += "I"
        elif p[0:6] == "float ":
            sig += "f"
        else:
            sig += "P"

    module = drv.module_from_file(os.path.join(cubin_dir, kernel_name + ".cubin"))
    func   = module.get_function(kernel_name)
    func.prepare(sig)
    # print("Loaded: " + kernel)
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
