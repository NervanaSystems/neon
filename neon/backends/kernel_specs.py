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
import subprocess
import pycuda.driver as drv
from pycuda.tools import context_dependent_memoize

# helpful for kernel development
debug = 0

base_dir  = os.path.dirname(__file__)
ptx_dir   = os.path.join(base_dir, "kernels", "ptx")
sass_dir  = os.path.join(base_dir, "kernels", "sass")
pre_dir   = os.path.join(base_dir, "kernels", "pre")
cubin_dir = os.path.join(base_dir, "kernels", "cubin")
dump_dir  = os.path.join(base_dir, "kernels", "dump")


kernels = {

    #TODO: perhaps get rid of these old conv kernels
    "hconv_bprop_C128_N128":  {"threads": 256, "sass": "hconv_xprop_X128_N128", "params": "bprop",  "share": "128*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "hconv_bprop_C128_N64":   {"threads": 128, "sass": "hconv_xprop_X128_N64",  "params": "bprop",  "share": "128*8*2 +  64*8*2 + 8", "args": {"prop": "b"}},
    "hconv_bprop_C32_N128":   {"threads":  64, "sass": "hconv_xprop_X32_N128",  "params": "bprop",  "share": " 32*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "hconv_bprop_C64_N128":   {"threads": 128, "sass": "hconv_xprop_X64_N128",  "params": "bprop",  "share": " 64*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "hconv_bprop_C64_N64":    {"threads":  64, "sass": "hconv_xprop_X64_N64",   "params": "bprop",  "share": " 64*8*2 +  64*8*2 + 8", "args": {"prop": "b"}},
    "hconv_fprop_K128_N128":  {"threads": 256, "sass": "hconv_xprop_X128_N128", "params": "fprop",  "share": "128*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "hconv_fprop_K128_N64":   {"threads": 128, "sass": "hconv_xprop_X128_N64",  "params": "fprop",  "share": "128*8*2 +  64*8*2 + 8", "args": {"prop": "f"}},
    "hconv_fprop_K32_N128":   {"threads":  64, "sass": "hconv_xprop_X32_N128",  "params": "fprop",  "share": " 32*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "hconv_fprop_K64_N128":   {"threads": 128, "sass": "hconv_xprop_X64_N128",  "params": "fprop",  "share": " 64*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "hconv_fprop_K64_N64":    {"threads":  64, "sass": "hconv_xprop_X64_N64",   "params": "fprop",  "share": " 64*8*2 +  64*8*2 + 8", "args": {"prop": "f"}},

    "sconv_bprop_C128_N128":  {"threads": 256, "sass": "sconv_xprop_X128_N128", "params": "bprop",  "share": "128*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "sconv_bprop_C128_N64":   {"threads": 128, "sass": "sconv_xprop_X128_N64",  "params": "bprop",  "share": "128*8*2 +  64*8*2 + 8", "args": {"prop": "b"}},
    "sconv_bprop_C32_N128":   {"threads":  64, "sass": "sconv_xprop_X32_N128",  "params": "bprop",  "share": " 32*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "sconv_bprop_C64_N128":   {"threads": 128, "sass": "sconv_xprop_X64_N128",  "params": "bprop",  "share": " 64*8*2 + 128*8*2 + 8", "args": {"prop": "b"}},
    "sconv_bprop_C64_N64":    {"threads":  64, "sass": "sconv_xprop_X64_N64",   "params": "bprop",  "share": " 64*8*2 +  64*8*2 + 8", "args": {"prop": "b"}},
    "sconv_fprop_K128_N128":  {"threads": 256, "sass": "sconv_xprop_X128_N128", "params": "fprop",  "share": "128*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "sconv_fprop_K128_N64":   {"threads": 128, "sass": "sconv_xprop_X128_N64",  "params": "fprop",  "share": "128*8*2 +  64*8*2 + 8", "args": {"prop": "f"}},
    "sconv_fprop_K32_N128":   {"threads":  64, "sass": "sconv_xprop_X32_N128",  "params": "fprop",  "share": " 32*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "sconv_fprop_K64_N128":   {"threads": 128, "sass": "sconv_xprop_X64_N128",  "params": "fprop",  "share": " 64*8*2 + 128*8*2 + 8", "args": {"prop": "f"}},
    "sconv_fprop_K64_N64":    {"threads":  64, "sass": "sconv_xprop_X64_N64",   "params": "fprop",  "share": " 64*8*2 +  64*8*2 + 8", "args": {"prop": "f"}},

    "sgemm_nn_128x128": {"threads": 256, "sass": "sgemm_nn_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "sgemm_nt_128x128": {"threads": 256, "sass": "sgemm_nt_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "sgemm_tn_128x128": {"threads": 256, "sass": "sgemm_tn_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "hgemm_nn_128x128": {"threads": 256, "sass": "hgemm_nn_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "hgemm_nt_128x128": {"threads": 256, "sass": "hgemm_nt_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "hgemm_tn_128x128": {"threads": 256, "sass": "hgemm_tn_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},

    "sgemm_nn_128x64":  {"threads": 128, "sass": "sgemm_nn_128x64",  "params": "gemm", "share": "128*8*2 +  64*8*2 + 4"},
    "sgemm_tn_128x64":  {"threads": 128, "sass": "sgemm_tn_128x64",  "params": "gemm", "share": "128*8*2 +  64*8*2 + 4"},
    "hgemm_nn_128x64":  {"threads": 128, "sass": "hgemm_nn_128x64",  "params": "gemm", "share": "128*8*2 +  64*8*2 + 4"},
    "hgemm_tn_128x64":  {"threads": 128, "sass": "hgemm_tn_128x64",  "params": "gemm", "share": "128*8*2 +  64*8*2 + 4"},

    "sgemm_nn_128x32":  {"threads": 128, "sass": "sgemm_nn_128x32",  "params": "gemm", "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "sgemm_tn_128x32":  {"threads": 128, "sass": "sgemm_tn_128x32",  "params": "gemm", "share": "(128*16 +  0)*2 + 32*16*2 + 4"},
    "hgemm_nn_128x32":  {"threads": 128, "sass": "hgemm_nn_128x32",  "params": "gemm", "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "hgemm_tn_128x32":  {"threads": 128, "sass": "hgemm_tn_128x32",  "params": "gemm", "share": "(128*16 +  0)*2 + 32*16*2 + 4"},

    "sgemm_nn_32x128":  {"threads": 128, "sass": "sgemm_nn_32x128",  "params": "gemm", "share": "(32*16 + 32)*2 + (128*16 +  0)*2 + 4"},
    "sgemm_nt_32x128":  {"threads": 128, "sass": "sgemm_nt_32x128",  "params": "gemm", "share": "(32*16 + 32)*2 + (128*16 + 32)*2 + 4"},
    "hgemm_nn_32x128":  {"threads": 128, "sass": "hgemm_nn_32x128",  "params": "gemm", "share": "(32*16 + 32)*2 + (128*16 +  0)*2 + 4"},
    "hgemm_nt_32x128":  {"threads": 128, "sass": "hgemm_nt_32x128",  "params": "gemm", "share": "(32*16 + 32)*2 + (128*16 + 32)*2 + 4"},

    "sconv_winograd_2x2_3x3_32x32":   {"threads": 256, "sass": "xconv_winograd_2x2_3x3_32x32",   "params": "fpropw", "share": "512*4*4", "args": {"type": "s"}},
    "hconv_winograd_2x2_3x3_32x32":   {"threads": 256, "sass": "xconv_winograd_2x2_3x3_32x32",   "params": "fpropw", "share": "512*4*4", "args": {"type": "h"}},
    "sconv_winograd_3x3_2x2_32x32":   {"threads": 256, "sass": "xconv_winograd_3x3_2x2_32x32",   "params": "updatw", "share": "(512*4 + 32)*4 + 8", "args": {"type": "s"}},
    "hconv_winograd_3x3_2x2_32x32":   {"threads": 256, "sass": "xconv_winograd_3x3_2x2_32x32",   "params": "updatw", "share": "(512*4 + 32)*4 + 8", "args": {"type": "h"}},

    "sconv_winograd_4x4_3x3_32x32":   {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32",   "params": "fpropw4",  "share": "32*36*2*4 + 8", "args": {"type": "s"}},
    "hconv_winograd_4x4_3x3_32x32":   {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32",   "params": "fpropw4",  "share": "32*36*2*4 + 8", "args": {"type": "h"}},
    "sconv_winograd_4x4_3x3_32x32_X": {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32_X", "params": "fpropw4X", "share": "32*36*2*4 + 8", "args": {"type": "s"}},
    "hconv_winograd_4x4_3x3_32x32_X": {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32_X", "params": "fpropw4X", "share": "32*36*2*4 + 8", "args": {"type": "h"}},
    "sconv_winograd_3x3_4x4_32x32":   {"threads": 640, "sass": "xconv_winograd_3x3_4x4_32x32",   "params": "updatw4",  "share": "32*36*2*4 + 8", "args": {"type": "s"}},
    "hconv_winograd_3x3_4x4_32x32":   {"threads": 640, "sass": "xconv_winograd_3x3_4x4_32x32",   "params": "updatw4",  "share": "32*36*2*4 + 8", "args": {"type": "h"}},

    "sconv_direct_fprop_64x32": {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "fprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"s","prop":"f"}},
    "sconv_direct_bprop_64x32": {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "bprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"s","prop":"b"}},
    "hconv_direct_fprop_64x32": {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "fprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"h","prop":"f"}},
    "hconv_direct_bprop_64x32": {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "bprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"h","prop":"b"}},
    "sconv_direct_updat_64x32": {"threads": 128, "sass": "xconv_direct_updat_64x32",  "params": "updat2", "share": "(32 + 64)*33*2 + 8", "args": {"type": "s",}},
    "hconv_direct_updat_64x32": {"threads": 128, "sass": "xconv_direct_updat_64x32",  "params": "updat2", "share": "(32 + 64)*33*2 + 8", "args": {"type": "h",}},
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
        "int param_KRST",
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
    "fprop2": [
        "float* param_Sum",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "int param_flags",
        "int param_C",
        "int param_D",
        "int param_H",
        "int param_W",
        "int param_N",
        "int param_K",
        "int param_M",
        "int param_P",
        "int param_Q",
        "int param_str_d",
        "int param_str_h",
        "int param_str_w",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "int param_DHWN",
        "int param_HWN",
        "int param_WN",
        "int param_MPQN",
        "int param_PQN",
        "int param_QN",
        "int param_PQnk",
        "int param_Qnk",
        "int param_nk",
        "int param_n",
        "int param_k",
        "int param_magic_PQnk",
        "int param_shift_PQnk",
        "int param_magic_Qnk",
        "int param_shift_Qnk",
        "int param_magic_nk",
        "int param_shift_nk",
        "int param_magic_k",
        "int param_shift_k",
        "int param_Km32",
        "int param_K32p",
        "int param_TRSK",
        "int param_TRS",
        "int param_RS",
        "int param_S",
        "int param_magic_RS",
        "int param_shift_RS",
        "int param_magic_S",
        "int param_shift_S",
        "int param_gridP2",
        "int param_gridQ",
        "int param_superM",
        "int param_superP",
        "int param_superQ",
        "int param_superN",
        "int param_shiftM",
        "int param_shiftP",
        "int param_shiftQ",
        "int param_shiftN",
        "int param_SuperM",
        "int param_SuperP",
        "int param_SuperQ",
        "int param_SuperN",
    ],
    "updat2": [
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "int param_C",
        "int param_D",
        "int param_H",
        "int param_W",
        "int param_N",
        "int param_K",
        "int param_M",
        "int param_P",
        "int param_Q",
        "int param_str_d",
        "int param_str_h",
        "int param_str_w",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "int param_DHWN",
        "int param_HWN",
        "int param_WN",
        "int param_MPQN16p",
        "int param_MPQN",
        "int param_PQN",
        "int param_QN",
        "int param_PQkc",
        "int param_Qkc",
        "int param_kc",
        "int param_c",
        "int param_k",
        "int param_magic_PQkc",
        "int param_shift_PQkc",
        "int param_magic_Qkc",
        "int param_shift_Qkc",
        "int param_magic_kc",
        "int param_shift_kc",
        "int param_magic_c",
        "int param_shift_c",
        "int param_CRSTK",
        "int param_CRST",
        "int param_TRS",
        "int param_RS",
        "int param_S",
        "int param_magic_TRS",
        "int param_shift_TRS",
        "int param_magic_RS",
        "int param_shift_RS",
        "int param_magic_S",
        "int param_shift_S",
        "int param_superM",
        "int param_superP",
        "int param_superQ",
        "int param_superN",
        "int param_shiftM",
        "int param_shiftP",
        "int param_shiftQ",
        "int param_strideP",
        "int param_strideQ",
        "int param_stridePQ",
        "int param_gridP",
        "int param_gridQ",
        "int param_loopX",
        "int param_loopXp",
        "int param_loopQ",
        "int param_loopQp",
        "int param_loopN",
        "int param_loopNp",
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
    "fpropw": [
        "float* param_Sum",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "int param_flags",
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
        "int param_4C_batchKp",
        "int param_pad_y",
        "int param_pad_x",
    ],
    "fpropw4X": [
        "float* param_Sum",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "int param_flags",
        "int param_C",
        "int param_K",
        "int param_N",
        "int param_Xk",
        "int param_k",
        "int param_magic_Xk",
        "int param_shift_Xk",
        "int param_magic_k",
        "int param_shift_k",
        "int param_C_1152",
        "int param_GXS_C_1152",
        "int param_GYS_GXS_C_1152",
        "int param_P",
        "int param_Q",
        "int param_QN",
        "int param_PQN",
        "int param_Np",
        "int param_QNp",
        "int param_QN3p",
        "int param_PQN1_QN3p",
        "int param_PQN15_QN3p",
        "int param_maskN",
        "int param_shiftX",
        "int param_shiftY",
        "int param_superX",
        "int param_superY",
    ],
    "fpropw4": [
        "float* param_Sum",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "int param_flags",
        "int param_C",
        "int param_K",
        "int param_N",
        "int param_Y",
        "int param_X",
        "int param_YXN",
        "int param_XN",
        "int param_Y2",
        "int param_GX",
        "int param_Xk",
        "int param_k",
        "int param_magic_Xk",
        "int param_shift_Xk",
        "int param_magic_k",
        "int param_shift_k",
        "int param_P",
        "int param_Q",
        "int param_QN",
        "int param_PQN",
        "int param_Np",
        "int param_QNp",
        "int param_QN3p",
        "int param_PQN1_QN3p",
        "int param_PQN15_QN3p",
        "int param_maskN",
        "int param_shiftX",
        "int param_shiftY",
        "int param_superX",
        "int param_superY",
        "int param_pad_x",
        "int param_pad_y",
        "int param_RSK",
        "int param_RSK2p",
        "int param_YXN2p",
    ],
    "updatw": [
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "int param_Y",
        "int param_X",
        "int param_P",
        "int param_Q",
        "int param_C",
        "int param_K",
        "int param_N",
        "int param_pad_y",
        "int param_pad_x",
        "int param_GY",
        "int param_GX",
        "int param_GYS",
        "int param_GXS",
        "int param_shiftYI",
        "int param_shiftXI",
        "int param_superYI",
        "int param_superXI",
        "int param_superNI",
        "int param_shiftY",
        "int param_shiftX",
        "int param_superY",
        "int param_superX",
        "int param_superN",
        "int param_loopXI",
        "int param_loopX",
        "int param_loopN",
        "int param_strideY",
        "int param_strideX",
        "int param_XN",
        "int param_YXN",
        "int param_QN",
        "int param_PQN",
        "int param_SK",
        "int param_RSK",
        "int param_Np",
        "int param_XNp",
        "int param_2XNp",
        "int param_QNp",
        "int param_CPQkc",
        "int param_PQkc",
        "int param_Qkc",
        "int param_kc",
        "int param_c",
        "int param_k",
        "int param_magic_CPQkc",
        "int param_shift_CPQkc",
        "int param_magic_PQkc",
        "int param_shift_PQkc",
        "int param_magic_Qkc",
        "int param_shift_Qkc",
        "int param_magic_kc",
        "int param_shift_kc",
        "int param_magic_c",
        "int param_shift_c",
        "int param_CRSK",
    ],
    "updatw4": [
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "int param_K",
        "int param_C",
        "int param_k",
        "int param_c",
        "int param_kc",
        "int param_magic_kc",
        "int param_shift_kc",
        "int param_magic_c",
        "int param_shift_c",
        "int param_YXN2",
        "int param_sYXN",
        "int param_magic_sYXN",
        "int param_shift_sYXN",
        "int param_stride_YXNp",
        "int param_YXN",
        "int param_YXN_1152",
        "int param_RSK",
        "int param_CRSK",
        "int param_Kp",
        "int param_SKp",
        "int param_RSK15_SK2p",
    ],
}

_params["bprop2"] = _params["fprop2"] + [
        "int param_magic_str_d",
        "int param_shift_str_d",
        "int param_magic_str_h",
        "int param_shift_str_h",
        "int param_magic_str_w",
        "int param_shift_str_w",
    ]

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

// args: {5}

.visible .entry  {1}(
{2}
)
.reqntid {3}
{{
{4}
    ret;
}}
"""

def get_ptx_file(kernel_spec, kernel_name, arch):

    thread_spec = kernel_spec["threads"]
    args_spec   = str(kernel_spec.get("args",""))
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

    if not os.path.exists(ptx_dir):
        os.mkdir(ptx_dir)

    kernel_text = _kernel_template.format(arch, kernel_name, kernel_params, thread_spec, share, args_spec)
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


include_re = re.compile(r'^<INCLUDE\s+file="([^"]+)"\s*/>')

def extract_includes(name, includes=None):
    if not includes:
        includes = list()
    sass_file = os.path.join(sass_dir, name)
    includes.append((sass_file, os.path.getmtime(sass_file)))
    for line in open(sass_file, "r"):
        match = include_re.search(line)
        if match:
            extract_includes(match.group(1), includes)
    return includes

def run_command(cmdlist):
    cmdline = " ".join(cmdlist)
    proc = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    code = proc.wait()
    if code:
        error = proc.stderr.read()
        raise RuntimeError("Error(%d):\n%s\n%s" % (code, cmdline, error))
    if debug:
        out = proc.stdout.read()
        print cmdline
        if out: print out

@context_dependent_memoize
def get_kernel(base_name, options=None):

    attributes = drv.Context.get_device().get_attributes()
    major = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MAJOR]
    minor = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MINOR]
    if major < 5:
        raise RuntimeError("sass kernels require Maxwell or greater class hardware")

    arch = "sm_%d%d" % (major, minor)

    maxas_i = ["maxas.pl -i"]
    maxas_p = ["maxas.pl -p"]

    kernel_spec = kernels[base_name]
    kernel_name = base_name

    if "args" in kernel_spec:
        for pair in kernel_spec["args"].items():
            maxas_i.append("-D%s %s" % pair)
            maxas_p.append("-D%s %s" % pair)

    if options is not None:
        for opt in options:
            if type(opt) is tuple:
                maxas_i.append("-D%s %s" % opt)
                maxas_p.append("-D%s %s" % opt)
                kernel_name += "_%s%s" % opt
            else:
                maxas_i.append("-D%s 1" % opt)
                maxas_p.append("-D%s 1" % opt)
                kernel_name += "_%s" % opt

    maxas_i.insert(1, "-k " + kernel_name)

    sass_name  = kernel_spec["sass"] + ".sass"
    cubin_name = kernel_name + ".cubin"

    ptx_file   = get_ptx_file(kernel_spec, kernel_name, arch)
    sass_file  = os.path.join(sass_dir, sass_name)
    cubin_file = os.path.join(cubin_dir, cubin_name)

    if not os.path.exists(cubin_dir):
        os.mkdir(cubin_dir)
    if not os.path.exists(sass_file):
        raise RuntimeError("Missing: %s for kernel: %s" % (sass_name, kernel_name))

    ptx_age   = os.path.getmtime(ptx_file)
    cubin_age = os.path.getmtime(cubin_file) if os.path.exists(cubin_file) else 0

    if ptx_age > cubin_age:
        run_command([ "ptxas -v -arch", arch, "-o", cubin_file, ptx_file ])
        cubin_age = 0

    includes = extract_includes(sass_name)

    for include, include_age in includes:
        if include_age > cubin_age:
            run_command(maxas_i + [sass_file, cubin_file])
            cubin_age = include_age
            break

    if debug:
        if not os.path.exists(pre_dir):  os.mkdir(pre_dir)
        if not os.path.exists(dump_dir): os.mkdir(dump_dir)
        pre_file  = os.path.join(pre_dir,  kernel_name + "_pre.sass")
        dump_file = os.path.join(dump_dir, kernel_name + "_dump.sass")
        pre_age   = os.path.getmtime(pre_file)  if os.path.exists(pre_file)  else 0
        dump_age  = os.path.getmtime(dump_file) if os.path.exists(dump_file) else 0

        for include, include_age in includes:
            if include_age > pre_age:
                run_command(maxas_p + [sass_file, pre_file])
                break

        if cubin_age > dump_age:
            run_command(["nvdisasm -raw", cubin_file, ">", dump_file])

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


