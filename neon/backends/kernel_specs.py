# Copyright 2014-2016 Nervana Systems Inc. All rights reserved.
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
from builtins import str
import re
import os.path
import subprocess
import pycuda.driver as drv
from pycuda.tools import context_dependent_memoize
from neon import logger as neon_logger
from neon.util.persist import get_cache_dir

# helpful for kernel development
debug = 0

base_dir  = os.path.dirname(__file__)
maxas_dir = os.path.join(base_dir, "kernels", "maxas")
sass_dir  = os.path.join(base_dir, "kernels", "sass")

ptx_dir   = get_cache_dir(['kernels', 'ptx'])
pre_dir   = get_cache_dir(['kernels', 'pre'])
cubin_dir = get_cache_dir(['kernels', 'cubin'])
dump_dir  = get_cache_dir(['kernels', 'dump'])

kernels = {
    "sconv_direct_fprop_128x128": {"threads": 256, "sass": "sconv_xprop_X128_N128", "params": "fprop",  "share": "128*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_128x128": {"threads": 256, "sass": "sconv_xprop_X128_N128", "params": "bprop",  "share": "128*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_128x128": {"threads": 256, "sass": "hconv_xprop_X128_N128", "params": "fprop",  "share": "128*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_128x128": {"threads": 256, "sass": "hconv_xprop_X128_N128", "params": "bprop",  "share": "128*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_64x128":  {"threads": 128, "sass": "sconv_xprop_X64_N128",  "params": "fprop",  "share": " 64*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_64x128":  {"threads": 128, "sass": "sconv_xprop_X64_N128",  "params": "bprop",  "share": " 64*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_64x128":  {"threads": 128, "sass": "hconv_xprop_X64_N128",  "params": "fprop",  "share": " 64*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_64x128":  {"threads": 128, "sass": "hconv_xprop_X64_N128",  "params": "bprop",  "share": " 64*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_32x128":  {"threads":  64, "sass": "sconv_xprop_X32_N128",  "params": "fprop",  "share": " 32*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_32x128":  {"threads":  64, "sass": "sconv_xprop_X32_N128",  "params": "bprop",  "share": " 32*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_32x128":  {"threads":  64, "sass": "hconv_xprop_X32_N128",  "params": "fprop",  "share": " 32*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_32x128":  {"threads":  64, "sass": "hconv_xprop_X32_N128",  "params": "bprop",  "share": " 32*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_128x64":  {"threads": 128, "sass": "sconv_xprop_X128_N64",  "params": "fprop",  "share": "128*8*2 +  64*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_128x64":  {"threads": 128, "sass": "sconv_xprop_X128_N64",  "params": "bprop",  "share": "128*8*2 +  64*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_128x64":  {"threads": 128, "sass": "hconv_xprop_X128_N64",  "params": "fprop",  "share": "128*8*2 +  64*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_128x64":  {"threads": 128, "sass": "hconv_xprop_X128_N64",  "params": "bprop",  "share": "128*8*2 +  64*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_64x64":   {"threads":  64, "sass": "sconv_xprop_X64_N64",   "params": "fprop",  "share": " 64*8*2 +  64*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_64x64":   {"threads":  64, "sass": "sconv_xprop_X64_N64",   "params": "bprop",  "share": " 64*8*2 +  64*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_64x64":   {"threads":  64, "sass": "hconv_xprop_X64_N64",   "params": "fprop",  "share": " 64*8*2 +  64*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_64x64":   {"threads":  64, "sass": "hconv_xprop_X64_N64",   "params": "bprop",  "share": " 64*8*2 +  64*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_64x32":   {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "fprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"s","prop":"f"}},
    "sconv_direct_bprop_64x32":   {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "bprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"s","prop":"b"}},
    "hconv_direct_fprop_64x32":   {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "fprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"h","prop":"f"}},
    "hconv_direct_bprop_64x32":   {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "bprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"h","prop":"b"}},
    "sconv_direct_updat_64x32":   {"threads": 128, "sass": "xconv_direct_updat_64x32",  "params": "updat2", "share": "(32 + 64)*33*2 + 8", "args": {"type": "s",}},
    "hconv_direct_updat_64x32":   {"threads": 128, "sass": "xconv_direct_updat_64x32",  "params": "updat2", "share": "(32 + 64)*33*2 + 8", "args": {"type": "h",}},


    "sconv_winograd_2x2_3x3_32x32":   {"threads": 256, "sass": "xconv_winograd_2x2_3x3_32x32",   "params": "fpropw", "share": "512*4*4", "args": {"type": "s"}},
    "hconv_winograd_2x2_3x3_32x32":   {"threads": 256, "sass": "xconv_winograd_2x2_3x3_32x32",   "params": "fpropw", "share": "512*4*4", "args": {"type": "h"}},
    "sconv_winograd_3x3_2x2_32x32":   {"threads": 256, "sass": "xconv_winograd_3x3_2x2_32x32",   "params": "updatw", "share": "(512*4 + 32)*4 + 8", "args": {"type": "s"}},
    "hconv_winograd_3x3_2x2_32x32":   {"threads": 256, "sass": "xconv_winograd_3x3_2x2_32x32",   "params": "updatw", "share": "(512*4 + 32)*4 + 8", "args": {"type": "h"}},

    "sconv_winograd_4x4_3x3_32x32":   {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32",   "params": "fpropw4",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "s"}},
    "hconv_winograd_4x4_3x3_32x32":   {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32",   "params": "fpropw4",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "h"}},
    "sconv_winograd_4x4_3x3_32x32_X": {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32_X", "params": "fpropw4X", "share": "32*36*2*4 + 64 + 8", "args": {"type": "s"}},
    "hconv_winograd_4x4_3x3_32x32_X": {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32_X", "params": "fpropw4X", "share": "32*36*2*4 + 64 + 8", "args": {"type": "h"}},
    "sconv_winograd_3x3_4x4_32x32":   {"threads": 640, "sass": "xconv_winograd_3x3_4x4_32x32",   "params": "updatw4",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "s"}},
    "hconv_winograd_3x3_4x4_32x32":   {"threads": 640, "sass": "xconv_winograd_3x3_4x4_32x32",   "params": "updatw4",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "h"}},

    "sconv_winograd_2x2_5x5_32x32":   {"threads": 640, "sass": "xconv_winograd_2x2_5x5_32x32",   "params": "fpropw5",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "s"}},
    "hconv_winograd_2x2_5x5_32x32":   {"threads": 640, "sass": "xconv_winograd_2x2_5x5_32x32",   "params": "fpropw5",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "h"}},


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

    "hgemm_nt_32x32": {"threads": 128, "sass": "hgemm_nt_32x32", "params": "gemm", "share": "32*65*4 + 4" },
    "hgemm_nt_16x64": {"threads": 128, "sass": "hgemm_nt_16x64", "params": "gemm", "share": "(16*64 + 32)*2 + (64*64 + 32)*2 + 4" },
    "hgemm_nn_32x64": {"threads": 128, "sass": "hgemm_nn_32x64", "params": "gemm", "share": "32*33*2 + 64*32*2 + 2048" },  #artificially limit occpancy
    "hgemm_nn_16x64": {"threads": 128, "sass": "hgemm_nn_16x64", "params": "gemm", "share": "(16*64 + 32)*2 + 64*64*2 + 4" },

    "sgemm_rnn_nn_128x32":    {"threads": 128, "sass": "sgemm_nn_rnn_128x32",       "params": "gemm_rnn",   "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "sgemm_rnn_nn_vec_128x32":    {"threads": 128, "sass": "sgemm_nn_rnn_128x32",       "params": "gemm_rnn",   "share": "(128*16 + 32)*2 + 32*16*2 + 4", "args": {"vec": "1"}},

    "sgemm_rnn_bprop_tn_128x32":    {"threads": 128, "sass": "sgemm_tn_rnn_bprop_128x32",       "params": "gemm_rnn_bprop",   "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "sgemm_rnn_bprop_tn_vec_128x32":    {"threads": 128, "sass": "sgemm_tn_rnn_bprop_128x32",       "params": "gemm_rnn_bprop",   "share": "(128*16 + 32)*2 + 32*16*2 + 4", "args": {"vec": "1"}},

    "persistent_rnn_fprop": {"threads": 256, "sass": "persistent_rnn_fprop", "params": "rnn_fprop", "share": "(64*48) + 4"},
    "persistent_rnn_bprop": {"threads": 256, "sass": "persistent_rnn_bprop", "params": "rnn_bprop", "share": "(64*48) + 4"},
}

_params = {
    "fprop": [
        "float* param_Sum",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_N",
        "unsigned param_K",
        "unsigned param_D",
        "unsigned param_H",
        "unsigned param_W",
        "unsigned param_WN",
        "unsigned param_HWN",
        "unsigned param_DHWN",
        "unsigned param_C",
        "unsigned param_KRST",
        "unsigned param_RST",
        "unsigned param_RS",
        "unsigned param_T",
        "unsigned param_R",
        "unsigned param_S",
        "unsigned param_magic_RS",
        "unsigned param_shift_RS",
        "unsigned param_magic_S",
        "unsigned param_shift_S",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "unsigned param_str_d",
        "unsigned param_str_h",
        "unsigned param_str_w",
        "unsigned param_dil_d",
        "unsigned param_dil_h",
        "unsigned param_dil_w",
        "unsigned param_P2",
        "unsigned param_Q",
        "unsigned param_PQk",
        "unsigned param_Qk",
        "unsigned param_k",
        "unsigned param_magic_PQk",
        "unsigned param_shift_PQk",
        "unsigned param_magic_Qk",
        "unsigned param_shift_Qk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_MPQN",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
        "unsigned param_gridMPQN",
    ],
    "fprop2": [
        "float* param_Sum",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_D",
        "unsigned param_H",
        "unsigned param_W",
        "unsigned param_N",
        "unsigned param_K",
        "unsigned param_M",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_str_d",
        "unsigned param_str_h",
        "unsigned param_str_w",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "unsigned param_dil_d",
        "unsigned param_dil_h",
        "unsigned param_dil_w",
        "unsigned param_DHWN",
        "unsigned param_HWN",
        "unsigned param_WN",
        "unsigned param_MPQN",
        "unsigned param_PQN",
        "unsigned param_QN",
        "unsigned param_PQnk",
        "unsigned param_Qnk",
        "unsigned param_nk",
        "unsigned param_n",
        "unsigned param_k",
        "unsigned param_magic_PQnk",
        "unsigned param_shift_PQnk",
        "unsigned param_magic_Qnk",
        "unsigned param_shift_Qnk",
        "unsigned param_magic_nk",
        "unsigned param_shift_nk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_Km32",
        "unsigned param_K32p",
        "unsigned param_TRSK",
        "unsigned param_TRS",
        "unsigned param_RS",
        "unsigned param_S",
        "unsigned param_magic_RS",
        "unsigned param_shift_RS",
        "unsigned param_magic_S",
        "unsigned param_shift_S",
        "unsigned param_gridP2",
        "unsigned param_gridQ",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
        "unsigned param_gridMPQN",
        "unsigned param_superM",
        "unsigned param_superP",
        "unsigned param_superQ",
        "unsigned param_superN",
        "unsigned param_shiftM",
        "unsigned param_shiftP",
        "unsigned param_shiftQ",
        "unsigned param_shiftN",
        "unsigned param_SuperM",
        "unsigned param_SuperP",
        "unsigned param_SuperQ",
        "unsigned param_SuperN",
    ],
    "updat2": [
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "unsigned param_C",
        "unsigned param_D",
        "unsigned param_H",
        "unsigned param_W",
        "unsigned param_N",
        "unsigned param_K",
        "unsigned param_M",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_str_d",
        "unsigned param_str_h",
        "unsigned param_str_w",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "unsigned param_dil_d",
        "unsigned param_dil_h",
        "unsigned param_dil_w",
        "unsigned param_DHWN",
        "unsigned param_HWN",
        "unsigned param_WN",
        "unsigned param_MPQN16p",
        "unsigned param_MPQN",
        "unsigned param_PQN",
        "unsigned param_QN",
        "unsigned param_PQkc",
        "unsigned param_Qkc",
        "unsigned param_kc",
        "unsigned param_c",
        "unsigned param_k",
        "unsigned param_magic_PQkc",
        "unsigned param_shift_PQkc",
        "unsigned param_magic_Qkc",
        "unsigned param_shift_Qkc",
        "unsigned param_magic_kc",
        "unsigned param_shift_kc",
        "unsigned param_magic_c",
        "unsigned param_shift_c",
        "unsigned param_CRSTK",
        "unsigned param_CRST",
        "unsigned param_TRS",
        "unsigned param_RS",
        "unsigned param_S",
        "unsigned param_magic_TRS",
        "unsigned param_shift_TRS",
        "unsigned param_magic_RS",
        "unsigned param_shift_RS",
        "unsigned param_magic_S",
        "unsigned param_shift_S",
        "unsigned param_superM",
        "unsigned param_superP",
        "unsigned param_superQ",
        "unsigned param_superN",
        "unsigned param_shiftM",
        "unsigned param_shiftP",
        "unsigned param_shiftQ",
        "unsigned param_strideP",
        "unsigned param_strideQ",
        "unsigned param_stridePQ",
        "unsigned param_gridP",
        "unsigned param_gridQ",
        "unsigned param_loopX",
        "unsigned param_loopXp",
        "unsigned param_loopQ",
        "unsigned param_loopQp",
        "unsigned param_loopN",
        "unsigned param_loopNp",
    ],
    "gemm": [
        "float* param_C",
        "float* param_A",
        "float* param_B",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_lda",
        "unsigned param_ldb",
        "unsigned param_ldc",
        "unsigned param_m",
        "unsigned param_n",
        "unsigned param_k",
        "unsigned param_ldaz",
        "unsigned param_ldbz",
        "unsigned param_ldcz",
        "unsigned param_batch_loops",
    ],
    "gemm_rnn": [
        "float* param_C",
        "float* param_A",
        "float* param_B",
        "float* param_bias",
        "float* param_lock",
        "float param_alpha",
        "float param_beta",
        "float param_xcutoff",
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
        "int   param_dimB",
        "int   param_dimC",
        "int   param_unrolling",
        "int   param_numBlks",
        "int   param_numAblks"
    ],
    "gemm_rnn_bprop": [
        "float* param_C",
        "float* param_A",
        "float* param_B",
        "float* param_H",
        "float* param_lock",
        "float param_alpha",
        "float param_beta",
        "float param_xcutoff",
        "int   param_flags",
        "int   param_lda",
        "int   param_ldb",
        "int   param_ldc",
        "int   param_ldh",
        "int   param_m",
        "int   param_n",
        "int   param_k",
        "int   param_ldaz",
        "int   param_ldbz",
        "int   param_ldcz",
        "int   param_batch_loops",
        "int   param_dimB",
        "int   param_dimC",
        "int   param_dimH",
        "int   param_unrolling",
        "int   param_numBlks",
        "int   param_numAblks"
    ],
    "fpropw": [
        "float* param_S",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_H",
        "unsigned param_P",
        "int param_pad_h",
        "int param_pad_w",
        "unsigned param_HWN",
        "unsigned param_WN",
        "unsigned param_PQN",
        "unsigned param_QN",
        "unsigned param_Qnk",
        "unsigned param_nk",
        "unsigned param_n",
        "unsigned param_k",
        "unsigned param_magic_Qnk",
        "unsigned param_shift_Qnk",
        "unsigned param_magic_nk",
        "unsigned param_shift_nk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_RSK",
        "unsigned param_4RSKp",
        "unsigned param_4HWNp",
        "unsigned param_gridK",
        "unsigned param_gridP2",
        "unsigned param_gridQ",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
        "unsigned param_superP",
        "unsigned param_superQ",
        "unsigned param_superN",
        "unsigned param_shiftP",
        "unsigned param_shiftQ",
        "unsigned param_shiftN",
    ],
    "fpropw4X": [
        "float* param_S",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_K",
        "unsigned param_N",
        "unsigned param_Xk",
        "unsigned param_k",
        "unsigned param_magic_Xk",
        "unsigned param_shift_Xk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_C_1152",
        "unsigned param_GXS_C_1152",
        "unsigned param_GYS_GXS_C_1152",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_PQN15",
        "unsigned param_maskN",
        "unsigned param_shiftX",
        "unsigned param_shiftY",
        "unsigned param_superX",
        "unsigned param_superY",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
    ],
    "fpropw4": [
        "float* param_S",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_K",
        "unsigned param_N",
        "unsigned param_Y",
        "unsigned param_W",
        "unsigned param_YXN",
        "unsigned param_XN",
        "unsigned param_Y2",
        "unsigned param_GX",
        "unsigned param_Xk",
        "unsigned param_k",
        "unsigned param_magic_Xk",
        "unsigned param_shift_Xk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_PQN15",
        "unsigned param_maskN",
        "unsigned param_shiftX",
        "unsigned param_shiftY",
        "unsigned param_superX",
        "unsigned param_superY",
        "int param_pad_x",
        "int param_pad_y",
        "unsigned param_RSK",
        "unsigned param_RSK2p",
        "unsigned param_YXN2p",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
    ],
    "fpropw5": [
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_K",
        "unsigned param_N",
        "unsigned param_H",
        "unsigned param_W",
        "unsigned param_HWN",
        "unsigned param_WN",
        "unsigned param_Y2",
        "unsigned param_GX",
        "unsigned param_Xk",
        "unsigned param_k",
        "unsigned param_magic_Xk",
        "unsigned param_shift_Xk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_PQNp",
        "unsigned param_PQN15p",
        "unsigned param_shiftY",
        "unsigned param_shiftX",
        "unsigned param_shiftN",
        "unsigned param_superY",
        "unsigned param_superX",
        "unsigned param_superN",
        "unsigned param_SuperY",
        "unsigned param_SuperX",
        "unsigned param_SuperN",
        "int param_pad_x",
        "int param_pad_y",
        "unsigned param_HWN2p",
        "unsigned param_C_1152",
    ],
    "updatw": [
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "unsigned param_Y",
        "unsigned param_X",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_C",
        "unsigned param_K",
        "unsigned param_N",
        "int param_pad_y",
        "int param_pad_x",
        "unsigned param_GY",
        "unsigned param_GX",
        "unsigned param_GYS",
        "unsigned param_GXS",
        "unsigned param_shiftYI",
        "unsigned param_shiftXI",
        "unsigned param_superYI",
        "unsigned param_superXI",
        "unsigned param_superNI",
        "unsigned param_shiftY",
        "unsigned param_shiftX",
        "unsigned param_superY",
        "unsigned param_superX",
        "unsigned param_superN",
        "unsigned param_loopXI",
        "unsigned param_loopX",
        "unsigned param_loopN",
        "unsigned param_strideY",
        "unsigned param_strideX",
        "unsigned param_XN",
        "unsigned param_YXN",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_SK",
        "unsigned param_RSK",
        "unsigned param_Np",
        "unsigned param_XNp",
        "unsigned param_2XNp",
        "unsigned param_QNp",
        "unsigned param_CPQkc",
        "unsigned param_PQkc",
        "unsigned param_Qkc",
        "unsigned param_kc",
        "unsigned param_c",
        "unsigned param_k",
        "unsigned param_magic_CPQkc",
        "unsigned param_shift_CPQkc",
        "unsigned param_magic_PQkc",
        "unsigned param_shift_PQkc",
        "unsigned param_magic_Qkc",
        "unsigned param_shift_Qkc",
        "unsigned param_magic_kc",
        "unsigned param_shift_kc",
        "unsigned param_magic_c",
        "unsigned param_shift_c",
        "unsigned param_CRSK",
    ],
    "updatw4": [
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "unsigned param_K",
        "unsigned param_C",
        "unsigned param_k",
        "unsigned param_c",
        "unsigned param_kc",
        "unsigned param_magic_kc",
        "unsigned param_shift_kc",
        "unsigned param_magic_c",
        "unsigned param_shift_c",
        "unsigned param_YXN2",
        "unsigned param_sYXN",
        "unsigned param_magic_sYXN",
        "unsigned param_shift_sYXN",
        "unsigned param_stride_YXNp",
        "unsigned param_YXN",
        "unsigned param_YXN_1152",
        "unsigned param_RSK",
        "unsigned param_CRSK",
        "unsigned param_Kp",
        "unsigned param_SKp",
        "unsigned param_RSK15_SK2p",
    ],
    "rnn_fprop": [
        "float* param_h",
        "float* param_hprev",
        "float* param_bias",
        "float* param_w",
        "int* param_lockAddr",
        "int param_ldh",
        "int param_ldw",
        "int param_bsz",
        "int param_seqLength",
        "int param_numBlks",
        "int param_rowSize",
        "int param_reverse",
        "float param_reluclip"
    ]
    ,
    "rnn_bprop": [
        "float* param_d",
        "float* param_dnext",
        "float* param_h",
        "float* param_w",
        "int* param_lockAddr",
        "int param_ldd",
        "int param_ldh",
        "int param_ldw",
        "int param_bsz",
        "int param_seqLength",
        "int param_numBlks",
        "int param_rowSize",
        "int param_reverse",
        "float param_reluclip"
    ]
}

_params["bprop"] = _params["fprop"] + [
        "unsigned param_magic_str_d",
        "unsigned param_shift_str_d",
        "unsigned param_magic_str_h",
        "unsigned param_shift_str_h",
        "unsigned param_magic_str_w",
        "unsigned param_shift_str_w",
    ]
_params["bprop2"] = _params["fprop2"] + [
        "unsigned param_magic_str_d",
        "unsigned param_shift_str_d",
        "unsigned param_magic_str_h",
        "unsigned param_shift_str_h",
        "unsigned param_magic_str_w",
        "unsigned param_shift_str_w",
    ]

_space_re = re.compile(r"\s+")

_share_template = r"""
    .shared .align 4 .b32 share[{0}];
"""

_kernel_template = r"""
.version {6}
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

def get_ptx_file(kernel_spec, kernel_name, arch, ptx_ver):

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

    kernel_text = _kernel_template.format(arch, kernel_name, kernel_params, thread_spec, share, args_spec, ptx_ver)
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
    cmd  = " ".join(cmdlist)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError("Error(%d):\n%s\n%s" % (proc.returncode, cmd, err))
    if debug:
        neon_logger.display(cmd)
        if out: neon_logger.display(out)
        if err: neon_logger.display(err)

@context_dependent_memoize
def get_kernel(base_name, options=None):

    attributes = drv.Context.get_device().get_attributes()
    major = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MAJOR]
    minor = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MINOR]
    if major < 5:
        raise RuntimeError("sass kernels require Maxwell or greater class hardware")

    arch = "sm_%d%d" % (major, minor)

    libprefix = "PERL5LIB=%s" % (maxas_dir)
    maxas_i = [libprefix, os.path.join(maxas_dir, "maxas.pl") + " -i -w"]
    maxas_p = [libprefix, os.path.join(maxas_dir, "maxas.pl") + " -p"]

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

    maxas_i.insert(2, "-k " + kernel_name)

    sass_name  = kernel_spec["sass"] + ".sass"
    cubin_name = kernel_name + ".cubin"

    ptx_version = "4.2" if major < 6 else "5.0"
    ptx_file   = get_ptx_file(kernel_spec, kernel_name, arch, ptx_version)
    sass_file  = os.path.join(sass_dir, sass_name)
    cubin_file = os.path.join(cubin_dir, cubin_name)

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
        elif ptype == 'unsigned':
            sig += "I"
        else:
            sig += "i"

    module = drv.module_from_file(os.path.join(cubin_dir, kernel_name + ".cubin"))
    func   = module.get_function(kernel_name)
    func.prepare(sig)
    func.threads = kernel_spec["threads"]
    return func


