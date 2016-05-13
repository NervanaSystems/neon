# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------
"""
Templates for cuda kernels:
    _ew_template:           generic template?
    _stage_template:        "loop"
                            "red32"
                            "red"
                            "red_ops"
                            "red_out"
    _fin_template
    _init_rand_func:        Initialize LFSR's
    _init_rand_round_func
    _finish_rand_func
    _common_urand_gen
    _common_frand
    _common_round random:  f4, f2, i4, i2, i1
                  nearest: f2, i4, u4, i2, u2, i1, u1
    _common_fp16_to_fp32:  inline assembly conversion function
    _ew_types:             f4,f2,i4,u4,i2,u2,i1,u1
    _ew_strings:
    _is_finite:            inline assembly test function
    _float_ops:            unary and binary element operations
    _reduction_ops:        sum, max, min, argmax, argmin
"""
from builtins import zip

# RAND_POOL_SIZE set to 65536 == 2048 * 32

_ew_template = r"""

#define FLT_MAX 3.402823466E+38F
#define RAND_POOL_SIZE 65536

%(common)s

#define THREADS %(threads)s

__global__ void %(name)s (
    unsigned* rand_state,
    %(arguments)s)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    extern __shared__ float sPartials[];

    %(inits)s
"""


_stage_template = {
    "loop": r"""

    for (int i = tid; i < n{0}; i += THREADS)
    {{
        %(loads{0})s

        %(ops{0})s
    }}
""",

    "red32": r"""

    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
    {{
        %(shfl_red{0})s
    }}

""",

    "red": r"""

    sPartials[tid] = %(var_red{0})s;
    __syncthreads();

    #pragma unroll
    for (int a = THREADS >> 1; a > 32; a >>= 1)
    {{
        if ( tid < a )
            %(share1_red{0})s
        __syncthreads();
    }}

    if ( tid < 32 )
    {{
        %(share2_red{0})s

        // __syncthreads(); // Seems to prevent a race condition but causes other problems

        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
            %(shfl_red{0})s

        sPartials[tid] = %(var_red{0})s;
    }}
    __syncthreads();
    %(var_red{0})s = sPartials[0];
""",

    "red_ops": r"""

        %(ops{0})s
""",

    "red_out": r"""

    if ( tid == 0 )
    {{
        %(ops{0})s
    }}
"""
}


_fin_template = r"""
    %(finish)s
}
"""


_init_rand_func = r"""
    unsigned lfsr0, lfsr1, lfsr2;
    unsigned idx = bid * THREADS + tid;
    rand_state += idx % RAND_POOL_SIZE;
    lfsr0 = *(rand_state + 0*RAND_POOL_SIZE);
    lfsr1 = *(rand_state + 1*RAND_POOL_SIZE);
    lfsr2 = *(rand_state + 2*RAND_POOL_SIZE);
"""


_init_rand_round_func = r"""
    int i_rand_scale = (127 - 32 - mantissa_bits) << 23;
    float rand_scale = *(float*)&i_rand_scale;
    unsigned rand_mask = 0xffffffff << (23 - mantissa_bits);
"""


_finish_rand_func = r"""
    *(rand_state + 0*RAND_POOL_SIZE) = lfsr0;
    *(rand_state + 1*RAND_POOL_SIZE) = lfsr1;
    *(rand_state + 2*RAND_POOL_SIZE) = lfsr2;
"""

_common_kepler = r"""
#define __ldg(x) (*(x))
"""

_common_urand_gen = r"""
__device__ unsigned urand_gen(unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
    lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
    lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
    return lfsr0 ^ lfsr1 ^ lfsr2;
}
"""


_common_frand = r"""
__device__ __forceinline__ float frand(unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    float val;
    asm("cvt.rn.f32.u32 %0, %1;\n\t"
        "mul.f32 %0, %0, 0F2f800000;"
        : "=f"(val) : "r"(urand));
    return val;
}
"""


_common_round = {

    "random": {

        "f4": r"""
__device__ float fp32_to_fp32_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2, float rand_scale,
          unsigned rand_mask)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);

    float ret;
    asm("{\n\t"
        ".reg .f32 exponent, frand, result;\n\t"
        "and.b32 exponent, %1, 0xff800000;\n\t"
        "mul.f32 exponent, exponent, %2;\n\t"
        "cvt.rz.f32.u32 frand, %3;\n\t"
        "fma.rz.f32 result, exponent, frand, %1;\n\t"
        "and.b32 %0, result, %4;\n\t"
        "}" : "=f"(ret) : "f"(val), "f"(rand_scale), "r"(urand), "r"(rand_mask));

    return ret;
}
""",
        "f2": r"""
__device__ unsigned short fp32_to_fp16_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2, float rand_scale,
          unsigned rand_mask)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);

    unsigned short half;
    asm("{\n\t"
        ".reg .f16 result16;\n\t"
        ".reg .f32 exponent, frand, result32;\n\t"
        "and.b32 exponent, %1, 0xff800000;\n\t"
        "mul.f32 exponent, exponent, %2;\n\t"
        "cvt.rz.f32.u32 frand, %3;\n\t"
        "fma.rz.f32 result32, exponent, frand, %1;\n\t"
        "and.b32 result32, result32, %4;\n\t"
        "cvt.rz.f16.f32 result16, result32;\n\t"
        "mov.b16 %0, result16;\n\t"
        "}" : "=h"(half) : "f"(val), "f"(rand_scale), "r"(urand), "r"(rand_mask));

    return half;
}
""",
        "i4": r"""
__device__ __forceinline__ int fp32_to_int32_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    int ret;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s32.f32 %0, result32;\n\t"
        "}" : "=r"(ret) : "f"(val), "r"(urand));
    return ret;
}
""",
        "i2": r"""
__device__ __forceinline__ short fp32_to_int16_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    short half;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s16.f32 %0, result32;\n\t"
        "}" : "=h"(half) : "f"(val), "r"(urand));
    return half;
}
""",
        "i1": r"""
__device__ __forceinline__ char fp32_to_int8_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    int ret;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s8.f32 %0, result32;\n\t"
        "}" : "=r"(ret) : "f"(val), "r"(urand));
    return ret;
}
""",
    },
    "nearest": {

        "f2": r"""
__device__ __forceinline__ unsigned short fp32_to_fp16(float val)
{
    unsigned short ret;
    asm("{\n\t"
        ".reg .f16 f16;\n\t"
        "cvt.rn.f16.f32 f16, %1;"
        "mov.b16 %0, f16;\n\t"
        "}" : "=h"(ret) : "f"(val));
    return ret;
}
""",
        "i4": r"""
__device__ __forceinline__ int fp32_to_int32(float val)
{
    int ret;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
        "u4": r"""
__device__ __forceinline__ unsigned fp32_to_uint32(float val)
{
    unsigned ret;
    asm("cvt.rni.u32.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
        "i2": r"""
__device__ __forceinline__ short fp32_to_int16(float val)
{
    short ret;
    asm("cvt.rni.s16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}
""",
        "u2": r"""
__device__ __forceinline__ unsigned short fp32_to_uint16(float val)
{
    unsigned short ret;
    asm("cvt.rni.u16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}
""",
        "i1": r"""
__device__ __forceinline__ char fp32_to_int8(float val)
{
    int ret;
    asm("cvt.rni.s8.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
        "u1": r"""
__device__ __forceinline__ unsigned char fp32_to_uint8(float val)
{
    unsigned ret;
    asm("cvt.rni.u8.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
    },
}
# random rounding not yet used for these types
for dtype in ("u4", "u2", "u1"):
    _common_round["random"][dtype] = _common_round["nearest"][dtype]

for mode in ("random", "nearest"):
    for xtype, itype in zip(("x4", "x2", "x1"), ("i4", "i2", "i1")):
        _common_round[mode][xtype] = _common_round[mode][itype]


_common_fp16_to_fp32 = r"""
__device__ __forceinline__ float fp16_to_fp32(unsigned short val)
{
    float ret;
    asm("{\n\t"
        ".reg .f16 f16;\n\t"
        "mov.b16 f16, %1;\n\t"
        "cvt.f32.f16 %0, f16\n\t;"
        "}" : "=f"(ret) : "h"(val));
    return ret;
}
"""

_common_max_abs = r"""
__device__ __forceinline__ float max_abs(int max_abs, int val)
{
    asm("{\n\t"
        ".reg .s32 abs_val;\n\t"
        "abs.s32 abs_val, %1;\n\t"
        "max.s32 %0, %0, abs_val;\n\t"
        "}" : "+r"(max_abs) : "r"(val));
    return max_abs;
}
"""

_ew_types = {
    "f4": {
        "type": "float",
        "type4": "float4",
        "cvt": "",
        "cvt_out": "",
    },
    "f2": {
        "type": "unsigned short",
        "type4": "ushort4",
        "cvt": "fp16_to_fp32",
        "cvt_out": "fp32_to_fp16",
    },
    "i4": {
        "type": "int",
        "cvt": "(float)",
    },
    "u4": {
        "type": "unsigned int",
        "cvt": "(float)",
    },
    "i2": {
        "type": "short",
        "cvt": "(float)",
    },
    "u2": {
        "type": "unsigned short",
        "cvt": "(float)",
    },
    "i1": {
        "type": "char",
        "cvt": "(float)",
    },
    "u1": {
        "type": "unsigned char",
        "cvt": "(float)",
    },
    "x4": {
        "type": "int",
        "cvt": "scale{0} * (float)",
    },
    "x2": {
        "type": "short",
        "cvt": "scale{0} * (float)",
    },
    "x1": {
        "type": "char",
        "cvt": "scale{0} * (float)",
    },
}


_ew_strings = {

    # 0: arg_id, 1: stage, 2: type, 3: cvt
    "in0": {
        "arguments": "const {2}* a{0}_in, int row_strd{0}, int col_strd{0}",
        "inits": "const {2}* a{0}_in{1} = a{0}_in + bid * row_strd{0} + tid * col_strd{0};\n"
        "    int a{0}_inc{1} = THREADS * col_strd{0};",
        "loads": "float a{0} = {3}(__ldg(a{0}_in{1}));\n"
        "        a{0}_in{1} += a{0}_inc{1};",
    },
    "in1": {
        "arguments": "const {2}* a{0}_in, int row_strd{0}, int col_strd{0}, const int* take{0}_in",
        "inits": """const {2}* a{0}_in{1} = a{0}_in + __ldg(take{0}_in + bid) * row_strd{0}
                                            + tid * col_strd{0};\n"""
        "    int a{0}_inc{1} = THREADS * col_strd{0};",
        "loads": "float a{0} = {3}(__ldg(a{0}_in{1}));\n"
        "        a{0}_in{1} += a{0}_inc{1};",
    },
    "in2": {
        "arguments": "const {2}* a{0}_in, int row_strd{0}, int col_strd{0}, const int* take{0}_in",
        "inits": "const {2}* a{0}_in{1} = a{0}_in + bid * row_strd{0};\n"
        "    const int* take{0}_in{1} = take{0}_in + tid;",
        "loads": "float a{0} = {3}(__ldg(a{0}_in{1} + __ldg(take{0}_in{1})));\n"
        "        take{0}_in{1} += THREADS;",
    },
    "out0": {
        "arguments": "{2}* a_out, int row_strd, int col_strd",
        "inits": "a_out += bid * row_strd + tid * col_strd;\n"
        "    int out_inc = THREADS * col_strd;",
        "output": "*a_out = {0};\n        a_out += out_inc;",
    },
    "out1": {
        "arguments": "{2}* a_out, int row_strd, int col_strd, const int* take_out",
        "inits": "a_out += __ldg(take_out + bid) * row_strd + tid * col_strd;\n"
        "    int out_inc = THREADS * col_strd;",
        "output": "*a_out = {0};\n        a_out += out_inc;",

    },
    "out2": {
        "arguments": "{2}* a_out, int row_strd, int col_strd, const int* take_out",
        "inits": "a_out += bid * row_strd;\n"
        "    take_out += tid;",

        "output": "*(a_out + __ldg(take_out)) = {0};\n        take_out += THREADS;",

    },
    "onehot0": {
        "arguments": "const int* onehot{0}_in",
        "inits": "onehot{0}_in += tid;",
        "loads": "int onehot{0} = __ldg(onehot{0}_in);\n"
        "        onehot{0}_in += THREADS;",
    },
    "onehot1": {
        "arguments": "const int* onehot{0}_in",
        "inits": "int onehot{0} = __ldg(onehot{0}_in + bid);\n",
        "loads": "",
    },
    "const": {
        "arguments": "float c{0}",
    },
    "round": {
        "random": {
            "f4": """float {0}          = fp32_to_fp32_rand({1}, lfsr0, lfsr1,
                                                            lfsr2, rand_scale, rand_mask);""",
            "f2": """unsigned short {0} = fp32_to_fp16_rand({1}, lfsr0, lfsr1,
                                                            lfsr2, rand_scale, rand_mask);""",
            "u4": "unsigned int {0}     = fp32_to_uint32({1});",
            "u2": "unsigned short {0}   = fp32_to_uint16({1});",
            "u1": "unsigned char {0}    = fp32_to_uint8({1});",
            "i4": "int {0}              = fp32_to_int32_rand({1}, lfsr0, lfsr1, lfsr2);",
            "i2": "short {0}            = fp32_to_int16_rand({1}, lfsr0, lfsr1, lfsr2);",
            "i1": "char {0}             = fp32_to_int8_rand( {1}, lfsr0, lfsr1, lfsr2);",
            "x4": "int {0}              = fp32_to_int32_rand({1}, lfsr0, lfsr1, lfsr2);",
            "x2": "short {0}            = fp32_to_int16_rand({1}, lfsr0, lfsr1, lfsr2);",
            "x1": "char {0}             = fp32_to_int8_rand( {1}, lfsr0, lfsr1, lfsr2);",
        },
        "nearest": {
            "f2": "unsigned short {0}   = fp32_to_fp16({1});",
            "u4": "unsigned int {0}     = fp32_to_uint32({1});",
            "u2": "unsigned short {0}   = fp32_to_uint16({1});",
            "u1": "unsigned char {0}    = fp32_to_uint8({1});",
            "i4": "int {0}              = fp32_to_int32({1});",
            "i2": "short {0}            = fp32_to_int16({1});",
            "i1": "char {0}             = fp32_to_int8({1});",
            "x4": "int {0}              = fp32_to_int32({1});",
            "x2": "short {0}            = fp32_to_int16({1});",
            "x1": "char {0}             = fp32_to_int8({1});",

        },
    },
}


_is_finite = r"""
float {0};
asm("{{\n\t"
    ".reg .pred is_finite;\n\t"
    "testp.finite.f32 is_finite, %1;\n\t"
    "selp.f32 %0, 0F3f800000, 0F00000000, is_finite;\n\t"
    "}}" : "=f"({0}) : "f"({1}));
"""


# Note: binary operands come off the stack in reverse order
_float_ops = {
    "assign": (2, "unused"),
    "add": (2, 'float {0} = {2} + {1};'),
    "sub": (2, 'float {0} = {2} - {1};'),
    "mul": (2, 'float {0} = {2} * {1};'),
    "div": (2, 'float {0} = {2} / {1};'),
    "eq": (2, "float {0} = {2} == {1};"),
    "ne": (2, "float {0} = {2} != {1};"),
    "lt": (2, "float {0} = {2} <  {1};"),
    "le": (2, "float {0} = {2} <= {1};"),
    "gt": (2, "float {0} = {2} >  {1};"),
    "ge": (2, "float {0} = {2} >= {1};"),
    "minimum": (2, "float {0} = fminf({2},{1});"),
    "maximum": (2, "float {0} = fmaxf({2},{1});"),
    "pow": (2, "float {0} = powf({2},{1});"),
    "finite": (1, _is_finite),
    "neg": (1, "float {0} = -{1};"),
    "abs": (1, "float {0} = abs({1});"),
    "sgn": (1, "float {0} = ({1} == 0.0f) ? (0.0f) : (copysignf(1.0f, {1}));"),
    "sqrt": (1, "float {0} = sqrtf({1});"),
    "sqr": (1, "float {0} = {1} * {1};"),
    "exp": (1, "float {0} = expf({1});"),
    "log": (1, "float {0} = logf({1});"),
    "safelog": (1, "float {0} = ({1} > 0.0f) ? logf({1}) : -50.0f;"),
    "exp2": (1, "float {0} = exp2f({1});"),
    "log2": (1, "float {0} = log2f({1});"),
    "sig": (1, "float {0} = 1.0f/(1.0f + expf(-{1}));"),
    "sig2": (1, "float {0} = 1.0f/(1.0f + exp2f(-{1}));"),
    "tanh": (1, "float {0} = tanhf({1});"),
    "tanh2": (1, "float {0} = (exp2f(2.0f*{1}) - 1.0f) / (exp2f(2.0f*{1}) + 1.0f);"),
    "rand": (0, "float {0} = frand(lfsr0, lfsr1, lfsr2);"),
    "onehot": (0, "float {0} = {1} == {2};"),
}


_reduction_ops = {
    "sum": {
        "inits": "float {0} = 0.0f;",
        "ops": "{0} += {1};",
        "shfl_red": "{0} += __shfl_xor({0}, i);",
        "share1_red": "sPartials[tid] += sPartials[tid + a];",
        "share2_red": "{0} = sPartials[tid] + sPartials[tid + 32];",
    },
    "max": {
        "inits": "float {0} = -FLT_MAX;",
        "ops": "{0} = fmaxf({0}, {1});",
        "shfl_red": "{0} = fmaxf({0}, __shfl_xor({0}, i));",
        "share1_red": "sPartials[tid] = fmaxf(sPartials[tid], sPartials[tid + a]);",
        "share2_red": "{0} = fmaxf(sPartials[tid], sPartials[tid + 32]);",
    },
    "min": {
        "inits": "float {0} = FLT_MAX;",
        "ops": "{0} = fminf({0}, {1});",
        "shfl_red": "{0} = fminf({0}, __shfl_xor({0}, i));",
        "share1_red": "sPartials[tid] = fminf(sPartials[tid], sPartials[tid + a]);",
        "share2_red": "{0} = fminf(sPartials[tid], sPartials[tid + 32]);",
    },
    "argmax": {
        "inits": "int {0} = -1; float max = -FLT_MAX;",
        "ops": "if ({1} > max) {{ max = {1}; {0} = i; }}",
        "shfl_red": "float max2 = __shfl_xor(max, i); int argMax2 = __shfl_xor({0}, i);\n"
        "        if (max2 > max) {{ max = max2; {0} = argMax2; }}"
        "        else if (max2 == max && argMax2 < {0}) {{ {0} = argMax2; }}",
    },
    "argmin": {
        "inits": "int {0} = -1; float min = FLT_MAX;",
        "ops": "if ({1} < min) {{ min = {1}; {0} = i; }}",
        "shfl_red": "float min2 = __shfl_xor(min, i); int argMin2 = __shfl_xor({0}, i);\n"
        "        if (min2 < min) {{ min = min2; {0} = argMin2; }}"
        "        else if (min2 == min && argMin2 < {0}) {{ {0} = argMin2; }}",
    },
}
