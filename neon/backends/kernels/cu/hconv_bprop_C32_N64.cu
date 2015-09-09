/*
 * Copyright 2014 Nervana Systems Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// nvcc -arch sm_50 -cubin sconv_bprop_C128_N64.cu

extern "C"
__global__ void __launch_bounds__(32) hconv_bprop_C32_N64
(
    unsigned short*        param_I,
    const unsigned short*  param_F,
    const unsigned short*  param_E,
    float param_alpha,
    int param_flags,
    int param_N,
    int param_K,
    int param_D,
    int param_H,
    int param_W,
    int param_WN,
    int param_HWN,
    int param_DHWN,
    int param_C,
    int param_CRST,
    int param_RST,
    int param_magic_RST,
    int param_shift_RST,
    int param_RS,
    int param_magic_RS,
    int param_shift_RS,
    int param_S,
    int param_magic_S,
    int param_shift_S,
    int param_pad_d,
    int param_pad_h,
    int param_pad_w,
    int param_str_d,
    int param_str_h,
    int param_str_w,
    int param_Q,
    int param_PQ,
    int param_QN,
    int param_PQN,
    int param_MPQN,
    int param_magic_Q,
    int param_shift_Q,
    int param_magic_PQ,
    int param_shift_PQ,
    int param_CRST8,
    int param_MPQN8
)
{
    __shared__ float share[32*8*2 + 64*8*2];

    int tid = threadIdx.x;

    share[tid] = 1;

    *param_I = share[31-tid];
}
