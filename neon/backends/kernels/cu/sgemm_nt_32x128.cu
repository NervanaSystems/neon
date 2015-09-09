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

extern "C"
__global__ void __launch_bounds__(128) sgemm_nt_32x128
(
    short*       param_Rand,
    const float* param_A,
    const float* param_B,
    float*       param_C,
    int          param_lda,  
    int          param_ldb,  
    int          param_ldc,
    int          param_m,
    int          param_n,
    int          param_k,
    float        param_alpha,
    float        param_beta,
    int          param_flags,
    int          param_ldaz,
    int          param_ldbz,
    int          param_ldcz,
    int          param_batch_loops
)
{
    __shared__ float share[(128*16 + 32)*2 + (32*16 + 32)*2 + 4];

    int tid = threadIdx.x;

    share[tid] = 1;

    param_C[tid] = share[127-tid];
}
