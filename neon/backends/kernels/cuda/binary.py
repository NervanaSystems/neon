# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize

"""
Binary CUDA kernels.
"""


# XNOR GEMM adapted from https://github.com/MatthieuCourbariaux/BinaryNet,
# licensed under the BSD 3-clause license. See details at
# https://github.com/MatthieuCourbariaux/BinaryNet/blob/master/LICENSE.txt
#
# A is shape (m,n), B is shape (n,k) and C is shape (m,k)
# 1 bit operations are simulated by packing 32 bits into a single integer
# and then performing bitwise operations. Matrice dimensions correspond to
# packed integers.
@context_dependent_memoize
def XNOR_gemm():

    code = r"""
#define BLOCK_SIZE 16
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // BLOCK_SIZE = 16 -> 256 threads, one per Csub element
    unsigned int Cvalue = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A
        unsigned int* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Bsub of B
        unsigned int* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int j = 0; j < BLOCK_SIZE; ++j) {
             Cvalue += __popc(As[row][j]^Bs[j][col]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol * BLOCK_SIZE < k && row + blockRow * BLOCK_SIZE < m) {
        Csub[row * k + col] = -(2 * (float) Cvalue - 32 * n);
    }
}
"""

    module = SourceModule(code)
    kernel = module.get_function("xnor_gemm")
    sig = "3P 3I"
    kernel.prepare(sig)
    return kernel


def pack():

    code = r"""
__device__ unsigned int pack(float* array) {
    unsigned int rvalue = 0;
    unsigned int sign;

    for (int i = 0; i < 32; i++) {
        sign = (array[i] >= 0);
        rvalue = rvalue | (sign << i);
    }

    return rvalue;
}
"""

    return code


@context_dependent_memoize
def pack_rows():

    code = pack() + r"""
__global__ void pack_rows(float *a, unsigned int *b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        b[i] = pack(&a[i * 32]);
    }
}
"""

    module = SourceModule(code)
    kernel = module.get_function("pack_rows")
    sig = "2P I"
    kernel.prepare(sig)
    return kernel


@context_dependent_memoize
def pack_cols():

    code = r"""
__global__ void pack_cols(float *a, unsigned int *b, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && 32 * i < m) {
        float num;
        unsigned int rvalue = 0;
        unsigned int sign;

        for(int k = 0; k < 32; k++) {
            num = a[j + n * (32 * i + k)];
            sign = (num >= 0);
            rvalue = rvalue | (sign << k);
        }
        b[j + n * i] = rvalue;
    }
}
"""

    module = SourceModule(code)
    kernel = module.get_function("pack_cols")
    sig = "2P 2I"
    kernel.prepare(sig)
    return kernel


def shift_element():

    code = r"""
__device__ float shift_element(float a, float b, bool value) {
    float result;

    int expb;
    if (value && b == 0) {
        return 0;
    } else if (value) {
        expb = round(log2(abs(b)));
    } else {
        expb = b;
    }

    int expa;
    double mantissa = frexp(a, &expa);
    result = ldexp(mantissa, expa + expb);

    if (value && b < 0) result = -result;

    return result;
}
"""

    return code


@context_dependent_memoize
def shift():

    code = shift_element() + r"""
__global__ void shift(
    float *a, float *b, float *c, bool value, int sizea, int b_rows, int b_cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < sizea) {
        float bi;
        if (b_rows > 1 && b_cols > 1) {
            bi = b[i];
        } else if (b_rows > 1 && b_cols == 1) {
            int step = sizea/b_rows;
            bi = b[i/step];
        } else if (b_rows == 1 && b_cols > 1) {
            bi = b[i % b_cols];
        } else if (b_rows == 1 && b_cols == 1) {
            bi = b[0];
        }

        c[i] = shift_element(a[i], bi, value);
    }
}
"""

    module = SourceModule(code)
    kernel = module.get_function("shift")
    sig = "3P 4I"
    kernel.prepare(sig)
    return kernel
