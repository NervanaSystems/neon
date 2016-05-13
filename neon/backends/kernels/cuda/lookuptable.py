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
from future.utils import native_str
from pycuda.tools import context_dependent_memoize

from neon.backends.cuda_templates import _ew_types
from neon.backends.util.source_module import SourceModule

"""
CUDA kernels for lookup table layers. Kernels are only given for bprop, since
fprop is just a take operation. There is a deterministic kernel and
non-deterministic kernel (using atomics) provided. Sorting kernels are also
provided to help with the deterministic version.
"""


@context_dependent_memoize
def _get_lut_bprop_kernel(dtype, deterministic=False):
    """
    Builds the bprop kernel for lookup table layers based on templated code.
    If the deterministic version is requested, an index buffer must be passed
    as an argument. This index buffer re-orders items in the input tensor
    so that word_ids are sorted. This is required since we need to be sure that
    each thread only updates weights for one word id.

    Arguments:
        dtype (np.dtype): The data which the kernel will operate on.
        deterministic (boolean): Builds the deterministic kernel when this is
            set to True.
    """
    if not deterministic:
        code = r"""
__global__ void lut_bprop(
    int* inputs, %(type)s* dW, %(type)s* errors, const int nin,
    const int embedding_dim, const int vocab_size, const int pad_idx)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    int word_id = inputs[bid];
    int error_row = bid * embedding_dim;
    int output_row = word_id * embedding_dim;

    if(word_id != pad_idx)
    {
        for(int i = tid; i < embedding_dim; i += blockDim.x)
        {
            atomicAdd(&dW[output_row + i], errors[error_row + i]);
        }
    }
}
"""

        code = code % {
            "type": _ew_types[dtype]["type"]
        }

        module = SourceModule(code, options=["--use_fast_math"])
        kernel = module.get_function("lut_bprop")
        kernel.prepare("PPPIIIi")
    else:
        code = r"""
__global__ void lut_bprop(
    int* inputs, int* index_buffer, %(type)s* dW, %(type)s* errors,
    const int nin, const int embedding_dim, const int vocab_size,
    const int pad_idx)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    int index_position = bid;
    int index = index_buffer[index_position];
    int word_id = inputs[index];

    if((bid == 0 || word_id != inputs[index_buffer[bid - 1]]) && word_id != pad_idx)
    {
        int output_row = word_id * embedding_dim;

        do {
            int error_row = index * embedding_dim;

            for(int i = tid; i < embedding_dim; i += blockDim.x)
            {
                dW[output_row + i] += errors[error_row + i];
            }

            index_position++;
            if(index_position == gridDim.x)
            {
                break;
            }
            index = index_buffer[index_position];
        } while(inputs[index] == word_id);
    }
}
"""

        code = code % {
            "type": _ew_types[dtype]["type"]
        }

        module = SourceModule(code, options=["--use_fast_math"])
        kernel = module.get_function("lut_bprop")
        kernel.prepare("PPPPIIIi")

    kernel.name = "lut_bprop"
    return kernel


def _get_sorting_kernel(kernel_id, block_size):
    """
    Builds kernels used for sorting inputs. There are several kernels here
    corresponding to the steps in the algorithm. The algorithm works by
    determining the sorted position for each input item. This is done with
    a bucket sort algorithm, where each word_id is a bucket. The first step
    determines the size of each bucket (number of occurences of each word_id).
    Next, a prefix some is computed over the list of bucket sizes to find
    where each bucket will be placed in the output buffer. Finally, each thread
    places it's index into the correct sorted position based on the bucket
    start index (computed from the prefix sum) and that thread's offset into
    the bucket (which is taken from the output of the atomic add done in the
    first step.)

    Arguments:
        kernel_id (Integer): Which step to build the kernel for [0, 4]
        block_size (Integer): Number of threads per block for the prefix sum
            kernels.
    """
    code = r"""
#define THREADS %(threads)s
#define STORE_BLOCKSUM %(store_blocksum)s
__global__ void sort_inputs0(
        int* inputs, int* index_buffer, int* offset_buffer, int* word_counts, const int vocab_size,
        const int input_length)
{
    const int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    int word_id;

    if(tid < input_length)
    {
        word_id = inputs[tid];
        offset_buffer[tid] = atomicAdd(&word_counts[word_id], 1);
    }
}

__device__ void scan(int* buffer, int* blocksum, int global_length)
{
    const int tid = (threadIdx.x << 1) + 1;
    const int gid = ((threadIdx.x + (blockIdx.x * blockDim.x)) << 1) + 1;

    __shared__ int local_counts[THREADS * 2];
    local_counts[tid] = buffer[gid];
    local_counts[tid - 1] = buffer[gid - 1];

    #pragma unroll
    for(int skip = 1; skip <= THREADS; skip <<= 1)
    {
        int mask = (skip << 1) - 1;
        if((tid & mask) == mask)
        {
            local_counts[tid] += local_counts[tid - skip];
        }

        __syncthreads();
    }

    if(tid == (THREADS * 2 - 1))
    {
#if STORE_BLOCKSUM
        blocksum[blockIdx.x] = local_counts[tid];
#endif
        local_counts[tid] = 0;
    }

    #pragma unroll
    for(int skip = THREADS; skip > 0; skip >>= 1)
    {
        int mask = (skip << 1) - 1;
        if((tid & mask) == mask)
        {
            int temp = local_counts[tid - skip];
            local_counts[tid - skip] = local_counts[tid];
            local_counts[tid] += temp;
        }

        __syncthreads();
    }

    if(gid < global_length)
    {
        buffer[gid] = local_counts[tid];
        buffer[gid - 1] = local_counts[tid - 1];
    }
}

__global__ void sort_inputs1(
        int* inputs, int* index_buffer, int* offset_buffer, int* word_counts, const int vocab_size,
        const int input_length)
{
    scan(word_counts, word_counts + vocab_size, vocab_size);
}

__global__ void sort_inputs2(
        int* inputs, int* index_buffer, int* offset_buffer, int* word_counts, const int vocab_size,
        const int input_length)
{
    scan(word_counts + vocab_size, 0, blockDim.x);
}

__global__ void sort_inputs3(
        int* inputs, int* index_buffer, int* offset_buffer, int* word_counts, const int vocab_size,
        const int input_length)
{
    const int gid = (threadIdx.x + (blockIdx.x * blockDim.x)) << 1;

    if(gid < vocab_size)
    {
        word_counts[gid] += word_counts[vocab_size + blockIdx.x];
        word_counts[gid + 1] += word_counts[vocab_size + blockIdx.x];
    }
}

__global__ void sort_inputs4(
        int* inputs, int* index_buffer, int* offset_buffer, int* word_counts, const int vocab_size,
        const int input_length)
{
    const int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    int word_id;

    if(tid < input_length)
    {
        word_id = inputs[tid];
        int sorted_position = word_counts[word_id] + offset_buffer[tid];
        index_buffer[sorted_position] = tid;
    }
}
"""
    code = code % {
        "threads": block_size,
        "store_blocksum": (1 if kernel_id == 1 else 0)
    }
    module = SourceModule(code, options=["--use_fast_math"])

    function_name = "sort_inputs" + native_str(kernel_id)
    kernel = module.get_function(function_name)
    kernel.prepare("PPPPII")
    kernel.name = "sort_inputs"
    return kernel
