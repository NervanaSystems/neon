from pycuda.tools import context_dependent_memoize
from neon.backends.cuda_templates import _ew_types
from neon.backends.layer_gpu import _magic64
from neon.backends.util.source_module import SourceModule

"""
CUDA kernels for convolutional layers. This code will return one of 3 kernels:
fprop, bprop, or update. Data format is expected to be (from least to most
contiguous dimension):

I (input activations): (C, H, W, N)
F (filters): (C, R, S, K)
O (output activations): (K, P, Q, N)

Where:
    N = batch size
    C = number of input feature maps/channels
    K = number of output feature maps
    H, W = height/width of input feature map
    R, S = height/width of filter
    P, Q = height/width of output feature map

Currently 3D convolution is not supported so it is expected that (D == T == M == 1)
"""


@context_dependent_memoize
def _get_conv_kernel(dtype, filter_size, bsum, operation, filter_bounds_check=False, debug=False):
    """
    Builds the convolution kernel for a specified filter size.

    Arguments:
        dtype (np.dtype): The data type which the kernel will operate on.
        filter_size (int): Total number of elements per filter (R * S)
        bsum (boolean): If set to true, kernel will include code to compute
            batch sum during fprop
        operation (string): Determines which kernel to build. options follow:
            'fprop': Forward propagation of activations.
            'bprop': Backward propagation of error.
            'update': Computes gradients for filter weights based on error and inputs.
        filter_bounds_check (boolean): Checks if filter weight is in bounds when K is
            not a multiple of 32.
        debug (boolean): When set to true, kernels will be compiled with debug symbols.
    """
    assert operation in ["fprop", "bprop", "update"]
    if operation == "fprop" or operation == "update":
        lut_code = r"""
    if(tid < 32)
    {
        int rs = tid;
        int base_x, base_y;

        base_x = output_pixel_x * stride_w - padding_w;
        base_y = output_pixel_y * stride_h - padding_h;

        unsigned int mask = (1 << tid) - 1;

        while(rs < FILTER_SIZE)
        {
            int filter_x, filter_y;
            _idiv_magic32(rs, magic_s, shift_s, S, filter_y, filter_x);

            int index_x = base_x + filter_x * dilation_w;
            int index_y = base_y + filter_y * dilation_h;

            //Check if the index is valid
            int in_bounds = (index_x >= 0 && index_x < W && index_y >= 0 && index_y < H);
            unsigned int threads_in_bounds = __ballot(in_bounds);

            //Store lookup table entry
            if(in_bounds)
            {
                int2 lut_entry;
                lut_entry.x = ((index_y * W + index_x) * N) >> 2;
                lut_entry.y = (rs * K) >> 2;

                int index = lut_size_local + __popc(threads_in_bounds & mask);
                lookup_table[index] = lut_entry;
            }

            lut_size_local += __popc(threads_in_bounds);

            rs += 32;
        }
    }
"""
    elif operation == "bprop":
        lut_code = r"""
    if(tid < 32)
    {
        int rs = tid;
        int base_q, base_p;

        base_q = output_pixel_x - ((S - 1) * dilation_w - padding_w);
        base_p = output_pixel_y - ((R - 1) * dilation_h - padding_h);

        unsigned int mask = (1 << tid) - 1;

        while(rs < FILTER_SIZE)
        {
            int filter_x, filter_y;
            _idiv_magic32(rs, magic_s, shift_s, S, filter_y, filter_x);

            int index_q = base_q + filter_x * dilation_w;
            int index_p = base_p + filter_y * dilation_h;

            //Check if the index is valid
            int in_bounds = (((index_q % stride_w) | (index_p % stride_h)) == 0);
            index_q /= stride_w;
            index_p /= stride_h;
            in_bounds = in_bounds && (index_q >= 0 && index_q < W
                                      && index_p >= 0 && index_p < H);
            unsigned int threads_in_bounds = __ballot(in_bounds);

            //Store lookup table entry
            if(in_bounds)
            {
                int2 lut_entry;
                lut_entry.x = (((index_p * W) + index_q) * N) >> 2;
                lut_entry.y = (rs * K) >> 2;

                int index = lut_size_local + __popc(threads_in_bounds & mask);
                lookup_table[index] = lut_entry;
            }

            lut_size_local += __popc(threads_in_bounds);

            rs += 32;
        }
    }
"""
    if bsum:
        bsum_code = r"""
            float local_bsum = result[q_offset].f[0] + result[q_offset].f[1] +
                               result[q_offset].f[2] + result[q_offset].f[3];
            atomicAdd(&bsum[filter_id], local_bsum);
"""
    else:
        bsum_code = ""

    if operation == "update":
        a_name = "image"
        b_name = "error"
    else:
        if operation == "fprop":
            a_name = "image"
            b_name = "filter"
        elif operation == "bprop":
            a_name = "error"
            b_name = "filter"

    if filter_bounds_check:
        filter_load_cond = "int filter_load_in_bounds = (((filter_id + threadIdx.x) << 2) < K);"
        check_filter_cond = "(!filter_load_in_bounds) ? make_float4(0, 0, 0, 0) :"
    else:
        filter_load_cond = ""
        check_filter_cond = ""

    header_code = r"""
#define TILE_DIM            32
#define ITEMS_PER_THREAD    4
#define THREADS_DIM         8

#define REG_TILE_X          4
#define REG_TILE_Y          4
#define THREADS_DIM_X       8
#define THREADS_DIM_Y       8
#define SM_TILE_X           (REG_TILE_X * THREADS_DIM_X)
#define SM_TILE_Y           (REG_TILE_Y * THREADS_DIM_Y)

#define NUM_ROWS            8
#define FILTER_SIZE         %(filter_size)s
#define MAGIC_FILTER_SIZE   %(magic_filter_size)s
#define SHIFT_FILTER_SIZE   %(shift_filter_size)s

typedef union Matrix {
    %(type)s4 f4;
    %(type)s f[4];
} Matrix;

__device__ inline void _idiv_fast(int numerator, int denominator, float rcp,
                                 int& result, int& remainder)
{
    result = (int)((float)numerator * rcp);
    remainder = numerator - (result * denominator);
    result = (remainder >= denominator) ? (result + 1) : result;
    remainder = (remainder >= denominator) ? (remainder - denominator) : remainder;
}

__device__ inline void _idiv_magic(int numerator, unsigned int magic, unsigned int shift,
                                   int denominator, int& result, int& remainder)
{
    if(magic == 1)
    {
        result = numerator >> shift;
    }
    else
    {
        unsigned long long res64 = numerator * (unsigned long long)magic;
        result = ((int)(res64 >> 32) >> shift);
    }
    remainder = numerator - (result * denominator);
}

__device__ inline void _idiv_magic32(int numerator, unsigned int magic, unsigned int shift,
                                     int denominator, int& result, int& remainder)
{
    if(magic == 1)
    {
        result = numerator >> shift;
    }
    else
    {
        result = ((numerator * magic) >> shift);
    }
    remainder = numerator - (result * denominator);
}

//Note: N and K must be multiples of 4
//blockIdx.x is gemm tile id (K dimension) and output pixel id
//blockIdx.y is gemm tile id (N dimension)
//threadIdx.x is gemm tile offset (K dimension)
//threadIdx.y is gemm tile offset (N dimension)
__global__ void conv_%(operation)s(
                           %(type)s alpha, %(type)s beta,
                           Matrix *I, Matrix *F, Matrix *O, float* bsum,
                           int C, int D, int H, int W, int N,
                           int T, int R, int S, int K,
                           int M, int P, int Q,
                           int stride_w, int stride_h, int padding_w, int padding_h,
                           int dilation_w, int dilation_h,
                           int input_channel_size, int filter_channel_size,
                           int output_filter_size,
                           int output_pixels, int grid_p, int grid_q,
                           unsigned int magic_pq, unsigned int shift_pq,
                           unsigned int magic_q, unsigned int shift_q,
                           unsigned int magic_s, unsigned int shift_s)

"""
    code = r"""
{
    __shared__ int2 lookup_table[FILTER_SIZE];
    __shared__ int lut_size;
    __shared__ Matrix %(a_name)s_data[NUM_ROWS][THREADS_DIM_X];
    __shared__ Matrix %(b_name)s_data[NUM_ROWS][THREADS_DIM_Y];

    int lut_size_local = 0;

    //TODO: Use square access pattern to image data to increase cache hits
    int output_pixel, image_id;
    _idiv_magic(blockIdx.x, magic_pq, shift_pq, output_pixels, image_id, output_pixel);
    image_id = (image_id * blockDim.x);

    //Zig zag along x axis to increase cache hits
    int temp_x, temp_y;
    _idiv_magic(output_pixel, magic_q, shift_q, Q, temp_y, temp_x);
    int output_pixel_x = (temp_y & 1) ? (Q - temp_x - 1) : temp_x;
    int output_pixel_y = temp_y;
    output_pixel = output_pixel_x + (output_pixel_y * Q);

    int filter_id = blockIdx.y * blockDim.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    //Offset buffers based on thread id
    I = &(I[image_id  + threadIdx.x]);
    F = &(F[filter_id + threadIdx.x]);

    %(filter_load_cond)s

    //Compute lookup table for filter/image data
%(lut_code)s

    if(tid == 0)
    {
        lut_size = lut_size_local;
    }

    __syncthreads();

    lut_size_local = lut_size;
    Matrix result[REG_TILE_Y] = {0};
    output_pixel = (output_pixel * N) >> 2;
    if(lut_size_local > 0)
    {
        //Evaluate gemm with outer product dimensions N, K and inner product CRS
        int CRS = lut_size_local * C;

        //Compute magic numbers for division by lut_size
        float reciprocal = 1.0f / (float)lut_size_local;

        //Initialize shared mem for first block
        int crs = CRS %% NUM_ROWS;
        crs = (crs == 0) ? 8 : crs;

        int c, rs;
        _idiv_fast(CRS - threadIdx.y - 1, lut_size_local, reciprocal, c, rs);

        int2 lut_entry = ((threadIdx.y & 7) >= crs) ? make_int2(0, 0) : lookup_table[rs];
        %(a_name)s_data[threadIdx.y][threadIdx.x].f4 =
            ((threadIdx.y & 7) >= crs) ? make_float4(0, 0, 0, 0) :
            I[(c * input_channel_size)  + lut_entry.x].f4;
        %(b_name)s_data[threadIdx.y][threadIdx.x].f4 = %(check_filter_cond)s
            ((threadIdx.y & 7) >= crs) ? make_float4(0, 0, 0, 0) :
            F[(c * filter_channel_size) + lut_entry.y].f4;

        //Iterate over entire filter
        for(crs = CRS - crs - 1; crs > 0; crs -= NUM_ROWS)
        {
            __syncthreads();

            #pragma unroll
            for(int i = 0; i < NUM_ROWS; i++)
            {
                Matrix load_row;
                Matrix load_col;

                load_row.f4 = %(a_name)s_data[i][threadIdx.x].f4;
                load_col.f4 = %(b_name)s_data[i][threadIdx.y].f4;

                //Accumulate product
                #pragma unroll
                for(int q_offset = 0; q_offset < REG_TILE_Y; q_offset++)
                {
                    #pragma unroll
                    for(int p_offset = 0; p_offset < REG_TILE_X; p_offset++)
                    {
                        result[q_offset].f[p_offset] += (load_row.f[p_offset] *
                                                         load_col.f[q_offset]);
                    }
                }
            }

            __syncthreads();

            //Load new image data and filter weights
            _idiv_fast(crs - threadIdx.y, lut_size_local, reciprocal, c, rs);

            lut_entry = lookup_table[rs];
            %(a_name)s_data[threadIdx.y][threadIdx.x].f4 =
                I[(c * input_channel_size)  + lut_entry.x].f4;
            %(b_name)s_data[threadIdx.y][threadIdx.x].f4 =
                %(check_filter_cond)s F[(c * filter_channel_size) + lut_entry.y].f4;
        }

        __syncthreads();

        //Accumulate product for last iteration
        #pragma unroll
        for(int i = 0; i < NUM_ROWS; i++)
        {
            Matrix load_row;
            Matrix load_col;

            load_row.f4 = %(a_name)s_data[i][threadIdx.x].f4;
            load_col.f4 = %(b_name)s_data[i][threadIdx.y].f4;

            //Accumulate product
            #pragma unroll
            for(int q_offset = 0; q_offset < REG_TILE_Y; q_offset++)
            {
                #pragma unroll
                for(int p_offset = 0; p_offset < REG_TILE_X; p_offset++)
                {
                    result[q_offset].f[p_offset] += (load_row.f[p_offset] * load_col.f[q_offset]);
                }
            }
        }
    }

    //Store result
    filter_id = (filter_id + threadIdx.y) << 2;
    if(filter_id < K)
    {
        image_id += threadIdx.x;

        #pragma unroll
        for(int q_offset = 0; q_offset < 4; q_offset++)
        {
            if(filter_id < K)
            {
                int out_index = (filter_id * output_filter_size) + output_pixel + image_id;
                %(bsum_code)s

                Matrix cur_value = {0};
                if(beta > 0.0f)
                {
                    cur_value.f4 = O[out_index].f4;
                }

                result[q_offset].f[0] = (result[q_offset].f[0] * alpha) + (cur_value.f[0] * beta);
                result[q_offset].f[1] = (result[q_offset].f[1] * alpha) + (cur_value.f[1] * beta);
                result[q_offset].f[2] = (result[q_offset].f[2] * alpha) + (cur_value.f[2] * beta);
                result[q_offset].f[3] = (result[q_offset].f[3] * alpha) + (cur_value.f[3] * beta);

                O[out_index].f4 = result[q_offset].f4;
            }
            filter_id++;
        }
    }
}
"""

    update_code = r"""
{
    __shared__ Matrix %(a_name)s_data[TILE_DIM / 4][THREADS_DIM * 4 + 4];
    __shared__ Matrix %(b_name)s_data[TILE_DIM / 4][THREADS_DIM * 4 + 4];

    //TODO: Use square access pattern to image data to increase cache hits
    int output_pixel, filter_id;
    _idiv_magic(blockIdx.x, magic_pq, shift_pq, output_pixels, filter_id, output_pixel);
    filter_id = filter_id * TILE_DIM;
    int load_filter_id = filter_id + threadIdx.y;

    int filter_pixel_id = blockIdx.y * TILE_DIM;

    //TODO: Zig zag along x axis to increase cache hits
    int output_pixel_x, output_pixel_y;
    _idiv_magic(output_pixel, magic_q, shift_q, grid_q, output_pixel_y, output_pixel_x);

    //Compute input image and filter offsets for this pixel
    int c, rs;
    int crs = filter_pixel_id + threadIdx.y;
    _idiv_magic(crs, MAGIC_FILTER_SIZE, SHIFT_FILTER_SIZE, FILTER_SIZE, c, rs);

    int filter_x, filter_y;
    _idiv_magic32(rs, magic_s, shift_s, S, filter_y, filter_x);

    int output_pixel_x_save = output_pixel_x;
    for(; output_pixel_y < P; output_pixel_y += grid_p)
    {
        for(output_pixel_x = output_pixel_x_save; output_pixel_x < Q; output_pixel_x += grid_q)
        {
            int base_x = output_pixel_x * stride_w - padding_w + filter_x * dilation_w;
            int base_y = output_pixel_y * stride_h - padding_h + filter_y * dilation_h;
            int crs_in_bounds = (c < C) && (base_x >= 0) && (base_x < W) &&
                                (base_y >= 0) && (base_y < H);
            int input_pixel = W * base_y + base_x;
            output_pixel = output_pixel_x + (Q * output_pixel_y);

            //Pre-multiply offset to simplify indexing
            input_pixel = (input_pixel * N) >> 2;
            output_pixel = (output_pixel * N) >> 2;

            //Evaluate gemm with outer product dimensions N, K and inner product CRS
            Matrix result[ITEMS_PER_THREAD] = {0};

            //Load image data and transpose into shared mem
            //TODO: pad shared memory to avoid bank conflicts
            Matrix buffer;
            buffer.f4 = crs_in_bounds ?
                        I[(c * input_channel_size) + input_pixel + threadIdx.x].f4 :
                        make_float4(0, 0, 0, 0);
            %(a_name)s_data[threadIdx.x][ 0 | threadIdx.y >> 2].f[threadIdx.y & 3] = buffer.f[0];
            %(a_name)s_data[threadIdx.x][ 8 | threadIdx.y >> 2].f[threadIdx.y & 3] = buffer.f[1];
            %(a_name)s_data[threadIdx.x][16 | threadIdx.y >> 2].f[threadIdx.y & 3] = buffer.f[2];
            %(a_name)s_data[threadIdx.x][24 | threadIdx.y >> 2].f[threadIdx.y & 3] = buffer.f[3];

            //Load error data and transpose into shared mem
            buffer.f4 = (load_filter_id < K) ?
                        F[(load_filter_id * output_filter_size) + output_pixel + threadIdx.x].f4 :
                        make_float4(0, 0, 0, 0);
            %(b_name)s_data[threadIdx.x][ 0 | threadIdx.y >> 2].f[threadIdx.y & 3] = buffer.f[0];
            %(b_name)s_data[threadIdx.x][ 8 | threadIdx.y >> 2].f[threadIdx.y & 3] = buffer.f[1];
            %(b_name)s_data[threadIdx.x][16 | threadIdx.y >> 2].f[threadIdx.y & 3] = buffer.f[2];
            %(b_name)s_data[threadIdx.x][24 | threadIdx.y >> 2].f[threadIdx.y & 3] = buffer.f[3];

            //Iterate over entire minibatch
            for(int n = threadIdx.x + (TILE_DIM >> 2); n < (N >> 2); n += (TILE_DIM >> 2))
            {
                __syncthreads();

                #pragma unroll
                for(int i = 0; i < (TILE_DIM >> 2); i++)
                {
                    Matrix row_image;
                    Matrix row_error;

                    row_image.f4 =
                        %(a_name)s_data[i][((threadIdx.y & 3) << 3) | threadIdx.y >> 2].f4;
                    row_error.f4 =
                        %(b_name)s_data[i][((threadIdx.y & 3) << 3) | threadIdx.x].f4;

                    //Accumulate product
                    #pragma unroll
                    for(int q_offset = 0; q_offset < ITEMS_PER_THREAD; q_offset++)
                    {
                        #pragma unroll
                        for(int p_offset = 0; p_offset < ITEMS_PER_THREAD; p_offset++)
                        {
                            result[p_offset].f[q_offset] +=
                                (row_image.f[p_offset] * row_error.f[q_offset]);
                        }
                    }
                }

                __syncthreads();

                //Load image data and transpose into shared mem
                buffer.f4 = crs_in_bounds ?
                    I[(c * input_channel_size) + input_pixel + n].f4 :
                    make_float4(0, 0, 0, 0);
                %(a_name)s_data[threadIdx.x][ 0 | threadIdx.y >> 2].f[threadIdx.y & 3] =
                    buffer.f[0];
                %(a_name)s_data[threadIdx.x][ 8 | threadIdx.y >> 2].f[threadIdx.y & 3] =
                    buffer.f[1];
                %(a_name)s_data[threadIdx.x][16 | threadIdx.y >> 2].f[threadIdx.y & 3] =
                    buffer.f[2];
                %(a_name)s_data[threadIdx.x][24 | threadIdx.y >> 2].f[threadIdx.y & 3] =
                    buffer.f[3];

                //Load error data and transpose into shared mem
                buffer.f4 = (load_filter_id < K) ?
                    F[(load_filter_id * output_filter_size) + output_pixel + n].f4 :
                    make_float4(0, 0, 0, 0);
                %(b_name)s_data[threadIdx.x][ 0 | threadIdx.y >> 2].f[threadIdx.y & 3] =
                    buffer.f[0];
                %(b_name)s_data[threadIdx.x][ 8 | threadIdx.y >> 2].f[threadIdx.y & 3] =
                    buffer.f[1];
                %(b_name)s_data[threadIdx.x][16 | threadIdx.y >> 2].f[threadIdx.y & 3] =
                    buffer.f[2];
                %(b_name)s_data[threadIdx.x][24 | threadIdx.y >> 2].f[threadIdx.y & 3] =
                    buffer.f[3];
            }

            __syncthreads();

            //Accumulate product for last iteration
            #pragma unroll
            for(int i = 0; i < (TILE_DIM >> 2); i++)
            {
                Matrix row_image;
                Matrix row_error;

                row_image.f4 = %(a_name)s_data[i][((threadIdx.y & 3) << 3) | threadIdx.y >> 2].f4;
                row_error.f4 = %(b_name)s_data[i][((threadIdx.y & 3) << 3) | threadIdx.x].f4;

                //Accumulate product
                #pragma unroll
                for(int q_offset = 0; q_offset < ITEMS_PER_THREAD; q_offset++)
                {
                    #pragma unroll
                    for(int p_offset = 0; p_offset < ITEMS_PER_THREAD; p_offset++)
                    {
                        result[p_offset].f[q_offset] +=
                            (row_image.f[p_offset] * row_error.f[q_offset]);
                    }
                }
            }

            //Reduce result between threads in warp
            Matrix outbound;
            int warp_y = threadIdx.y & 3;
            int warp_id = threadIdx.x + (threadIdx.y << 3);
            buffer.f4 = (warp_y == 0) ? result[0].f4 :
                        (warp_y == 1) ? result[1].f4 :
                        (warp_y == 2) ? result[2].f4 :
                        result[3].f4;

            outbound.f4 = (warp_y == 0) ? result[3].f4 :
                          (warp_y == 1) ? result[0].f4 :
                          (warp_y == 2) ? result[1].f4 :
                          result[2].f4;
            buffer.f[0] += __shfl(outbound.f[0], warp_id + 8);
            buffer.f[1] += __shfl(outbound.f[1], warp_id + 8);
            buffer.f[2] += __shfl(outbound.f[2], warp_id + 8);
            buffer.f[3] += __shfl(outbound.f[3], warp_id + 8);

            outbound.f4 = (warp_y == 0) ? result[2].f4 :
                          (warp_y == 1) ? result[3].f4 :
                          (warp_y == 2) ? result[0].f4 :
                          result[1].f4;
            buffer.f[0] += __shfl(outbound.f[0], warp_id + 16);
            buffer.f[1] += __shfl(outbound.f[1], warp_id + 16);
            buffer.f[2] += __shfl(outbound.f[2], warp_id + 16);
            buffer.f[3] += __shfl(outbound.f[3], warp_id + 16);

            outbound.f4 = (warp_y == 0) ? result[1].f4 :
                          (warp_y == 1) ? result[2].f4 :
                          (warp_y == 2) ? result[3].f4 :
                          result[0].f4;
            buffer.f[0] += __shfl(outbound.f[0], warp_id + 24);
            buffer.f[1] += __shfl(outbound.f[1], warp_id + 24);
            buffer.f[2] += __shfl(outbound.f[2], warp_id + 24);
            buffer.f[3] += __shfl(outbound.f[3], warp_id + 24);

            //Store result
            int idx_filter_id = filter_id + (threadIdx.x << 2);
            if(idx_filter_id < K && crs_in_bounds)
            {
                int out_index = (c * filter_channel_size) + (((rs * K) + (idx_filter_id)) >> 2);

                atomicAdd(&O[out_index].f[0], buffer.f[0]);
                atomicAdd(&O[out_index].f[1], buffer.f[1]);
                atomicAdd(&O[out_index].f[2], buffer.f[2]);
                atomicAdd(&O[out_index].f[3], buffer.f[3]);
            }
        }
    }
}
"""
    if operation == "update":
        code = header_code + update_code
    else:
        code = header_code + code

    magic = _magic64(filter_size)

    code = code % {
        "filter_size": filter_size,
        "magic_filter_size": magic[0],
        "shift_filter_size": magic[1],
        "type": _ew_types[dtype]["type"],
        "lut_code": lut_code,
        "bsum_code": bsum_code,
        "operation": operation,
        "a_name": a_name,
        "b_name": b_name,
        "filter_load_cond": filter_load_cond,
        "check_filter_cond": check_filter_cond
    }

    options = ["--use_fast_math"]
    if debug and operation == "bprop":
        options = options + ["-g", "-G"]
    module = SourceModule(code, options=options)

    kernel = module.get_function("conv_" + operation)
    kernel.prepare("ffPPPPIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    kernel.name = "conv_" + operation
    return kernel
