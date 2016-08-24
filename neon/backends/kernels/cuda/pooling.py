from builtins import str
from pycuda.tools import context_dependent_memoize
from neon.backends import cuda_templates

from neon.backends.cuda_templates import (_common_fp16_to_fp32,
                                          _common_round,  # for fp32_to_fp16 converter
                                          _common_max_abs,
                                          _common_kepler,
                                          _ew_types)
from neon.backends.util.source_module import SourceModule

"""
CUDA kernels for pooling layers, with support for max pooling and average pooling.
For each pooling type, there is an fprop function, a bprop function, and a bprop
for overlapping kernels. Each of the six kernels uses templating to perform dtype
conversion so it works for all data types (currently fp32 and fp16 are supported).
Additionally, there are templates for statistics collection, currently supporting
a global max abs over the output tensor which is passed in as an additional kernel
argument.
"""


def map_string2func(funcname, clss, compute_capability):
    """
    Helper function that converts string function names to function calls
    """
    if "_get_" + funcname not in globals():
        raise AttributeError("kernel type '" + funcname + "' not understood")
    return globals()["_get_" + funcname](clss, compute_capability)


# this template is used to hide variables that are only defined conditionally.
atomic_max = r"""
atomicMax(maxabs, intermediate_max);
"""

_common_divmod = r"""
__device__ __forceinline__ int div16(int numerator, int magic, int shift)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, 0;" : "=r"(res) : "r"(numerator), "r"(magic));
    return res >> shift;
}
__device__ __forceinline__ int mod16(int numerator, int div, int maxdiv)
{
    int res;
    asm("vmad.s32.u32.u32 %0, -%1.h0, %2.h0, %3;" : "=r"(res) : "r"(div), "r"(maxdiv), "r"(numerator));
    return res;
}
__device__ __forceinline__ int mad16(int a, int b, int c)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}
__device__ __forceinline__ int msub16(int a, int b, int c)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, -%3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}
"""

def prepare_template_vals(dtype, compute_capability, rounding=False):
    """
    Set up template code snippets that are reused across multiple kernels.
    Most are data type conversion and statistics collection related.
    """
    template_vals = dict()
    for key in ("inits", "finish", "stats_args", "mul_by_scale", "atomic_max", "cvt_out"):
        template_vals[key] = ""

    template_vals["common"] = _common_divmod

    if rounding:
        template_vals["common"] += _common_urand_gen
        template_vals["common"] += _common_round["nearest"].get(dtype, "")
        template_vals["inits"] += _init_rand_func + _init_rand_round_func
        template_vals["finish"] += _finish_rand_func
        mode = "random"
    else:
        mode = "nearest"

    template_vals["common"] += _common_round[mode].get(dtype, "")
    template_vals["common"] += _common_max_abs

    if (compute_capability[0] == 3 and compute_capability[1] < 5) or compute_capability[0] < 3:
        template_vals["common"] += _common_kepler

    template_vals["type"] = _ew_types[dtype]["type"]
    template_vals["cvt"] = _ew_types[dtype]["cvt"]

    if dtype == "f2":
        template_vals["common"] += _common_fp16_to_fp32
        template_vals["cvt_out"] = "fp32_to_fp16"
    elif dtype == "x2":
        template_vals["stats_args"] += ", int* maxabs, float scale0"
        template_vals["cvt"] = "(float)"
        template_vals["cvt_out"] = "fp32_to_int16"
        template_vals["mul_by_scale"] += "1/scale0 *"
        template_vals["atomic_max"] += atomic_max
    elif dtype == "f4":
        pass
    else:
        raise ValueError("Did not understand clss dtype " + str(dtype))

    return template_vals


# This section of the code contains templated CUDA-C code for the kernels.
@context_dependent_memoize
def _get_fprop_max(clss, compute_capability):

    code = r"""
#define FLT_MAX 3.402823466E+38F

%(common)s

__global__ void spool_fprop_max(
    const %(type)s* I, %(type)s* O, unsigned char* A,
    float alpha, float beta, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN, int P, int Q,
    int magic_P, int shift_P, int QN, int PQN, int MPQN,
    int pad_c, int pad_d, int pad_h, int pad_w,
    int str_c, int str_d, int str_h, int str_w,
    int S, int RS, int RST, int JRST,
    int magic_S, int shift_S,
    int magic_RS, int shift_RS, int magic_RST, int shift_RST,
    int supP, int supQ, int shlP, int maskP, int shrP,
    int shlQ, int maskQ, int shrQ, int maskN, int shrN
    %(stats_args)s
    )
{
    extern __shared__ int lut[];
    int tid = threadIdx.x;

    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*supP;

    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = supQ - q - 1;

    // Superblock P and Q
    p = (p << shlP) + ((tid & maskP) >> shrP);
    q = (q << shlQ) + ((tid & maskQ) >> shrQ);
    int n = tid & maskN;

    int sb = tid >> shrN;

    int offset = k*MPQN + m*PQN + p*QN + mad16(q, N, n);
    I += n;
    O += offset;
    A += offset;

    float O_val = beta != 0.0f && p < P && q < Q && n < N ? %(cvt)s(__ldg(O)) : 0.0f;

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int inc = min(maskN + 1, 32);

        int jrst = n;
        while (jrst < JRST)
        {
            int j   = div16(jrst, magic_RST, shift_RST);
            int rst = mod16(jrst, j, RST);

            int t   = div16(rst, magic_RS, shift_RS);
            int rs  = mod16(rst, t, RS);

            int r   = div16(rs, magic_S, shift_S);
            int s   = mod16(rs, r, S);

            int x = qs + s;
            int y = pr + r;
            int z = mt + t;
            int c = kj + j;

            bool bounds_x  = x >= 0 && x < W;
            bool bounds_y  = y >= 0 && y < H;
            bool bounds_z  = z >= 0 && z < D;
            bool bounds_c  = c >= 0 && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            int lut_offset = mad16(sb, JRST, jrst);

            lut[lut_offset] = in_bounds ? sliceI : -1;
            jrst += inc;
        }
    }
    __syncthreads();

    int intermediate_max = 0;

    if (p < P && q < Q && n < N)
    {
        int jrst = 0;
        int argmax = 0;
        float max = -FLT_MAX;
        while (jrst < JRST)
        {
            int lut_offset = mad16(sb, JRST, jrst);

            int slice0 = lut[lut_offset + 0];
            int slice1 = lut[lut_offset + 1];
            int slice2 = lut[lut_offset + 2];
            int slice3 = lut[lut_offset + 3];

            // val needs to stay in fp32 or can't be se to FLT_MAX
            float val0 = jrst + 0 < JRST && slice0 >= 0 ? %(cvt)s(__ldg(I + slice0)) : -FLT_MAX;
            float val1 = jrst + 1 < JRST && slice1 >= 0 ? %(cvt)s(__ldg(I + slice1)) : -FLT_MAX;
            float val2 = jrst + 2 < JRST && slice2 >= 0 ? %(cvt)s(__ldg(I + slice2)) : -FLT_MAX;
            float val3 = jrst + 3 < JRST && slice3 >= 0 ? %(cvt)s(__ldg(I + slice3)) : -FLT_MAX;

            if (val0 > max) {
                max = val0;
                argmax = jrst + 0;
            }
            if (val1 > max) {
                max = val1;
                argmax = jrst + 1;
            }
            if (val2 > max) {
                max = val2;
                argmax = jrst + 2;
            }
            if (val3 > max) {
                max = val3;
                argmax = jrst + 3;
            }

            jrst += 4;
        }
        // convert back to fp to write out
        %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (max*alpha + O_val*beta));
        if (!(flags & 1)) {
            *O = temp_out;
            *A = (unsigned char)argmax;
        }

        intermediate_max = max_abs(0, temp_out);  // compute abs
    }
    intermediate_max += 0;
    %(atomic_max)s
}
"""

    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_fprop_max")
    sig = "3P 2f 44I" + ("Pf" if (clss[0] == "x") else "")
    kernel.prepare(sig)
    return kernel


@context_dependent_memoize
def _get_fprop_avg(clss, compute_capability):

    code = r"""
%(common)s

__global__ void spool_fprop_avg(
    const %(type)s* I, %(type)s* O, unsigned char* A,
    float alpha, float beta, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN, int P, int Q,
    int magic_P, int shift_P, int QN, int PQN, int MPQN,
    int pad_c, int pad_d, int pad_h, int pad_w,
    int str_c, int str_d, int str_h, int str_w,
    int S, int RS, int RST, int JRST,
    int magic_S, int shift_S,
    int magic_RS, int shift_RS, int magic_RST, int shift_RST,
    int supP, int supQ, int shlP, int maskP, int shrP,
    int shlQ, int maskQ, int shrQ, int maskN, int shrN
    %(stats_args)s
    )
{
    __shared__ float rcpWindowSize[32];
    extern __shared__ int lut[];

    int tid = threadIdx.x;

    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*supP;

    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = supQ - q - 1;

    // Superblock P and Q
    p = (p << shlP) + ((tid & maskP) >> shrP);
    q = (q << shlQ) + ((tid & maskQ) >> shrQ);
    int n = tid & maskN;

    int sb = tid >> shrN;

    I += n;
    O += k*MPQN + m*PQN + p*QN + mad16(q, N, n);

    float O_val = beta != 0.0f && p < P && q < Q && n < N ? %(cvt)s(__ldg(O)) : 0.0f;

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int inc    = min(maskN + 1, 32);
        int sbBits = 1 << min(shrN, 5);
        int sbMask = ~(-1 << sbBits) << mad16(sb, sbBits, 0);

        int window_size = 0;
        int jrst = n;
        while (jrst < JRST)
        {
            int j   = div16(jrst, magic_RST, shift_RST);
            int rst = mod16(jrst, j, RST);

            int t   = div16(rst, magic_RS, shift_RS);
            int rs  = mod16(rst, t, RS);

            int r   = div16(rs, magic_S, shift_S);
            int s   = mod16(rs, r, S);

            int x = qs + s;
            int y = pr + r;
            int z = mt + t;
            int c = kj + j;

            bool bounds_x  = x >= 0 && x < W;
            bool bounds_y  = y >= 0 && y < H;
            bool bounds_z  = z >= 0 && z < D;
            bool bounds_c  = c >= 0 && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            // Count the total valid slices
            window_size += __popc(sbMask & __ballot(in_bounds));

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            int lut_offset = mad16(sb, JRST, jrst);

            lut[lut_offset] = in_bounds ? sliceI : -1;
            jrst += inc;
        }

        // TODO confirm kepler OK
        unsigned int shrN_mask = (shrN < 32) ? max(0, ((1 << shrN) - 1)) : 0xffffffff;
        if((tid & shrN_mask) == 0)
            rcpWindowSize[sb] = 1.0f / (float)window_size;
    }
    __syncthreads();

    float rcp_window_size = rcpWindowSize[sb];

    int intermediate_max = 0;

    if (p < P && q < Q && n < N)
    {
        int jrst = 0;
        float sum = 0.0f;
        while (jrst < JRST)
        {
            int lut_offset = mad16(sb, JRST, jrst);

            int slice0 = lut[lut_offset + 0];
            int slice1 = lut[lut_offset + 1];
            int slice2 = lut[lut_offset + 2];
            int slice3 = lut[lut_offset + 3];

            sum += jrst + 0 < JRST && slice0 >= 0 ? %(cvt)s(__ldg(I + slice0)) : 0.0f;
            sum += jrst + 1 < JRST && slice1 >= 0 ? %(cvt)s(__ldg(I + slice1)) : 0.0f;
            sum += jrst + 2 < JRST && slice2 >= 0 ? %(cvt)s(__ldg(I + slice2)) : 0.0f;
            sum += jrst + 3 < JRST && slice3 >= 0 ? %(cvt)s(__ldg(I + slice3)) : 0.0f;

            jrst += 4;
        }

        // convert back to fp to write out
        %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (sum*rcp_window_size*alpha + O_val*beta));
        if (!(flags & 1)) {
            *O = temp_out;
        }
        // collect max abs stats
        intermediate_max = max_abs(0, temp_out); // compute abs
    }
    intermediate_max += 0;
    %(atomic_max)s
}
"""

    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code, options=["--use_fast_math"])
    kernel = module.get_function("spool_fprop_avg")
    kernel.prepare("3P 2f 44I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel


@context_dependent_memoize
def _get_fprop_lrn(clss, compute_capability):
    """
    Local Response Normalization (LRN) layer.
    Implementation based on fprop_avg kernel.

    Implements the following operation:
    for each output pixel
        x' = x / response
    where the response is
        response = [1 + alpha/N * sum_neighbors x_neighbor**2 ]**beta
    so we compute the pooling output
    """
    code = r"""
%(common)s

__global__ void spool_fprop_lrn(
    const %(type)s* I, %(type)s* O, %(type)s* A,
    float alpha, float beta, float ascale, float bpower, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN, int P, int Q,
    int magic_P, int shift_P, int QN, int PQN, int MPQN,
    int pad_c, int pad_d, int pad_h, int pad_w,
    int str_c, int str_d, int str_h, int str_w,
    int S, int RS, int RST, int JRST,
    int magic_S, int shift_S,
    int magic_RS, int shift_RS, int magic_RST, int shift_RST,
    int supP, int supQ, int shlP, int maskP, int shrP,
    int shlQ, int maskQ, int shrQ, int maskN, int shrN
    %(stats_args)s
    )
{
    __shared__ float rcpWindowSize;
    extern __shared__ int lut[];

    int tid = threadIdx.x;

    // paralellism is over QMPK dimensions (output pixels and ofm's)
    int n  = tid;
    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*P;

    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = Q - q - 1;

    const %(type)s* IonO = I;  // input pixel at output location
    I += n;
    IonO += k*MPQN + m*PQN + p*QN + q*N + n;
    O += k*MPQN + m*PQN + p*QN + q*N + n;
    A += k*MPQN + m*PQN + p*QN + q*N + n;

    float O_val = beta != 0.0f ? %(cvt)s(__ldg(O)) : 0.0f;

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int window_size = 0;
        int jrst = tid;
        // this loop generates the LUT (same for pooling and normalization)
        while (jrst < JRST)
        {
            int j = jrst * magic_RST; j >>= shift_RST;
            int rst = jrst - j * RST;

            int t = rst * magic_RS; t >>= shift_RS;
            int rs = rst - t * RS;

            int r = rs * magic_S; r >>= shift_S;
            int s = rs - r*S;

            int x = qs + s;
            int y = pr + r;
            int z = mt + t;
            int c = kj + j;

            bool bounds_x  = x >= 0 && x < W;
            bool bounds_y  = y >= 0 && y < H;
            bool bounds_z  = z >= 0 && z < D;
            bool bounds_c  = c >= 0 && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            // Count the total valid slices
            window_size += __popc(__ballot(in_bounds));

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            lut[jrst] = in_bounds ? sliceI : -1;
            jrst += 32;
        }

        if(tid == 0)
        {
            //rcpWindowSize = 1.0f / (float)window_size;
            rcpWindowSize = (float)RST/(float)JRST;
        }
    }
    __syncthreads();

    float out = 0.0f;
    float denom;
    float sumsquare = 0.0f;
    float input = 0.0f;
    int jrst = 0;
    while (jrst < JRST)
    {
        int slice0 = lut[jrst + 0];
        int slice1 = lut[jrst + 1];
        int slice2 = lut[jrst + 2];
        int slice3 = lut[jrst + 3];

        // TODO: May not need to load all slices if they are not used.
        input =      jrst + 0 < JRST && slice0 >= 0 ? %(cvt)s(__ldg(I + slice0)) : 0.0f;
        sumsquare += jrst + 0 < JRST && slice0 >= 0 ? input * input: 0.0f;
        input =      jrst + 1 < JRST && slice1 >= 0 ? %(cvt)s(__ldg(I + slice1)) : 0.0f;
        sumsquare += jrst + 1 < JRST && slice1 >= 0 ? input * input: 0.0f;
        input =      jrst + 2 < JRST && slice2 >= 0 ? %(cvt)s(__ldg(I + slice2)) : 0.0f;
        sumsquare += jrst + 2 < JRST && slice2 >= 0 ? input * input: 0.0f;
        input =      jrst + 3 < JRST && slice3 >= 0 ? %(cvt)s(__ldg(I + slice3)) : 0.0f;
        sumsquare += jrst + 3 < JRST && slice3 >= 0 ? input * input: 0.0f;

        jrst += 4;
    }

    denom = (1 + ascale*sumsquare*rcpWindowSize);
    out = %(cvt)s(__ldg(IonO)) / powf(denom, bpower);


    // convert back to fp to write out
    %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (out*alpha + O_val*beta));

    // predicate write with no-op flag
    if (!(flags & 1)) {
        *O = temp_out;
        *A = %(cvt_out)s( %(mul_by_scale)s denom );  // write the denomiantor to address
    }

    // collect max abs stats
    int intermediate_max = max_abs(0, temp_out); // compute abs
    %(atomic_max)s
}
"""

    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_fprop_lrn")
    kernel.prepare("3P 4f 34I 10I" + ("Pf" if (clss[0] == "x") else ""))  # add superblocking parameter
    return kernel


@context_dependent_memoize
def _get_bprop_lrn_overlap(clss, compute_capability):

    code = r"""
%(common)s

union LutEntry {
    struct {
        int slice;
        int argmax;
    } data;
    int2 data2;
};

__global__ void spool_bprop_lrn_overlap(
    const %(type)s* I, const %(type)s* O, const %(type)s* E, %(type)s* delta, const %(type)s* A,
    float alpha, float beta, float ascale, float bpower, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN,
    int magic_H, int shift_H,
    int pad_w, int pad_h, int pad_d, int pad_c,
    int str_w, int str_h, int str_d, int str_c,
    int magic_str_w, int shift_str_w,
    int magic_str_h, int shift_str_h,
    int magic_str_d, int shift_str_d,
    int magic_str_c, int shift_str_c,
    int S, int R, int T, int J, int RS, int RST, int JRST,
    int magic_S, int shift_S, int magic_RS, int shift_RS,
    int magic_RST, int shift_RST,
    int Q, int P, int M, int K, int QN, int PQN, int MPQN
    %(stats_args)s  // template for "int* maxabs, float scale0"
    )
{
    extern __shared__ int2 lut[];
    __shared__ float rcpWindowSize;

    int tid = threadIdx.x;

    int n  = tid;
    int x  = blockIdx.x;
    int yz = blockIdx.y;
    int c  = blockIdx.z;

    int z = yz * magic_H; z >>= shift_H;
    int y = yz - z*H;

    // zigzag q back and forth to improve L2 cache perf
    if (y & 1)
        x = W - x - 1;

    // O E A used inside JRST loop
    O += n;  // output
    E += n;  // error
    A += n;  // denom

    // I E A used for output
    const %(type)s* E_out = E;
    const %(type)s* A_out = A;
    delta += c*DHWN + z*HWN + y*WN + x*N + n;
    I     += c*DHWN + z*HWN + y*WN + x*N + n;
    E_out += c*DHWN + z*HWN + y*WN + x*N;
    A_out += c*DHWN + z*HWN + y*WN + x*N;

    float delta_val = (beta != 0.0f) ? %(cvt)s(__ldg(delta)) : 0.0f;

    // build the lookup table
    if (tid < 32)
    {
        int kj = c - J + pad_c + 1;
        int mt = z - T + pad_d + 1;
        int pr = y - R + pad_h + 1;
        int qs = x - S + pad_w + 1;

        int jrst = tid;
        while (jrst < JRST)
        {
            int j = jrst * magic_RST; j >>= shift_RST;
            int rst = jrst - j * RST;

            int t = rst * magic_RS; t >>= shift_RS;
            int rs = rst - t * RS;

            int r = rs * magic_S; r >>= shift_S;
            int s = rs - r*S;

            int k_prime = kj + j;
            int m_prime = mt + t;
            int p_prime = pr + r;
            int q_prime = qs + s;

            int k     = k_prime * magic_str_c; k >>= shift_str_c;
            int k_mod = k_prime - k*str_c;
            bool k_bounds = k_mod == 0 && k >= 0 && k < K;

            int m     = m_prime * magic_str_d; m >>= shift_str_d;
            int m_mod = m_prime - m*str_d;
            bool m_bounds = m_mod == 0 && m >= 0 && m < M;

            int p     = p_prime * magic_str_h; p >>= shift_str_h;
            int p_mod = p_prime - p*str_h;
            bool p_bounds = p_mod == 0 && p >= 0 && p < P;

            int q     = q_prime * magic_str_w; q >>= shift_str_w;
            int q_mod = q_prime - q*str_w;
            bool q_bounds = q_mod == 0 && q >= 0 && q < Q;

            bool in_bounds = k_bounds && m_bounds && p_bounds && q_bounds;

            int j_prime = c - k_prime + pad_c;
            int t_prime = z - m_prime + pad_d;
            int r_prime = y - p_prime + pad_h;
            int s_prime = x - q_prime + pad_w;

            int sliceI  = k*MPQN + m*PQN + p*QN + q*N;
            int argmaxI = j_prime*RST + t_prime*RS + r_prime*S + s_prime;

            LutEntry entry;
            entry.data.slice  = sliceI;
            entry.data.argmax = in_bounds ? argmaxI : -1;

            lut[jrst] = entry.data2;
            jrst += 32;
        }

        if(tid == 0)
        {
            rcpWindowSize = (float)RST/(float)JRST;
        }
    }
    __syncthreads();

    int jrst = 0;
    // float out = 0.0f;
    float array_delta = 0.0f;
    int intermediate_max = 0;

    while (jrst < JRST)
    {
        LutEntry entry0;
        LutEntry entry1;
        LutEntry entry2;
        LutEntry entry3;

        entry0.data2 = lut[jrst + 0];
        entry1.data2 = lut[jrst + 1];
        entry2.data2 = lut[jrst + 2];
        entry3.data2 = lut[jrst + 3];

        array_delta += (jrst + 0 < JRST && entry0.data.argmax >= 0) ? %(cvt)s(__ldg(O + entry0.data.slice)) * %(cvt)s(__ldg(E + entry0.data.slice)) / %(cvt)s(__ldg(A + entry0.data.slice)) : 0.0f;
        array_delta += (jrst + 1 < JRST && entry1.data.argmax >= 0) ? %(cvt)s(__ldg(O + entry1.data.slice)) * %(cvt)s(__ldg(E + entry1.data.slice)) / %(cvt)s(__ldg(A + entry1.data.slice)) : 0.0f;
        array_delta += (jrst + 2 < JRST && entry2.data.argmax >= 0) ? %(cvt)s(__ldg(O + entry2.data.slice)) * %(cvt)s(__ldg(E + entry2.data.slice)) / %(cvt)s(__ldg(A + entry2.data.slice)) : 0.0f;
        array_delta += (jrst + 3 < JRST && entry3.data.argmax >= 0) ? %(cvt)s(__ldg(O + entry3.data.slice)) * %(cvt)s(__ldg(E + entry3.data.slice)) / %(cvt)s(__ldg(A + entry3.data.slice)) : 0.0f;

        jrst += 4;
    }

    array_delta = -2 * bpower * ascale * __ldg(I) * array_delta * rcpWindowSize + (__ldg(E_out) * powf(__ldg(A_out), -bpower));

    %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (array_delta*alpha + delta_val*beta));
    if (!(flags & 1)) {
        *delta = temp_out;
    }

    // compute max-abs
    intermediate_max = max_abs(intermediate_max, temp_out);  // used for abs
    %(atomic_max)s
}
"""

    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_bprop_lrn_overlap")
    kernel.prepare("5P 4f 47I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel

@context_dependent_memoize
def _get_bprop_max(clss, compute_capability):

    code = r"""

%(common)s

__global__ void spool_bprop_max(
    const %(type)s* I, %(type)s* O, const unsigned char* A,
    float alpha, float beta, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN, int P, int Q,
    int magic_P, int shift_P, int QN, int PQN, int MPQN,
    int pad_c, int pad_d, int pad_h, int pad_w,
    int str_c, int str_d, int str_h, int str_w,
    int S, int RS, int RST, int JRST,
    int magic_S, int shift_S,
    int magic_RS, int shift_RS, int magic_RST, int shift_RST,
    int supP, int supQ, int shlP, int maskP, int shrP,
    int shlQ, int maskQ, int shrQ, int maskN, int shrN
    %(stats_args)s
    )
{
    extern __shared__ int lut[];

    int tid = threadIdx.x;

    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*supP;

    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = supQ - q - 1;

    // Superblock P and Q
    p = (p << shlP) + ((tid & maskP) >> shrP);
    q = (q << shlQ) + ((tid & maskQ) >> shrQ);
    int n = tid & maskN;

    int sb = tid >> shrN;

    int offset = k*MPQN + m*PQN + p*QN + mad16(q, N, n);
    O += n;
    I += offset;
    A += offset;

    float delta = 0.0f;
    int argmax  = -1;
    if (p < P && q < Q && n < N)
    {
        delta  = %(cvt)s(__ldg(I));
        argmax = __ldg(A);
    }

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int inc = min(maskN + 1, 32);

        int jrst = n;
        while (jrst < JRST)
        {
            int j   = div16(jrst, magic_RST, shift_RST);
            int rst = mod16(jrst, j, RST);

            int t   = div16(rst, magic_RS, shift_RS);
            int rs  = mod16(rst, t, RS);

            int r   = div16(rs, magic_S, shift_S);
            int s   = mod16(rs, r, S);

            int x = qs + s;
            int y = pr + r;
            int z = mt + t;
            int c = kj + j;

            bool bounds_x  = x >= 0 && x < W;
            bool bounds_y  = y >= 0 && y < H;
            bool bounds_z  = z >= 0 && z < D;
            bool bounds_c  = c >= 0 && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            int lut_offset = mad16(sb, JRST, jrst);

            lut[lut_offset] = in_bounds ? sliceI : -1;
            jrst += inc;
        }
    }
    __syncthreads();

    int intermediate_max = 0;

    if (p < P && q < Q && n < N)
    {
        delta *= alpha;
        bool load_beta = beta != 0.0f;
        int jrst = 0;

        while (jrst < JRST)
        {
            int lut_offset = mad16(sb, JRST, jrst);

            int offset0 = lut[lut_offset + 0];
            int offset1 = lut[lut_offset + 1];
            int offset2 = lut[lut_offset + 2];
            int offset3 = lut[lut_offset + 3];

            // need to figure out how to write into output. Can't be float * if we write fp16
            // load fp16 from O, so it's an fp16 pointer
            %(type)s* out0 = O + offset0;
            %(type)s* out1 = O + offset1;
            %(type)s* out2 = O + offset2;
            %(type)s* out3 = O + offset3;

            bool valid0 = jrst + 0 < JRST && offset0 >= 0;
            bool valid1 = jrst + 1 < JRST && offset1 >= 0;
            bool valid2 = jrst + 2 < JRST && offset2 >= 0;
            bool valid3 = jrst + 3 < JRST && offset3 >= 0;

            // load input dtype, convert to float32.
            float beta0 = valid0 && load_beta ? %(cvt)s(__ldg(out0)) * beta : 0.0f;
            float beta1 = valid1 && load_beta ? %(cvt)s(__ldg(out1)) * beta : 0.0f;
            float beta2 = valid2 && load_beta ? %(cvt)s(__ldg(out2)) * beta : 0.0f;
            float beta3 = valid3 && load_beta ? %(cvt)s(__ldg(out3)) * beta : 0.0f;

            // convert float32 back into input format to write out
            %(type)s temp_out0 = valid0 ? %(cvt_out)s(%(mul_by_scale)s(jrst + 0 == argmax ? delta + beta0 : beta0)) : 0.0f;
            %(type)s temp_out1 = valid1 ? %(cvt_out)s(%(mul_by_scale)s(jrst + 1 == argmax ? delta + beta1 : beta1)) : 0.0f;
            %(type)s temp_out2 = valid2 ? %(cvt_out)s(%(mul_by_scale)s(jrst + 2 == argmax ? delta + beta2 : beta2)) : 0.0f;
            %(type)s temp_out3 = valid3 ? %(cvt_out)s(%(mul_by_scale)s(jrst + 3 == argmax ? delta + beta3 : beta3)) : 0.0f;

            // predicate writes with no-op flag.
            if (!(flags & 1)) {
                if (valid0) *out0 = temp_out0;
                if (valid1) *out1 = temp_out1;
                if (valid2) *out2 = temp_out2;
                if (valid3) *out3 = temp_out3;
            }
            intermediate_max = max_abs(intermediate_max, temp_out0);
            intermediate_max = max_abs(intermediate_max, temp_out1);
            intermediate_max = max_abs(intermediate_max, temp_out2);
            intermediate_max = max_abs(intermediate_max, temp_out3);

            jrst += 4;
        }
    }
    intermediate_max += 0;
    %(atomic_max)s
}
"""
    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_bprop_max")
    # f = open("spool_bprop_max.cu", "w")
    # print >>f, code
    # f.close()
    kernel.prepare("3P 2f 44I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel


@context_dependent_memoize
def _get_bprop_avg(clss, compute_capability):

    code = r"""

%(common)s

__global__ void spool_bprop_avg(
    const %(type)s* I, %(type)s* O, const unsigned char* A,
    float alpha, float beta, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN, int P, int Q,
    int magic_P, int shift_P, int QN, int PQN, int MPQN,
    int pad_c, int pad_d, int pad_h, int pad_w,
    int str_c, int str_d, int str_h, int str_w,
    int S, int RS, int RST, int JRST,
    int magic_S, int shift_S,
    int magic_RS, int shift_RS, int magic_RST, int shift_RST,
    int supP, int supQ, int shlP, int maskP, int shrP,
    int shlQ, int maskQ, int shrQ, int maskN, int shrN
    %(stats_args)s
    )
{
    __shared__ float rcpWindowSize[32];
    extern __shared__ int lut[];

    int tid = threadIdx.x;

    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*supP;

    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = supQ - q - 1;

    // Superblock P and Q
    p = (p << shlP) + ((tid & maskP) >> shrP);
    q = (q << shlQ) + ((tid & maskQ) >> shrQ);
    int n = tid & maskN;

    int sb = tid >> shrN;

    O += n;
    I += k*MPQN + m*PQN + p*QN + mad16(q, N, n);

    float delta = 0.0f;
    if (p < P && q < Q && n < N)
        delta  = %(cvt)s(__ldg(I));

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int inc    = min(maskN + 1, 32);
        int sbBits = 1 << min(shrN, 5);
        int sbMask = ~(-1 << sbBits) << mad16(sb, sbBits, 0);

        int window_size = 0;
        int jrst = n;
        while (jrst < JRST)
        {
            int j   = div16(jrst, magic_RST, shift_RST);
            int rst = mod16(jrst, j, RST);

            int t   = div16(rst, magic_RS, shift_RS);
            int rs  = mod16(rst, t, RS);

            int r   = div16(rs, magic_S, shift_S);
            int s   = mod16(rs, r, S);

            int x = qs + s;
            int y = pr + r;
            int z = mt + t;
            int c = kj + j;

            bool bounds_x  = x >= 0 && x < W;
            bool bounds_y  = y >= 0 && y < H;
            bool bounds_z  = z >= 0 && z < D;
            bool bounds_c  = c >= 0 && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            // Count the total valid slices
            window_size += __popc(sbMask & __ballot(in_bounds));

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            int lut_offset = mad16(sb, JRST, jrst);

            lut[lut_offset] = in_bounds ? sliceI : -1;
            jrst += inc;
        }
        // TODO confirm kepler OK
        unsigned int shrN_mask = (shrN < 32) ? max(0, ((1 << shrN) - 1)) : 0xffffffff;
        if((tid & shrN_mask) == 0)
            rcpWindowSize[sb] = 1.0f / (float)window_size;
    }
    __syncthreads();

    int intermediate_max = 0;

    if (p < P && q < Q && n < N)
    {
        delta *= alpha * rcpWindowSize[sb];
        bool load_beta = beta != 0.0f;
        int jrst = 0;

        while (jrst < JRST)
        {
            int lut_offset = mad16(sb, JRST, jrst);

            int offset0 = lut[lut_offset + 0];
            int offset1 = lut[lut_offset + 1];
            int offset2 = lut[lut_offset + 2];
            int offset3 = lut[lut_offset + 3];

            %(type)s* out0 = O + offset0;
            %(type)s* out1 = O + offset1;
            %(type)s* out2 = O + offset2;
            %(type)s* out3 = O + offset3;

            bool valid0 = jrst + 0 < JRST && offset0 >= 0;
            bool valid1 = jrst + 1 < JRST && offset1 >= 0;
            bool valid2 = jrst + 2 < JRST && offset2 >= 0;
            bool valid3 = jrst + 3 < JRST && offset3 >= 0;

            float beta0 = valid0 && load_beta ? %(cvt)s(__ldg(out0)) * beta : 0.0f;
            float beta1 = valid1 && load_beta ? %(cvt)s(__ldg(out1)) * beta : 0.0f;
            float beta2 = valid2 && load_beta ? %(cvt)s(__ldg(out2)) * beta : 0.0f;
            float beta3 = valid3 && load_beta ? %(cvt)s(__ldg(out3)) * beta : 0.0f;

            %(type)s temp_out0 = valid0 ? %(cvt_out)s(%(mul_by_scale)s(delta + beta0)) : 0.0f;
            %(type)s temp_out1 = valid1 ? %(cvt_out)s(%(mul_by_scale)s(delta + beta1)) : 0.0f;
            %(type)s temp_out2 = valid2 ? %(cvt_out)s(%(mul_by_scale)s(delta + beta2)) : 0.0f;
            %(type)s temp_out3 = valid3 ? %(cvt_out)s(%(mul_by_scale)s(delta + beta3)) : 0.0f;

            // predicate writes with no-op flag.
            if (!(flags & 1)) {
                if (valid0) *out0 = temp_out0;
                if (valid1) *out1 = temp_out1;
                if (valid2) *out2 = temp_out2;
                if (valid3) *out3 = temp_out3;
            }
            intermediate_max = max_abs(intermediate_max, temp_out0);
            intermediate_max = max_abs(intermediate_max, temp_out1);
            intermediate_max = max_abs(intermediate_max, temp_out2);
            intermediate_max = max_abs(intermediate_max, temp_out3);

            jrst += 4;
        }
    }
    intermediate_max += 0;
    %(atomic_max)s
}
"""
    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code, options=["--use_fast_math"])
    kernel = module.get_function("spool_bprop_avg")
    kernel.prepare("3P 2f 44I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel


@context_dependent_memoize
def _get_bprop_max_overlap(clss, compute_capability):

    code = r"""
%(common)s

union LutEntry {
    struct {
        int slice;
        int argmax;
    } data;
    int2 data2;
};

__global__ void spool_bprop_max_overlap(
    const %(type)s* I, %(type)s* O, const unsigned char* A,
    float alpha, float beta, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN,
    int magic_H, int shift_H,
    int pad_w, int pad_h, int pad_d, int pad_c,
    int str_w, int str_h, int str_d, int str_c,
    int magic_str_w, int shift_str_w,
    int magic_str_h, int shift_str_h,
    int magic_str_d, int shift_str_d,
    int magic_str_c, int shift_str_c,
    int S, int R, int T, int J, int RS, int RST, int JRST,
    int magic_S, int shift_S, int magic_RS, int shift_RS,
    int magic_RST, int shift_RST,
    int Q, int P, int M, int K, int QN, int PQN, int MPQN
    %(stats_args)s  // template for "int* maxabs, float scale0"
    )
{
    int __shared__ lutSize;
    extern __shared__ int2 lut[];

    int tid = threadIdx.x;

    int n  = tid;
    int x  = blockIdx.x;
    int yz = blockIdx.y;
    int c  = blockIdx.z;

    int z = yz * magic_H; z >>= shift_H;
    int y = yz - z*H;

    // zigzag q back and forth to improve L2 cache perf
    if (y & 1)
        x = W - x - 1;

    I += n;
    A += n;
    O += c*DHWN + z*HWN + y*WN + x*N + n;

    float O_val = (beta != 0.0f) ? %(cvt)s(__ldg(O)) : 0.0f;

    int lut_size;
    if (tid < 32)
    {
        int kj = c - J + pad_c + 1;
        int mt = z - T + pad_d + 1;
        int pr = y - R + pad_h + 1;
        int qs = x - S + pad_w + 1;

        unsigned dep_thd_mask = 0xffffffff;
        dep_thd_mask >>= 32 - tid;

        lut_size = 0;

        int jrst = tid;
        while (jrst < JRST)
        {
            int j   = div16(jrst, magic_RST, shift_RST);
            int rst = mod16(jrst, j, RST);

            int t   = div16(rst, magic_RS, shift_RS);
            int rs  = mod16(rst, t, RS);

            int r   = div16(rs, magic_S, shift_S);
            int s   = mod16(rs, r, S);

            int k_prime = kj + j;
            int m_prime = mt + t;
            int p_prime = pr + r;
            int q_prime = qs + s;

            int  k        = div16(k_prime, magic_str_c, shift_str_c);
            int  k_mod    = mod16(k_prime, k, str_c);
            bool k_bounds = k_mod == 0 && k >= 0 && k < K;

            int  m        = div16(m_prime, magic_str_d, shift_str_d);
            int  m_mod    = mod16(m_prime, m, str_d);
            bool m_bounds = m_mod == 0 && m >= 0 && m < M;

            int  p        = div16(p_prime, magic_str_h, shift_str_h);
            int  p_mod    = mod16(p_prime, p, str_h);
            bool p_bounds = p_mod == 0 && p >= 0 && p < P;

            int  q        = div16(q_prime, magic_str_w, shift_str_w);
            int  q_mod    = mod16(q_prime, q, str_w);
            bool q_bounds = q_mod == 0 && q >= 0 && q < Q;

            bool in_bounds = k_bounds && m_bounds && p_bounds && q_bounds;

            // Get a mask of all valid slices in the warp
            unsigned ballot = __ballot(in_bounds);

            // Count the total valid slices
            unsigned warp_slices = __popc(ballot);

            if (in_bounds)
            {
                // Count all the valid slices below this threadid
                unsigned dep_thd_cnt = __popc(dep_thd_mask & ballot);

                int j_prime = c - k_prime + pad_c;
                int t_prime = z - m_prime + pad_d;
                int r_prime = y - p_prime + pad_h;
                int s_prime = x - q_prime + pad_w;

                LutEntry entry;
                entry.data.slice  = k*MPQN + m*PQN + p*QN + mad16(q, N, 0);
                entry.data.argmax = j_prime*RST + mad16(t_prime, RS, mad16(r_prime, S, s_prime));

                lut[lut_size + dep_thd_cnt] = entry.data2;
            }
            lut_size += warp_slices;
            jrst += 32;
        }
        if(tid == 0)
            lutSize = lut_size;
    }
    __syncthreads();

    lut_size = lutSize;

    int jrst = 0;
    float out = 0.0f;
    int intermediate_max = 0;

    while (jrst < lut_size)
    {
        LutEntry entry0;
        LutEntry entry1;
        LutEntry entry2;
        LutEntry entry3;

        entry0.data2 = lut[jrst + 0];
        entry1.data2 = lut[jrst + 1];
        entry2.data2 = lut[jrst + 2];
        entry3.data2 = lut[jrst + 3];

        // argmax
        int fprop_argmax0 = jrst + 0 < lut_size ? __ldg(A + entry0.data.slice) : -2;
        int fprop_argmax1 = jrst + 1 < lut_size ? __ldg(A + entry1.data.slice) : -2;
        int fprop_argmax2 = jrst + 2 < lut_size ? __ldg(A + entry2.data.slice) : -2;
        int fprop_argmax3 = jrst + 3 < lut_size ? __ldg(A + entry3.data.slice) : -2;

        out += jrst + 0 < lut_size && fprop_argmax0 == entry0.data.argmax ? %(cvt)s(__ldg(I + entry0.data.slice)) : 0.0f;
        out += jrst + 1 < lut_size && fprop_argmax1 == entry1.data.argmax ? %(cvt)s(__ldg(I + entry1.data.slice)) : 0.0f;
        out += jrst + 2 < lut_size && fprop_argmax2 == entry2.data.argmax ? %(cvt)s(__ldg(I + entry2.data.slice)) : 0.0f;
        out += jrst + 3 < lut_size && fprop_argmax3 == entry3.data.argmax ? %(cvt)s(__ldg(I + entry3.data.slice)) : 0.0f;

        jrst += 4;
    }
    %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (out*alpha + O_val*beta));
    if (!(flags & 1)) {
        *O = temp_out;
    }
    // compute max-abs
    intermediate_max = max_abs(intermediate_max, temp_out);  // used for abs
    %(atomic_max)s
}
"""
    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_bprop_max_overlap")
    kernel.prepare("3P 2f 47I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel

@context_dependent_memoize
def _get_bprop_avg_overlap(clss, compute_capability):

    code = r"""

%(common)s

union LutEntry {
    struct {
        int   slice;
        float rcp_in;
    } data;
    int2 data2;
};

__device__ __forceinline__ int imin(int val1, int val2)
{
    int ret;
    asm("min.s32 %%0, %%1, %%2;" : "=r"(ret) : "r"(val1), "r"(val2));
    return ret;
}

__global__ void spool_bprop_avg_overlap(
    const %(type)s* I, %(type)s* O, const unsigned char* A,
    float alpha, float beta, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN,
    int magic_H, int shift_H,
    int pad_w, int pad_h, int pad_d, int pad_c,
    int str_w, int str_h, int str_d, int str_c,
    int magic_str_w, int shift_str_w,
    int magic_str_h, int shift_str_h,
    int magic_str_d, int shift_str_d,
    int magic_str_c, int shift_str_c,
    int S, int R, int T, int J, int RS, int RST, int JRST,
    int magic_S, int shift_S, int magic_RS, int shift_RS,
    int magic_RST, int shift_RST,
    int Q, int P, int M, int K, int QN, int PQN, int MPQN
    %(stats_args)s  // template for "int* maxabs, float scale0"
    )
{
    int    __shared__ lutSize;
    extern __shared__ int2 lut[];

    int tid = threadIdx.x;

    int n  = tid;
    int x  = blockIdx.x;
    int yz = blockIdx.y;
    int c  = blockIdx.z;

    int z = yz * magic_H; z >>= shift_H;
    int y = yz - z*H;

    // zigzag q back and forth to improve L2 cache perf
    if (y & 1)
        x = W - x - 1;

    I += n;
    O += c*DHWN + z*HWN + y*WN + x*N + n;

    float O_val = beta != 0.0f ? %(cvt)s(__ldg(O)) : 0.0f;
    int lut_size;

    if (tid < 32)
    {
        int kj = c - J + pad_c + 1;
        int mt = z - T + pad_d + 1;
        int pr = y - R + pad_h + 1;
        int qs = x - S + pad_w + 1;

        unsigned dep_thd_mask = 0xffffffff;
        dep_thd_mask >>= 32 - tid;

        lut_size = 0;

        int jrst = tid;
        while (jrst < JRST)
        {
            int j   = div16(jrst, magic_RST, shift_RST);
            int rst = mod16(jrst, j, RST);

            int t   = div16(rst, magic_RS, shift_RS);
            int rs  = mod16(rst, t, RS);

            int r   = div16(rs, magic_S, shift_S);
            int s   = mod16(rs, r, S);

            int k_prime = kj + j;
            int m_prime = mt + t;
            int p_prime = pr + r;
            int q_prime = qs + s;

            int  k        = div16(k_prime, magic_str_c, shift_str_c);
            int  k_mod    = mod16(k_prime, k, str_c);
            bool k_bounds = k_mod == 0 && k >= 0 && k < K;

            int  m        = div16(m_prime, magic_str_d, shift_str_d);
            int  m_mod    = mod16(m_prime, m, str_d);
            bool m_bounds = m_mod == 0 && m >= 0 && m < M;

            int  p        = div16(p_prime, magic_str_h, shift_str_h);
            int  p_mod    = mod16(p_prime, p, str_h);
            bool p_bounds = p_mod == 0 && p >= 0 && p < P;

            int  q        = div16(q_prime, magic_str_w, shift_str_w);
            int  q_mod    = mod16(q_prime, q, str_w);
            bool q_bounds = q_mod == 0 && q >= 0 && q < Q;

            bool in_bounds = k_bounds && m_bounds && p_bounds && q_bounds;

            // Get a mask of all valid slices in the warp
            unsigned ballot = __ballot(in_bounds);

            // Count the total valid slices
            unsigned warp_slices = __popc(ballot);

            if (in_bounds)
            {
                // Count all the valid slices below this threadid
                unsigned dep_thd_cnt = __popc(dep_thd_mask & ballot);

                int c_left = msub16(k, str_c, pad_c);
                int z_left = msub16(m, str_d, pad_d);
                int y_left = msub16(p, str_h, pad_h);
                int x_left = msub16(q, str_w, pad_w);

                float k_in = (float)imin( imin(J + c_left, J), imin(C - c_left, J) );
                float m_in = (float)imin( imin(T + z_left, T), imin(D - z_left, T) );
                float p_in = (float)imin( imin(R + y_left, R), imin(H - y_left, R) );
                float q_in = (float)imin( imin(S + x_left, S), imin(W - x_left, S) );

                float total_in = q_in * p_in * m_in * k_in;

                float rcp_in = total_in > 0.0f ? 1.0f / total_in : 0.0f;

                LutEntry entry;
                entry.data.slice  = k*MPQN + m*PQN + p*QN + mad16(q, N, 0);
                entry.data.rcp_in = rcp_in;

                lut[lut_size + dep_thd_cnt] = entry.data2;
            }
            lut_size += warp_slices;
            jrst += 32;
        }
        if(tid==0)
            lutSize = lut_size;
    }
    __syncthreads();

    lut_size = lutSize;

    int jrst = 0;
    float out = 0.0f;
    int intermediate_max = 0;

    while (jrst < lut_size)
    {
        LutEntry entry0;
        LutEntry entry1;
        LutEntry entry2;
        LutEntry entry3;

        entry0.data2 = lut[jrst + 0];
        entry1.data2 = lut[jrst + 1];
        entry2.data2 = lut[jrst + 2];
        entry3.data2 = lut[jrst + 3];

        out += jrst + 0 < lut_size ? %(cvt)s(__ldg(I + entry0.data.slice)) * entry0.data.rcp_in : 0.0f;
        out += jrst + 1 < lut_size ? %(cvt)s(__ldg(I + entry1.data.slice)) * entry1.data.rcp_in : 0.0f;
        out += jrst + 2 < lut_size ? %(cvt)s(__ldg(I + entry2.data.slice)) * entry2.data.rcp_in : 0.0f;
        out += jrst + 3 < lut_size ? %(cvt)s(__ldg(I + entry3.data.slice)) * entry3.data.rcp_in : 0.0f;

        jrst += 4;
    }

    %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (out*alpha + O_val*beta));
    if (!(flags & 1)) {
        *O = temp_out;
    }

    // max-abs over unrolls
    intermediate_max = max_abs(intermediate_max, temp_out);  // used for abs

    %(atomic_max)s

}
"""
    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code, options=["--use_fast_math"])
    kernel = module.get_function("spool_bprop_avg_overlap")
    kernel.prepare("3P 2f 47I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel


@context_dependent_memoize
def _get_bprop_max_overlap_smallN(clss, compute_capability):

    code = r"""
%(common)s

union LutEntry {
    struct {
        int slice;
        int argmax;
    } data;
    int2 data2;
};

__global__ void spool_bprop_max_overlap_smallN(
    const %(type)s* I, %(type)s* O, const unsigned char* A,
    float alpha, float beta, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN,
    int magic_H, int shift_H,
    int pad_w, int pad_h, int pad_d, int pad_c,
    int str_w, int str_h, int str_d, int str_c,
    int magic_str_w, int shift_str_w,
    int magic_str_h, int shift_str_h,
    int magic_str_d, int shift_str_d,
    int magic_str_c, int shift_str_c,
    int S, int R, int T, int J, int RS, int RST, int JRST,
    int magic_S, int shift_S, int magic_RS, int shift_RS,
    int magic_RST, int shift_RST,
    int Q, int P, int M, int K, int QN, int PQN, int MPQN,
    int supH, int supW, int shlH, int maskH, int shrH,
    int shlW, int maskW, int shrW, int maskN, int shrN, int maxLutSize
    %(stats_args)s  // template for "int* maxabs, float scale0"
    )
{
    extern __shared__ int2 lut[];

    int tid = threadIdx.x;

    int x  = blockIdx.x;
    int yz = blockIdx.y;
    int c  = blockIdx.z;

    int z = yz * magic_H; z >>= shift_H;
    int y = yz - z*supH;

    // zigzag q back and forth to improve L2 cache perf
    if (y & 1)
        x = supW - x - 1;

    // Superblock H and W
    y = (y << shlH) + ((tid & maskH) >> shrH);
    x = (x << shlW) + ((tid & maskW) >> shrW);
    int n = tid & maskN;

    int sb = tid >> shrN;

    I += n;
    A += n;
    O += c*DHWN + z*HWN + y*WN + mad16(x, N, n);

    float O_val = beta != 0.0f && y < H && x < W && n < N ? %(cvt)s(__ldg(O)) : 0.0f;

    int kj = c - J + pad_c + 1;
    int mt = z - T + pad_d + 1;
    int pr = y - R + pad_h + 1;
    int qs = x - S + pad_w + 1;

    int sbSize = maskN + 1;
    int sbBits = mad16(sb, sbSize, 0);
    unsigned sbMask = ~(0xffffffff << sbSize) << sbBits;
    unsigned dep_thd_mask = (0xffffffff >> (32 - n)) << sbBits;

    int lut_offset = mad16(sb, maxLutSize, 0);
    int lut_size = 0;
    int jrst = n;
    int JRSTend = JRST;
    if (JRSTend & maskN)
        JRSTend += sbSize - (JRSTend & maskN);

    while (jrst < JRSTend)
    {
        int j   = div16(jrst, magic_RST, shift_RST);
        int rst = mod16(jrst, j, RST);

        int t   = div16(rst, magic_RS, shift_RS);
        int rs  = mod16(rst, t, RS);

        int r   = div16(rs, magic_S, shift_S);
        int s   = mod16(rs, r, S);

        int k_prime = kj + j;
        int m_prime = mt + t;
        int p_prime = pr + r;
        int q_prime = qs + s;

        int  k        = div16(k_prime, magic_str_c, shift_str_c);
        int  k_mod    = mod16(k_prime, k, str_c);
        bool k_bounds = k_mod == 0 && k >= 0 && k < K;

        int  m        = div16(m_prime, magic_str_d, shift_str_d);
        int  m_mod    = mod16(m_prime, m, str_d);
        bool m_bounds = m_mod == 0 && m >= 0 && m < M;

        int  p        = div16(p_prime, magic_str_h, shift_str_h);
        int  p_mod    = mod16(p_prime, p, str_h);
        bool p_bounds = p_mod == 0 && p >= 0 && p < P;

        int  q        = div16(q_prime, magic_str_w, shift_str_w);
        int  q_mod    = mod16(q_prime, q, str_w);
        bool q_bounds = q_mod == 0 && q >= 0 && q < Q;

        bool in_bounds = jrst < JRST && k_bounds && m_bounds && p_bounds && q_bounds;

        // Get a mask of all valid slices in the warp
        unsigned ballot = __ballot(in_bounds);

        // Count the total valid slices in this superblock
        int sb_slices = __popc(sbMask & ballot);

        if (in_bounds)
        {
            // Count all the valid slices below this threadid
            int dep_thd_cnt = __popc(dep_thd_mask & ballot);

            int j_prime = c - k_prime + pad_c;
            int t_prime = z - m_prime + pad_d;
            int r_prime = y - p_prime + pad_h;
            int s_prime = x - q_prime + pad_w;

            int sliceI  = k*MPQN + m*PQN + p*QN + mad16(q, N, 0);
            int argmaxI = j_prime*RST + mad16(t_prime, RS, mad16(r_prime, S, s_prime));

            LutEntry entry;
            entry.data.slice  = sliceI;
            entry.data.argmax = argmaxI;

            lut[lut_offset + lut_size + dep_thd_cnt] = entry.data2;
        }
        lut_size += sb_slices;
        jrst += sbSize;
    }

    int intermediate_max = 0;

    if (y < H && x < W && n < N)
    {
        int jrst = 0;
        float out = 0.0f;

        while (jrst < maxLutSize)
        {
            LutEntry entry0;
            LutEntry entry1;
            LutEntry entry2;
            LutEntry entry3;

            lut_offset = mad16(sb, maxLutSize, jrst);

            entry0.data2 = lut[lut_offset + 0];
            entry1.data2 = lut[lut_offset + 1];
            entry2.data2 = lut[lut_offset + 2];
            entry3.data2 = lut[lut_offset + 3];

            // argmax
            int fprop_argmax0 = jrst + 0 < lut_size ? __ldg(A + entry0.data.slice) : -2;
            int fprop_argmax1 = jrst + 1 < lut_size ? __ldg(A + entry1.data.slice) : -2;
            int fprop_argmax2 = jrst + 2 < lut_size ? __ldg(A + entry2.data.slice) : -2;
            int fprop_argmax3 = jrst + 3 < lut_size ? __ldg(A + entry3.data.slice) : -2;

            out += jrst + 0 < lut_size && fprop_argmax0 == entry0.data.argmax ? %(cvt)s(__ldg(I + entry0.data.slice)) : 0.0f;
            out += jrst + 1 < lut_size && fprop_argmax1 == entry1.data.argmax ? %(cvt)s(__ldg(I + entry1.data.slice)) : 0.0f;
            out += jrst + 2 < lut_size && fprop_argmax2 == entry2.data.argmax ? %(cvt)s(__ldg(I + entry2.data.slice)) : 0.0f;
            out += jrst + 3 < lut_size && fprop_argmax3 == entry3.data.argmax ? %(cvt)s(__ldg(I + entry3.data.slice)) : 0.0f;

            jrst += 4;
        }
        %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (out*alpha + O_val*beta));
        if (!(flags & 1))
            *O = temp_out;

        // compute max-abs
        intermediate_max = max_abs(intermediate_max, temp_out);  // used for abs
    }
    intermediate_max += 0;
    %(atomic_max)s
}
"""
    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_bprop_max_overlap_smallN")
    kernel.prepare("3P 2f 58I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel


@context_dependent_memoize
def _get_bprop_avg_overlap_smallN(clss, compute_capability):

    code = r"""

%(common)s

union LutEntry {
    struct {
        int   slice;
        float rcp_in;
    } data;
    int2 data2;
};

__device__ __forceinline__ int imin(int val1, int val2)
{
    int ret;
    asm("min.s32 %%0, %%1, %%2;" : "=r"(ret) : "r"(val1), "r"(val2));
    return ret;
}

__global__ void spool_bprop_avg_overlap_smallN(
    const %(type)s* I, %(type)s* O, const unsigned char* A,
    float alpha, float beta, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN,
    int magic_H, int shift_H,
    int pad_w, int pad_h, int pad_d, int pad_c,
    int str_w, int str_h, int str_d, int str_c,
    int magic_str_w, int shift_str_w,
    int magic_str_h, int shift_str_h,
    int magic_str_d, int shift_str_d,
    int magic_str_c, int shift_str_c,
    int S, int R, int T, int J, int RS, int RST, int JRST,
    int magic_S, int shift_S, int magic_RS, int shift_RS,
    int magic_RST, int shift_RST,
    int Q, int P, int M, int K, int QN, int PQN, int MPQN,
    int supH, int supW, int shlH, int maskH, int shrH,
    int shlW, int maskW, int shrW, int maskN, int shrN, int maxLutSize
    %(stats_args)s  // template for "int* maxabs, float scale0"
    )
{
    extern __shared__ int2 lut[];

    int tid = threadIdx.x;

    int x  = blockIdx.x;
    int yz = blockIdx.y;
    int c  = blockIdx.z;

    int z = yz * magic_H; z >>= shift_H;
    int y = yz - z*supH;

    // zigzag q back and forth to improve L2 cache perf
    if (y & 1)
        x = supW - x - 1;

    // Superblock H and W
    y = (y << shlH) + ((tid & maskH) >> shrH);
    x = (x << shlW) + ((tid & maskW) >> shrW);
    int n = tid & maskN;

    int sb = tid >> shrN;

    I += n;
    O += c*DHWN + z*HWN + y*WN + mad16(x, N, n);

    float O_val = beta != 0.0f && y < H && x < W && n < N ? %(cvt)s(__ldg(O)) : 0.0f;

    int kj = c - J + pad_c + 1;
    int mt = z - T + pad_d + 1;
    int pr = y - R + pad_h + 1;
    int qs = x - S + pad_w + 1;

    int sbSize = maskN + 1;
    int sbBits = mad16(sb, sbSize, 0);
    unsigned sbMask = ~(0xffffffff << sbSize) << sbBits;
    unsigned dep_thd_mask = (0xffffffff >> (32 - n)) << sbBits;

    int lut_offset = mad16(sb, maxLutSize, 0);
    int lut_size = 0;
    int jrst = n;
    int JRSTend = JRST;
    if (JRSTend & maskN)
        JRSTend += sbSize - (JRSTend & maskN);

    while (jrst < JRSTend)
    {
        int j   = div16(jrst, magic_RST, shift_RST);
        int rst = mod16(jrst, j, RST);

        int t   = div16(rst, magic_RS, shift_RS);
        int rs  = mod16(rst, t, RS);

        int r   = div16(rs, magic_S, shift_S);
        int s   = mod16(rs, r, S);

        int k_prime = kj + j;
        int m_prime = mt + t;
        int p_prime = pr + r;
        int q_prime = qs + s;

        int  k        = div16(k_prime, magic_str_c, shift_str_c);
        int  k_mod    = mod16(k_prime, k, str_c);
        bool k_bounds = k_mod == 0 && k >= 0 && k < K;

        int  m        = div16(m_prime, magic_str_d, shift_str_d);
        int  m_mod    = mod16(m_prime, m, str_d);
        bool m_bounds = m_mod == 0 && m >= 0 && m < M;

        int  p        = div16(p_prime, magic_str_h, shift_str_h);
        int  p_mod    = mod16(p_prime, p, str_h);
        bool p_bounds = p_mod == 0 && p >= 0 && p < P;

        int  q        = div16(q_prime, magic_str_w, shift_str_w);
        int  q_mod    = mod16(q_prime, q, str_w);
        bool q_bounds = q_mod == 0 && q >= 0 && q < Q;

        bool in_bounds = jrst < JRST && k_bounds && m_bounds && p_bounds && q_bounds;

        // Get a mask of all valid slices in the warp
        unsigned ballot = __ballot(in_bounds);

        // Count the total valid slices in this superblock
        int sb_slices = __popc(sbMask & ballot);

        if (in_bounds)
        {
            // Count all the valid slices below this threadid
            int dep_thd_cnt = __popc(dep_thd_mask & ballot);

            int c_left = msub16(k, str_c, pad_c);
            int z_left = msub16(m, str_d, pad_d);
            int y_left = msub16(p, str_h, pad_h);
            int x_left = msub16(q, str_w, pad_w);

            float k_in = (float)imin( imin(J + c_left, J), imin(C - c_left, J) );
            float m_in = (float)imin( imin(T + z_left, T), imin(D - z_left, T) );
            float p_in = (float)imin( imin(R + y_left, R), imin(H - y_left, R) );
            float q_in = (float)imin( imin(S + x_left, S), imin(W - x_left, S) );

            float total_in = q_in * p_in * m_in * k_in;

            LutEntry entry;
            entry.data.slice  = k*MPQN + m*PQN + p*QN + mad16(q, N, 0);
            entry.data.rcp_in = total_in > 0.0f ? 1.0f / total_in : 0.0f;

            lut[lut_offset + lut_size + dep_thd_cnt] = entry.data2;
        }
        lut_size += sb_slices;
        jrst += sbSize;
    }

    int intermediate_max = 0;

    if (y < H && x < W && n < N)
    {
        int jrst = 0;
        float out = 0.0f;

        while (jrst < maxLutSize)
        {
            LutEntry entry0;
            LutEntry entry1;
            LutEntry entry2;
            LutEntry entry3;

            lut_offset = mad16(sb, maxLutSize, jrst);

            entry0.data2 = lut[lut_offset + 0];
            entry1.data2 = lut[lut_offset + 1];
            entry2.data2 = lut[lut_offset + 2];
            entry3.data2 = lut[lut_offset + 3];

            out += jrst + 0 < lut_size ? %(cvt)s(__ldg(I + entry0.data.slice)) * entry0.data.rcp_in : 0.0f;
            out += jrst + 1 < lut_size ? %(cvt)s(__ldg(I + entry1.data.slice)) * entry1.data.rcp_in : 0.0f;
            out += jrst + 2 < lut_size ? %(cvt)s(__ldg(I + entry2.data.slice)) * entry2.data.rcp_in : 0.0f;
            out += jrst + 3 < lut_size ? %(cvt)s(__ldg(I + entry3.data.slice)) * entry3.data.rcp_in : 0.0f;

            jrst += 4;
        }
        %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (out*alpha + O_val*beta));
        if (!(flags & 1))
            *O = temp_out;

        // max-abs over unrolls
        intermediate_max = max_abs(intermediate_max, temp_out);  // used for abs
    }
    intermediate_max += 0;
    %(atomic_max)s

}
"""
    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    # f = open("pool.cu", "w")
    # print >>f, code
    # f.close()
    module = SourceModule(code, options=["--use_fast_math"])
    kernel = module.get_function("spool_bprop_avg_overlap_smallN")
    kernel.prepare("3P 2f 58I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel
