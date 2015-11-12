
from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize
from neon.backends import cuda_templates

from neon.backends.cuda_templates import (_common_fp16_to_fp32,
                                          _common_round,  # for fp32_to_fp16 converter
                                          _common_max_abs,
                                          _ew_types)

"""
CUDA kernels for pooling layers, with support for max pooling and average pooling.
For each pooling type, there is an fprop function, a bprop function, and a bprop
for overlapping kernels. Each of the six kernels uses templating to perform dtype
conversion so it works for all data types (currently fp32 and fp16 are supported).
Additionally, there are templates for statistics collection, currently supporting
a global max abs over the output tensor which is passed in as an additional kernel
argument.
"""


def map_string2func(funcname, clss):
    """
    Helper function that converts string function names to function calls
    """
    if funcname == "fprop_max":
        return _get_fprop_max(clss)
    if funcname == "bprop_max":
        return _get_bprop_max(clss)
    if funcname == "bprop_max_overlap":
        return _get_bprop_max_overlap(clss)
    if funcname == "fprop_avg":
        return _get_fprop_avg(clss)
    if funcname == "bprop_avg":
        return _get_bprop_avg(clss)
    if funcname == "bprop_avg_overlap":
        return _get_bprop_avg_overlap(clss)


# this template is used to hide variables that are only defined conditionally.
atomic_max = r"""
atomicMax(maxabs, intermediate_max);
"""


def prepare_template_vals(dtype, rounding=False):
    """
    Set up template code snippets that are reused across multiple kernels.
    Most are data type conversion and statistics collection related.
    """
    template_vals = dict()
    for key in ("common", "inits", "finish", "stats_args", "mul_by_scale", "atomic_max"):
        template_vals[key] = ""

    if dtype == "f2":
        template_vals["common"] += _common_fp16_to_fp32

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

    template_vals["type"] = _ew_types[dtype]["type"]
    template_vals["cvt"] = _ew_types[dtype]["cvt"]

    if dtype == "f2":
        template_vals["cvt_out"] = "fp32_to_fp16"
    elif dtype == "x2":
        template_vals["stats_args"] += ", int* maxabs, float scale0"
        template_vals["cvt"] = "(float)"
        template_vals["cvt_out"] = "fp32_to_int16"
        template_vals["mul_by_scale"] += "1/scale0 *"
        template_vals["atomic_max"] += atomic_max
    elif dtype == "f4":
        template_vals["cvt_out"] = ""
    else:
        raise ValueError("Did not understand clss dtype " + str(dtype))

    return template_vals


# This section of the code contains templated CUDA-C code for the kernels.
@context_dependent_memoize
def _get_fprop_max(clss):

    code = r"""
#define FLT_MAX 3.402823466E+38F

// signature 3P2f33I

// get the convert functions
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
    int magic_RS, int shift_RS, int magic_RST, int shift_RST
    %(stats_args)s
    )
{
    extern __shared__ int lut[];
    int tid = threadIdx.x;

    int n  = tid;
    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*P;


    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = Q - q - 1;

    int offset = k*MPQN + m*PQN + p*QN + q*N + n;
    I += n;
    O += offset;
    A += offset;

    float O_val = beta != 0.0f ? %(cvt)s(__ldg(O)) : 0.0f;

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int jrst = tid;
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

            bool bounds_x  = x >= 0.0f && x < W;
            bool bounds_y  = y >= 0.0f && y < H;
            bool bounds_z  = z >= 0.0f && z < D;
            bool bounds_c  = c >= 0.0f && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            lut[jrst] = in_bounds ? sliceI : -1;
            jrst += 32;
        }
    }
    __syncthreads();

    int jrst = 0;
    int argmax = 0;
    float max = -FLT_MAX;
    while (jrst < JRST)
    {
        int slice0 = lut[jrst + 0];
        int slice1 = lut[jrst + 1];
        int slice2 = lut[jrst + 2];
        int slice3 = lut[jrst + 3];

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
    }

    int intermediate_max = max_abs(0, temp_out);  // compute abs
    %(atomic_max)s

    *A = (unsigned char)argmax;
}
"""

    template_vals = prepare_template_vals(clss)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_fprop_max")
    sig = "3P 2f 34I" + ("Pf" if (clss[0] == "x") else "")
    kernel.prepare(sig)
    return kernel


@context_dependent_memoize
def _get_fprop_avg(clss):

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
    int magic_RS, int shift_RS, int magic_RST, int shift_RST
    %(stats_args)s
    )
{
    __shared__ float rcpWindowSize;
    extern __shared__ int lut[];

    int tid = threadIdx.x;

    int n  = tid;
    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*P;

    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = Q - q - 1;

    I += n;
    O += k*MPQN + m*PQN + p*QN + q*N + n;

    float O_val = beta != 0.0f ? %(cvt)s(__ldg(O)) : 0.0f;

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int window_size = 0;
        int jrst = tid;
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

            bool bounds_x  = x >= 0.0f && x < W;
            bool bounds_y  = y >= 0.0f && y < H;
            bool bounds_z  = z >= 0.0f && z < D;
            bool bounds_c  = c >= 0.0f && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            // Count the total valid slices
            window_size += __popc(__ballot(in_bounds));

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            lut[jrst] = in_bounds ? sliceI : -1;
            jrst += 32;
        }

        rcpWindowSize = 1.0f / (float)window_size;
    }
    __syncthreads();

    int jrst = 0;
    float sum = 0.0f;
    while (jrst < JRST)
    {
        int slice0 = lut[jrst + 0];
        int slice1 = lut[jrst + 1];
        int slice2 = lut[jrst + 2];
        int slice3 = lut[jrst + 3];

        sum += jrst + 0 < JRST && slice0 >= 0 ? %(cvt)s(__ldg(I + slice0)) : 0.0f;
        sum += jrst + 1 < JRST && slice1 >= 0 ? %(cvt)s(__ldg(I + slice1)) : 0.0f;
        sum += jrst + 2 < JRST && slice2 >= 0 ? %(cvt)s(__ldg(I + slice2)) : 0.0f;
        sum += jrst + 3 < JRST && slice3 >= 0 ? %(cvt)s(__ldg(I + slice3)) : 0.0f;

        jrst += 4;
    }

    // convert back to fp to write out
    %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (sum*rcpWindowSize*alpha + O_val*beta));
    if (!(flags & 1)) {
        *O = temp_out;
    }
    // collect max abs stats
    int intermediate_max = max_abs(0, temp_out); // compute abs
    %(atomic_max)s
}
"""

    template_vals = prepare_template_vals(clss)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_fprop_avg")
    kernel.prepare("3P 2f 34I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel


@context_dependent_memoize
def _get_bprop_max(clss):

    code = r"""
// sig 3P2f33I

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
    int magic_RS, int shift_RS, int magic_RST, int shift_RST
    %(stats_args)s
    )
{
    extern __shared__ int lut[];

    int tid = threadIdx.x;

    int n  = tid;
    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*P;

    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = Q - q - 1;

    int offset = k*MPQN + m*PQN + p*QN + q*N + n;
    O += n;
    I += offset;
    A += offset;

    float delta  = %(cvt)s(__ldg(I));
    int argmax   = __ldg(A);

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int jrst = tid;
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

            bool bounds_x  = x >= 0.0f && x < W;
            bool bounds_y  = y >= 0.0f && y < H;
            bool bounds_z  = z >= 0.0f && z < D;
            bool bounds_c  = c >= 0.0f && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            lut[jrst] = in_bounds ? sliceI : -1;
            jrst += 32;
        }
    }
    __syncthreads();

    delta *= alpha;
    bool load_beta = beta != 0.0f;
    int jrst = 0;
    int intermediate_max = 0;

    %(type)s temp_out0 = 0;
    %(type)s temp_out1 = 0;
    %(type)s temp_out2 = 0;
    %(type)s temp_out3 = 0;

    while (jrst < JRST)
    {
        int offset0 = lut[jrst + 0];
        int offset1 = lut[jrst + 1];
        int offset2 = lut[jrst + 2];
        int offset3 = lut[jrst + 3];

        // need to figure out how to write into output. Can't be float * if we write fp16
        // load fp16 from O, so it's an fp16 pointer
        %(type)s* out0 = O + offset0;
        %(type)s* out1 = O + offset1;
        %(type)s* out2 = O + offset2;
        %(type)s* out3 = O + offset3;

        // load input dtype, convert to float32.
        float beta0 = jrst + 0 < JRST && offset0 >= 0 && load_beta ? %(cvt)s(__ldg(out0)) * beta : 0.0f;
        float beta1 = jrst + 1 < JRST && offset1 >= 0 && load_beta ? %(cvt)s(__ldg(out1)) * beta : 0.0f;
        float beta2 = jrst + 2 < JRST && offset2 >= 0 && load_beta ? %(cvt)s(__ldg(out2)) * beta : 0.0f;
        float beta3 = jrst + 3 < JRST && offset3 >= 0 && load_beta ? %(cvt)s(__ldg(out3)) * beta : 0.0f;

        // convert float32 back into input format to write out
        if (jrst + 0 < JRST && offset0 >= 0)
            temp_out0 = %(cvt_out)s(%(mul_by_scale)s (jrst + 0 == argmax ? delta + beta0 : beta0));
        if (jrst + 1 < JRST && offset1 >= 0)
            temp_out1 = %(cvt_out)s(%(mul_by_scale)s (jrst + 1 == argmax ? delta + beta1 : beta1));
        if (jrst + 2 < JRST && offset2 >= 0)
            temp_out2 = %(cvt_out)s(%(mul_by_scale)s (jrst + 2 == argmax ? delta + beta2 : beta2));
        if (jrst + 3 < JRST && offset3 >= 0)
            temp_out3 = %(cvt_out)s(%(mul_by_scale)s (jrst + 3 == argmax ? delta + beta3 : beta3));

        // predicate writes with no-op flag.
        if (!(flags & 1)) {
            if (jrst + 0 < JRST && offset0 >= 0)
                *out0 = temp_out0;
            if (jrst + 1 < JRST && offset1 >= 0)
                *out1 = temp_out1;
            if (jrst + 2 < JRST && offset2 >= 0)
                *out2 = temp_out2;
            if (jrst + 3 < JRST && offset3 >= 0)
                *out3 = temp_out3;
        }

        intermediate_max = max_abs(intermediate_max, temp_out0);
        intermediate_max = max_abs(intermediate_max, temp_out1);
        intermediate_max = max_abs(intermediate_max, temp_out2);
        intermediate_max = max_abs(intermediate_max, temp_out3);

        jrst += 4;

    }
    %(atomic_max)s

}
"""
    template_vals = prepare_template_vals(clss)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_bprop_max")
    # f = open("spool_bprop_max.cu", "w")
    # print >>f, code
    # f.close()
    kernel.prepare("3P 2f 34I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel


@context_dependent_memoize
def _get_bprop_avg(clss):

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
    int magic_RS, int shift_RS, int magic_RST, int shift_RST
    %(stats_args)s
    )
{
    __shared__ float rcpWindowSize;
    extern __shared__ int lut[];

    int tid = threadIdx.x;

    int n  = tid;
    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*P;

    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = Q - q - 1;

    O += n;
    I += k*MPQN + m*PQN + p*QN + q*N + n;

    float delta  = %(cvt)s(__ldg(I));

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int window_size = 0;
        int jrst = tid;
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

            bool bounds_x  = x >= 0.0f && x < W;
            bool bounds_y  = y >= 0.0f && y < H;
            bool bounds_z  = z >= 0.0f && z < D;
            bool bounds_c  = c >= 0.0f && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            window_size += __popc(__ballot(in_bounds));

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            lut[jrst] = in_bounds ? sliceI : -1;
            jrst += 32;
        }
        rcpWindowSize = 1.0f / (float)window_size;

    }
    __syncthreads();

    delta *= alpha * rcpWindowSize;
    bool load_beta = beta != 0.0f;
    int jrst = 0;
    int intermediate_max = 0;

    %(type)s temp_out0 = 0;
    %(type)s temp_out1 = 0;
    %(type)s temp_out2 = 0;
    %(type)s temp_out3 = 0;

    while (jrst < JRST)
    {
        int offset0 = lut[jrst + 0];
        int offset1 = lut[jrst + 1];
        int offset2 = lut[jrst + 2];
        int offset3 = lut[jrst + 3];

        %(type)s* out0 = O + offset0;
        %(type)s* out1 = O + offset1;
        %(type)s* out2 = O + offset2;
        %(type)s* out3 = O + offset3;

        float beta0 = jrst + 0 < JRST && offset0 >= 0 && load_beta ? %(cvt)s(__ldg(out0)) * beta : 0.0f;
        float beta1 = jrst + 1 < JRST && offset1 >= 0 && load_beta ? %(cvt)s(__ldg(out1)) * beta : 0.0f;
        float beta2 = jrst + 2 < JRST && offset2 >= 0 && load_beta ? %(cvt)s(__ldg(out2)) * beta : 0.0f;
        float beta3 = jrst + 3 < JRST && offset3 >= 0 && load_beta ? %(cvt)s(__ldg(out3)) * beta : 0.0f;

        if (jrst + 0 < JRST && offset0 >= 0)
            temp_out0 = %(cvt_out)s(%(mul_by_scale)s(delta + beta0));
        if (jrst + 1 < JRST && offset1 >= 0)
            temp_out1 = %(cvt_out)s(%(mul_by_scale)s(delta + beta1));
        if (jrst + 2 < JRST && offset2 >= 0)
            temp_out2 = %(cvt_out)s(%(mul_by_scale)s(delta + beta2));
        if (jrst + 3 < JRST && offset3 >= 0)
            temp_out3 = %(cvt_out)s(%(mul_by_scale)s(delta + beta3));

        // predicate writes with no-op flag.
        if (!(flags & 1)) {
            if (jrst + 0 < JRST && offset0 >= 0)
                *out0 = temp_out0;
            if (jrst + 1 < JRST && offset1 >= 0)
                *out1 = temp_out1;
            if (jrst + 2 < JRST && offset2 >= 0)
                *out2 = temp_out2;
            if (jrst + 3 < JRST && offset3 >= 0)
                *out3 = temp_out3;
        }

        jrst += 4;

        intermediate_max = max_abs(intermediate_max, temp_out0);
        intermediate_max = max_abs(intermediate_max, temp_out1);
        intermediate_max = max_abs(intermediate_max, temp_out2);
        intermediate_max = max_abs(intermediate_max, temp_out3);

    }
    %(atomic_max)s
}
"""
    template_vals = prepare_template_vals(clss)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_bprop_avg")
    kernel.prepare("3P 2f 34I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel


@context_dependent_memoize
def _get_bprop_max_overlap(clss):

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
    }
    __syncthreads();

    int jrst = 0;
    float out = 0.0f;
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

        // argmax
        int fprop_argmax0 = jrst + 0 < JRST && entry0.data.argmax >= 0 ? __ldg(A + entry0.data.slice) : -2;
        int fprop_argmax1 = jrst + 1 < JRST && entry1.data.argmax >= 0 ? __ldg(A + entry1.data.slice) : -2;
        int fprop_argmax2 = jrst + 2 < JRST && entry2.data.argmax >= 0 ? __ldg(A + entry2.data.slice) : -2;
        int fprop_argmax3 = jrst + 3 < JRST && entry3.data.argmax >= 0 ? __ldg(A + entry3.data.slice) : -2;

        out += jrst + 0 < JRST && fprop_argmax0 == entry0.data.argmax ? %(cvt)s(__ldg(I + entry0.data.slice)) : 0.0f;
        out += jrst + 1 < JRST && fprop_argmax1 == entry1.data.argmax ? %(cvt)s(__ldg(I + entry1.data.slice)) : 0.0f;
        out += jrst + 2 < JRST && fprop_argmax2 == entry2.data.argmax ? %(cvt)s(__ldg(I + entry2.data.slice)) : 0.0f;
        out += jrst + 3 < JRST && fprop_argmax3 == entry3.data.argmax ? %(cvt)s(__ldg(I + entry3.data.slice)) : 0.0f;



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

    template_vals = prepare_template_vals(clss)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_bprop_max_overlap")
    kernel.prepare("3P 2f 47I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel


@context_dependent_memoize
def _get_bprop_avg_overlap(clss):

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

            int c_left = k * str_c - pad_c;
            int z_left = m * str_d - pad_d;
            int y_left = p * str_h - pad_h;
            int x_left = q * str_w - pad_w;

            int k_in = imin( imin(J + c_left, J), imin(C - c_left, J) );
            int m_in = imin( imin(T + z_left, T), imin(D - z_left, T) );
            int p_in = imin( imin(R + y_left, R), imin(H - y_left, R) );
            int q_in = imin( imin(S + x_left, S), imin(W - x_left, S) );

            int total_in = q_in * p_in * m_in * k_in;

            float rcp_in = total_in > 0 ? 1.0f / (float)total_in : 0.0f;

            int sliceI  = k*MPQN + m*PQN + p*QN + q*N;

            LutEntry entry;
            entry.data.slice  = in_bounds ? sliceI : -1;
            entry.data.rcp_in = rcp_in;

            lut[jrst] = entry.data2;
            jrst += 32;
        }
    }
    __syncthreads();

    int jrst = 0;
    float out = 0.0f;
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

        out += jrst + 0 < JRST && entry0.data.slice >= 0 ? %(cvt)s(__ldg(I + entry0.data.slice)) * entry0.data.rcp_in : 0.0f;
        out += jrst + 1 < JRST && entry1.data.slice >= 0 ? %(cvt)s(__ldg(I + entry1.data.slice)) * entry1.data.rcp_in : 0.0f;
        out += jrst + 2 < JRST && entry2.data.slice >= 0 ? %(cvt)s(__ldg(I + entry2.data.slice)) * entry2.data.rcp_in : 0.0f;
        out += jrst + 3 < JRST && entry3.data.slice >= 0 ? %(cvt)s(__ldg(I + entry3.data.slice)) * entry3.data.rcp_in : 0.0f;

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

    template_vals = prepare_template_vals(clss)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_bprop_avg_overlap")
    kernel.prepare("3P 2f 47I" + ("Pf" if (clss[0] == "x") else ""))
    return kernel
