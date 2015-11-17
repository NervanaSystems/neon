from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize

# from neon.backends.cuda_templates import (_ew_template,
#                                           _stage_template,
#                                           _fin_template,
#                                           _init_rand_func,
#                                           _init_rand_round_func,
#                                           _finish_rand_func,
#                                           _common_urand_gen,
#                                           _common_frand,
#                                           _common_round,
#                                           _common_fp16_to_fp32,
#                                           _ew_types,
#                                           _ew_strings,
#                                           _is_finite,
#                                           _float_ops,
#                                           _reduction_ops)
from neon.backends.cuda_templates import (_common_round,
                                          _ew_types,
                                          _common_fp16_to_fp32,
                                          _ew_strings)


@context_dependent_memoize
def _get_bn_fprop_kernel(dtype, threads):

    if threads > 32:
        shr_code = "__shared__ float sPartials[THREADS];"
        red_code = r"""
    sPartials[tid] = xvar;
    __syncthreads();

    #pragma unroll
    for (int a = THREADS >> 1; a > 32; a >>= 1)
    {
        if ( tid < a )
            sPartials[tid] += sPartials[tid + a];
        __syncthreads();
    }
    if ( tid < 32 )
    {
        xvar = sPartials[tid] + sPartials[tid + 32];
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
            xvar += __shfl_xor(xvar, i);

        sPartials[tid] = xvar * rcpN;
    }
    __syncthreads();
    xvar = sPartials[0];
"""
    else:
        shr_code = ""
        red_code = r"""
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        xvar += __shfl_xor(xvar, i);
    xvar *= rcpN;
"""

    code = r"""
#define THREADS %(threads)s

%(common)s

__global__ void batchnorm_fprop (
    %(type)s* y_out, %(type)s* xvar_out, %(type)s* gmean_out, %(type)s* gvar_out,
    const %(type)s* x_in, const float* xsum_in, const %(type)s* gmean_in,
    const %(type)s* gvar_in, const %(type)s* gamma_in, const %(type)s* beta_in,
    const float eps, const float rho, const int N, const int relu)
{
    %(share)s

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    int offset = bid * N;

    const %(type)s* x_in0 = x_in + offset + tid;

    const float rcpN = 1.0f/(float)N;

    float xmean = __ldg(xsum_in + bid) * rcpN;

    float xvar = 0.0f;
    for (int i = tid; i < N; i += THREADS)
    {
        float x = %(cvt)s(__ldg(x_in0));
        x_in0 += THREADS;

        x -= xmean;
        xvar += x * x;
    }
    %(red)s

    float gamma = %(cvt)s(__ldg(gamma_in + bid));
    float beta  = %(cvt)s(__ldg(beta_in  + bid));

    if ( tid == 0 )
    {
        float gmean = %(cvt)s(__ldg(gmean_in + bid));
        float gvar  = %(cvt)s(__ldg(gvar_in  + bid));

        gmean = gmean * rho + (1.0f - rho) * xmean;
        gvar  = gvar  * rho + (1.0f - rho) * xvar;
        %(xvar_out)s
        %(gmean_out)s
        %(gvar_out)s
        *(xvar_out  + bid) = xvar_val;
        *(gmean_out + bid) = gmean_val;
        *(gvar_out  + bid) = gvar_val;
    }

    float xvar_rcp_sqrt = 1.0f / sqrtf(xvar + eps);

    int start = N - (THREADS*4 - tid);
    offset += start;
    x_in   += offset;
    y_out  += offset;

    for (int i = start; i >= -THREADS*3; i -= THREADS*4)
    {
        float x0 = i >= -THREADS*0 ? %(cvt)s(__ldg(x_in + THREADS*0)) : 0.0f;
        float x1 = i >= -THREADS*1 ? %(cvt)s(__ldg(x_in + THREADS*1)) : 0.0f;
        float x2 = i >= -THREADS*2 ? %(cvt)s(__ldg(x_in + THREADS*2)) : 0.0f;
        float x3 =                   %(cvt)s(__ldg(x_in + THREADS*3));

        x_in -= THREADS*4;

        float xhat0 = (x0 - xmean) * xvar_rcp_sqrt;
        float xhat1 = (x1 - xmean) * xvar_rcp_sqrt;
        float xhat2 = (x2 - xmean) * xvar_rcp_sqrt;
        float xhat3 = (x3 - xmean) * xvar_rcp_sqrt;

        float y0 = xhat0 * gamma + beta;
        float y1 = xhat1 * gamma + beta;
        float y2 = xhat2 * gamma + beta;
        float y3 = xhat3 * gamma + beta;

        if (relu)
        {
            y0 = fmaxf(y0, 0.0f);
            y1 = fmaxf(y1, 0.0f);
            y2 = fmaxf(y2, 0.0f);
            y3 = fmaxf(y3, 0.0f);
        }

        %(y0_out)s
        %(y1_out)s
        %(y2_out)s
        %(y3_out)s
        if (i >= -THREADS*0) *(y_out + THREADS*0) = y0_val;
        if (i >= -THREADS*1) *(y_out + THREADS*1) = y1_val;
        if (i >= -THREADS*2) *(y_out + THREADS*2) = y2_val;
                             *(y_out + THREADS*3) = y3_val;
        y_out -= THREADS*4;
    }
}
"""
    out_code = _ew_strings["round"]["nearest"].get(dtype, "float {0} = {1};")
    common_code  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common_code += _common_fp16_to_fp32

    code = code % {
        "common"    : common_code,
        "share"     : shr_code,
        "red"       : red_code,
        "threads"   : threads,
        "type"      : _ew_types[dtype]["type"],
        "cvt"       : _ew_types[dtype]["cvt"],
        "xvar_out"  : out_code.format("xvar_val",  "xvar"),
        "gmean_out" : out_code.format("gmean_val", "gmean"),
        "gvar_out"  : out_code.format("gvar_val",  "gvar"),
        "y0_out"    : out_code.format("y0_val",     "y0"),
        "y1_out"    : out_code.format("y1_val",     "y1"),
        "y2_out"    : out_code.format("y2_val",     "y2"),
        "y3_out"    : out_code.format("y3_val",     "y3"),
    }
    module = SourceModule(code, options=["--use_fast_math"])
    kernel = module.get_function("batchnorm_fprop")
    kernel.prepare("PPPPPPPPPPffII")
    kernel.name = "batchnorm_fprop"
    return kernel


@context_dependent_memoize
def _get_bn_bprop_kernel(dtype, threads):

    if threads > 32:
        shr_code = "__shared__ float sPartials[THREADS * 2];"
        red_code = r"""
    sPartials[tid + THREADS*0] = grad_gamma;
    sPartials[tid + THREADS*1] = grad_beta;
    __syncthreads();

    #pragma unroll
    for (int a = THREADS >> 1; a > 32; a >>= 1)
    {
        if ( tid < a )
        {
            sPartials[tid + THREADS*0] += sPartials[tid + a + THREADS*0];
            sPartials[tid + THREADS*1] += sPartials[tid + a + THREADS*1];
        }
        __syncthreads();
    }
    if ( tid < 32 )
    {
        grad_gamma = sPartials[tid + THREADS*0] + sPartials[tid + 32 + THREADS*0];
        grad_beta  = sPartials[tid + THREADS*1] + sPartials[tid + 32 + THREADS*1];

        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
        {
            grad_gamma += __shfl_xor(grad_gamma, i);
            grad_beta  += __shfl_xor(grad_beta,  i);
        }
        sPartials[tid + THREADS*0] = grad_gamma;
        sPartials[tid + THREADS*1] = grad_beta;
    }
    __syncthreads();
    grad_gamma = sPartials[THREADS*0];
    grad_beta  = sPartials[THREADS*1];
"""
    else:
        shr_code = ""
        red_code = r"""
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
    {
        grad_gamma += __shfl_xor(grad_gamma, i);
        grad_beta  += __shfl_xor(grad_beta,  i);
    }
"""

    code = r"""
#define THREADS %(threads)s

%(common)s

__global__ void batchnorm_bprop (
    %(type)s* delta_out, %(type)s* grad_gamma_out, %(type)s* grad_beta_out,
    const %(type)s* delta_in, const %(type)s* x_in, const float* xsum_in,
    const %(type)s* xvar_in, const %(type)s* gamma_in,
    const float eps, const int N)
{
    %(share)s

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const float rcpN = 1.0f/(float)N;
    int offset = bid * N;

    const %(type)s* x_in0 = x_in     + offset + tid;
    const %(type)s* d_in0 = delta_in + offset + tid;

    float xmean = __ldg(xsum_in  + bid) * rcpN;
    float xvar  = %(cvt)s(__ldg(xvar_in  + bid));
    float gamma = %(cvt)s(__ldg(gamma_in + bid));

    float xvar_rcp_sqrt = 1.0f / sqrtf(xvar + eps);
    float grad_gamma    = 0.0f;
    float grad_beta     = 0.0f;

    for (int i = tid; i < N; i += THREADS)
    {
        float x = %(cvt)s(__ldg(x_in0));
        x_in0 += THREADS;
        float d = %(cvt)s(__ldg(d_in0));
        d_in0 += THREADS;

        float xhat = (x - xmean) * xvar_rcp_sqrt;

        grad_gamma += xhat * d;
        grad_beta  += d;
    }
    %(red)s

    if ( tid == 0 )
    {
        %(grad_gamma_out)s
        %(grad_beta_out)s
        *(grad_gamma_out + bid) = grad_gamma_val;
        *(grad_beta_out  + bid) = grad_beta_val;
    }

    int start = N - (THREADS*4 - tid);
    offset += start;
    const %(type)s* x_in1 = x_in     + offset;
    const %(type)s* d_in1 = delta_in + offset;
    delta_out += offset;

    for (int i = start; i >= -THREADS*3; i -= THREADS*4)
    {
        float x0 = i >= -THREADS*0 ? %(cvt)s(__ldg(x_in1 + THREADS*0)) : 0.0f;
        float x1 = i >= -THREADS*1 ? %(cvt)s(__ldg(x_in1 + THREADS*1)) : 0.0f;
        float x2 = i >= -THREADS*2 ? %(cvt)s(__ldg(x_in1 + THREADS*2)) : 0.0f;
        float x3 =                   %(cvt)s(__ldg(x_in1 + THREADS*3));

        float d0 = i >= -THREADS*0 ? %(cvt)s(__ldg(d_in1 + THREADS*0)) : 0.0f;
        float d1 = i >= -THREADS*1 ? %(cvt)s(__ldg(d_in1 + THREADS*1)) : 0.0f;
        float d2 = i >= -THREADS*2 ? %(cvt)s(__ldg(d_in1 + THREADS*2)) : 0.0f;
        float d3 =                   %(cvt)s(__ldg(d_in1 + THREADS*3));

        x_in1 -= THREADS*4;
        d_in1 -= THREADS*4;

        float xhat0 = (x0 - xmean) * xvar_rcp_sqrt;
        float xhat1 = (x1 - xmean) * xvar_rcp_sqrt;
        float xhat2 = (x2 - xmean) * xvar_rcp_sqrt;
        float xhat3 = (x3 - xmean) * xvar_rcp_sqrt;

        float xtmp0 = (xhat0 * grad_gamma + grad_beta) * rcpN;
        float xtmp1 = (xhat1 * grad_gamma + grad_beta) * rcpN;
        float xtmp2 = (xhat2 * grad_gamma + grad_beta) * rcpN;
        float xtmp3 = (xhat3 * grad_gamma + grad_beta) * rcpN;

        float delta0 = gamma * (d0 - xtmp0) * xvar_rcp_sqrt;
        float delta1 = gamma * (d1 - xtmp1) * xvar_rcp_sqrt;
        float delta2 = gamma * (d2 - xtmp2) * xvar_rcp_sqrt;
        float delta3 = gamma * (d3 - xtmp3) * xvar_rcp_sqrt;

        %(delta0_out)s
        %(delta1_out)s
        %(delta2_out)s
        %(delta3_out)s
        if (i >= -THREADS*0) *(delta_out + THREADS*0) = delta0_val;
        if (i >= -THREADS*1) *(delta_out + THREADS*1) = delta1_val;
        if (i >= -THREADS*2) *(delta_out + THREADS*2) = delta2_val;
                             *(delta_out + THREADS*3) = delta3_val;
        delta_out -= THREADS*4;
    }
}
"""
    out_code = _ew_strings["round"]["nearest"].get(dtype, "float {0} = {1};")
    common_code = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common_code += _common_fp16_to_fp32

    code = code % {
        "common"         : common_code,
        "share"          : shr_code,
        "red"            : red_code,
        "threads"        : threads,
        "type"           : _ew_types[dtype]["type"],
        "cvt"            : _ew_types[dtype]["cvt"],
        "grad_gamma_out" : out_code.format("grad_gamma_val", "grad_gamma"),
        "grad_beta_out"  : out_code.format("grad_beta_val",  "grad_beta"),
        "delta0_out"     : out_code.format("delta0_val",     "delta0"),
        "delta1_out"     : out_code.format("delta1_val",     "delta1"),
        "delta2_out"     : out_code.format("delta2_val",     "delta2"),
        "delta3_out"     : out_code.format("delta3_val",     "delta3"),
    }
    module = SourceModule(code, options=["--use_fast_math"])
    kernel = module.get_function("batchnorm_bprop")
    kernel.prepare("PPPPPPPPfI")
    kernel.name = "batchnorm_bprop"
    return kernel
