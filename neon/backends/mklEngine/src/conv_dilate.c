/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "MKLDNN.h"

void im2col_cpu(
    const float* data_im,
    float* data_col,
    const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w);

void forward_dilated_conv(
    float* input,
    float* output,
    float* weight,
    float* bias, float* input_col,
    int N,  int inC, int inH, int inW,
    int kH, int kW,
    int dH, int dW,
    int padH, int padW,
    int dilH, int dilW,
    int outC, int outH, int outW)
{
    const long input_batch_size = inC * inH * inW;
    const long output_batch_size = outC * outH * outW;
    const int dil_kernel_h = (kH - 1) * dilH + 1;
    const int dil_kernel_w = (kW - 1) * dilW + 1;
    const int colH = (inH + 2 * padH - dil_kernel_h) / dH + 1;
    const int colW = (inW + 2 * padW - dil_kernel_w) / dW + 1;
    const long map_size = outH * outW;

    //temp transpose weight
    const long width = inC*kH*kW;
    float* weight_t = malloc(outC*width*sizeof(float));
    mkl_somatcopy('r', 't', width, outC, 1.0, weight, outC, weight_t, width);
    for (int n = 0; n < N; ++n)
    {
        float* input_batch = input + n * input_batch_size;
        float* output_batch = output + n * output_batch_size;
        float* input_col_batch = input_col + n * inC * kH * kW * map_size;
        im2col_cpu(input_batch, input_col_batch,
                   inC, inH, inW, kH, kW, padH, padW, dH, dW, dilH, dilW);
        if(bias)
        {
            // add bias to each output channel
            for(int c = 0; c < outC; ++c)
            {
                float* output_ch = output_batch + c * map_size;
                for(int i = 0; i < map_size; ++i)
                    output_ch[i] = bias[c];
            }
        }
        else
        {
            // add bias to each output channel
            for(int c = 0; c < outC; ++c)
            {
                float* output_ch = output_batch + c * map_size;
                for(int i = 0; i < map_size; ++i)
                    output_ch[i] = 0.0;
            }

        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, outC, map_size, width, 1.0, weight_t,
                    width, input_col_batch, map_size, 1.0, output_batch, map_size);
    }
    free(weight_t);
}

void col2im_cpu(const float* data_col, float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w);

void backward_dilated_conv(
    float* gradout,
    float* gradin,
    float* weight,
    float* gradin_col,
    int N,  int inC, int inH, int inW,
    int kH, int kW,
    int dH,  int dW,
    int padH, int padW,
    int dilH, int dilW,
    int outC, int outH, int outW)
{
    const long width = inC * kH * kW;
    const long input_batch_size = inC * inH * inW;
    const long output_batch_size = outC * outH * outW;
    const int dil_kernel_h = (kH - 1) * dilH + 1;
    const int dil_kernel_w = (kW - 1) * dilW + 1;
    const int colH = (inH + 2 * padH - dil_kernel_h) / dH + 1;
    const int colW = (inW + 2 * padW - dil_kernel_w) / dW + 1;
    const long map_size = outH * outW;

    for (int n = 0; n < N; ++n)
    {
        float* gradout_batch = gradout + n * output_batch_size;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width, map_size, outC, 1.0, weight,
                     outC, gradout_batch, map_size, 0.0, gradin_col, map_size);

        float* gradin_batch = gradin + n * input_batch_size;
        col2im_cpu(gradin_col, gradin_batch,
                   inC, inH, inW, kH, kW, padH, padW, dH, dW, dilH, dilW);
    }
}

// bdata_dilated_conv(gradout_NCHW, gradbias_ptr, gradweight_ptr, input_column,
void bfilter_dilated_conv(
    float* gradout,
    float* grad_bias,
    float* grad_weight,
    float* input_col,
    int N,  int inC, int inH, int inW,
    int kH, int kW,
    int dH,  int dW,
    int padH, int padW,
    int dilH, int dilW,
    int outC, int outH, int outW)
{
    const long width = inC * kH * kW;
    const long input_batch_size = width*outH*outW;
    const long output_batch_size = outC * outH * outW;
    const int dil_kernel_h = (kH - 1) * dilH + 1;
    const int dil_kernel_w = (kW - 1) * dilW + 1;
    const int colH = (inH + 2 * padH - dil_kernel_h) / dH + 1;
    const int colW = (inW + 2 * padW - dil_kernel_w) / dW + 1;
    const long output_mapsize = outH * outW;

    //zero the grad
    for (long i = 0; i < outC*width; ++i)
    {
        grad_weight[i] = 0;
    }

    //temp transpose input_column
    float* gradout_batch_t = malloc(output_batch_size * sizeof(float));
    for (int n = 0; n < N; ++n)
    {
        float* gradout_batch = gradout + n * output_batch_size;
        float* input_batch = input_col + n * input_batch_size;
        mkl_somatcopy('r', 't', outC, output_mapsize, 1.0, gradout_batch, output_mapsize, gradout_batch_t, outC);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width, outC, output_mapsize, 1.0, input_batch, output_mapsize,
                    gradout_batch_t, outC, 1.0, grad_weight, outC);
    }

    //for grad_bias
    if(grad_bias != NULL)
    {
        for(int i=0; i<outC; ++i)
            grad_bias[i] = 0;

        for(int b=0; b<N; ++b)
        {
            for(int c=0; c<outC; ++c)
            {
                for(long s=0; s<output_mapsize; ++s)
                {
                    grad_bias[c] += gradout[b * output_batch_size + c * output_mapsize + s];
                }

            }
        }
    }

    free(gradout_batch_t);
}

void im2col_cpu(
    const float* data_im,
    float* data_col,
    const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w)
{
    int dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
    int dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
    int height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;

    #pragma omp parallel for if (channels_col > 1)
    for (int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;

        const int hc0 = h_offset * dilation_h - pad_h;
        const int wc0 = w_offset * dilation_w - pad_w;
        for (int h = 0; h < height_col; ++h)
        {
            int h_pad = h * stride_h + hc0;

            const int row_offset = (c * height_col + h) * width_col;
            const int srow_offset = (c_im * height + h_pad) * width;
            for (int w = 0; w < width_col; ++w)
            {
                int w_pad = w * stride_w + wc0;
                if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width)))
                    data_col[row_offset + w] = data_im[srow_offset + w_pad];
                else
                    data_col[row_offset + w] = 0.;
            }
        }
    }
}

void col2im_cpu(
    const float* data_col,
    float* data_im,
    const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w)
{
    int dil_patch_h = (kernel_h - 1) * dilation_h + 1;
    int dil_patch_w = (kernel_w - 1) * dilation_w + 1;
    int height_col = (height + 2 * pad_h - dil_patch_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - dil_patch_w) / stride_w + 1;
    long chunk_len = kernel_h * kernel_w;

    const long length = height * width * channels;
    #pragma omp parallel for
    for (long i = 0; i < length; ++i)
    {
        data_im[i] = 0;
    }

    #pragma omp parallel for if (channels > 1)
    for (int idx = 0; idx < channels; ++idx)
    {
        for (int inner_idx = 0; inner_idx < chunk_len; ++inner_idx)
        {
            int c = idx * chunk_len + inner_idx;
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / kernel_h / kernel_w;

            const int hc0 = h_offset * dilation_h - pad_h;
            const int wc0 = w_offset * dilation_w - pad_w;
            for (int h = 0; h < height_col; ++h)
            {
                for (int w = 0; w < width_col; ++w)
                {
                    int h_pad = h * stride_h + hc0;
                    const int srow_offset = (c_im * height + h_pad) * width;
                    const int row_offset = (c * height_col + h) * width_col;
                    int w_pad = w * stride_w + wc0;
                    if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width)))
                        data_im[srow_offset + w_pad] += data_col[row_offset + w];

                }
            }
        }
    }
}