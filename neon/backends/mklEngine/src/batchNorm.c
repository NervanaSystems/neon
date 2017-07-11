/*
 Copyright 2017 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include "MKLDNN.h"

static void BatchNormInit(
          long long* input,
          long long* output,
          long long* primitives,
          int N,
          int inC,
          int inH,
          int inW,
	      double eps)
{
    dnnError_t err;

    //CHWN
    size_t inputSize[DIM4] = 	{inW,  inH,         inC,   N};
    size_t inputStrides[DIM4] = {N,  N*inW,   N*inW*inH,   1};

    // we need prepare two layout(plain layout and primitive layout) for input(forp input and backp gradinput)
    dnnLayout_t lt_in_forp_backp_plain  = NULL;
    dnnLayout_t lt_out_forp_backp_plain = NULL;

    //create plain layout
    CHECK_ERR( dnnLayoutCreate_F32(&lt_in_forp_backp_plain,  DIM4, inputSize, inputStrides), err );    //plain layout input
    CHECK_ERR( dnnLayoutCreate_F32(&lt_out_forp_backp_plain, DIM4, inputSize, inputStrides), err );    //plain layout output

    //input may have user layout (from raw image data)
    //or maybe mkl layout (is previous mkl-based layer's output)
    dnnLayout_t lt_in_forp_raw = (dnnLayout_t)input[MKLLayout];                                 //try to get primitive layout
    if(lt_in_forp_raw == NULL)
    {
        lt_in_forp_raw = lt_in_forp_backp_plain;
    }

    //primitive to store info of related OP
    dnnPrimitive_t bn_forp     = NULL;
    dnnPrimitive_t bn_backp    = NULL;
    dnnPrimitive_t bn_forp_inf = NULL;
    
    //create forward and backward primitives and operation
    CHECK_ERR( dnnBatchNormalizationCreateForward_v2_F32(&bn_forp,     NULL, lt_in_forp_raw, eps, dnnUseScaleShift), err );
    CHECK_ERR( dnnBatchNormalizationCreateForward_v2_F32(&bn_forp_inf, NULL, lt_in_forp_raw, eps, dnnUseInputMeanVariance), err );
    CHECK_ERR( dnnBatchNormalizationCreateBackward_v2_F32(&bn_backp,   NULL, lt_in_forp_raw, eps, dnnUseScaleShift), err );

    //primitive layout 
    dnnLayout_t lt_forp_input_prmt;
    dnnLayout_t lt_mean_prmt;
    dnnLayout_t lt_variance_prmt;
    dnnLayout_t lt_forp_scaleshift_prmt;
    dnnLayout_t lt_forp_output_prmt;
    dnnLayout_t lt_backp_input_prmt;
    dnnLayout_t lt_backp_output_prmt;
    dnnLayout_t lt_backp_scaleshift_prmt;

    //create custom layout from primitives
    dnnLayoutCreateFromPrimitive_F32(&lt_forp_input_prmt,       bn_forp,  dnnResourceSrc);
    dnnLayoutCreateFromPrimitive_F32(&lt_mean_prmt,             bn_forp,  dnnResourceMean);
    dnnLayoutCreateFromPrimitive_F32(&lt_variance_prmt,         bn_forp,  dnnResourceVariance);
    dnnLayoutCreateFromPrimitive_F32(&lt_forp_scaleshift_prmt,  bn_forp,  dnnResourceScaleShift);
    dnnLayoutCreateFromPrimitive_F32(&lt_forp_output_prmt,      bn_forp,  dnnResourceDst);
    dnnLayoutCreateFromPrimitive_F32(&lt_backp_input_prmt,      bn_backp, dnnResourceDiffSrc);
    dnnLayoutCreateFromPrimitive_F32(&lt_backp_output_prmt,     bn_backp, dnnResourceDiffDst);
    dnnLayoutCreateFromPrimitive_F32(&lt_backp_scaleshift_prmt, bn_backp, dnnResourceDiffScaleShift);

    //buffer to store primitive layout
    float* buffer_forp_input       = NULL;
    float* buffer_mean             = NULL;
    float* buffer_variance         = NULL;
    float* buffer_forp_scaleshift  = NULL;
    float* buffer_forp_output      = NULL;
    float* buffer_backp_input      = NULL;
    float* buffer_backp_output     = NULL;
    float* buffer_backp_scaleshift = NULL;

    //allocate buffer for inner buffer
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forp_input),      lt_forp_input_prmt), err );
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_mean),            lt_mean_prmt),       err );
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_variance),        lt_variance_prmt),   err );
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forp_scaleshift), lt_forp_scaleshift_prmt), err );
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forp_output),     lt_forp_output_prmt),  err );
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_backp_input),     lt_backp_input_prmt),  err );
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_backp_output),    lt_backp_output_prmt), err );
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_backp_scaleshift),lt_backp_scaleshift_prmt), err );

    //plain layout will not be stored in primitives, it must be assigned here
    //primitive layout could be stored in primitives, it could be assigned laterly.
    output[CPULayout] = (long long)lt_out_forp_backp_plain;

    //save the dnnPrimitive
    //output layout
    primitives[BN_LT_IN_FORP_BACKP_PLAIN]  = (long long)lt_in_forp_backp_plain;
    primitives[BN_LT_OUT_FORP_BACKP_PLAIN] = (long long)lt_out_forp_backp_plain;

    primitives[BN_LT_FORP_INPUT]   = (long long)lt_forp_input_prmt;
    primitives[BN_LT_FORP_OUTPUT]  = (long long)lt_forp_output_prmt;
    primitives[BN_LT_BACKP_INPUT]  = (long long)lt_backp_input_prmt;
    primitives[BN_LT_BACKP_OUTPUT] = (long long)lt_backp_output_prmt;

    primitives[BN_FORP]     = (long long)bn_forp;
    primitives[BN_FORP_INF] = (long long)bn_forp_inf;
    primitives[BN_BACKP]    = (long long)bn_backp;

    //layout buffer
    primitives[BN_BUFFER_MEAN]             = (long long)buffer_mean;
    primitives[BN_BUFFER_VARIANCE]         = (long long)buffer_variance;
    primitives[BN_BUFFER_FORP_INPUT]       = (long long)buffer_forp_input;
    primitives[BN_BUFFER_FORP_SCALESHIFT]  = (long long)buffer_forp_scaleshift;
    primitives[BN_BUFFER_FORP_OUTPUT]      = (long long)buffer_forp_output;
    primitives[BN_BUFFER_BACKP_INPUT]      = (long long)buffer_backp_input;
    primitives[BN_BUFFER_BACKP_OUTPUT]     = (long long)buffer_backp_output;
    primitives[BN_BUFFER_BACKP_SCALESHIFT] = (long long)buffer_backp_scaleshift;
ERR_RETURN:
    return;
}

void BatchNormFprop(
  unsigned long long input,
  unsigned long long output,
  unsigned long long weight,
  unsigned long long bias,
  unsigned long long run_mean,
  unsigned long long run_var,
  float decay,
  int N,
  int inC,
  int inH,
  int inW,
  double eps,
  unsigned long long dnnprimitives,
  int initOk,
  int inference)
{
    dnnError_t err;
    long long * primitives = (long long * )dnnprimitives;
    if (initOk == 0)
    {
	    BatchNormInit((long long*)input, (long long*)output, primitives, N, inC, inH, inW, eps);
    }
    float* buffer_forp_scaleshift = (float*)primitives[BN_BUFFER_FORP_SCALESHIFT];
    float* weightPtr  = GetPtr(weight);
    float* biasPtr    = GetPtr(bias);
    int i = 0;
    for (i = 0; i < inC; ++i)
    {
	    buffer_forp_scaleshift[i]     = weightPtr ? weightPtr[i] : 1;
	    buffer_forp_scaleshift[i+inC] = biasPtr   ?   biasPtr[i] : 0;
    }

    void* BatchNorm_res[dnnResourceNumber] = {0};
    BatchNorm_res[dnnResourceSrc]  = GetPtr(input);
    BatchNorm_res[dnnResourceDst]  = (float*)primitives[BN_BUFFER_FORP_OUTPUT];

    BatchNorm_res[dnnResourceScaleShift] = buffer_forp_scaleshift;
    dnnPrimitive_t bn_forp = NULL;

    if (inference)
    {
        BatchNorm_res[dnnResourceMean]     = (float*)run_mean;
        BatchNorm_res[dnnResourceVariance] = (float*)run_var;
        bn_forp = (dnnPrimitive_t)primitives[BN_FORP_INF];
    }
    else
    {
        BatchNorm_res[dnnResourceMean]     = (float*)primitives[BN_BUFFER_MEAN];
        BatchNorm_res[dnnResourceVariance] = (float*)primitives[BN_BUFFER_VARIANCE];
        bn_forp = (dnnPrimitive_t)primitives[BN_FORP];
    }

    CHECK_ERR( dnnExecute_F32(bn_forp, (void*)BatchNorm_res), err );
    ((long long*)output)[MKLPtr]    = primitives[BN_BUFFER_FORP_OUTPUT];
    ((long long*)output)[MKLLayout] = primitives[BN_LT_FORP_OUTPUT];

	//calcuate global mean and val
	if (!inference)
	{
        const float reborn = 1.f - decay;
        cblas_saxpby(inC, reborn, BatchNorm_res[dnnResourceMean], 1, decay, (float*)run_mean, 1);
        cblas_saxpby(inC, reborn, BatchNorm_res[dnnResourceVariance], 1, decay, (float*)run_var, 1);
	}
ERR_RETURN:
    return;
}

static void Batch_InitB(long long* primitives, long long* gradOut)
{
    dnnError_t err;
    dnnLayout_t lt_out = (dnnLayout_t)gradOut[MKLLayout];
    if (lt_out==NULL)
    {
        lt_out = (dnnLayout_t)primitives[BN_LT_OUT_FORP_BACKP_PLAIN];
    }
    dnnPrimitive_t cv_out_b = NULL; float* buf_out_b = NULL;
    CHECK_ERR( try_convert(&cv_out_b, &buf_out_b, lt_out, (dnnLayout_t)primitives[BN_LT_BACKP_OUTPUT]) , err );
	primitives[BN_CV_BACKP_OUTPUT]      = (long long)cv_out_b;
	primitives[BN_BUFFER_BACKP_OUTPUT]  = (long long)buf_out_b;
ERR_RETURN:
    return;
}

void BatchNormBackp(
  unsigned long long input,
  unsigned long long gradOutput,
  unsigned long long gradInput,
  unsigned long long gradWeight,
  unsigned long long gradBias,
  int inC,
  unsigned long long dnnprimitives,
  int initOk)
{
    dnnError_t err;
    long long* primitives = (long long* )dnnprimitives;
    float* buffer_mean 	            = (float* )primitives[BN_BUFFER_MEAN];
    float* buffer_variance      	= (float* )primitives[BN_BUFFER_VARIANCE];
    float* buffer_forp_scaleshift 	= (float* )primitives[BN_BUFFER_FORP_SCALESHIFT];
    float* buffer_backp_scaleshift 	= (float* )primitives[BN_BUFFER_BACKP_SCALESHIFT];

    float* gradOutput_data  = GetPtr(gradOutput);
    float* gradWeight_data  = GetPtr(gradWeight);
    float* gradBias_data    = GetPtr(gradBias);
    void* BatchNorm_res[dnnResourceNumber] = {0};

    dnnPrimitive_t bn_backp = (dnnPrimitive_t)primitives[BN_BACKP];
    if (initOk==0)
    {
        Batch_InitB(primitives, (long long*)gradOutput);
    }

    //check gradOut
    BatchNorm_res[dnnResourceDiffDst] = gradOutput_data;
    dnnPrimitive_t cv_out_b = (dnnPrimitive_t)primitives[BN_CV_BACKP_OUTPUT];
    if (cv_out_b)
    {
        float* buf = (float *)(primitives[BN_BUFFER_BACKP_OUTPUT]);
        CHECK_ERR( dnnConversionExecute_F32(cv_out_b, gradOutput_data, buf), err );
        BatchNorm_res[dnnResourceDiffDst] = buf;
    }
    else
    {
        BatchNorm_res[dnnResourceDiffDst] = gradOutput_data;
    }

    BatchNorm_res[dnnResourceSrc]        = GetPtr(input);
    BatchNorm_res[dnnResourceDiffSrc]    = (float* )primitives[BN_BUFFER_BACKP_INPUT];
    BatchNorm_res[dnnResourceMean]       = buffer_mean;
    BatchNorm_res[dnnResourceVariance]   = buffer_variance;
    BatchNorm_res[dnnResourceScaleShift] = buffer_forp_scaleshift;
    BatchNorm_res[dnnResourceDiffScaleShift] = buffer_backp_scaleshift;

    CHECK_ERR( dnnExecute_F32(bn_backp, (void*)BatchNorm_res), err );
    for (int i = 0; i < inC; ++i)
    {
        gradWeight_data[i]  = buffer_backp_scaleshift[i];
        gradBias_data[i]    = buffer_backp_scaleshift[i+inC];
    }

    ((long long*)gradInput)[MKLPtr]  = (long long)primitives[BN_BUFFER_BACKP_INPUT];
    ((long long*)gradInput)[MKLLayout] = (long long)primitives[BN_LT_BACKP_INPUT];
    ((long long*)gradInput)[CPULayout] = (long long)primitives[BN_LT_IN_FORP_BACKP_PLAIN];

ERR_RETURN:
    return;
}
