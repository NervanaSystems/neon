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

//convert tensor in MKL layout back to Numpy NCHW layout
//if layout diff, do conversion, else, copy memory directly
void ConvertBack(unsigned long long tensor_, int N, int C, int H, int W)
{
    long long * tensor = (long long *)tensor_;
    if (tensor[CPUPtr] == 0 )
    {
        printf("error to converback tensor!\n");
        return;
    }

	if (tensor[MKLLayout] == 0)    return;//do not need convert
    dnnError_t err;
	size_t inSize[DIM4]   =   { W, H, C, N};
	size_t inStride[DIM4] =   { 1, W, W*H, W*H*C};
	dnnLayout_t lt_NCHW = NULL, lt_CHWN = NULL;
	float* newPtr = NULL;
	CHECK_ERR( dnnLayoutCreate_F32(&lt_NCHW, DIM4, inSize, inStride),  err );
	if (!dnnLayoutCompare_F32((dnnLayout_t)tensor[MKLLayout], (dnnLayout_t)tensor[CPULayout]))
	{
		float* cpuPtr = (float *)tensor[CPUPtr];
		float* mklPtr = (float *)tensor[MKLPtr];

		if (!dnnLayoutCompare_F32((dnnLayout_t)tensor[MKLLayout], lt_NCHW))
		{
		    dnnPrimitive_t cv;
            CHECK_ERR( dnnConversionCreate_F32(&cv, (dnnLayout_t)tensor[MKLLayout],lt_NCHW), err );
            newPtr = (float*)malloc(N*C*H*W*sizeof(float));
            CHECK_ERR( dnnConversionExecute_F32(cv, mklPtr, newPtr), err );
            mklPtr = newPtr;
		}
        mkl_somatcopy('r', 't', N, C*H*W, 1.0, mklPtr, C*H*W, cpuPtr, N);
	}
	else
	{
	    long long grad_in_len = (long long)dnnLayoutGetMemorySize_F32((dnnLayout_t)tensor[MKLLayout]) ;
        float * destPtr = (float*)tensor[CPUPtr];
        float * srcPtr = (float*)tensor[MKLPtr];
        #pragma omp parallel for
        for (long long i = 0; i < grad_in_len/4; ++i)
        {
            destPtr[i] = srcPtr[i];
        }
    }
ERR_RETURN:
    if (newPtr!=NULL)
 	{
 	    free(newPtr);
 	}
}

void ConvertToMKL(unsigned long long tensor_)
{
    long long * tensor = (long long *)tensor_;
    if (tensor[CPUPtr] == 0 )
    {
        printf("error to conver to MKL tensor!\n");
        return;
    }

	if (tensor[MKLLayout] == 0)    return;//do not need convert

	if (!dnnLayoutCompare_F32((dnnLayout_t)tensor[MKLLayout],
			(dnnLayout_t)tensor[CPULayout]))
	{
		dnnError_t err; dnnPrimitive_t cv;
		CHECK_ERR( dnnConversionCreate_F32(&cv, (dnnLayout_t)tensor[CPULayout],
		            (dnnLayout_t)tensor[MKLLayout]), err );
		CHECK_ERR( dnnConversionExecute_F32(cv, (float *)tensor[CPUPtr], (float *)tensor[MKLPtr]), err );
	}
	else
	{
	    memcpy((void*)tensor[MKLPtr], (void*)tensor[CPUPtr], dnnLayoutGetMemorySize_F32((dnnLayout_t)tensor[MKLLayout]));
    }
ERR_RETURN:
    return;
}

void CleanPrimitive(long long * primitives, int length)
{
    for(int i=0; i<length; i++)
    {
        if(primitives[i] != 0)
        {
            void* p = (void*)primitives[i];
            free(p);
        }
    }
}

void MatTrans(
  unsigned long long input,
  unsigned long long output,
  unsigned long long m,
  unsigned long long n)
{
    mkl_somatcopy('r', 't', m, n, 1.0, (float*)input, n, (float*)output, m);
}

static int Conv_f_init(
  long long * input,
  long long * output,
  long long * weight,
  long long * primitives,
  int N, int inC, int inH, int inW,
  int kH, int kW,
  int dH, int dW,
  int padH, int padW,
  int outC, int outH, int outW,
  int hasBias)
{
    dnnError_t err;

    //init dimensions
	size_t inputSize[DIM4]  =   { inW,  inH,  inC, N};
	size_t outputSize[DIM4] =   {outW, outH, outC, N};
	size_t filterSize[DIM4] =   {  kW,   kH,  inC, outC};
	size_t stride[DIM2]     =   {  dW,   dH};
	int    pad[DIM2]        =   {-padW, -padH};
	size_t biasSize[1]      =   {outC};
	size_t biasStrides[1]   =   { 1 };

    //using NCHW layout
	size_t filterStridesNCHW[DIM4] = {1,  kW,      kW*kH,     kW*kH*inC};
    size_t inputStridesNCHW[DIM4]  = {1,  inW,   inW*inH,    inW*inH*inC};
    size_t outputStridesNCHW[DIM4] = {1, outW, outW*outH, outW*outH*outC};

    //CHWN
    size_t filterStridesCHWN[DIM4] = {outC,  outC*kW,      outC*kW*kH,     1};
    size_t inputStridesCHWN[DIM4]  = {N,  N*inW,   N*inW*inH,    1};
    size_t outputStridesCHWN[DIM4] = {N,  N*outW,  N*outW*outH,  1};

    //create execute and save into primitives
    dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
	dnnPrimitive_t conv_f       = NULL;    //forward operation
	dnnPrimitive_t conv_bdata   = NULL;    //backward calculate gradient input
	dnnPrimitive_t conv_bfilter = NULL;    //backward calculate gradient filter(weight)
    dnnPrimitive_t conv_b_bias  = NULL;    //backward bias

    //create layout and save
    //lt_in, layout of input in NCHW form
    //lt_filter_f, required layout (MKL layout) for forward for weight
    //lt_out_bfilter, required layout for backward weight update for output
	dnnLayout_t lt_in_NCHW,    lt_filter,         lt_out_NCHW,   lt_in_CHWN,  lt_out_CHWN, lt_bias_CHWN=NULL;
	dnnLayout_t lt_in_f,       lt_filter_f,       lt_out_f,      lt_bias_f;
	dnnLayout_t lt_in_bdata,   lt_filter_bdata,   lt_out_bdata,  lt_bias_bdata;
    dnnLayout_t lt_in_bfilter, lt_filter_bfilter, lt_out_bfilter,lt_bias_bias, lt_out_bias;

 	if (hasBias)
 	{
 	    CHECK_ERR(dnnConvolutionCreateForwardBias_F32(   &conv_f,   attributes, dnnAlgorithmConvolutionDirect, DIM4, inputSize, outputSize, filterSize, stride, pad, dnnBorderZeros),err);
 	    CHECK_ERR(dnnConvolutionCreateForwardBias_F32(   &conv_f,   attributes, dnnAlgorithmConvolutionDirect, DIM4, inputSize, outputSize, filterSize, stride, pad, dnnBorderZeros),err);
        CHECK_ERR( dnnLayoutCreate_F32(&lt_bias_CHWN, 1, biasSize, biasStrides), err );
    	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bias_f,   conv_f, dnnResourceBias   ) , err );
        CHECK_ERR(dnnConvolutionCreateBackwardBias_F32(  &conv_b_bias,  attributes, dnnAlgorithmConvolutionDirect, DIM4, outputSize),err);
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bias_bias,   conv_b_bias,  dnnResourceDiffBias) , err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_out_bias,    conv_b_bias,  dnnResourceDiffDst) , err );
    }
    else
    	CHECK_ERR(dnnConvolutionCreateForward_F32(       &conv_f,   attributes, dnnAlgorithmConvolutionDirect, DIM4, inputSize, outputSize, filterSize, stride, pad, dnnBorderZeros),err);
 	CHECK_ERR(dnnConvolutionCreateBackwardData_F32(  &conv_bdata,   attributes, dnnAlgorithmConvolutionDirect, DIM4, inputSize, outputSize, filterSize, stride, pad, dnnBorderZeros),err);
	CHECK_ERR(dnnConvolutionCreateBackwardFilter_F32(&conv_bfilter, attributes, dnnAlgorithmConvolutionDirect, DIM4, inputSize, outputSize, filterSize, stride, pad, dnnBorderZeros),err);

    primitives[FORWARD_INDEX]    = (long long)conv_f;
    primitives[BWD_DATA_INDEX]   = (long long)conv_bdata;
    primitives[BWD_FILTER_INDEX] = (long long)conv_bfilter;
    primitives[BDW_BIAS_INDEX]   = (long long)conv_b_bias;

	CHECK_ERR( dnnLayoutCreate_F32(&lt_in_NCHW,     DIM4, inputSize,  inputStridesNCHW),  err );
	CHECK_ERR( dnnLayoutCreate_F32(&lt_in_CHWN,     DIM4, inputSize,  inputStridesCHWN),  err );
	CHECK_ERR( dnnLayoutCreate_F32(&lt_filter,      DIM4, filterSize, filterStridesCHWN), err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_out_NCHW,    DIM4, outputSize, outputStridesNCHW), err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_out_CHWN,    DIM4, outputSize, outputStridesCHWN), err );

    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_in_f,     conv_f, dnnResourceSrc   ) , err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_filter_f, conv_f, dnnResourceFilter), err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_out_f,    conv_f, dnnResourceDst   ) , err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_in_bdata,     conv_bdata, dnnResourceDiffSrc) , err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_filter_bdata, conv_bdata, dnnResourceFilter) , err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_out_bdata,    conv_bdata, dnnResourceDiffDst) , err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_in_bfilter,     conv_bfilter, dnnResourceSrc) , err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_filter_bfilter, conv_bfilter, dnnResourceDiffFilter) , err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_out_bfilter,    conv_bfilter, dnnResourceDiffDst) , err );

    //here assume NCHW (CHWN will be transposed)
    primitives[L_I]    = (long long)lt_in_NCHW;
    primitives[L_O]    = (long long)lt_out_NCHW;
    primitives[L_W]    = (long long)lt_filter;
    primitives[L_B]    = (long long)lt_bias_CHWN;
    primitives[L_F_I]  = (long long)lt_in_f;
    primitives[L_F_O]  = (long long)lt_out_f;
    primitives[L_F_W]  = (long long)lt_filter_f;
    primitives[L_F_B]  = (long long)lt_bias_f;
    primitives[L_BD_I] = (long long)lt_in_bdata;
    primitives[L_BD_O] = (long long)lt_out_bdata;
    primitives[L_BD_W] = (long long)lt_filter_bdata;
    primitives[L_BF_I] = (long long)lt_in_bfilter;
    primitives[L_BF_O] = (long long)lt_out_bfilter;
    primitives[L_BF_W] = (long long)lt_filter_bfilter;
    primitives[L_I_CHWN] = (long long)lt_in_CHWN;
    primitives[L_O_CHWN] = (long long)lt_out_CHWN;
    primitives[L_B_B]    = (long long)lt_bias_bias;
    primitives[L_B_O]    = (long long)lt_out_bias;
    //input may have user layout (from raw image data,continuous NCHW )
    //  or maybe mkl layout (is previous mkl-based layer's output)
    dnnLayout_t lt_in_real = (dnnLayout_t)input[MKLLayout];
    if(lt_in_real==NULL) lt_in_real = lt_in_NCHW;
    //create conversion and buff if necessary
    dnnPrimitive_t cv_in_f = NULL; float * buf_in_f = NULL;
    CHECK_ERR( try_convert(&cv_in_f, &buf_in_f, lt_in_real, lt_in_f) , err );

    //create transpose if necessary
    float* newPtr = NULL;
	if (input[MKLLayout] == 0)
	{
		newPtr = (float*)malloc(inC*inH*inW*N*sizeof(float));
	}
    primitives[BUFFER_TRANS_INPUT] = (long long)newPtr;

    //save conversion and buff
    primitives[BUFFER_FORWARD_INPUT]  = (long long)buf_in_f;
    primitives[CONVERT_FORWARD_INPUT] = (long long)cv_in_f;

    //filter layout
    dnnPrimitive_t cv_filter_f = NULL; float * buf_filter_f = NULL;
    CHECK_ERR( try_convert(&cv_filter_f, &buf_filter_f, lt_filter, lt_filter_f), err );
    primitives[CONVERT_FORWARD_FILTER] = (long long)cv_filter_f;
    primitives[BUFFER_FORWARD_FILTER]  = (long long)buf_filter_f;

	//save user layout for output, and create mkl buffer
	//output always has mkl buffer and recorded in layer's primitive
	output[CPULayout] = (long long)lt_out_CHWN;
	float* buf_out_f = NULL;
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buf_out_f), lt_out_f), err );
    primitives[BUFFER_FORWARD_OUTPUT] = (long long)buf_out_f;

    //for bias
    dnnPrimitive_t cv_bias_f = NULL; float * buf_bias_f = NULL;
    dnnPrimitive_t cv_bias_b = NULL; float * buf_bias_b = NULL;
    if (hasBias)
    {
        CHECK_ERR( try_convert(&cv_bias_f, &buf_bias_f, lt_bias_CHWN, lt_bias_f), err );
    }
    primitives[CONVERT_FORWARD_BIAS] = (long long)cv_bias_f;
    primitives[BUFFER_FORWARD_BIAS]  = (long long)buf_bias_f;

    return 0;

ERR_RETURN:

    return 1;
}

int Conv_forward(
  unsigned long long input,
  unsigned long long output,
  unsigned long long weight,
  unsigned long long bias,
  unsigned long long dnnprimitives,
  int initOk,
  int N,  int inC, int inH, int inW,
  int kH, int kW,
  int dH,  int dW,
  int padH, int padW,
  int outC, int outH, int outW)
{
	dnnError_t err;
	long long * primitives = (long long*)dnnprimitives;
	if(initOk == 0)
	{
		int hasBias = 1;
        if (bias == 0) hasBias = 0;
		//for the first time, initialize layout and conversion
		int res = Conv_f_init((long long *)input,	(long long *)output, (long long *)weight, primitives,
		            N, inC, inH, inW, kH, kW, dH, dW, padH, padW, outC, outH, outW, hasBias);
	    if(res)
	    {
	        return 1;
	    }
	}

	//get memory as resource
	float* resConv[dnnResourceNumber]={0};
	float* outPtr    = (float*)primitives[BUFFER_FORWARD_OUTPUT];
	float* filterPtr = GetPtr(weight);
    float* biasPtr   = NULL;

	resConv[dnnResourceFilter] = filterPtr;
	resConv[dnnResourceDst]    = outPtr;
	float* inPtr = GetPtr(input);
	if(bias != 0) resConv[dnnResourceBias] = GetPtr(bias);

	//do conversion for input if necessary
	float* newPtr = (float*)primitives[BUFFER_TRANS_INPUT];
	if( newPtr != NULL)
	{
		mkl_somatcopy('r', 't', inC*inH*inW, N, 1.0, inPtr, N, newPtr, inC*inH*inW);
        inPtr = newPtr;
    }
	resConv[dnnResourceSrc]  = inPtr;

	dnnPrimitive_t cv_in_f = (dnnPrimitive_t)primitives[CONVERT_FORWARD_INPUT];
    if(cv_in_f)
	{
		//if no MKL layout, first transpose CHWN into NCHW
		float* buf_in_f  = (float *)(primitives[BUFFER_FORWARD_INPUT]);
		CHECK_ERR( dnnConversionExecute_F32(cv_in_f, inPtr, buf_in_f), err );
	    resConv[dnnResourceSrc] = buf_in_f;
	}

	//do conversion for filter if necessary
	dnnPrimitive_t cv_filter_f = (dnnPrimitive_t)primitives[CONVERT_FORWARD_FILTER];
	if(cv_filter_f)
	{
		float* buf_filter_f = (float *)(primitives[BUFFER_FORWARD_FILTER]);
		CHECK_ERR( dnnConversionExecute_F32(cv_filter_f, filterPtr, buf_filter_f), err );
        resConv[dnnResourceFilter] = buf_filter_f;
    }

    dnnPrimitive_t cv_bias_f  = (dnnPrimitive_t)primitives[CONVERT_FORWARD_BIAS];

    if (cv_bias_f)
    {
        biasPtr = GetPtr(bias);
        float* buf_bias_f = (float *)primitives[BUFFER_FORWARD_BIAS];
        CHECK_ERR( dnnConversionExecute_F32(cv_bias_f, biasPtr, buf_bias_f), err );
        resConv[dnnResourceBias] = buf_bias_f;
    }

    //real execute operation
	CHECK_ERR(dnnExecute_F32((dnnPrimitive_t)primitives[FORWARD_INDEX],(void**)resConv),err);

    //always fill in MKL information for output
	((long long*)output)[MKLPtr]    = primitives[BUFFER_FORWARD_OUTPUT];
    ((long long*)output)[MKLLayout] = (long long)primitives[L_F_O];

    return 0;

ERR_RETURN:
    return 1;
}

//gradOut: output gradient of CONV layer, known parameters
//gradIn:  input gradient, to be calculated
static void Conv_bdata_init(
  long long * gradIn,
  long long * gradOut,
  int N, int oC, int oH, int oW,
  long long * weight,
  long long * primitives)
{
    dnnError_t err;

    //get gradOut layout, create conversion if necessary
    dnnLayout_t lt_out  = (dnnLayout_t)gradOut[MKLLayout];
    if(lt_out==NULL)
    {
        lt_out  = (dnnLayout_t)primitives[L_O];
    }
    dnnPrimitive_t cv_out_bdata = NULL;	float * buf_out_bdata = NULL;
    CHECK_ERR( try_convert(&cv_out_bdata, &buf_out_bdata, lt_out, (dnnLayout_t)primitives[L_BD_O]) , err );
	primitives[CONVERT_BWDDATA_OUTPUT] = (long long)cv_out_bdata;
	primitives[BUFFER_BWDDATA_OUTPUT]  = (long long)buf_out_bdata;

	//for filter
	dnnLayout_t lt_filter       = (dnnLayout_t)primitives[L_W];
	dnnLayout_t lt_filter_bdata = (dnnLayout_t)primitives[L_BD_W];
    dnnPrimitive_t cv_filter_bdata = NULL; float * buf_filter_bdata = NULL;
    CHECK_ERR( try_convert(&cv_filter_bdata, &buf_filter_bdata, lt_filter, lt_filter_bdata), err );
	primitives[BUFFER_BWDDATA_FILTER]  = (long long)buf_filter_bdata;
	primitives[CONVERT_BWDDATA_FILTER] = (long long)cv_filter_bdata;

    //create gradInput layout and memory
    dnnLayout_t lt_in_bdata = (dnnLayout_t)primitives[L_BD_I];
    float * buf_in_bdata = (float*)(gradIn[CPUPtr]);
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buf_in_bdata), lt_in_bdata), err );
    primitives[BUFFER_BWDDATA_INPUT] = (long long)buf_in_bdata;
    gradIn[CPULayout] = (long long)(dnnLayout_t)primitives[L_I_CHWN];

    float* gradOutTransPtr = NULL;
    if (gradOut[MKLLayout] == 0)
	{
		gradOutTransPtr = (float*)malloc(N*oC*oH*oW*sizeof(float));
	}
	primitives[BUFFER_TRANS_OUTPUT] = (long long)gradOutTransPtr;

ERR_RETURN:
    return;
}

//gradOutput, conv output gradient, also as the input
//gradInput, to be calculated
void Conv_bwdData(
  unsigned long long gradOutput,
  unsigned long long gradInput,
  unsigned long long weight,
  unsigned long long dnnprimitives,
  int N, int oC, int oH, int oW,
  int initOk, float beta)
{
    dnnError_t err;
    long long * primitives = (long long *)dnnprimitives;

	if(initOk == 0)
	{
		Conv_bdata_init((long long *)gradInput,	(long long *)gradOutput, N, oC, oH, oW, (long long *)weight, primitives);
	}

    //get resource
    float* inPtr     = (float*)primitives[BUFFER_BWDDATA_INPUT];
	float* outPtr    = GetPtr(gradOutput);
	float* filterPtr = GetPtr(weight);

	float * resConv[dnnResourceNumber]={0};
	resConv[dnnResourceDiffSrc] = inPtr;
	resConv[dnnResourceFilter]  = filterPtr;

    //do transpose if necessary
    float* newPtr = (float*)primitives[BUFFER_TRANS_OUTPUT];
    if (newPtr!=NULL)
    {
		mkl_somatcopy('r', 't', oC*oH*oW, N, 1.0, outPtr, N, newPtr, oC*oH*oW);
        outPtr = newPtr;
    }
	resConv[dnnResourceDiffDst] = outPtr;

	//do conversion if necessary
	dnnPrimitive_t cv_out_bdata = (dnnPrimitive_t)primitives[CONVERT_BWDDATA_OUTPUT];
	if (cv_out_bdata)
	{
		float* buf_out_bdata = (float *)(primitives[BUFFER_BWDDATA_OUTPUT]);
		CHECK_ERR( dnnConversionExecute_F32(cv_out_bdata, outPtr, buf_out_bdata), err );
		resConv[dnnResourceDiffDst] = buf_out_bdata;
	}

	dnnPrimitive_t cv_filter_bdata = (dnnPrimitive_t)primitives[CONVERT_BWDDATA_FILTER];
	if (cv_filter_bdata)
	{
		float* buf_filter_bdata = (float *)(primitives[BUFFER_BWDDATA_FILTER]);
		CHECK_ERR( dnnConversionExecute_F32(cv_filter_bdata, filterPtr, buf_filter_bdata), err );
	    resConv[dnnResourceFilter] = buf_filter_bdata;
	}

	CHECK_ERR(dnnExecute_F32((dnnPrimitive_t)primitives[BWD_DATA_INDEX], (void**)resConv),err);

    ((long long*)gradInput)[MKLLayout] = (long long)primitives[L_BD_I];
    ((long long*)gradInput)[MKLPtr]    = (long long)primitives[BUFFER_BWDDATA_INPUT];

ERR_RETURN:
    return;
}

static void Conv_bfilter_init(
  long long * input,
  long long * gradOutput,
  long long * gradWeight,
  long long * primitives,
  int N, int oC, int oH, int oW)
{
    dnnError_t err;

    //for gradOut
    dnnLayout_t lt_out = (dnnLayout_t)(gradOutput[MKLLayout]);
    if(lt_out==NULL) lt_out = (dnnLayout_t)primitives[L_O];
    dnnPrimitive_t cv_out_bfilter = NULL; float* buf_out_bfilter = NULL;
    CHECK_ERR( try_convert(&cv_out_bfilter, &buf_out_bfilter, lt_out, (dnnLayout_t)primitives[L_BF_O]) , err );
	primitives[CONVERT_BWDFILTER_OUTPUT] = (long long)cv_out_bfilter;
	primitives[BUFFER_BWDFILTER_OUTPUT]  = (long long)buf_out_bfilter;

	//for the first layer without delta, input gradOut should first be transposed
    float* gradOutTransPtr = NULL;
    if ( gradOutput[MKLLayout] == 0 && primitives[BUFFER_TRANS_OUTPUT] == 0)
	{
		gradOutTransPtr = (float*)malloc(N*oC*oH*oW*sizeof(float));
		primitives[BUFFER_TRANS_OUTPUT] = (long long)gradOutTransPtr;
	}

    //for filter
	dnnLayout_t lt_filter         = (dnnLayout_t)primitives[L_W];
	dnnLayout_t lt_filter_bfilter = (dnnLayout_t)primitives[L_BF_W];
    dnnPrimitive_t cv_filter_bfilter = NULL; float * buf_filter_bfilter = NULL;
    if(!dnnLayoutCompare_F32(lt_filter_bfilter, lt_filter))
    {
        CHECK_ERR( dnnConversionCreate_F32(&cv_filter_bfilter, lt_filter_bfilter, lt_filter), err);
		CHECK_ERR( dnnAllocateBuffer_F32((void**)&buf_filter_bfilter, lt_filter_bfilter), err);
    }
    primitives[BUFFER_BWDFILTER_FILTER]  = (long long)buf_filter_bfilter;
	primitives[CONVERT_BWDFILTER_FILTER] = (long long)cv_filter_bfilter;

    //for input
    dnnLayout_t lt_in_real = (dnnLayout_t)input[MKLLayout];
    if(lt_in_real==NULL)
    {
        lt_in_real = (dnnLayout_t)primitives[L_I];
    }
    dnnLayout_t lt_in_bfilter = (dnnLayout_t)primitives[L_BF_I];
    dnnPrimitive_t cv_in_bfilter = NULL; float* buf_in_bfilter = (float*)(input[CPUPtr]);
    CHECK_ERR( try_convert(&cv_in_bfilter, &buf_in_bfilter, lt_in_real, lt_in_bfilter), err );
    primitives[BUFFER_BWDFILTER_INPUT]  = (long long)buf_in_bfilter;
    primitives[CONVERT_BWDFILTER_INPUT] = (long long)cv_in_bfilter;

    //if has bias
    if (primitives[BDW_BIAS_INDEX] != 0)
    {
        //convert for grad_bias if necessary
        dnnLayout_t lt_bias_bias  = (dnnLayout_t)primitives[L_B_B];
        dnnLayout_t lt_bias       = (dnnLayout_t)primitives[L_B];
        dnnPrimitive_t cv_bias_bias = NULL; float * buf_bias_bias = NULL;
        CHECK_ERR( dnnConversionCreate_F32(&cv_bias_bias, lt_bias_bias, lt_bias), err);
		CHECK_ERR( dnnAllocateBuffer_F32((void**)&buf_bias_bias, lt_bias_bias), err);
        primitives[BUFFER_BIAS_BIAS]  = (long long)buf_bias_bias;
        primitives[CV_BIAS_BIAS] =      (long long)cv_bias_bias;

        //convert for grad_out if necessary
        dnnLayout_t lt_bias_out = (dnnLayout_t)primitives[L_B_O];
        dnnPrimitive_t cv_out_bias = NULL; float* buf_out_bias = (float*)(input[CPUPtr]);
        CHECK_ERR( try_convert(&cv_out_bias, &buf_out_bias, lt_out, lt_bias_out), err );
        primitives[BUFFER_BIAS_OUT]  = (long long)buf_out_bias;
        primitives[CV_BIAS_OUT] = (long long)cv_out_bias;
    }


ERR_RETURN:
    return;
}

void Conv_bwdFilter(
  unsigned long long input,
  unsigned long long gradOutput,
  unsigned long long gradWeight,
  unsigned long long gradBias,
  unsigned long long dnnprimitives,
  int N, int oC, int oH, int oW,
  int initOk,
  int has_delta)
{
    dnnError_t err;
    long long * primitives = (long long * )dnnprimitives;
	if (initOk == 0)
	{
		Conv_bfilter_init((long long *)input,(long long *)gradOutput,(long long *)gradWeight, primitives, N,  oC, oH, oW);
	}

	float * inPtr     = GetPtr(input);
	float * filterPtr = GetPtr(gradWeight);
	float * outPtr    = GetPtr(gradOutput);

	float * resConv[dnnResourceNumber]={0};
	float * resBias[dnnResourceNumber]={0};
	resConv[dnnResourceDiffFilter] = filterPtr;

    //do input conversion if necessary
    float* newInputPtr = (float*)primitives[BUFFER_TRANS_INPUT];
	if (newInputPtr != NULL)
	{
	    inPtr = newInputPtr;
	}
    resConv[dnnResourceSrc] = inPtr;

	dnnPrimitive_t cv_in_bfilter = (dnnPrimitive_t)primitives[CONVERT_BWDFILTER_INPUT];
	if (cv_in_bfilter)
	{
		float* buf_in_bfilter = (float *)(primitives[BUFFER_BWDFILTER_INPUT]);
		CHECK_ERR( dnnConversionExecute_F32(cv_in_bfilter, inPtr, buf_in_bfilter), err );
	    resConv[dnnResourceSrc] = buf_in_bfilter;
	}

    //for gradout in cpu layout
    float* newGradOutPtr = (float*)primitives[BUFFER_TRANS_OUTPUT];
	if (newGradOutPtr != NULL)
	{
	    if (!has_delta)   //for the first layer without delta
	    {
	    	mkl_somatcopy('r', 't', oC*oH*oW, N, 1.0, outPtr, N, newGradOutPtr, oC*oH*oW);
	    }
	    outPtr = newGradOutPtr;//use transposed NCHW layout
	}
    resConv[dnnResourceDiffDst] = outPtr;

	//do gradOutput conversion if necessary
	dnnPrimitive_t cv_out_bfilter = (dnnPrimitive_t)primitives[CONVERT_BWDFILTER_OUTPUT];
	if (cv_out_bfilter)
	{
        float* buf_out_bfilter = (float *)(primitives[BUFFER_BWDFILTER_OUTPUT]);
		CHECK_ERR( dnnConversionExecute_F32(cv_out_bfilter, outPtr, buf_out_bfilter), err );
	    resConv[dnnResourceDiffDst] = buf_out_bfilter;
	    resBias[dnnResourceDiffDst] = buf_out_bfilter;
	}

	dnnPrimitive_t cv_filter_bfilter = (dnnPrimitive_t)primitives[CONVERT_BWDFILTER_FILTER];
	float* buf_filter_bfilter = (float *)(primitives[BUFFER_BWDFILTER_FILTER]);
	if (cv_filter_bfilter)
	{
	    resConv[dnnResourceDiffFilter] = buf_filter_bfilter;
	}

	CHECK_ERR(dnnExecute_F32((dnnPrimitive_t)primitives[BWD_FILTER_INDEX],
	                        (void**)resConv), err);

    //bias
    if (gradBias != 0)
    {
        float * biasPtr = GetPtr(gradBias);
        dnnPrimitive_t cv_bias_bias = (dnnPrimitive_t)primitives[CV_BIAS_BIAS];
        resBias[dnnResourceDiffBias] = biasPtr;
        if (cv_bias_bias)
        {
            resBias[dnnResourceDiffBias] = (float*)primitives[BUFFER_BIAS_BIAS];
        }

	    resBias[dnnResourceDiffDst] = outPtr;
	    dnnPrimitive_t cv_out_bias = (dnnPrimitive_t)primitives[CV_BIAS_OUT];
	    if (cv_out_bias)
	    {
	     	float* buf_out_bias = (float*)primitives[BUFFER_BIAS_OUT];
	        CHECK_ERR( dnnConversionExecute_F32(cv_out_bias, outPtr, buf_out_bias), err );
	        resBias[dnnResourceDiffDst] = outPtr;
	    }

	    CHECK_ERR(dnnExecute_F32((dnnPrimitive_t)primitives[BDW_BIAS_INDEX], (void**)resBias), err);

        if (cv_bias_bias)
        {
            CHECK_ERR( dnnConversionExecute_F32(cv_bias_bias,resBias[dnnResourceDiffBias], biasPtr), err );
        }
    }

    //do gradWeight conversion if necessary
    if (cv_filter_bfilter)
    {
       CHECK_ERR( dnnConversionExecute_F32(cv_filter_bfilter, buf_filter_bfilter, filterPtr), err );
    }
ERR_RETURN:
    return;
}