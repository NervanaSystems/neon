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
#include <math.h>

//useMaxPooling, or averagePooling
//useCaffe use ceil mode for pooling output dim
static void Init_f(
  long long * input,
  long long * output,
  long long * primitives,
  int N, int inC, int inH, int inW,
  int kH, int kW, int dH, int dW,
  int padH,int padW,
  int outC, int outH,int outW,
  int useMaxPooling,
  int useCaffe)
{

	dnnError_t err;

	//dimension
	size_t inputSize[DIM4]     = { inW,  inH,  inC, N};
	size_t outputSize[DIM4]    = {outW, outH, outC, N};
	size_t inputStrides1[DIM4]  = {1,  inW,   inW*inH,    inW*inH*inC};
	size_t outputStrides1[DIM4] = {1, outW, outW*outH, outW*outH*outC};

	//CHWN
	size_t inputStrides[DIM4]  = {N,  N*inW,   N*inW*inH,    1};
	size_t outputStrides[DIM4] = {N, N*outW, N*outW*outH, 1};

    size_t kernelSize[2]       = {   kW,   kH};
	size_t kernelStride[2]     = {   dW,   dH};

	//calculate pad
	int padH2 = (outH-1)*dH + kH - inH - padH;
	int padW2 = (outW-1)*dW + kW - inW - padW;
	int symm = 0;
	if (padH2==padH && padW2==padW) symm = 1;
	if (padH2<0)    padH2 = 0;
	if (padW2<0)    padW2 = 0;

	int pad_dim4[DIM4]      = {-padW, -padH, -padW2,-padH2};
	int pad_dim2[DIM2]      = {-padW, -padH};
	int inputOffset[DIM2]   = {    0,    0};

    //create user layout
    dnnLayout_t lt_out = NULL, lt_in = NULL;
	CHECK_ERR( dnnLayoutCreate_F32(&lt_in,  DIM4,  inputSize,  inputStrides) , err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_out, DIM4, outputSize, outputStrides) , err );
    primitives[POOL_L_I] = (long long)lt_in;
    primitives[POOL_L_O] = (long long)lt_out;

    //create MKL input layout
	dnnLayout_t lt_in_f = (dnnLayout_t)input[MKLLayout];
    if(lt_in_f==NULL)
    {
        lt_in_f = lt_in;
    }
    primitives[POOL_L_F_I] = (long long)lt_in_f;

	//create operation
	dnnPrimitive_t pool_f = NULL, pool_b = NULL;
	dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );

	if (useMaxPooling==1)
	{
		if(useCaffe || symm)
		{
		    CHECK_ERR( dnnPoolingCreateForward_F32 (&pool_f, attributes, dnnAlgorithmPoolingMax,lt_in_f, kernelSize, kernelStride, pad_dim2, dnnBorderZeros), err );
	        CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_b, attributes, dnnAlgorithmPoolingMax,lt_in_f, kernelSize, kernelStride, pad_dim2, dnnBorderZeros), err );
		}
		else
		{
            CHECK_ERR( dnnPoolingCreateForward_F32 (&pool_f, attributes, dnnAlgorithmPoolingMax,lt_in_f, kernelSize, kernelStride, pad_dim4, dnnBorderZerosAsymm), err );
            CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_b, attributes, dnnAlgorithmPoolingMax,lt_in_f, kernelSize, kernelStride, pad_dim4, dnnBorderZerosAsymm), err );
	    }
	}
	else
	{
		if(useCaffe || symm)
		{
		    CHECK_ERR( dnnPoolingCreateForward_F32 (&pool_f, attributes, dnnAlgorithmPoolingAvg,lt_in_f, kernelSize, kernelStride, pad_dim2, dnnBorderZeros), err );
	        CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_b, attributes, dnnAlgorithmPoolingAvg,lt_in_f, kernelSize, kernelStride, pad_dim2, dnnBorderZeros), err );

		}
		else
		{
            CHECK_ERR( dnnPoolingCreateForward_F32 (&pool_f, attributes, dnnAlgorithmPoolingAvg,lt_in_f, kernelSize, kernelStride, pad_dim4, dnnBorderZerosAsymm), err );
            CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_b, attributes, dnnAlgorithmPoolingAvg,lt_in_f, kernelSize, kernelStride, pad_dim4, dnnBorderZerosAsymm), err );
	    }
	}
	primitives[POOLING_FORWARD]  = (long long)pool_f;
	primitives[POOLING_BACKWARD] = (long long)pool_b;

    //create mkl layout for output
    dnnLayout_t lt_out_f = NULL, lt_out_b = NULL, lt_in_b = NULL;
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_out_f, pool_f, dnnResourceDst),   err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_in_b,  pool_f, dnnResourceSrc),   err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_out_b, pool_f, dnnResourceDst),   err );
    primitives[POOL_L_F_O] = (long long)lt_out_f;
    primitives[POOL_L_B_I] = (long long)lt_in_b;
    primitives[POOL_L_B_O] = (long long)lt_out_b;

	//create work space , to record max location?
	dnnLayout_t lt_space = NULL; float* buf_space = NULL;
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_space, pool_f, dnnResourceWorkspace), err );
	CHECK_ERR( dnnAllocateBuffer_F32((void**)&buf_space, lt_space) , err );
    primitives[BUFFER_POOLING_FORWARD_WORKSPACE] = (long long)buf_space;

	//output layout
    output[CPULayout] = (long long)lt_out;
    float* buf_out_f = NULL;
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buf_out_f), lt_out_f), err );
    primitives[BUFFER_POOLING_FORWARD_OUTPUT] = (long long)buf_out_f;

ERR_RETURN:
    return;
}

void MaxPooling_fprop(
  unsigned long long input,
  unsigned long long output,
  unsigned long long dnnprimitives,
  int initOk,
  int useMaxPooling,
  int N, int inC,
  int inH, int inW,
  int kH, int kW,
  int dH, int dW,
  int padH, int padW,
  int outC, int outH, int outW,
  int bCeil)
{
	dnnError_t err;
    long long* primitives = (long long*)dnnprimitives;

 	if(initOk == 0)
	{
		Init_f((long long *)input, (long long *)output, primitives,N, inC, inH, inW, kH, kW, dH, dW, padH, padW, outC, outH, outW, useMaxPooling, bCeil);
	}

    //get resource
    float* resPool[dnnResourceNumber] = {0};
 	float* input_data = GetPtr(input);
	resPool[dnnResourceSrc]       = input_data;
    resPool[dnnResourceDst]       = (float*)primitives[BUFFER_POOLING_FORWARD_OUTPUT];
	resPool[dnnResourceWorkspace] = (float*)primitives[BUFFER_POOLING_FORWARD_WORKSPACE];

    //do input conversion if necessary
	dnnPrimitive_t cv_in_f 	= (dnnPrimitive_t)primitives[CV_POOLING_FORWARD_INPUT];
	if(cv_in_f)
	{
	    float* buf_in_f	 = (float*) (primitives[BUFFER_POOLING_FORWARD_INPUT]);
		CHECK_ERR( dnnConversionExecute_F32(cv_in_f, input_data, buf_in_f), err );
	    resPool[dnnResourceSrc] = buf_in_f;
	}

	CHECK_ERR( dnnExecute_F32((dnnPrimitive_t)primitives[POOLING_FORWARD], (void**)resPool), err );

    ((long long*)output)[MKLPtr]    = primitives[BUFFER_POOLING_FORWARD_OUTPUT];
    ((long long*)output)[MKLLayout] = primitives[POOL_L_F_O];

ERR_RETURN:
    return;
}

static void Init_b(long long * gradIn, long long * gradOut, long long * primitives)
{
    dnnError_t err;

    //gradOut, layout is user or mkl
    dnnLayout_t lt_out_b     = (dnnLayout_t)primitives[POOL_L_B_O];
    dnnLayout_t lt_out       = (dnnLayout_t)gradOut[MKLLayout];
    if (lt_out==NULL) lt_out  = (dnnLayout_t)primitives[POOL_L_O];
    //create conversion and buff if necessary
    dnnPrimitive_t cv_out_b = NULL;	float * buf_out_b = NULL;
    CHECK_ERR( try_convert(&cv_out_b, &buf_out_b, lt_out, lt_out_b) , err );
	//save
	primitives[CV_POOLING_BACKWARD_OUTPUT]      = (long long)cv_out_b;
	primitives[BUFFER_POOLING_BACKWARD_OUTPUT]  = (long long)buf_out_b;

	//gradIn, layout
    gradIn[CPULayout]   = primitives[POOL_L_I];
    dnnLayout_t lt_in_b = (dnnLayout_t)primitives[POOL_L_B_I];
    float* buf_in_b = NULL;
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buf_in_b), lt_in_b), err );
    primitives[BUFFER_POOLING_BACKWARD_INPUT] = (long long)buf_in_b;

ERR_RETURN:
    return;
}

void MaxPooling_bprop(
  unsigned long long gradOutput,  //input, N*outC*outH*outW
  unsigned long long gradInput,   //output result
  unsigned long long dnnprimitives,
  int initOK)
{
	dnnError_t err;
	long long* primitives = (long long*)dnnprimitives;
	if (initOK == 0)
	{
		Init_b((long long *)gradInput, (long long *)gradOutput, primitives);
	}

    //get resource
    float* resPool[dnnResourceNumber] = {0};
	float* OutPtr= GetPtr(gradOutput);

	resPool[dnnResourceDiffSrc]   = (float*)primitives[BUFFER_POOLING_BACKWARD_INPUT];
	resPool[dnnResourceDiffDst]   = OutPtr;
    resPool[dnnResourceWorkspace] = (float*)primitives[BUFFER_POOLING_FORWARD_WORKSPACE];

    //make conversion for gradeOut if necessary
    dnnPrimitive_t cv_out_b = (dnnPrimitive_t)(primitives[CV_POOLING_BACKWARD_OUTPUT]);
	if (cv_out_b)
	{
		float* buf_out_b = (float*)primitives[BUFFER_POOLING_BACKWARD_OUTPUT];
		CHECK_ERR( dnnConversionExecute_F32(cv_out_b, OutPtr, buf_out_b), err );
	    resPool[dnnResourceDiffDst] = buf_out_b;
	}

    long long grad_in_len = (long long)dnnLayoutGetMemorySize_F32((dnnLayout_t)primitives[POOL_L_B_I]) ;
    float * tempPtr = (float*)primitives[BUFFER_POOLING_BACKWARD_INPUT];
    #pragma omp parallel for
    for (long long i = 0; i < grad_in_len/4; ++i)
    {
        tempPtr[i] = 0;
    }

	CHECK_ERR( dnnExecute_F32((dnnPrimitive_t)primitives[POOLING_BACKWARD], (void**)resPool), err );

	((long long *)gradInput)[MKLLayout] = primitives[POOL_L_B_I];
    ((long long *)gradInput)[MKLPtr]    = primitives[BUFFER_POOLING_BACKWARD_INPUT];

ERR_RETURN:
    return;
}