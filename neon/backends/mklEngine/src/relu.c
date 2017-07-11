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

static void Init_f(
  long long * input,
  long long * primitives,
  int N, int inC, int inH, int inW)
{
	const float threshold  = 0;
	dnnError_t err;

	//init dimensions
	size_t inputSize[DIM4] = 	{inW, inH, inC, N};
	size_t inputStridesNCHW[DIM4] = {1, inW, inH*inW, inC*inH*inW };

	//CHWN
	size_t inputStridesCHWN[DIM4] = {N, N*inW, N*inH*inW, 1};

    //create NCHW layout
	dnnLayout_t lt_in_NCHW = NULL, lt_in_CHWN = NULL;
	CHECK_ERR( dnnLayoutCreate_F32(&lt_in_NCHW, DIM4, inputSize, inputStridesNCHW), err );
	CHECK_ERR( dnnLayoutCreate_F32(&lt_in_CHWN, DIM4, inputSize, inputStridesCHWN), err );

    //get input real layout which is used to create execute
    dnnLayout_t lt_in_f = (dnnLayout_t)input[MKLLayout];
	if (lt_in_f==NULL)
	{
	    lt_in_f = lt_in_NCHW;
	    primitives[RELU_L_I] = (long long)lt_in_NCHW;
	}
    else
    {
        primitives[RELU_L_I] = (long long)lt_in_CHWN;
    }
    primitives[RELU_L_F_I] = (long long)lt_in_f;

    //create execute and save
	dnnPrimitive_t relu_f = NULL, relu_b = NULL;
	dnnPrimitiveAttributes_t attributes = NULL;
	CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
	CHECK_ERR( dnnReLUCreateForward_F32 (&relu_f, attributes, lt_in_f, threshold), err );
	CHECK_ERR( dnnReLUCreateBackward_F32(&relu_b, attributes, lt_in_f, lt_in_f, threshold), err );
	primitives[RELU_FORWARD]  = (long long)relu_f;
	primitives[RELU_BACKWARD] = (long long)relu_b;

    //create backward layout from created execute
	dnnLayout_t lt_out_b = NULL, lt_in_b = NULL;
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_out_b, relu_b, dnnResourceDiffDst), err );
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_in_b,  relu_b, dnnResourceDiffSrc), err );
	primitives[RELU_L_B_O] = (long long)lt_out_b;
    primitives[RELU_L_B_I] = (long long)lt_in_b;

    primitives[BUF_RELU_FORWARD_IN] = (long long)GetPtr((unsigned long long)input);

ERR_RETURN:
    return;
}

void Relu_f(
  unsigned long long input,
  unsigned long long dnnprimitives,
  int initOk,
  int N, int inC, int inH, int inW)
{
	dnnError_t err;
	long long * primitives = (long long * )dnnprimitives;
	if (initOk == 0)
	{
	    Init_f((long long*)input,primitives,N,inC,inH,inW);
	}

	float *resRelu[dnnResourceNumber];
    resRelu[dnnResourceSrc] = GetPtr(input);
    resRelu[dnnResourceDst] = GetPtr(input);

	CHECK_ERR( dnnExecute_F32((dnnPrimitive_t)(primitives[RELU_FORWARD]), (void**)resRelu), err );

ERR_RETURN:
    return;
}

static void Init_b(long long * gradOut, long long * primitives)
{
	dnnError_t err;

    //get layout
 	dnnLayout_t lt_out   = (dnnLayout_t)primitives[RELU_L_I];
	dnnLayout_t lt_out_b = (dnnLayout_t)primitives[RELU_L_B_O];

    dnnLayout_t lt_gradout = (dnnLayout_t)gradOut[MKLLayout];
    if (lt_gradout == NULL) lt_gradout = lt_out;

	if (!dnnLayoutCompare_F32(lt_gradout, lt_out_b))
    {
        float* buf_out_b = NULL; dnnPrimitive_t cv = NULL;
        CHECK_ERR( dnnConversionCreate_F32(&cv, lt_gradout, lt_out_b), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buf_out_b), lt_out_b), err );
        primitives[CV_RELU_BACKWARD_OUT]  = (long long)cv;
        primitives[BUF_RELU_BACKWARD_OUT] = (long long)buf_out_b;
    }

    if (gradOut[CPULayout]==0)
    {
        gradOut[CPULayout] = (long long)lt_out;
    }

ERR_RETURN:
    return;
}

void Relu_b(
  unsigned long long input,
  unsigned long long gradOutput,
  unsigned long long dnnprimitives,
  int initOk)
{

	dnnError_t err;
	long long * primitives = (long long *)dnnprimitives;
	if (initOk == 0)
	{
		Init_b((long long*)gradOutput,primitives);
	}

    float *resRelu[dnnResourceNumber] = {0};
	resRelu[dnnResourceSrc]     = (float*)primitives[BUF_RELU_FORWARD_IN];
	resRelu[dnnResourceDiffDst] = GetPtr(gradOutput);
    resRelu[dnnResourceDiffSrc] = GetPtr(gradOutput);
    dnnPrimitive_t cv_out_b = (dnnPrimitive_t)primitives[CV_RELU_BACKWARD_OUT] ;

    if (cv_out_b)
    {
        float* buf = (float*)primitives[BUF_RELU_BACKWARD_OUT];
        CHECK_ERR( dnnConversionExecute_F32(cv_out_b, resRelu[dnnResourceDiffDst], buf), err );
	    resRelu[dnnResourceDiffDst] = (float*)buf;
	    resRelu[dnnResourceDiffSrc] = (float*)buf;
        //change input's layout if the conversion is done
        ((long long*)gradOutput)[MKLLayout] = primitives[RELU_L_B_I];
        ((long long*)gradOutput)[MKLPtr] = (long long)resRelu[dnnResourceDiffSrc];
    }

	CHECK_ERR( dnnExecute_F32((dnnPrimitive_t)primitives[RELU_BACKWARD], (void**)resRelu), err );

ERR_RETURN:
    return;
}
