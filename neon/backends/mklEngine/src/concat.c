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
#include "omp.h"

/*****************************************************************
Concat will concatenate output of different branch into one
always in the Channel dimension

forward:
input->branch0,N*C0*H*W->output,N*(C0+C1)*H*W
     ->branch1,N*C1*H*W

backward:

gradOut=>split=>gradOutBranch[1]-->branch1-->gradIn[1]=>sum=>gradIn
                gradOutBranch[0]-->branch0-->gradIn[0]

Concat_f do the concatenation
Concat_b split the gradient out into branch outputs
SumTensor then sum the branch gradient input into one
******************************************************************/

//param
//inputs, point to an array of pointer
//inputs[0],[1],[2],[3] is a tensor's cpu layout, cpu memeory, mkl layout, mkl memory
//      [4],[5],[6],[7] is the next tensors
//and there is moduleNum tensors in inputs
static void Init_f(
  long long* inputs,
  long long* output,
  int  moduleNum,
  long long * primitives,
  long long* channels,
  int N, int C, int H, int W)
{
	dnnError_t err;
	long long inPtr = 0;

	//get input layout for each output of branch, MKL
	dnnLayout_t *layouts = malloc(moduleNum * sizeof(dnnLayout_t));
	for (int i=0; i < moduleNum; ++i)
	{
		long long* temp = inputs + i*4;
        dnnLayout_t lt_in = (dnnLayout_t)temp[MKLLayout];
        if (lt_in==NULL)
        {
            lt_in = (dnnLayout_t)temp[CPULayout];
        }
        if (lt_in==NULL)
        {
            size_t inC = (size_t)channels[i];
            size_t inSize[DIM4]    = {W, H, inC, N};
	        size_t inStrides1[DIM4] = {1, W, W*H, inC*W*H};

	        //CHWN
	        size_t inStrides[DIM4] = {N, N*W, N*W*H, 1};
	        CHECK_ERR( dnnLayoutCreate_F32(&lt_in, DIM4, inSize, inStrides) , err );
        }
        layouts[i] = lt_in;
	}

	//create NCHW layout
	size_t outputSize[DIM4]     = {W, H, C, N};
	size_t outputStrides1[DIM4] = {1, W, W*H, C*W*H};

	//CHWN
	size_t outputStrides[DIM4]  = {N, N*W, N*W*H, 1};
    dnnLayout_t lt_out = NULL;
    CHECK_ERR( dnnLayoutCreate_F32(&lt_out, DIM4, outputSize, outputStrides) , err );
	primitives[CONCAT_LAYOUT_OUTPUT] = (long long)lt_out;

	//create op using input real layout
    dnnPrimitive_t concat_f = NULL;
	CHECK_ERR(dnnConcatCreate_F32(&concat_f, NULL, moduleNum, layouts), err);
    primitives[CONCAT_FORWARD] = (long long)concat_f;

    //create forward out layout
	dnnLayout_t lt_f = NULL;
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_f, concat_f, dnnResourceDst), err );
	primitives[CONCAT_LAYOUT_FORWARD_OUTPUT] = (long long)lt_f;

	//create buffer
	float* buf = NULL;
	CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buf), lt_f), err );
	primitives[CONCAT_BUF_F_OUT] = (long long)buf;
    output[CPULayout] = (long long)lt_out;


ERR_RETURN:
    if(layouts != NULL)
        free(layouts);
}

void Concat_f(
  unsigned long long inputs,
  int moduleNum,
  unsigned long long output,
  unsigned long long dnnprimitives,
  long long* channels,
  int initOk, int N, int C, int H, int W)
{
	dnnError_t err;
    long long* primitives = (long long*)dnnprimitives;
	if (initOk == 0)
	{
		Init_f((long long*)inputs, (long long*)output, moduleNum, primitives, channels, N, C, H, W);
    }

    //get resource
    void *concat_res[dnnResourceNumber] = {0};
	for (int i = 0; i < moduleNum; ++i)
	{
		long long* ptr = (long long *)inputs + i*4;
		long long ptr2 = (long long)ptr;
		concat_res[dnnResourceMultipleSrc + i] = GetPtr(ptr2);
	}

	concat_res[dnnResourceDst] = (float*)primitives[CONCAT_BUF_F_OUT];
	CHECK_ERR( dnnExecute_F32((dnnPrimitive_t)primitives[CONCAT_FORWARD],(void*)concat_res), err );

    ((long long*)output)[MKLLayout] = (long long)primitives[CONCAT_LAYOUT_FORWARD_OUTPUT];
    ((long long*)output)[MKLPtr]    = (long long)primitives[CONCAT_BUF_F_OUT];

ERR_RETURN:
    return;

}

static void Init_b(
  long long* gradIns,
  int moduleNum,
  long long* gradOut,
  long long* primitives,
  long long* channels,
  int N, int H, int W)
{
	dnnError_t err;

    //get gradOut layout
	dnnLayout_t layout = (dnnLayout_t)gradOut[MKLLayout];
	if (layout == 0)
	{
	    layout = (dnnLayout_t)primitives[CONCAT_LAYOUT_OUTPUT];
	}

    //load channels
	size_t* split_channels = malloc(moduleNum*sizeof(size_t));
	for (int i = 0; i < moduleNum; ++i)
	{
		split_channels[i] = (size_t)(channels[i]);
	}

	//create OP
	dnnPrimitive_t concat_split = NULL;
	CHECK_ERR(dnnSplitCreate_F32(&concat_split, NULL, moduleNum, layout, split_channels), err);
	primitives[CONCAT_BACKWARD] = (long long)concat_split;

	//create memory for layout and memory pointer for each branch
	long long* pMem = malloc(moduleNum*sizeof(long long));
    primitives[CONCAT_BUF_BRANCHES] = (long long)pMem;
    long long* pLayout = malloc(moduleNum*sizeof(long long));
    primitives[CONCAT_LT_BRANCHES] = (long long)pLayout;

    //for gradout of each branch
	for(int i=0; i < moduleNum; ++i)
	{
		long long* ptr = (long long *)gradIns + i*4;

		//create cpu layout
		int C = (int)channels[i];
		size_t outputSize[DIM4] = {W, H, C, N};
	    size_t outputStrides1[DIM4] = {1, W, W*H, C*W*H};

		//CHWN
	    size_t outputStrides[DIM4] = {N, N*W, N*W*H, 1};

        dnnLayout_t lt_usr = NULL;
        CHECK_ERR( dnnLayoutCreate_F32( &lt_usr, DIM4, outputSize, outputStrides) , err );
		ptr[CPULayout] = (long long)lt_usr;

        //create memory and MKL layout
		dnnLayout_t lt_out_branch = NULL;
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_out_branch, concat_split, dnnResourceMultipleDst + i) , err );
		float* buf = NULL;
		CHECK_ERR( dnnAllocateBuffer_F32((void**)&buf, lt_out_branch) , err );
		pLayout[i] = (long long)lt_out_branch;
		pMem[i]    = (long long)buf;
	}


ERR_RETURN:
    if(split_channels != NULL)
        free(split_channels);
}

void Concat_b(
  unsigned long long gradOutBranch,
  int moduleNum,
  unsigned long long gradOut,
  unsigned long long dnnprimitives,
  unsigned long long channels,
  int initOk, int N, int H, int W)
{
	dnnError_t err;
    long long* primitives = (long long*)dnnprimitives;
	if(initOk == 0)
    {
        Init_b((long long*)gradOutBranch, moduleNum, (long long*)gradOut, primitives, (long long*)channels, N, H, W);
    }

	void *split_res[dnnResourceNumber] = {0};

    //get out branch source
	long long* pMem    = (long long*)primitives[CONCAT_BUF_BRANCHES];
    long long* pLayout = (long long*)primitives[CONCAT_LT_BRANCHES];
	for (int i = 0; i < moduleNum; ++i)
	{
		long long* ptr = (long long*)gradOutBranch + i*4;
		split_res[dnnResourceMultipleDst + i] = (float*)pMem[i];
        ptr[MKLLayout] = pLayout[i];
        ptr[MKLPtr]    = pMem[i];
	}

    //get gradOut
    split_res[dnnResourceSrc] = GetPtr(gradOut);
	CHECK_ERR(dnnExecute_F32((dnnPrimitive_t)primitives[CONCAT_BACKWARD], split_res), err);

ERR_RETURN:
    return;
}

//inputs:        tensors to be summed up
//sum:           resulting tensor
//primitives:    MKL memory, will give to sum tensor
void MklSumTensor(
  int iN,
  unsigned long long inputs,
  unsigned long long len,
  unsigned long long sum,
  unsigned long long dnnprimitives)
{
    dnnError_t err;
    long long* primitives = (long long*)dnnprimitives;
    long long* pInput     = (long long*)inputs;
    long long* pOut       = (long long*)sum;

    // use layout of first tensor as default layout
    dnnLayout_t lt_in_first = (dnnLayout_t)pInput[MKLLayout];
    if (lt_in_first == NULL)  lt_in_first = (dnnLayout_t)pInput[CPULayout];

    // allocate memory for resulting sum, if necessary
    // save it in primitive
    float* pSumBuf= (float*)primitives[0];
    if (pSumBuf == NULL)
    {
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&pSumBuf, lt_in_first) , err );
        primitives[0] = (long long)pSumBuf;
    }
    // set layout and memory, will be changed by Neon Python, thus need to
    // be set every time
    pOut[MKLPtr] = (long long)pSumBuf;
    pOut[MKLLayout] = (long long)lt_in_first;
    pOut[CPULayout] = pInput[CPULayout];

    // create dnnSum
    dnnPrimitive_t p_sum = NULL;
    float* coeffs = (float*) malloc(iN*sizeof(float));
    for (int i = 0; i < iN; i++) coeffs[i] = 1.0;
	err = dnnSumCreate_F32(&p_sum, NULL, iN, lt_in_first, coeffs);

    // For some layout dnnSumCreate fails
    if (err == E_SUCCESS)
    {
        void* sum_res[dnnResourceNumber] = {0};
        sum_res[dnnResourceDst] = pSumBuf;
        void* temp_memory[dnnResourceNumber]  = {0};

        //convert if necessary
        for (int i = 0; i < iN; ++i)
        {
            long long* temp = pInput +  i * 4;
            float* pBuf = GetPtr((unsigned long long)temp);

            //if layout diff with the first tensor, do conversion
            dnnLayout_t lt_src = (dnnLayout_t)temp[MKLLayout];
            if(lt_src == NULL) lt_src = (dnnLayout_t)temp[CPULayout];
            if (!dnnLayoutCompare_F32(lt_in_first, lt_src))
            {
                dnnPrimitive_t cv;
                float* pNewBuf = NULL;
                CHECK_ERR( dnnAllocateBuffer_F32((void**)&pNewBuf, lt_in_first) , err );
                CHECK_ERR( dnnConversionCreate_F32(&cv, lt_src, lt_in_first), err );
                CHECK_ERR( dnnConversionExecute_F32(cv, pBuf, pNewBuf), err );
                pBuf = pNewBuf;
                temp_memory[i] = pBuf;
            }
            sum_res[dnnResourceMultipleSrc + i] = pBuf;
        }
	    CHECK_ERR( dnnExecute_F32(p_sum,(void*)sum_res), err );

        for (int i=0; i<iN; ++i)
        {
            if (temp_memory[i] != NULL) dnnReleaseBuffer_F32(temp_memory[i]);
        }
        free(coeffs);
        dnnDelete_F32(p_sum);
    }

    else //use blas to sum tensors
    {
        float* pFirstBuf = GetPtr(inputs);
        len = (long long)dnnLayoutGetMemorySize_F32(lt_in_first) / 4 ;
        cblas_scopy(len, pFirstBuf, 1, pSumBuf, 1);

        float* temp_memory = NULL;
        for (int i = 1; i < iN; ++i) //do sum of the remaining tensors
        {
            long long* temp = pInput +  i * 4;
            float* pBuf = GetPtr((unsigned long long)temp);

            //if layout diff with the first tensor, do conversion
            dnnLayout_t lt_src = (dnnLayout_t)temp[MKLLayout];
            if(lt_src == NULL) lt_src = (dnnLayout_t)temp[CPULayout];
            if (!dnnLayoutCompare_F32(lt_in_first, lt_src))
            {
                if (temp_memory == NULL)
                    CHECK_ERR( dnnAllocateBuffer_F32((void**)&temp_memory, lt_in_first) , err );
                dnnPrimitive_t cv = NULL;
                CHECK_ERR( dnnConversionCreate_F32(&cv, lt_src, lt_in_first), err );
                CHECK_ERR( dnnConversionExecute_F32(cv, pBuf, temp_memory), err );
                pBuf = temp_memory;
            }
            cblas_saxpy(len, 1.0, pBuf, 1, pSumBuf, 1);
        }
        if (temp_memory != NULL)
            dnnReleaseBuffer_F32(temp_memory);
    }

ERR_RETURN:
    return;
}
