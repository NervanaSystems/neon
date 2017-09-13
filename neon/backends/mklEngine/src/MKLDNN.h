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
#ifndef _NEON_MKLDNN_H
#define _NEON_MKLDNN_H

//flags to log in C codes
#define  LOG_ENABLE 0

#include <stdio.h>
#include <stdlib.h>
#include <mkl_dnn.h>
#include <mkl_trans.h>
#include <mkl_cblas.h>
#include <sys/time.h>
#include <omp.h>
#include <string.h>


#define DIM2 (2)
#define DIM4 (4)

typedef enum {
    FORWARD_INDEX   			    = 0,
    BWD_DATA_INDEX  			    = 1,
    BWD_FILTER_INDEX  			    = 2,
    CONVERT_FORWARD_INPUT 		    = 3,
    CONVERT_FORWARD_FILTER        	= 4,
    CONVERT_FORWARD_BIAS     		= 5,
    CONVERT_FORWARD_OUTPUT   		= 6,
    CONVERT_BWDDATA_INPUT 		    = 7,
    CONVERT_BWDDATA_FILTER        	= 8,
    CONVERT_BWDDATA_OUTPUT   		= 9,
    CONVERT_BWDFILTER_INPUT 		= 10,
    CONVERT_BWDFILTER_FILTER        = 11,
    CONVERT_BWDFILTER_OUTPUT   		= 12,
    BUFFER_FORWARD_INPUT 		    = 13,
    BUFFER_FORWARD_FILTER        	= 14,
    BUFFER_FORWARD_BIAS     		= 15,
    BUFFER_FORWARD_OUTPUT   		= 16,
    BUFFER_BWDDATA_INPUT 		    = 17,
    BUFFER_BWDDATA_FILTER        	= 18,
    BUFFER_BWDDATA_OUTPUT   		= 19,
    BUFFER_BWDFILTER_INPUT 		    = 20,
    BUFFER_BWDFILTER_FILTER        	= 21,
    BUFFER_BWDFILTER_OUTPUT   		= 22,
    BUFFER_TRANS_INPUT   	     	= 38,
    BUFFER_TRANS_OUTPUT   		    = 40,
    L_F_I  = 23,
    L_F_W  = 24,
    L_F_O  = 25,
    L_F_B  = 26,
    L_BD_I = 27,
    L_BD_O = 28,
    L_BD_W = 29,
    L_BD_B = 30,
    L_BF_I = 31,
    L_BF_O = 32,
    L_BF_W = 33,
    L_BF_B = 34,
    L_I    = 35,
    L_O    = 36,
    L_W    = 37,
    L_I_CHWN = 39,
    L_O_CHWN = 41,
    BDW_BIAS_INDEX   = 42,
    L_B              = 44,
    L_B_B            = 45,
    L_B_O            = 46,
    CV_BIAS_BIAS     = 47,
    BUFFER_BIAS_BIAS = 48,
    CV_BIAS_OUT      = 49,
    BUFFER_BIAS_OUT  = 50,


} mkldnnConvolutionIndex_t;

typedef enum {
    RELU_FORWARD          = 0,
    RELU_BACKWARD         = 1,
    RELU_L_I              = 2,
    RELU_L_O              = 3,
    RELU_L_F_I            = 4,
    RELU_L_F_O            = 5,
    RELU_L_B_I            = 6,
    RELU_L_B_O            = 7,
    BUF_RELU_BACKWARD_OUT = 8,
    CV_RELU_BACKWARD_OUT  = 9,
    BUF_RELU_FORWARD_IN   = 10,
} mkldnnReLUIndex_t;

typedef enum {
    POOLING_FORWARD                     = 0,  
    POOLING_BACKWARD                    = 1,
    CV_POOLING_FORWARD_INPUT            = 2,
    CV_POOLING_FORWARD_OUTPUT           = 3,
    CV_POOLING_BACKWARD_INPUT           = 4,
    CV_POOLING_BACKWARD_OUTPUT          = 5,
    BUFFER_POOLING_FORWARD_INPUT        = 6,
    BUFFER_POOLING_FORWARD_OUTPUT       = 7,
    BUFFER_POOLING_FORWARD_WORKSPACE    = 8,
    BUFFER_POOLING_BACKWARD_INPUT       = 9,
    BUFFER_POOLING_BACKWARD_OUTPUT      = 10,
    BUFFER_POOLING_BACKWARD_WORKSPACE   = 11,
    POOL_L_I   = 12,
    POOL_L_O   = 13,
    POOL_L_F_I = 14,
    POOL_L_F_O = 15,
    POOL_L_B_I = 14,
    POOL_L_B_O = 15,
} mkldnnPoolingIndex_t; 
    

typedef enum {
    BN_LT_IN_FORP_BACKP_PLAIN       = 0,
    BN_LT_OUT_FORP_BACKP_PLAIN      = 1,
    BN_LT_FORP_INPUT                = 2,
    BN_LT_FORP_OUTPUT               = 3,
    BN_LT_BACKP_INPUT               = 4,
    BN_LT_BACKP_OUTPUT              = 5,
    BN_FORP                         = 6,
    BN_BACKP                        = 7,
    BN_BUFFER_FORP_INPUT            = 8,
    BN_BUFFER_FORP_OUTPUT           = 9, 
    BN_BUFFER_MEAN                  = 10,
    BN_BUFFER_VARIANCE              = 11,
    BN_BUFFER_FORP_SCALESHIFT       = 12,
    BN_BUFFER_BACKP_OUTPUT          = 13,
    BN_BUFFER_BACKP_INPUT           = 14,
    BN_BUFFER_BACKP_SCALESHIFT      = 15,
    BN_CV_BACKP_OUTPUT              = 16,
    BN_FORP_INF                     = 17,
} mkldnnBNIndex_t;


typedef enum {
    CONCAT_LAYOUT_INPUT			    = 0,
    CONCAT_LAYOUT_OUTPUT		    = 1,
    CONCAT_LAYOUT_FORWARD_OUTPUT	= 2,
    CONCAT_LAYOUT_BACKWARD_INPUT	= 3,
    CONCAT_FORWARD			        = 4,
    CONCAT_BACKWARD			        = 5,
    CONCAT_BUF_BACKWARD_INPUT       = 6,
    CONCAT_BUF_BRANCHES             = 7,
    CONCAT_LT_BRANCHES              = 8,
    CONCAT_CV_B_OUT                 = 9,
    CONCAT_BUF_B_OUT                = 10,
    CONCAT_BUF_F_OUT                = 11,
} mkldnnConcatIndex_t;


typedef enum {
    CPULayout = 0,
    CPUPtr    = 1,
    MKLLayout = 2,
    MKLPtr    = 3
} mkldnnTensor_t;


#define CHECK_ERR(f, err) do { \
    (err) = (f); \
    if ((err) != E_SUCCESS) { \
        fprintf(stderr,"[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
        goto ERR_RETURN; \
    } \
} while(0)

//assume has MKLptr, if no, return CPUptr
static float* GetPtr(long long tensor)
{
    long long* ptr = (long long *)tensor;
    long long ptr1 = ptr[MKLPtr];
	if (ptr1 == 0)
	{
	    ptr1 = ptr[CPUPtr];
	}
	return (float*)ptr1;
}

//cv is created to convert some input with layout lt_src
//  to memory with layout lt_des
//param:
//cv, to be created if necessary, otherwise NULL
//memory, allocated if necessary, otherwise NULL
//lt_src, lt_des, two layout to compare, if lt_des diff with lt_src
static int try_convert(
  dnnPrimitive_t *cv,
  float **memory,
  dnnLayout_t lt_src,
  dnnLayout_t lt_des)
{
	dnnError_t err = E_SUCCESS;
	*memory = NULL;
	if((lt_src != NULL) && (lt_des != NULL))
	{
		if (!dnnLayoutCompare_F32(lt_src, lt_des))
		{
			err = dnnConversionCreate_F32(cv, lt_src, lt_des);
			if(err) return err;
			err = dnnAllocateBuffer_F32((void**)memory, lt_des);
			if(err) return err;
		}
		return err;
	}
	else
	{
	    fprintf(stderr, "wrong input for try_convert!\n");
	    return err;
	}
}
#endif
