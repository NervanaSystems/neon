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
#include <stdlib.h>
#include "MKLDNN.h"
#include "omp.h"
#include <math.h>

void SoftmaxNCHW(unsigned long long input, int N, long long len)
{
    float* inPtr = (float*)input;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        float *pTemp = inPtr + i * len;
        float pMax = pTemp[0];
        for(long long j = 0; j < len; ++j)
        {
            if (pMax < pTemp[j])
            {
                pMax = pTemp[j];
            }
        }
        float pSum = 0.0f;
        for(long long j=0; j<len; ++j)
        {
            pTemp[j] = exp(pTemp[j] - pMax);
            pSum += pTemp[j];
        }
        for(long long j=0; j < len; ++j)
        {
            pTemp[j] = pTemp[j] / pSum;
        }
    }
}

void SoftmaxCHWN(unsigned long long input, int N, long long len)
{
    float* inPtr = (float*)input;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        float *pTemp = inPtr + i;
        float pMax = pTemp[0];
        for (long long j = 0; j < len; ++j)
        {
            if ( pMax<pTemp[j*N]) pMax = pTemp[j*N];
        }
        float pSum = 0.0f;
        for (long long j = 0; j < len; ++j)
        {
            pTemp[j*N] = exp(pTemp[j*N] - pMax);
            pSum += pTemp[j*N];
        }
        for(long long j = 0; j < len; ++j)
        {
           pTemp[j*N] = pTemp[j*N] / pSum;
        }
    }
}
