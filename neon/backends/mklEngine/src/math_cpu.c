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
#include <omp.h>
#include <math.h>
#include "MKLDNN.h"

#define CMATH_(NAME)    cmath_##NAME

/* blas copy */
void CMATH_(copy)(const float* in_tensor, size_t sz, float* out_tensor) {
    cblas_scopy(sz, in_tensor, 1, out_tensor, 1);
}

/* tensor negative */
void CMATH_(neg)(const float* in_tensor, size_t sz, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = -in_tensor[i];
}

/* tensor sqrt */
void CMATH_(sqrt)(const float* in_tensor, size_t sz, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = sqrt(in_tensor[i]);
}

/* tensor square */
void CMATH_(square)(const float* in_tensor, size_t sz, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = in_tensor[i] * in_tensor[i];
}

/* tensor exp */
void CMATH_(exp)(const float* in_tensor, size_t sz, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = exp(in_tensor[i]);
}

/* tensor log */
void CMATH_(log)(const float* in_tensor, size_t sz, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = log(in_tensor[i]);
}

/* tensor safelog */
void CMATH_(safelog)(const float* in_tensor, size_t sz, float* out_tensor) {
    size_t i;
    float temp;
    float exp_minus50 = exp(-50.);

    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++) {
        temp = (in_tensor[i] > exp_minus50) ? in_tensor[i] : exp_minus50;
        out_tensor[i] = log(temp);
    }
}

/* mathematical tensor add */
void CMATH_(add)(const float* in_tensor_l, const float* in_tensor_r, size_t sz, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = in_tensor_l[i] + in_tensor_r[i];
}

/* mathematical tensor sub */
void CMATH_(sub)(const float* in_tensor_l, const float* in_tensor_r, size_t sz, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = in_tensor_l[i] - in_tensor_r[i];
}

/* mathematical tensor mul*/
void CMATH_(mul)(const float* in_tensor, size_t sz, double s, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = in_tensor[i] * s;
}

/* mathematical tensor div */
void CMATH_(div)(const float* in_tensor, size_t sz, double s, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = in_tensor[i] / s;
}

/* numpy broadcast tensor add */
void CMATH_(addmv)(const float* in_tensor2d, const float* in_tensor1d, size_t row, size_t col, float* out_tensor2d) {
    size_t i, j, index;
    #pragma omp parallel for collapse(2) private(i,j)
    for (j = 0; j < row; j++) {
        for (i = 0; i < col; i++) {
            index = j * col + i;
            out_tensor2d[index] = in_tensor2d[index] + in_tensor1d[j];
        }
    }
}

/* numpy broadcast tensor sub */
void CMATH_(submv)(const float* in_tensor2d, const float* in_tensor1d, size_t row, size_t col, float* out_tensor2d) {
    size_t i, j, index;
    #pragma omp parallel for collapse(2) private(i,j)
    for (j = 0; j < row; j++) {
        for (i = 0; i < col; i++) {
            index = j * col + i;
            out_tensor2d[index] = in_tensor2d[index] - in_tensor1d[j];
        }
    }
}

/* numpy broadcast tensor mul */
void CMATH_(mulmv)(const float* in_tensor2d, const float* in_tensor1d, size_t row, size_t col, float* out_tensor2d) {
    size_t i, j, index;
    #pragma omp parallel for collapse(2) private(i,j)
    for (j = 0; j < row; j++) {
        for (i = 0; i < col; i++) {
            index = j * col + i;
            out_tensor2d[index] = in_tensor2d[index] * in_tensor1d[j];
        }
    }
}

/* numpy broadcast tensor div */
void CMATH_(divmv)(const float* in_tensor2d, const float* in_tensor1d, size_t row, size_t col, float* out_tensor2d) {
    size_t i, j, index;
    #pragma omp parallel for collapse(2) private(i,j)
    for (j = 0; j < row; j++) {
        for (i = 0; i < col; i++) {
            index = j * col + i;
            out_tensor2d[index] = in_tensor2d[index] / in_tensor1d[j];
        }
    }
}

/* numpy dot multiply */
void CMATH_(mulmm)(const float* in_tensor_l, const float* in_tensor_r, size_t sz, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = in_tensor_l[i] * in_tensor_r[i];
}

/* numpy dot divide */
void CMATH_(divmm)(const float* in_tensor_l, const float* in_tensor_r, size_t sz, float* out_tensor) {
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < sz; i++)
        out_tensor[i] = in_tensor_l[i] / in_tensor_r[i];
}

/* blas gemm */
void CMATH_(gemm)(const char transa, const char transb, const size_t m, const size_t n, const size_t k, 
          const float alpha, const float* a, const size_t lda, const float* b, const size_t ldb,
          const float beta, float* c, const size_t ldc) {
    CBLAS_LAYOUT layout;
    CBLAS_TRANSPOSE transA, transB;

    layout = CblasRowMajor;
    transA = transa == 'n' ? CblasNoTrans : CblasTrans;
    transB = transb == 'n' ? CblasNoTrans : CblasTrans;

    cblas_sgemm(layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/* tensor sum */
void CMATH_(sum)(const float* in_tensor, int axis, size_t row, size_t col, float* out_tensor) {
    size_t i, j, index;
    float sum;

    if (axis == 1) {
        #pragma omp parallel for private(j)
        for (j = 0; j < row; j++) {
            sum = 0.0f;
            for (i = 0; i < col; i++) {
                index = j * col + i;
                sum += in_tensor[index];
            }
            out_tensor[j] = sum;
        }
    }else{
        #pragma omp parallel for private(i)
        for (i = 0; i < col; i++) {
            sum = 0.0f;
            for (j = 0; j < row; j++) {
                index = j * col + i;
                sum += in_tensor[index];
            }
            out_tensor[i] = sum;
        }
    }
}

/* blas axpby: y = a*x + b*y */
void CMATH_(axpby)(size_t sz, const float a, const float* x, const float b, float* y) {
    cblas_sscal(sz, b, y, 1);
    cblas_saxpy(sz, a, x, 1, y, 1);
}











