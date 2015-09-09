/*
 * Copyright 2015 Baidu USA, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


//Simple test to determine that the kernels can be loaded and run
//with some basic sanity checking.

#include <cuda.h>
#include <iostream>
#include <vector>
#include <type_traits>
#include "nervana_c_api.h"

void test_hgemm(short* d_a, short* d_b, short* d_c, bool a_t, bool b_t, int size) {
    if (!nervana_hgemm(d_a, d_b, d_c, a_t, b_t, size, size, size, size, size, size, 1.0, 0.0, NULL, false, false, 0))
        return;

    short* h_c = (short *)malloc(sizeof(short) * size * size);
    cudaMemcpy(h_c, d_c, sizeof(short) * size * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size * size; ++i) {
        if (h_c[i] != 0)
            std::cout << "Mismatch at " << i << " " << h_c[i] << " " << 0 << std::endl;
    }

    free(h_c);
}

void test_sgemm(float* d_a, float* d_b, float* d_c, bool a_t, bool b_t, int size) {
    if (!nervana_sgemm(d_a, d_b, d_c, a_t, b_t, size, size, size, size, size, size, 1.0, 0.0, NULL, false, false, 0))
        return;

    float* h_c = (float *)malloc(sizeof(float) * size * size);
    cudaMemcpy(h_c, d_c, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size * size; ++i) {
        if (h_c[i] != size)
            std::cout << "Mismatch at " << i << " " << h_c[i] << " " << size << std::endl;
    }

    free(h_c);
}

int main() {
    cudaError_t res = cudaFree(0);
    if (res != cudaSuccess) {
        std::cout << "CUDA did not initialize correctly" << std::endl;
        exit(1);
    }

    if (!nervana_loadKernels("../cubin/")) {
        std::cerr << "Couldn't load all kernels" << std::endl;
        exit(1);
    }

    //make sure we load and run all different blocking and vector variants
    std::vector<int> sizes {257, 256, 255, 129, 128, 127, 65, 64, 17, 16, 15};

    {
        float *d_a, *d_b, *d_c;
        int size = sizes[0];

        std::vector<float> h_a(size * size, 1.f);
        std::vector<float> h_b(size * size, 1.f);

        cudaMalloc(&d_a, sizeof(float) * size * size);
        cudaMalloc(&d_b, sizeof(float) * size * size);
        cudaMalloc(&d_c, sizeof(float) * size * size);

        cudaMemcpy(d_a, h_a.data(), sizeof(float) * size * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), sizeof(float) * size * size, cudaMemcpyHostToDevice);

        for (auto size : sizes) {
            test_sgemm(d_a, d_b, d_c, false, false, size);
            test_sgemm(d_a, d_b, d_c, false, true, size);
            test_sgemm(d_a, d_b, d_c, true, false, size);
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    {
        short *d_a, *d_b, *d_c;
        int size = sizes[0];

        std::vector<short> h_a(size * size, 0);
        std::vector<short> h_b(size * size, 0);

        cudaMalloc(&d_a, sizeof(short) * size * size);
        cudaMalloc(&d_b, sizeof(short) * size * size);
        cudaMalloc(&d_c, sizeof(short) * size * size);

        cudaMemcpy(d_a, h_a.data(), sizeof(short) * size * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), sizeof(short) * size * size, cudaMemcpyHostToDevice);

        for (auto size : sizes) {
            test_hgemm(d_a, d_b, d_c, false, false, size);
            test_hgemm(d_a, d_b, d_c, false, true, size);
            test_hgemm(d_a, d_b, d_c, true, false, size);
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    nervana_unloadKernels();
    return 0;
}
