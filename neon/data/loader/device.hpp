/*
 Copyright 2015 Nervana Systems Inc.
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

#include <exception>

typedef uint8_t uchar;
enum DeviceType { CPU=0, GPU=1 };

class Device {
public:
    Device(int type) : _type(type) {}
    virtual ~Device() {};
    virtual int init() = 0;
    virtual int copyData(int idx, uchar* data, int size) = 0;
    virtual int copyLabels(int idx, uchar* data, int size) = 0;
    virtual int copyDataBack(int idx, uchar* data, int size) = 0;
    virtual int copyLabelsBack(int idx, uchar* data, int size) = 0;

public:
    int                         _type;
};

class DeviceParams {
public:
    int                         _type;
    int                         _id;
};

class CpuParams : public DeviceParams {
public:
    uchar*                      _data[2];
    uchar*                      _labels[2];
};

#if HASGPU
#include <cuda.h>
#include <cuda_runtime.h>
using std::runtime_error;

#define checkCudaErrors(val) check( (val), cudaSuccess, #val, __FILE__, __LINE__)
#define checkDriverErrors(val) check( (val), CUDA_SUCCESS, #val, __FILE__, __LINE__)

template<typename T>
void check(T err, T sval, const char* const func,
           const char* const file, const int line) {
    if (err == sval) {
        return;
    }
    printf("CUDA error %d at: %s:%d\n", err, file, line);
    throw runtime_error("CUDA error\n");
}

class GpuParams : public DeviceParams {
public:
    CUdeviceptr                 _data[2];
    CUdeviceptr                 _labels[2];
};

class Gpu : public Device {
public:
    Gpu(int id, int dataSize, int labelSize)
    : Device(GPU), _alloc(true), _id(id) {
        init();
        for (int i = 0; i < 2; i++) {
            checkDriverErrors(cuMemAlloc(&_data[i], dataSize));
            checkDriverErrors(cuMemAlloc(&_labels[i], labelSize));
        }
    }

    Gpu(DeviceParams* params)
    : Device(GPU), _alloc(false), _id(params->_id) {
        GpuParams* gpuParams = reinterpret_cast<GpuParams*>(params);
        for (int i = 0; i < 2; i++) {
            _data[i] = gpuParams->_data[i];
            _labels[i] = gpuParams->_labels[i];
        }
    }

    ~Gpu() {
        if (_alloc == true) {
            for (int i = 0; i < 2; i++) {
                cuMemFree(_data[i]);
                cuMemFree(_labels[i]);
            }
        }
    }

    int init() {
        try {
            checkCudaErrors(cudaSetDevice(_id));
            checkCudaErrors(cudaFree(0));
        } catch(...) {
            return -1;
        }
        return 0;
    }

    int copyData(int idx, uchar* data, int size) {
        return copy(_data[idx], data, size);
    }

    int copyLabels(int idx, uchar* labels, int size) {
        return copy(_labels[idx], labels, size);
    }

    int copyDataBack(int idx, uchar* data, int size) {
        return copyBack(data, _data[idx], size);
    }

    int copyLabelsBack(int idx, uchar* labels, int size) {
        return copyBack(labels, _labels[idx], size);
    }

private:
    int copy(CUdeviceptr dst, uchar* src, int size) {
        try {
            checkDriverErrors(cuMemcpyHtoD(dst, src, size));
        } catch(...) {
            return -1;
        }
        return 0;
    }

    int copyBack(uchar* dst, CUdeviceptr src, int size) {
        try {
            checkDriverErrors(cuMemcpyDtoH(dst, src, size));
        } catch(...) {
            return -1;
        }
        return 0;
    }

private:
    CUdeviceptr                 _data[2];
    CUdeviceptr                 _labels[2];
    bool                        _alloc;
    int                         _id;
};
#endif

class Cpu : public Device {
public:
    Cpu(int id, int dataSize, int labelSize)
    : Device(CPU), _alloc(true) {
        init();
        for (int i = 0; i < 2; i++) {
            _data[i] = new uchar[dataSize];
            _labels[i] = new uchar[labelSize];
        }
    }

    Cpu(DeviceParams* params)
    : Device(CPU), _alloc(false) {
        CpuParams* cpuParams = reinterpret_cast<CpuParams*>(params);
        for (int i = 0; i < 2; i++) {
            _data[i] = cpuParams->_data[i];
            _labels[i] = cpuParams->_labels[i];
        }
    }

    ~Cpu() {
        if (_alloc == true) {
            for (int i = 0; i < 2; i++) {
                delete[] _data[i];
                delete[] _labels[i];
            }
        }
    }

    int init() {
        return 0;
    }

    int copyData(int idx, uchar* data, int size) {
        memcpy(_data[idx], data, size);
        return 0;
    }

    int copyLabels(int idx, uchar* labels, int size) {
        memcpy(_labels[idx], labels, size);
        return 0;
    }

    int copyDataBack(int idx, uchar* data, int size) {
        memcpy(data, _data[idx], size);
        return 0;
    }

    int copyLabelsBack(int idx, uchar* labels, int size) {
        memcpy(labels, _labels[idx], size);
        return 0;
    }

private:
    uchar*                      _data[2];
    uchar*                      _labels[2];
    bool                        _alloc;
};
