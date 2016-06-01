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

enum DeviceType { CPU=0, GPU=1 };

class DeviceParams {
public:
    DeviceParams(int type, int id) : _type(type), _id(id) {}

public:
    int                         _type;
    int                         _id;
};

class CpuParams : public DeviceParams {
public:
    CpuParams(int type, int id, char* data[2], char* targets[2])
    : DeviceParams(type, id) {
        for (int i = 0; i < 2; i++) {
            _data[i] = data[i];
            _targets[i] = targets[i];
        }
    }

public:
    char*                       _data[2];
    char*                       _targets[2];
};

class Device {
public:
    Device(int type) : _type(type) {}
    virtual ~Device() {};
    virtual int init() = 0;
    virtual int copyData(int idx, CharBuffer* buf) = 0;
    virtual int copyLabels(int idx, CharBuffer* buf) = 0;
    virtual int copyDataBack(int idx, CharBuffer* buf) = 0;
    virtual int copyLabelsBack(int idx, CharBuffer* buf) = 0;

    static Device* create(DeviceParams* params);

public:
    int                         _type;
};

#if HAS_GPU
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
    throw std::runtime_error("CUDA error\n");
}

class GpuParams : public DeviceParams {
public:
    CUdeviceptr                 _data[2];
    CUdeviceptr                 _targets[2];
};

class Gpu : public Device {
public:
    Gpu(int id, int dataSize, int targetSize)
    : Device(GPU), _alloc(true), _id(id) {
        init();
        for (int i = 0; i < 2; i++) {
            checkDriverErrors(cuMemAlloc(&_data[i], dataSize));
            checkDriverErrors(cuMemAlloc(&_targets[i], targetSize));
        }
    }

    Gpu(GpuParams* params)
    : Device(GPU), _alloc(false), _id(params->_id) {
        for (int i = 0; i < 2; i++) {
            _data[i] = params->_data[i];
            _targets[i] = params->_targets[i];
        }
    }

    virtual ~Gpu() {
        if (_alloc == true) {
            for (int i = 0; i < 2; i++) {
                cuMemFree(_data[i]);
                cuMemFree(_targets[i]);
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

    int copyData(int idx, CharBuffer* buf) {
        return copy(_data[idx], buf->_data, buf->_size);
    }

    int copyLabels(int idx, CharBuffer* buf) {
        return copy(_targets[idx], buf->_data, buf->_size);
    }

    int copyDataBack(int idx, CharBuffer* buf) {
        return copyBack(buf->_data, _data[idx], buf->_size);
    }

    int copyLabelsBack(int idx, CharBuffer* buf) {
        return copyBack(buf->_data, _targets[idx], buf->_size);
    }

private:
    int copy(CUdeviceptr dst, char* src, int size) {
        try {
            checkDriverErrors(cuMemcpyHtoD(dst, src, size));
        } catch(...) {
            return -1;
        }
        return 0;
    }

    int copyBack(char* dst, CUdeviceptr src, int size) {
        try {
            checkDriverErrors(cuMemcpyDtoH(dst, src, size));
        } catch(...) {
            return -1;
        }
        return 0;
    }

private:
    CUdeviceptr                 _data[2];
    CUdeviceptr                 _targets[2];
    bool                        _alloc;
    int                         _id;
};
#endif

class Cpu : public Device {
public:
    Cpu(int id, int dataSize, int targetSize)
    : Device(CPU), _alloc(true) {
        init();
        for (int i = 0; i < 2; i++) {
            _data[i] = new char[dataSize];
            _targets[i] = new char[targetSize];
        }
    }

    Cpu(CpuParams* params)
    : Device(CPU), _alloc(false) {
        for (int i = 0; i < 2; i++) {
            _data[i] = params->_data[i];
            _targets[i] = params->_targets[i];
        }
    }

    virtual ~Cpu() {
        if (_alloc == true) {
            for (int i = 0; i < 2; i++) {
                delete[] _data[i];
                delete[] _targets[i];
            }
        }
    }

    int init() {
        return 0;
    }

    int copyData(int idx, CharBuffer* buf) {
        memcpy(_data[idx], buf->_data, buf->_size);
        return 0;
    }

    int copyLabels(int idx, CharBuffer* buf) {
        memcpy(_targets[idx], buf->_data, buf->_size);
        return 0;
    }

    int copyDataBack(int idx, CharBuffer* buf) {
        memcpy(buf->_data, _data[idx], buf->_size);
        return 0;
    }

    int copyLabelsBack(int idx, CharBuffer* buf) {
        memcpy(buf->_data, _targets[idx], buf->_size);
        return 0;
    }

private:
    char*                       _data[2];
    char*                       _targets[2];
    bool                        _alloc;
};

Device* Device::create(DeviceParams* params) {
#if HAS_GPU
    if (params->_type == CPU) {
        return new Cpu(reinterpret_cast<CpuParams*>(params));
    }
    return new Gpu(reinterpret_cast<GpuParams*>(params));
#else
    assert(params->_type == CPU);
    return new Cpu(reinterpret_cast<CpuParams*>(params));
#endif
}
