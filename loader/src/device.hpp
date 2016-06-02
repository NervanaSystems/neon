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
    CpuParams(int type, int id, char* data[2], char* targets[2], int* meta[2])
    : DeviceParams(type, id) {
        for (int i = 0; i < 2; i++) {
            _data[i] = data[i];
            _targets[i] = targets[i];
            _meta[i] = meta[i];
        }
    }

public:
    char*                       _data[2];
    char*                       _targets[2];
    int*                        _meta[2];
};

class Device {
public:
    Device(int type) : _type(type) {}
    virtual ~Device() {};
    virtual int init() = 0;
    virtual void copyData(int idx, CharBuffer* buf) = 0;
    virtual void copyLabels(int idx, CharBuffer* buf) = 0;
    virtual void copyMeta(int idx, IntBuffer* buf) = 0;
    virtual void copyDataBack(int idx, CharBuffer* buf) = 0;
    virtual void copyLabelsBack(int idx, CharBuffer* buf) = 0;

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
    CUdeviceptr                 _meta[2];
};

class Gpu : public Device {
public:
    Gpu(int id, int dataSize, int targetSize, int metaSize)
    : Device(GPU), _alloc(true), _id(id) {
        init();
        for (int i = 0; i < 2; i++) {
            checkDriverErrors(cuMemAlloc(&_data[i], dataSize));
            checkDriverErrors(cuMemAlloc(&_targets[i], targetSize));
            checkDriverErrors(cuMemAlloc(&_meta[i], metaSize * sizeof(int)));
        }
    }

    Gpu(GpuParams* params)
    : Device(GPU), _alloc(false), _id(params->_id) {
        for (int i = 0; i < 2; i++) {
            _data[i] = params->_data[i];
            _targets[i] = params->_targets[i];
            _meta[i] = params->_meta[i];
        }
    }

    virtual ~Gpu() {
        if (_alloc == true) {
            for (int i = 0; i < 2; i++) {
                cuMemFree(_data[i]);
                cuMemFree(_targets[i]);
                cuMemFree(_meta[i]);
            }
        }
    }

    int init() {
        checkCudaErrors(cudaSetDevice(_id));
        checkCudaErrors(cudaFree(0));
        return 0;
    }

    void copyData(int idx, CharBuffer* buf) {
        copy(_data[idx], buf);
    }

    void copyLabels(int idx, CharBuffer* buf) {
        copy(_targets[idx], buf);
    }

    void copyMeta(int idx, IntBuffer* buf) {
        copy(_meta[idx], buf);
    }

    void copyDataBack(int idx, CharBuffer* buf) {
        copyBack(buf, _data[idx]);
    }

    void copyLabelsBack(int idx, CharBuffer* buf) {
        copyBack(buf, _targets[idx]);
    }

private:
    void copy(CUdeviceptr dst, CharBuffer* src) {
        checkDriverErrors(cuMemcpyHtoD(dst, src->_data, src->_totalLen));
    }

    void copy(CUdeviceptr dst, IntBuffer* src) {
        checkDriverErrors(cuMemcpyHtoD(dst, (char*) src->_data, src->_totalLen));
    }

    void copyBack(CharBuffer* dst, CUdeviceptr src) {
        checkDriverErrors(cuMemcpyDtoH(dst->_data, src, dst->_totalLen));
    }

private:
    CUdeviceptr                 _data[2];
    CUdeviceptr                 _targets[2];
    CUdeviceptr                 _meta[2];
    bool                        _alloc;
    int                         _id;
};
#endif

class Cpu : public Device {
public:
    Cpu(int id, int dataSize, int targetSize, int metaSize)
    : Device(CPU), _alloc(true) {
        init();
        for (int i = 0; i < 2; i++) {
            _data[i] = new char[dataSize];
            _targets[i] = new char[targetSize];
            _meta[i] = new int[metaSize];
        }
    }

    Cpu(CpuParams* params)
    : Device(CPU), _alloc(false) {
        for (int i = 0; i < 2; i++) {
            _data[i] = params->_data[i];
            _targets[i] = params->_targets[i];
            _meta[i] = params->_meta[i];
        }
    }

    virtual ~Cpu() {
        if (_alloc == true) {
            for (int i = 0; i < 2; i++) {
                delete[] _data[i];
                delete[] _targets[i];
                delete[] _meta[i];
            }
        }
    }

    int init() {
        return 0;
    }

    void copyData(int idx, CharBuffer* buf) {
        memcpy(_data[idx], buf->_data, buf->_totalLen);
    }

    void copyLabels(int idx, CharBuffer* buf) {
        memcpy(_targets[idx], buf->_data, buf->_totalLen);
    }

    void copyMeta(int idx, IntBuffer* buf) {
        if (_meta[idx] != 0) {
            memcpy(_meta[idx], buf->_data, buf->_totalLen);
        }
    }

    void copyDataBack(int idx, CharBuffer* buf) {
        memcpy(buf->_data, _data[idx], buf->_totalLen);
    }

    void copyLabelsBack(int idx, CharBuffer* buf) {
        memcpy(buf->_data, _targets[idx], buf->_totalLen);
    }

private:
    char*                       _data[2];
    char*                       _targets[2];
    int*                        _meta[2];
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
