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

#include <assert.h>
#if HAS_GPU
#include <cuda.h>
#endif

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "streams.hpp"

typedef uint8_t uchar;
using std::vector;
using std::pair;
using std::make_pair;
using std::mutex;
using std::unique_lock;
using std::condition_variable;

template<typename T>
class Buffer {
public:
    explicit Buffer(int size, bool pinned = false)
    : _size(size), _idx(0), _alloc(true), _pinned(pinned) {
        _data = alloc();
        _cur = _data;
    }

    Buffer(T* data, int size)
    : _data(data), _size(size), _cur(_data), _idx(0), _alloc(false) {
    }

    virtual ~Buffer() {
        if (_alloc == true) {
            dealloc(_data);
        }
    }

    void reset() {
        _cur = _data;
        _idx = 0;
        _items.clear();
        _lens.clear();
    }

    void dump() {
        uchar* data = reinterpret_cast<uchar*>(_data);
        int len = _size * sizeof(T);
        assert(len % 16 == 0);
        int index = 0;
        while (index < len) {
            printf("%08x", index);
            for (int i = 0; i < 8; i++) {
                printf(" %02x", data[i]);
            }
            printf("  ");
            for (int i = 8; i < 16; i++) {
                printf(" %02x", data[i]);
            }
            printf(" ");
            for (int i = 0; i < 16; i++) {
                printf("%c", (data[i] < 32)? '.' : data[i]);
            }
            printf("\n");
            data += 16;
            index += 16;
        }
    }

    void pushItem(int len) {
        _items.push_back(_idx);
        _lens.push_back(len);
        _cur += len;
        _idx += len;
    }

    T* getItem(int index, int& len) {
        if (index >= (int) _items.size()) {
            return 0;
        }
        len = _lens[index];
        return _data + _items[index];
    }

    int getItemCount() {
        return _items.size();
    }

    T* getCurrent() {
        return _cur;
    }

    uint getSize() {
        return _size;
    }

    uint getLevel() {
        return _idx;
    }

    void read(IfStream& ifs, int size) {
        resizeIfNeeded(size);
        ifs.read(_cur, size);
        pushItem(size);
    }

    void read(T* src, int size) {
        resizeIfNeeded(size);
        memcpy((void *) _cur, (void *) src, size * sizeof(T));
        pushItem(size);
    }

private:
    void resizeIfNeeded(int inc) {
        if (getLevel() + inc > getSize()) {
            resize(inc);
        }
    }

    void resize(int inc) {
        assert(_alloc == true);
        _size = getLevel() + inc;
        // Allocate a bit more to minimize reallocations.
        _size += _size / 8;
        T* data = alloc();
        memcpy(data, _data, getLevel() * sizeof(T));
        dealloc(_data);
        _data = data;
        _cur = _data + _idx;
    }

    T* alloc() {
        T*      data;
        assert(_alloc == true);
        if (_pinned == true) {
#if HAS_GPU
            CUresult status = cuMemAllocHost((void**)&data, _size * sizeof(T));
            if (status != CUDA_SUCCESS) {
                throw std::bad_alloc();
            }
#else
            data = new T[_size];
#endif
        } else {
            data = new T[_size];
        }
        return data;
    }

    void dealloc(T* data) {
        if (_pinned == true) {
#if HAS_GPU
            cuMemFreeHost(data);
#else
            delete[] data;
#endif
        } else {
            delete[] data;
        }
    }

public:
    T*                          _data;
    uint                        _size;

protected:
    T*                          _cur;
    int                         _idx;
    vector<int>                 _items;
    vector<int>                 _lens;
    bool                        _alloc;
    bool                        _pinned;
};

typedef Buffer<char>                                    CharBuffer;
typedef pair<CharBuffer*, CharBuffer*>                  BufferPair;

class BufferPool {
public:
    BufferPool(int dataSize, int targetSize, bool pinned = false, int count = 2)
    : _count(count), _used(0), _readPos(0), _writePos(0) {
        for (int i = 0; i < count; i++) {
            CharBuffer* dataBuffer = new CharBuffer(dataSize, pinned);
            CharBuffer* targetBuffer = new CharBuffer(targetSize, pinned);
            _bufs.push_back(make_pair(dataBuffer, targetBuffer));
        }
    }

    virtual ~BufferPool() {
        for (auto buf : _bufs) {
            delete buf.first;
            delete buf.second;
        }
    }

    BufferPair& getForWrite() {
        _bufs[_writePos].first->reset();
        _bufs[_writePos].second->reset();
        return _bufs[_writePos];
    }

    BufferPair& getForRead() {
        return _bufs[_readPos];
    }

    void advanceReadPos() {
        _used--;
        advance(_readPos);
    }

    void advanceWritePos() {
        _used++;
        advance(_writePos);
    }

    bool empty() {
        assert(_used >= 0);
        return (_used == 0);
    }

    bool full() {
        assert(_used <= _count);
        return (_used == _count);
    }

    mutex& getMutex() {
        return _mutex;
    }

    void waitForNonEmpty(unique_lock<mutex>& lock) {
        _nonEmpty.wait(lock);
    }

    void waitForNonFull(unique_lock<mutex>& lock) {
        _nonFull.wait(lock);
    }

    void signalNonEmpty() {
        _nonEmpty.notify_all();
    }

    void signalNonFull() {
        _nonFull.notify_all();
    }

protected:
    void advance(int& index) {
        if (++index == _count) {
            index = 0;
        }
    }

protected:
    int                         _count;
    int                         _used;
    vector<BufferPair>          _bufs;
    int                         _readPos;
    int                         _writePos;
    mutex                       _mutex;
    condition_variable          _nonFull;
    condition_variable          _nonEmpty;
};
