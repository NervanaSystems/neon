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
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>

#include "reader.hpp"
#include "decoder.hpp"
#include "matrix.hpp"
#include "device.hpp"

using std::string;
using std::thread;
using std::mutex;
using std::condition_variable;
using std::unique_lock;
using std::lock_guard;
using std::pair;
using std::make_pair;

template<typename DataType, typename LabelType>
class BufferPool {
public:
typedef pair<Buffer<DataType>*, Buffer<LabelType>*>     BufferPair;

public:
    BufferPool(int dataSize, int labelSize, bool pinned = false, int count = 2)
    : _count(count), _used(0), _readPos(0), _writePos(0) {
        for (int i = 0; i < count; i++) {
            Buffer<DataType>* dataBuffer =
                    new Buffer<DataType>(dataSize, pinned);
            Buffer<LabelType>* labelBuffer =
                    new Buffer<LabelType>(labelSize, pinned);
            _bufs.push_back(make_pair(dataBuffer, labelBuffer));
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

class ThreadPool {
public:
    explicit ThreadPool(int count)
    : _count(count), _done(false) {
    }

    virtual ~ThreadPool() {
        for (auto t : _threads) {
            t->join();
            delete t;
        }
    }

    virtual void start() {
        for (int i = 0; i < _count; i++) {
            _threads.push_back(new thread(&ThreadPool::run, this, i));
            _stopped.push_back(false);
        }
    }

    virtual void stop() {
        _done = true;
    }

    bool stopped() {
        for (int i = 0; i < _count; i++) {
            if (_stopped[i] == false) {
                return false;
            }
        }
        return true;
    }

    void join() {
        for (auto t : _threads) {
            t->join();
        }
    }

protected:
    virtual void work(int id) = 0;

    void run(int id) {
        while (_done == false) {
            work(id);
        }
        _stopped[id] = true;
    }

protected:
    int                         _count;
    vector<thread*>             _threads;
    bool                        _done;
    vector<bool>                _stopped;
};

template <typename DataType, typename LabelType>
class DecodeThreadPool : public ThreadPool {
private:
typedef BufferPool<DataType, LabelType>                         InBufferPool;
typedef BufferPool<DataType, LabelType>                         OutBufferPool;
typedef typename BufferPool<DataType, LabelType>::BufferPair    BufferPair;

public:
    DecodeThreadPool(int count, int minibatchSize,
                     int outputItemSize, int labelCount,
                     InBufferPool& in,
                     OutBufferPool& out,
                     Device* device, Decoder* decoder)
    : ThreadPool(count),
      _itemsPerThread((minibatchSize - 1) / count + 1),
      _in(in), _out(out), _endSignaled(0),
      _manager(0), _managerStopped(false), _inputBuf(0),
      _bufferIndex(0), _minibatchSize(minibatchSize),
      _outputItemSize(outputItemSize),
      _labelCount(labelCount),
      _device(device), _decoder(decoder) {
        assert(_itemsPerThread * count >= _minibatchSize);
        assert(_itemsPerThread * (count - 1) < _minibatchSize);
        for (int i = 0; i < count; i++) {
            _startSignaled.push_back(0);
            _itemCounts.push_back(0);
            _minibatchOffsets.push_back(0);
        }
    }

    virtual ~DecodeThreadPool() {
        if (_manager != 0) {
            _manager->join();
            delete _manager;
        }
        // The other thread objects are freed in the destructor
        // of the parent class.
    }

    virtual void start() {
        for (int i = 0; i < _count; i++) {
            _threads.push_back(new thread(&DecodeThreadPool::run, this, i));
            _stopped.push_back(false);
        }
        _manager = new thread(&DecodeThreadPool::manage, this);
    }

    virtual void stop() {
        ThreadPool::stop();
        while (_managerStopped == false) {
            std::this_thread::sleep_for(std::chrono::seconds(0));
            _in.signalNonEmpty();
        }
        _started.notify_all();
    }

protected:
    virtual void run(int id) {
        _minibatchOffsets[id] =  id * _itemsPerThread;
        if (id < _count - 1) {
            _itemCounts[id] = _itemsPerThread;
        } else {
            _itemCounts[id] = _minibatchSize - id * _itemsPerThread;
        }

        while (_done == false) {
            work(id);
        }

        _stopped[id] = true;
    }

    virtual void work(int id) {
        // Thread function.
        {
            unique_lock<mutex> lock(_mutex);
            while (_startSignaled[id] == 0) {
                _started.wait(lock);
                if (_done == true) {
                    return;
                }
            }
            _startSignaled[id]--;
            assert(_startSignaled[id] == 0);
        }
        BufferPair& outBuf = _out.getForWrite();
        DataType* dataBuf = outBuf.first->_data;
        LabelType* labelBuf = outBuf.second->_data;
        dataBuf += _minibatchOffsets[id] * _outputItemSize;
        labelBuf += _minibatchOffsets[id];
        int start = _minibatchOffsets[id];
        int end = start + _itemCounts[id];
        for (int i = start; i < end; i++) {
            int itemSize = 0;
            DataType* item = _inputBuf->first->getItem(i, itemSize);
            if (item == 0) {
                break;
            }
            _decoder->decode(item, itemSize, dataBuf);
            dataBuf += _outputItemSize;
        }
        int labelChunkSize = 0;
        LabelType* labels = _inputBuf->second->getItem(0, labelChunkSize);
        assert(labelChunkSize == outBuf.second->_size * sizeof(LabelType));
        for (int i = 0; i < _labelCount; i++) {
            int offset = i * _minibatchSize;
            memcpy(labelBuf + offset, labels + start + offset,
                   _itemCounts[id] * sizeof(LabelType));
        }
        {
            lock_guard<mutex> lock(_mutex);
            _endSignaled++;
            assert(_endSignaled <= _count);
        }
        _ended.notify_one();
    }

    void produce() {
        // Produce a minibatch.
        {
            unique_lock<mutex> lock(_out.getMutex());
            while (_out.full() == true) {
                _out.waitForNonFull(lock);
            }
            {
                lock_guard<mutex> lock(_mutex);
                assert(_endSignaled == 0);
                for (int i=0; i < _startSignaled.size(); i++) {
                    _startSignaled[i] = 1;
                }
            }
            _started.notify_all();
            {
                unique_lock<mutex> lock(_mutex);
                while (_endSignaled < _count) {
                    _ended.wait(lock);
                }
                _endSignaled = 0;
            }
            // At this point, we have decoded data for the whole minibatch.
            BufferPair& outBuf = _out.getForWrite();
            Matrix<DataType>::transpose(outBuf.first->_data,
                                        _minibatchSize, _outputItemSize);
            // Copy to device.
            _device->copyData(_bufferIndex, outBuf.first->_data,
                              outBuf.first->_size * sizeof(DataType));
            _device->copyLabels(_bufferIndex,
                                reinterpret_cast<uchar *>(outBuf.second->_data),
                                outBuf.second->_size * sizeof(LabelType));
            _bufferIndex = (_bufferIndex == 0) ? 1 : 0;
            _out.advanceWritePos();
        }
        _out.signalNonEmpty();
    }

    void consume() {
        // Consume an input buffer.
        {
            unique_lock<mutex> lock(_in.getMutex());
            while (_in.empty() == true) {
                _in.waitForNonEmpty(lock);
                if (_done == true) {
                    assert(_in.empty() == true);
                    return;
                }
            }
            _inputBuf = &_in.getForRead();
            produce();
            _in.advanceReadPos();
        }
        _in.signalNonFull();
    }

    void manage() {
        // Thread function.
        int result = _device->init();
        if (result != 0) {
            _done = true;
        }
        while (_done == false) {
            consume();
        }
        _managerStopped = true;
    }

private:
    int                         _itemsPerThread;
    InBufferPool&               _in;
    OutBufferPool&              _out;
    mutex                       _mutex;
    condition_variable          _started;
    condition_variable          _ended;
    vector<int>                 _startSignaled;
    int                         _endSignaled;
    vector<int>                 _itemCounts;
    thread*                     _manager;
    bool                        _managerStopped;
    BufferPair*                 _inputBuf;
    int                         _bufferIndex;
    int                         _minibatchSize;
    vector<int>                 _minibatchOffsets;
    int                         _outputItemSize;
    int                         _labelCount;
    Device*                     _device;
    Decoder*                    _decoder;
};

template<typename DataType, typename LabelType>
class ReadThreadPool : public ThreadPool {
private:
typedef typename BufferPool<DataType, LabelType>::BufferPair    BufferPair;

public:
    ReadThreadPool(BufferPool<DataType, LabelType>& out, Reader* reader)
    : ThreadPool(1), _out(out), _reader(reader) {
        assert(_count == 1);
    }

protected:
    virtual void work(int id) {
        produce();
    }

    void produce() {
        // Fill input buffers.
        {
            unique_lock<mutex> lock(_out.getMutex());
            while (_out.full() == true) {
                _out.waitForNonFull(lock);
            }
            BufferPair bufPair = _out.getForWrite();
            int result = _reader->read(bufPair.first, bufPair.second);
            if (result == -1) {
                _done = true;
                throw std::runtime_error("Could not read data\n");
            }
            _out.advanceWritePos();
        }
        _out.signalNonEmpty();
    }

private:
    BufferPool<DataType,
               LabelType>&      _out;
    Reader*                     _reader;
};

template<typename DataType, typename LabelType>
class Loader {
private:
typedef BufferPool<DataType, LabelType>         InBufferPool;
typedef BufferPool<DataType, LabelType>         OutBufferPool;
typedef ReadThreadPool<DataType, LabelType>     ReadPool;
typedef DecodeThreadPool<DataType, LabelType>   DecodePool;

public:
    Loader(int minibatchSize, int itemMaxSize, int labelCount,
           Device* device, Reader* reader, Decoder* decoder)
    : _first(true),
      _minibatchSize(minibatchSize), _itemMaxSize(itemMaxSize),
      _labelCount(labelCount), _readBufs(0), _readPool(0),
      _decodeBufs(0), _decodePool(0), _device(device),
      _reader(reader), _decoder(decoder) {
    }

    virtual ~Loader() {
        delete _readBufs;
        delete _readPool;
        delete _decodeBufs;
        delete _decodePool;
        delete _device;
        delete _reader;
        delete _decoder;
    }

    int start() {
        _first = true;
        try {
            _readBufs =
                new InBufferPool(_minibatchSize * _itemMaxSize,
                                 _labelCount * _minibatchSize);
            _readPool =
                new ReadPool(*_readBufs, _reader);
            bool pinned = (_device->_type != CPU);
            _decodeBufs =
                new OutBufferPool(_minibatchSize * _itemMaxSize,
                                  _labelCount * _minibatchSize, pinned);
            int numCores = thread::hardware_concurrency();
            int itemsPerThread = (_minibatchSize - 1) /  numCores + 1;
            int threadCount =  (_minibatchSize - 1) / itemsPerThread + 1;
            threadCount = std::min(threadCount, _minibatchSize);
            _decodePool =
                new DecodePool(threadCount, _minibatchSize, _itemMaxSize,
                               _labelCount, *_readBufs, *_decodeBufs,
                               _device, _decoder);
        } catch(std::bad_alloc&) {
            return -1;
        }
        _decodePool->start();
        _readPool->start();
        return 0;
    }

    void stop() {
        _readPool->stop();
        while (_readPool->stopped() == false) {
            std::this_thread::yield();
            drain();
        }
        while ((_decodeBufs->empty() == false) ||
               (_readBufs->empty() == false)) {
            drain();
        }
        _decodePool->stop();
        while (_decodePool->stopped() == false) {
            std::this_thread::yield();
        }
        delete _readBufs;
        delete _readPool;
        delete _decodeBufs;
        delete _decodePool;
        _readBufs = 0;
        _readPool = 0;
        _decodeBufs = 0;
        _decodePool = 0;
    }

    int reset() {
        stop();
        _reader->reset();
        start();
        return 0;
    }

    void next(Buffer<DataType>* dataBuf, Buffer<LabelType>* labelBuf) {
        // Copy minibatch data into the buffers passed in.
        // Only used for testing purposes.
        {
            unique_lock<mutex> lock(_decodeBufs->getMutex());
            while (_decodeBufs->empty()) {
                _decodeBufs->waitForNonEmpty(lock);
            }
            Buffer<DataType>* data = _decodeBufs->getForRead().first;
            memcpy(dataBuf->_data, data->_data,
                   dataBuf->_size * sizeof(DataType));
            Buffer<LabelType>* labels = _decodeBufs->getForRead().second;
            memcpy(labelBuf->_data, labels->_data,
                   labelBuf->_size * sizeof(LabelType));
            _decodeBufs->advanceReadPos();
        }
        _decodeBufs->signalNonFull();
    }

    void next() {
        unique_lock<mutex> lock(_decodeBufs->getMutex());
        if (_first == true) {
            _first = false;
        } else {
            // Unlock the buffer used for the previous minibatch.
            _decodeBufs->advanceReadPos();
            _decodeBufs->signalNonFull();
        }
        while (_decodeBufs->empty()) {
            _decodeBufs->waitForNonEmpty(lock);
        }
    }

private:
    void drain() {
        {
            unique_lock<mutex> lock(_decodeBufs->getMutex());
            if (_decodeBufs->empty() == true) {
                return;
            }
            _decodeBufs->advanceReadPos();
        }
        _decodeBufs->signalNonFull();
    }


private:
    bool                        _first;
    int                         _minibatchSize;
    int                         _itemMaxSize;
    int                         _labelCount;
    InBufferPool*               _readBufs;
    ReadPool*                   _readPool;
    OutBufferPool*              _decodeBufs;
    DecodePool*                 _decodePool;
    Device*                     _device;
    Reader*                     _reader;
    Decoder*                    _decoder;
};

typedef Loader<uchar, int>    ImageLoader;

#if STANDALONE

unsigned int sum(uchar* data, unsigned int len) {
    unsigned int result = 0;
    for (unsigned int i = 0; i < len; i++) {
        result += data[i];
    }
    return result;
}

int single(Reader* reader, Decoder* decoder, int epochCount,
           int macrobatchCount, int macrobatchSize,
           int minibatchSize, int itemMaxSize) {
    typedef pair<Buffer<uchar>*, Buffer<int>*>  BufferPair;
    typedef BufferPool<uchar, int>              InBufferPool;

    InBufferPool* readBufs = new InBufferPool(macrobatchSize * itemMaxSize,
                                              macrobatchSize);
    unsigned int sm = 0;
    uchar* dataBuf = new uchar[itemMaxSize];
    int* labelBuf = new int[macrobatchSize];
    for (int epoch = 0; epoch < epochCount; epoch++) {
        reader->reset();
        for (int i = 0; i < macrobatchCount; i++) {
            BufferPair bufPair = readBufs->getForWrite();
            reader->readAll(bufPair.first, bufPair.second);
            for (int j = 0; j < macrobatchSize; j++) {
                int itemSize = 0;
                uchar* item = bufPair.first->getItem(j, itemSize);
                assert(item != 0);
                decoder->decode(item, itemSize, dataBuf);
                sm += sum(dataBuf, itemMaxSize);
            }
            int labelChunkSize = 0;
            int* labels = bufPair.second->getItem(0, labelChunkSize);
            assert((uint) labelChunkSize == macrobatchSize * sizeof(int));
            memcpy(labelBuf, labels, macrobatchSize * sizeof(int));
            sm += sum(reinterpret_cast<uchar *>(labelBuf),
                      macrobatchSize * sizeof(int));
            reader->next();
        }
    }
    reader->close();
    delete[] dataBuf;
    delete[] labelBuf;
    delete readBufs;
    return sm;
}

int multi(ImageLoader* loader, Device* device, Reader* reader, Decoder* decoder,
          int epochCount, int macrobatchCount, int macrobatchSize,
          int minibatchSize, int itemMaxSize) {
    int result = loader->start();
    if (result != 0) {
        return result;
    }
    assert(macrobatchSize % minibatchSize == 0);
    int minibatchCount = macrobatchCount * macrobatchSize / minibatchSize;
    unsigned int sm = 0;
    uchar* data = new uchar[minibatchSize*itemMaxSize];
    uchar* labels = new uchar[minibatchSize*sizeof(int)];
    for (int epoch = 0; epoch < epochCount; epoch++) {
        for (int i = 0; i < minibatchCount; i++) {
            loader->next();
            int bufIdx = i % 2;
            device->copyDataBack(bufIdx, data, minibatchSize*itemMaxSize);
            device->copyLabelsBack(bufIdx, labels, minibatchSize*sizeof(int));
            sm += sum(data, minibatchSize * itemMaxSize);
            sm += sum(labels, minibatchSize*4);
        }
    }
    delete[] data;
    delete[] labels;
    loader->stop();
    return sm;
}

int test(int minibatchSize, int itemMaxSize, Device* device) {
    int epochCount = 1;
    int macrobatchCount = 1;
    int macrobatchSize = 3072;
    string pathPrefix = "/usr/local/data/I1K/imageset_batches_dw/data_batch_";
    Reader* reader = new MacrobatchReader(pathPrefix, 0,
                                          macrobatchCount*macrobatchSize,
                                          minibatchSize);
    Decoder* decoder = new ImageDecoder(224, false);
    ImageLoader loader(minibatchSize, itemMaxSize, 1, device, reader, decoder);
    unsigned int multiSum = multi(&loader, device, reader, decoder, epochCount,
                                  macrobatchCount, macrobatchSize,
                                  minibatchSize, itemMaxSize);
    unsigned int singleSum = single(reader, decoder, epochCount,
                                    macrobatchCount, macrobatchSize,
                                    minibatchSize, itemMaxSize);
    printf("sum %d true sum %d\n", multiSum, singleSum);
    assert(multiSum == singleSum);
    return 0;
}

int main() {
    int minibatchSize = 128;
    int itemMaxSize = 3*224*224;
#if HASGPU
    Device* gpu = new Gpu(0, minibatchSize*itemMaxSize,
                          minibatchSize*sizeof(int));
    test(minibatchSize, itemMaxSize, gpu);
#endif
    Device* cpu = new Cpu(0, minibatchSize*itemMaxSize,
                          minibatchSize*sizeof(int));
    test(minibatchSize, itemMaxSize, cpu);
}

#else  // STANDALONE else

extern "C" {

extern void* start(int img_size, int inner_size, bool center, bool flip,
                   bool rgb, bool multiview, int minibatch_size,
                   char *filename, int macro_start,
                   uint num_data, uint num_labels, bool macro,
                   DeviceParams* params) {
    static_assert(sizeof(int) == 4, "int is not 4 bytes");
    try {
        int nchannels = (rgb == true) ? 3 : 1;
        int item_max_size = nchannels*inner_size*inner_size;
        // These objects will get freed in the destructor of ImageLoader.
        Device* device;
#if HASGPU
        if (params->_type == CPU) {
            device = new Cpu(params);
        } else {
            device = new Gpu(params);
        }
#else
        assert(params->_type == CPU);
        device = new Cpu(params);
#endif
        Reader*     reader;
        if (macro == true) {
            reader = new MacrobatchReader(filename, macro_start,
                                          num_data, minibatch_size);
        } else {
            reader = new ImageFileReader(filename, num_data,
                                         minibatch_size, img_size);
        }
        Decoder* decoder = new ImageDecoder(inner_size, flip);
        ImageLoader* loader = new ImageLoader(minibatch_size, item_max_size,
                                              num_labels, device,
                                              reader, decoder);
        int result = loader->start();
        if (result != 0) {
            printf("Could not start data loader. Error %d", result);
            delete loader;
            exit(-1);
        }
        return (void *) loader;
    } catch(...) {
        return 0;
    }
}

extern int next(ImageLoader* loader) {
    try {
        loader->next();
        return 0;
    } catch(...) {
        return -1;
    }
}

extern int reset(ImageLoader* loader) {
    try {
        return loader->reset();
    } catch(...) {
        return -1;
    }
}

extern int stop(ImageLoader* loader) {
    try {
        loader->stop();
        delete loader;
        return 0;
    } catch(...) {
        return -1;
    }
}
}

#endif  // STANDALONE else
