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

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>

#include "reader.hpp"
#include "decoder.hpp"
#include "matrix.hpp"
#include "device.hpp"

using std::thread;
using std::mutex;
using std::condition_variable;
using std::unique_lock;
using std::lock_guard;

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

class DecodeThreadPool : public ThreadPool {
public:
    DecodeThreadPool(int count, int minibatchSize,
                     int outputItemSize, int labelSize, int labelCount,
                     BufferPool& in, BufferPool& out,
                     Device* device, Decoder* decoder)
    : ThreadPool(count),
      _itemsPerThread((minibatchSize - 1) / count + 1),
      _in(in), _out(out), _endSignaled(0),
      _manager(0), _managerStopped(false), _inputBuf(0),
      _bufferIndex(0), _minibatchSize(minibatchSize),
      _outputItemSize(outputItemSize),
      _labelChunkSize(labelSize * labelCount),
      _device(device), _decoder(decoder) {
        assert(_itemsPerThread * count >= _minibatchSize);
        assert(_itemsPerThread * (count - 1) < _minibatchSize);
        for (int i = 0; i < count; i++) {
            _startSignaled.push_back(0);
            _startInds.push_back(0);
            _endInds.push_back(0);
            _dataOffsets.push_back(0);
            _labelOffsets.push_back(0);
            _labelSpans.push_back(0);
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
        assert(id < _count);
        _startInds[id] = id * _itemsPerThread;
        int itemCount = _itemsPerThread;
        if (id == _count - 1) {
            itemCount = _minibatchSize - id * _itemsPerThread;
        }

        _endInds[id] = _startInds[id] + itemCount;
        _dataOffsets[id] = _startInds[id] * _outputItemSize;
        _labelOffsets[id] = _startInds[id] * _labelChunkSize;
        _labelSpans[id] = itemCount * _labelChunkSize;
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

        int start = _startInds[id];
        int end = _endInds[id];
        // No locking required because threads
        // write into non-overlapping regions.
        BufferPair& outBuf = _out.getForWrite();
        char* dataBuf = outBuf.first->_data + _dataOffsets[id];
        for (int i = start; i < end; i++) {
            // Handle the data.
            int itemSize = 0;
            char* item = _inputBuf->first->getItem(i, itemSize);
            assert(item != 0);
            _decoder->decode(item, itemSize, dataBuf);
            dataBuf += _outputItemSize;
        }

        // Handle the targets.
        char* labelDst = outBuf.second->_data + _labelOffsets[id];
        int labelChunkSize = 0;
        char* labelSrc = _inputBuf->second->getItem(start, labelChunkSize);
        assert(labelChunkSize == _labelChunkSize);
        memcpy(labelDst, labelSrc, _labelSpans[id]);

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
                for (unsigned int i = 0; i < _startSignaled.size(); i++) {
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
            Matrix<char>::transpose(outBuf.first->_data,
                                    _minibatchSize, _outputItemSize);
            // Copy to device.
            _device->copyData(_bufferIndex, outBuf.first->_data,
                              outBuf.first->_size);
            _device->copyLabels(_bufferIndex, outBuf.second->_data,
                                outBuf.second->_size);
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
    BufferPool&                 _in;
    BufferPool&                 _out;
    mutex                       _mutex;
    condition_variable          _started;
    condition_variable          _ended;
    vector<int>                 _startSignaled;
    int                         _endSignaled;
    thread*                     _manager;
    bool                        _managerStopped;
    BufferPair*                 _inputBuf;
    int                         _bufferIndex;
    int                         _minibatchSize;
    vector<int>                 _startInds;
    vector<int>                 _endInds;
    vector<int>                 _dataOffsets;
    vector<int>                 _labelOffsets;
    vector<int>                 _labelSpans;
    int                         _outputItemSize;
    int                         _labelChunkSize;
    Device*                     _device;
    Decoder*                    _decoder;
};

class ReadThreadPool : public ThreadPool {
public:
    ReadThreadPool(BufferPool& out, Reader* reader)
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
    BufferPool&                 _out;
    Reader*                     _reader;
};

class Loader {
public:
    Loader(int minibatchSize, int readMaxSize, int itemMaxSize, int labelSize,
           int labelCount, Device* device, Reader* reader, Decoder* decoder)
    : _first(true),
      _minibatchSize(minibatchSize),
      _readMaxSize(readMaxSize), _itemMaxSize(itemMaxSize),
      _labelSize(labelSize), _labelCount(labelCount),
      _readBufs(0), _decodeBufs(0),
      _readPool(0), _decodePool(0),
      _device(device), _reader(reader), _decoder(decoder) {
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
                new BufferPool(_minibatchSize * _readMaxSize,
                               _labelCount * _minibatchSize * _labelSize);
            _readPool =
                new ReadThreadPool(*_readBufs, _reader);
            bool pinned = (_device->_type != CPU);
            _decodeBufs =
                new BufferPool(_minibatchSize * _itemMaxSize,
                               _labelCount * _minibatchSize * _labelSize,
                               pinned);
            int numCores = thread::hardware_concurrency();
            int itemsPerThread = (_minibatchSize - 1) /  numCores + 1;
            int threadCount =  (_minibatchSize - 1) / itemsPerThread + 1;
            threadCount = std::min(threadCount, _minibatchSize);
            _decodePool =
                new DecodeThreadPool(threadCount, _minibatchSize, _itemMaxSize,
                                     _labelSize, _labelCount,
                                     *_readBufs, *_decodeBufs,
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

    void next(Buffer<char>* dataBuf, Buffer<char>* labelBuf) {
        // Copy minibatch data into the buffers passed in.
        // Only used for testing purposes.
        {
            unique_lock<mutex> lock(_decodeBufs->getMutex());
            while (_decodeBufs->empty()) {
                _decodeBufs->waitForNonEmpty(lock);
            }
            Buffer<char>* data = _decodeBufs->getForRead().first;
            memcpy(dataBuf->_data, data->_data, dataBuf->_size);
            Buffer<char>* labels = _decodeBufs->getForRead().second;
            memcpy(labelBuf->_data, labels->_data, labelBuf->_size);
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
    int                         _readMaxSize;
    int                         _itemMaxSize;
    int                         _labelSize;
    int                         _labelCount;
    BufferPool*                 _readBufs;
    BufferPool*                 _decodeBufs;
    ReadThreadPool*             _readPool;
    DecodeThreadPool*           _decodePool;
    Device*                     _device;
    Reader*                     _reader;
    Decoder*                    _decoder;
};

