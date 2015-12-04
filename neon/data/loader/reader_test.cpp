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

                int* item2 = bufPair.second->getItem(j, itemSize);
                assert(item2 !=0);
                labelBuf[j] = *item2;
            }
            sm += sum(reinterpret_cast<uchar *>(labelBuf), macrobatchSize * sizeof(int));
            reader->next();
        }
    }
    reader->close();
    delete[] dataBuf;
    delete[] labelBuf;
    delete readBufs;
    return sm;
}

int test(const string &pathPrefix, int macrobatchSize, int minibatchSize, int itemMaxSize,
         Device* device) {
    int epochCount = 1;
    int macrobatchCount = 1;
    Reader* reader = new MacrobatchReader(pathPrefix, 0,
                                          macrobatchCount * macrobatchSize,
                                          minibatchSize);
    AugmentationParams *agp = new AugmentationParams();
    Decoder* decoder = new ImageDecoder(agp);
    // ImageLoader loader(minibatchSize, itemMaxSize, 1, device, reader, decoder);
    unsigned int singleSum = single(reader, decoder, epochCount,
                                    macrobatchCount, macrobatchSize,
                                    minibatchSize, itemMaxSize);
    std::cout << "Single Sum " << singleSum << std::endl;
    delete reader;
    delete decoder;
    delete agp;
    return 0;
}

int main (int argc, char **argv) {
    int minibatchSize = 128;
    int itemMaxSize = 3*224*224;
    Device* cpu = new Cpu(0, minibatchSize*itemMaxSize, minibatchSize*sizeof(int));
    if (argc < 3) {
        std::cout << "Usage: reader_test prefix macrobatchSize" << std::endl;
        return -1;
    }

    string pathPrefix(argv[1]);
    int macrobatchSize = atoi(argv[2]);
    test(pathPrefix, macrobatchSize, minibatchSize, itemMaxSize, cpu);
    delete cpu;
}
