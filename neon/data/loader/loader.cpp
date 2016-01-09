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
#include <cstdio>

#include "loader.hpp"

#if STANDALONE

// Code for unit testing.

unsigned int sum(char* data, unsigned int len) {
    unsigned int result = 0;
    for (unsigned int i = 0; i < len; i++) {
        result += data[i];
    }
    return result;
}

int single(Reader* reader, Decoder* decoder, int epochCount,
           int macrobatchCount, int macrobatchSize,
           int minibatchSize, int itemMaxSize, int labelSize) {
    BufferPool* readBufs = new BufferPool(reader->totalDataSize(),
                                          reader->totalTargetsSize());
    unsigned int sm = 0;
    char* dataBuf = new char[itemMaxSize];
    char* labelBuf = new char[macrobatchSize * labelSize];
    memset(dataBuf, 0, itemMaxSize);
    memset(labelBuf, 0, macrobatchSize * labelSize);
    for (int epoch = 0; epoch < epochCount; epoch++) {
        reader->reset();
        for (int i = 0; i < macrobatchCount; i++) {
            BufferPair bufPair = readBufs->getForWrite();
            reader->readAll(bufPair.first, bufPair.second);
            for (int j = 0; j < macrobatchSize; j++) {
                int itemSize = 0;
                char* item = bufPair.first->getItem(j, itemSize);
                assert(item != 0);
                decoder->decode(item, itemSize, dataBuf);
                sm += sum(dataBuf, itemMaxSize);
                int labelChunkSize = 0;
                char* labels = bufPair.second->getItem(j, labelChunkSize);
                labelBuf[j] = *labels;
            }

            sm += sum(labelBuf, macrobatchSize * labelSize);
        }
    }
    delete[] dataBuf;
    delete[] labelBuf;
    delete readBufs;
    return sm;
}

int multi(Loader* loader, Device* device, Reader* reader, Decoder* decoder,
          int epochCount, int macrobatchCount, int macrobatchSize,
          int minibatchSize, int itemMaxSize, int labelSize) {
    int result = loader->start();
    assert(result == 0);
    assert(macrobatchSize % minibatchSize == 0);
    int minibatchCount = macrobatchCount * macrobatchSize / minibatchSize;
    unsigned int sm = 0;
    char* data = new char[minibatchSize*itemMaxSize];
    char* labels = new char[minibatchSize*labelSize];
    memset(data, 0, minibatchSize*itemMaxSize);
    memset(labels, 0, minibatchSize*labelSize);
    for (int epoch = 0; epoch < epochCount; epoch++) {
        for (int i = 0; i < minibatchCount; i++) {
            loader->next();
            int bufIdx = i % 2;
            device->copyDataBack(bufIdx, data, minibatchSize * itemMaxSize);
            device->copyLabelsBack(bufIdx, labels, minibatchSize * labelSize);
            sm += sum(data, minibatchSize * itemMaxSize);
            sm += sum(labels, minibatchSize * labelSize);
        }
    }
    delete[] data;
    delete[] labels;
    loader->stop();
    return sm;
}

int test(const char* pathPrefix, int minibatchSize, int itemMaxSize,
         int labelSize, Device* device) {
    int epochCount = 1;
    int macrobatchCount = 1;
    int readMaxSize = itemMaxSize;
    stringstream fileName;
    fileName << pathPrefix << 0;

    // Peek into macrobatch to check the size of the file.
    int macrobatchSize;
    {
        MacrobatchReader reader(pathPrefix, 0, 0, 0);
        macrobatchSize = reader.itemCount();
        readMaxSize = reader.maxDatumSize();
    }

    if (macrobatchSize % minibatchSize != 0) {
        printf("Macrobatch size %d is not a multiple of minibatch size %d\n",
               macrobatchSize, minibatchSize);
        return -1;
    }
    Reader* reader = new MacrobatchReader(pathPrefix, 0,
                                          macrobatchCount * macrobatchSize,
                                          minibatchSize);
    AugmentationParams* agp = new AugmentationParams();
    Decoder* decoder = new ImageDecoder(agp);
    Loader loader(minibatchSize, readMaxSize, itemMaxSize, labelSize, 1,
                  device, reader, decoder);
    unsigned int multiSum = multi(&loader, device, reader, decoder, epochCount,
                                  macrobatchCount, macrobatchSize,
                                  minibatchSize, itemMaxSize, labelSize);
    unsigned int singleSum = single(reader, decoder, epochCount,
                                    macrobatchCount, macrobatchSize,
                                    minibatchSize, itemMaxSize, labelSize);
    printf("sum %u true sum %u\n", multiSum, singleSum);
    assert(multiSum == singleSum);
    printf("OK\n");
    return 0;
}

int main(int argc, char** argv) {
    int itemMaxSize = 3*224*224;
    if (argc < 3) {
        printf("Usage: %s macrobatch_prefix minibatch_size\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    char* pathPrefix = argv[1];
    int minibatchSize = atoi(argv[2]);

#if HASGPU
    Device* gpu = new Gpu(0, minibatchSize*itemMaxSize,
                          minibatchSize*sizeof(int));
    test(pathPrefix, minibatchSize, itemMaxSize, 4, gpu);
#endif
    Device* cpu = new Cpu(0, minibatchSize*itemMaxSize,
                          minibatchSize*sizeof(int));
    test(pathPrefix, minibatchSize, itemMaxSize, 4, cpu);
}

#else  // STANDALONE else
#include "api.hpp"
#endif  // STANDALONE else
