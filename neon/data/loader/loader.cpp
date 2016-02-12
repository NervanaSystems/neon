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
           int batchSize, int datumSize, int targetSize) {
    BufferPool* readBufs = new BufferPool(reader->totalDataSize(),
                                          reader->totalTargetsSize());
    unsigned int sm = 0;
    char* dataBuf = new char[datumSize];
    char* targetBuf = new char[macrobatchSize * targetSize];
    memset(dataBuf, 0, datumSize);
    memset(targetBuf, 0, macrobatchSize * targetSize);
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
                sm += sum(dataBuf, datumSize);
                int targetChunkSize = 0;
                char* targets = bufPair.second->getItem(j, targetChunkSize);
                targetBuf[j] = *targets;
            }

            sm += sum(targetBuf, macrobatchSize * targetSize);
        }
    }
    delete[] dataBuf;
    delete[] targetBuf;
    delete readBufs;
    return sm;
}

int multi(Loader* loader, Device* device, Reader* reader, Decoder* decoder,
          int epochCount, int macrobatchCount, int macrobatchSize,
          int batchSize, int datumSize, int targetSize) {
    int result = loader->start();
    assert(result == 0);
    assert(macrobatchSize % batchSize == 0);
    int minibatchCount = macrobatchCount * macrobatchSize / batchSize;
    unsigned int sm = 0;
    char* data = new char[batchSize*datumSize];
    char* targets = new char[batchSize*targetSize];
    memset(data, 0, batchSize*datumSize);
    memset(targets, 0, batchSize*targetSize);
    for (int epoch = 0; epoch < epochCount; epoch++) {
        for (int i = 0; i < minibatchCount; i++) {
            loader->next();
            int bufIdx = i % 2;
            device->copyDataBack(bufIdx, data, batchSize * datumSize);
            device->copyLabelsBack(bufIdx, targets, batchSize * targetSize);
            sm += sum(data, batchSize * datumSize);
            sm += sum(targets, batchSize * targetSize);
        }
    }
    delete[] data;
    delete[] targets;
    loader->stop();
    return sm;
}

int test(const char* pathPrefix, int batchSize, int datumSize,
         int targetSize, Device* device) {
    int epochCount = 1;
    int macrobatchCount = 1;
    stringstream fileName;
    fileName << pathPrefix << 0;

    // Peek into macrobatch to check the size of the file.
    int macrobatchSize;
    {
        ArchiveReader reader(pathPrefix, 0, 0, 0);
        macrobatchSize = reader.itemCount();
    }

    if (macrobatchSize % batchSize != 0) {
        printf("Macrobatch size %d is not a multiple of minibatch size %d\n",
               macrobatchSize, batchSize);
        return -1;
    }
    Reader* reader = new ArchiveReader(pathPrefix, 0,
                                       macrobatchCount * macrobatchSize,
                                       batchSize);
    ImageParams* params = new ImageParams();
    Decoder* decoder = Decoder::create(params);
    Loader loader(batchSize, datumSize, targetSize,
                  device, reader, decoder);
    unsigned int multiSum = multi(&loader, device, reader, decoder, epochCount,
                                  macrobatchCount, macrobatchSize,
                                  batchSize, datumSize, targetSize);
    unsigned int singleSum = single(reader, decoder, epochCount,
                                    macrobatchCount, macrobatchSize,
                                    batchSize, datumSize, targetSize);
    printf("sum %u true sum %u\n", multiSum, singleSum);
    assert(multiSum == singleSum);
    printf("OK\n");
    return 0;
}

int main(int argc, char** argv) {
    int datumSize = 3*224*224;
    if (argc < 3) {
        printf("Usage: %s macrobatch_prefix minibatch_size\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    char* pathPrefix = argv[1];
    int batchSize = atoi(argv[2]);

#if HASGPU
    Device* gpu = new Gpu(0, batchSize*datumSize,
                          batchSize*sizeof(int));
    test(pathPrefix, batchSize, datumSize, 4, gpu);
#endif
    Device* cpu = new Cpu(0, batchSize*datumSize,
                          batchSize*sizeof(int));
    test(pathPrefix, batchSize, datumSize, 4, cpu);
}

#else  // STANDALONE else
#include "api.hpp"
#endif  // STANDALONE else
