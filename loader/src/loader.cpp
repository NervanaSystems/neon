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
#include "image.hpp"

#if HAS_VIDLIB
#include "video.hpp"
#endif

#include "audio.hpp"

#define UNSUPPORTED_MEDIA_MESSAGE "support not built-in. Please install the " \
                                  "pre-requisites and re-run the installer."
Media* Media::create(MediaParams* params, MediaParams* ingestParams) {
    switch (params->_mtype) {
    case IMAGE:
        return new Image(reinterpret_cast<ImageParams*>(params),
                         reinterpret_cast<ImageIngestParams*>(ingestParams));
    case VIDEO:
#if HAS_VIDLIB
        return new Video(reinterpret_cast<VideoParams*>(params));
#else
        {
            string message = "Video " UNSUPPORTED_MEDIA_MESSAGE;
            throw std::runtime_error(message);
        }
#endif
    case AUDIO:
        return new Audio(reinterpret_cast<AudioParams*>(params));
    default:
        throw std::runtime_error("Unknown media type");
    }
    return 0;
}

#if STANDALONE

// Code for unit testing.

unsigned int sum(char* data, unsigned int len) {
    unsigned int result = 0;
    for (unsigned int i = 0; i < len; i++) {
        result += data[i];
    }
    return result;
}

int single(Loader* loader, int epochCount, int minibatchCount,
           int batchSize, int datumSize, int targetSize) {
    unsigned int sm = 0;
    Reader* reader = loader->getReader();
    Media* media = loader->getMedia();
    char* dataBuf = new char[datumSize];
    memset(dataBuf, 0, datumSize);
    CharBuffer dataBuffer(0);
    CharBuffer targetBuffer(0);
    BufferPair bufPair = make_pair(&dataBuffer, &targetBuffer);
    for (int epoch = 0; epoch < epochCount; epoch++) {
        reader->reset();
        for (int i = 0; i < minibatchCount; i++) {
            bufPair.first->reset();
            bufPair.second->reset();
            reader->read(bufPair);
            for (int j = 0; j < batchSize; j++) {
                int itemSize = 0;
                char* item = bufPair.first->getItem(j, itemSize);
                assert(item != 0);
                media->transform(item, itemSize, dataBuf, datumSize);
                sm += sum(dataBuf, datumSize);
                int targetChunkSize = 0;
                char* targets = bufPair.second->getItem(j, targetChunkSize);
                sm += sum(targets, targetSize);
            }
        }
    }

    delete[] dataBuf;
    return sm;
}

int multi(Loader* loader, int epochCount, int minibatchCount,
          int batchSize, int datumSize, int targetSize) {
    int result = loader->start();
    assert(result == 0);
    unsigned int sm = 0;
    int dataSize = batchSize * datumSize;
    int targetsSize = batchSize * targetSize;
    char* data = new char[dataSize];
    char* targets = new char[targetsSize];
    memset(data, 0, dataSize);
    memset(targets, 0, targetsSize);
    Device* device = loader->getDevice();
    for (int epoch = 0; epoch < epochCount; epoch++) {
        loader->reset();
        for (int i = 0; i < minibatchCount; i++) {
            loader->next();
            int bufIdx = i % 2;
            device->copyDataBack(bufIdx, data, dataSize);
            device->copyLabelsBack(bufIdx, targets, targetsSize);
            sm += sum(data, dataSize);
            sm += sum(targets, targetsSize);
        }
    }
    loader->stop();
    delete[] data;
    delete[] targets;
    return sm;
}

int test(char* repoDir, char* indexFile,
         int batchSize, int nchan, int height, int width) {
    int datumSize = nchan * height * width;
    int targetSize = 4;
    int epochCount = 2;
    int minibatchCount = 65;
    int itemCount = 0;

    ImageParams mediaParams(3, 30, 30, true, false, 0, 0, 0, 0, 0, 0, 0,
                            false, 0, 0, 0, 0);
    char* dataBuffer[2];
    char* targetBuffer[2];
    for (int i = 0; i < 2; i++) {
        dataBuffer[i] = new char[batchSize * datumSize];
        targetBuffer[i] = new char[batchSize * targetSize];
    }

    string archiveDir(repoDir);
    archiveDir += "-ingested";
    string metaFile = "";
    CpuParams deviceParams(0, 0, dataBuffer, targetBuffer);
    ImageIngestParams ingestParams(false, true, 0, 0);
    Loader loader(&itemCount, batchSize, repoDir, archiveDir.c_str(),
                  indexFile, metaFile.c_str(),
                  "archive-", false, false, 0, datumSize, targetSize, 100,
                  &mediaParams, &deviceParams, &ingestParams);
    unsigned int singleSum = single(&loader, epochCount,
                                    minibatchCount, batchSize,
                                    datumSize, targetSize);
    unsigned int multiSum = multi(&loader, epochCount,
                                  minibatchCount, batchSize,
                                  datumSize, targetSize);
    for (int i = 0; i < 2; i++) {
        delete[] dataBuffer[i];
        delete[] targetBuffer[i];
    }
    printf("sum %u true sum %u\n", multiSum, singleSum);
    assert(multiSum == singleSum);
    printf("OK\n");
    return 0;
}

int main(int argc, char** argv) {
    int nchan = 3;
    int height = 30;
    int width = 30;
    int batchSize = 128;
    if (argc < 3) {
        printf("Usage: %s repo_dir index_file\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    char* repoDir = argv[1];
    char* indexFile = argv[2];

    test(repoDir, indexFile, batchSize, nchan, height, width);
}

#else  // STANDALONE else
#include "api.hpp"
#endif  // STANDALONE else
