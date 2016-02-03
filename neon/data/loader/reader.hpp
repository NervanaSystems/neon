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

#include <string.h>
#include <assert.h>
#include <libgen.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <memory>
#include <deque>
#include <random>
#include "buffer.hpp"
#include "batchfile.hpp"

using std::string;
using std::ifstream;
using std::stringstream;
using std::vector;

typedef std::pair<std::unique_ptr<ByteVect>,std::unique_ptr<ByteVect>> DataPair;

class Reader {
public:
    virtual ~Reader() {};
    virtual int read(CharBuffer* data, CharBuffer* labels) = 0;
    virtual int reset() = 0;
    // For unit testing.
    virtual void readAll(CharBuffer* data, CharBuffer* labels) = 0;
    virtual int totalDataSize() = 0;
    virtual int totalTargetsSize() = 0;
};

class MacrobatchReader : public Reader {
public:
    MacrobatchReader(string pathPrefix, int startFileIdx,
                     int itemCount, int batchSize, bool shuffle=false)
    : Reader(), _pathPrefix(pathPrefix), _startFileIdx(startFileIdx),
      _fileIdx(startFileIdx), _itemIdx(0), _itemCount(itemCount),
      _batchSize(batchSize), _itemsLeft(0), _shuffle(shuffle) {
        static_assert(sizeof(int) == 4, "int is not 4 bytes");
        open();
    }

    virtual ~MacrobatchReader() {
        close();
    }

    int read(CharBuffer* data, CharBuffer* labels) {
        int offset = 0;
        while (offset < _batchSize) {
            int count = _batchSize - offset;
            int result;
            if (_shuffle) {
                result = readShuffle(data, labels, count);
            } else {
                result = read(data, labels, count);
            }
            if (result == -1) {
                return -1;
            }
            offset += result;
        }

        assert(offset == _batchSize);
        assert(_itemIdx <= _itemCount);
        return 0;
    }

    int reset() {
        close();
        _fileIdx = _startFileIdx;
        _itemIdx = 0;
        open();
        return 0;
    }

    int itemCount() {
        return _batchFile.itemCount();
    }

    int maxDatumSize() {
        return _batchFile.maxDatumSize();
    }

    int maxTargetSize() {
        return _batchFile.maxTargetSize();
    }

    int totalDataSize() {
        return _batchFile.totalDataSize();
    }

    int totalTargetsSize() {
        return _batchFile.totalTargetsSize();
    }

    // For unit testing.
    void readAll(CharBuffer* data, CharBuffer* labels) {
        readExact(data, labels, _itemsLeft);
    }

private:
    int read(CharBuffer* data, CharBuffer* labels, int count) {
        if (_itemsLeft == 0) {
            next();
        }
        assert(_itemsLeft > 0);
        int realCount = std::min(count, _itemsLeft);
        if (_itemIdx + realCount >= _itemCount) {
            realCount = _itemCount - _itemIdx;
            readExact(data, labels, realCount);
            reset();
            return realCount;
        }
        readExact(data, labels, realCount);
        return realCount;
    }

    int replenishQueue(int count) {
        // Make sure we have at least count in our queue
        if ( (int) _shuffleQueue.size() >= count)
            return 0;

        while (_itemsLeft > 0 && _itemIdx < _itemCount) {
            DataPair d = _batchFile.readItem();
            _shuffleQueue.push_back(std::move(d));
            _itemIdx++;
            _itemsLeft--;
        }
        std::random_device rd;
        std::shuffle(_shuffleQueue.begin(), _shuffleQueue.end(),
                     std::mt19937(rd()));
        if (_itemIdx == _itemCount)
            reset();
        else
            next();
        return 0;
    }

    int readShuffle(CharBuffer* data, CharBuffer* labels, int count) {
        while ((int) _shuffleQueue.size() < count) {
            replenishQueue(count);
        }

        for (int i=0; i<count; ++i) {
            auto ee = std::move(_shuffleQueue.at(0));
            int dataSize = ee.first->size();
            int labelSize = ee.second->size();
            memcpy(data->getCurrent(), &(*ee.first)[0], dataSize);
            memcpy(labels->getCurrent(), &(*ee.second)[0], labelSize);
            data->pushItem(dataSize);
            labels->pushItem(labelSize);
            _shuffleQueue.pop_front();
        }
        return count;
    }

    void readExact(CharBuffer* data, CharBuffer* labels, int count) {
        assert(count <= _itemsLeft);
        for (int i = 0; i < count; ++i) {
            uint        dataSize;
            uint        labelSize;
            _batchFile.readItem(data->getCurrent(),
                                labels->getCurrent(),
                                &dataSize, &labelSize);
            data->pushItem(dataSize);
            labels->pushItem(labelSize);
        }
        _itemsLeft -= count;
        _itemIdx += count;
    }

    void next() {
        close();
        _fileIdx++;
        open();
    }

    void open() {
        stringstream fileName;
        fileName << _pathPrefix << _fileIdx << ".cpio";
        _batchFile.openForRead(fileName.str());
        _itemsLeft = _batchFile.itemCount();
    }

    void close() {
        _batchFile.close();
    }

private:
    string                      _pathPrefix;
    int                         _startFileIdx;
    // Index of current macrobatch file.
    int                         _fileIdx;
    // Index of current item.
    uint                        _itemIdx;
    // Total number of items to read.
    uint                        _itemCount;
    // Number of items to read at a time.
    int                         _batchSize;
    // Number of items left in the current macrobatch.
    int                         _itemsLeft;
    BatchFile                   _batchFile;
    bool                        _shuffle;
    std::deque<DataPair>        _shuffleQueue;
};

class ImageFileReader : public Reader {
public:
    ImageFileReader(string listFile, int itemCount, int batchSize, int outerSize,
                    bool shuffle=false)
    : Reader(), _listFile(listFile), _itemIdx(0), _itemCount(itemCount),
      _batchSize(batchSize), _outerSize(outerSize),
      _shuffle(shuffle) {

        _error = readFileLines(_listFile, _fileNames);
        _rootPath = _listFile.c_str();
        _rootPath = dirname((char*) _rootPath.c_str());
    }

    int read(CharBuffer* data, CharBuffer* labels) {
        if (_error != 0) {
            return _error;
        }
        if (_shuffle) {
            printf("Shuffling not supported yet\n");
            return -1;
        }
        vector<uchar> buf;
        vector<int> param = {CV_IMWRITE_JPEG_QUALITY, 95};
        for (int i = 0; i < _batchSize; ++i) {
            string path = _rootPath + '/' + _fileNames[_itemIdx];
            cv::Mat src = cv::imread(path);
            if (src.data == 0) {
                printf("Could not open %s\n", path.c_str());
                return -1;
            }
            cv::Mat dst(_outerSize, _outerSize, CV_8UC3);
            cv::resize(src, dst, dst.size());

            cv::imencode(".jpg", dst, buf, param);
            int imageSize = buf.size();
            memcpy(data->getCurrent(), &buf[0], imageSize);
            data->pushItem(imageSize);
            memset(labels->getCurrent(), 0, sizeof(int));
            labels->pushItem(sizeof(int));

            if (++_itemIdx == _itemCount) {
                // Wrap around.
                reset();
            }

        }
        return 0;
    }

    int reset() {
        _itemIdx = 0;
        _error = 0;
        return 0;
    }

    // For unit testing.
    virtual void readAll(CharBuffer* data, CharBuffer* labels) {
        assert(0);
    };

    int totalDataSize() {
        return 0;
    }

    int totalTargetsSize() {
        return 0;
    }

private:
    string                      _listFile;
    string                      _rootPath;
    vector<string>              _fileNames;
    uint                        _itemIdx;
    uint                        _itemCount;
    int                         _batchSize;
    int                         _outerSize;
    int                         _error;
    bool                        _shuffle;
};
