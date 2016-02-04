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

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
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
using std::ios;
using std::stringstream;
using std::vector;

typedef std::pair<std::unique_ptr<ByteVect>,std::unique_ptr<ByteVect>> DataPair;

#define INDEX_FILE_NAME "index.csv"

class IndexElement {
public:
    IndexElement() {
    }

public:
    string                      _fileName;
    vector<string>              _targets;
};

class Index {
public:
    Index() {}

    virtual ~Index() {
        for (auto elem : _elements) {
            delete elem;
        }
    }

    void addElement(string& line) {
        IndexElement* elem = new IndexElement();
        std::istringstream ss(line);
        string token;
        std::getline(ss, token, ',');
        elem->_fileName = token;
        while (std::getline(ss, token, ',')) {
            elem->_targets.push_back(token);
        }

        // For now, restrict to a single target.
        assert(elem->_targets.size() == 1);
        _elements.push_back(elem);
    }

    IndexElement* operator[] (int idx) {
        return _elements[idx];
    }

    uint size() {
        return _elements.size();
    }

    void shuffle() {
        std::srand(0);
        std::random_shuffle(_elements.begin(), _elements.end());
    }

public:
    vector<IndexElement*>       _elements;
};

class Reader {
public:
    virtual ~Reader() {};
    virtual int read(CharBuffer* data, CharBuffer* targets) = 0;
    virtual int reset() = 0;

    // For unit testing.
    virtual int readAll(CharBuffer* data, CharBuffer* targets) {
        return 0;
    }

    virtual int totalDataSize() {
        return 0;
    }

    virtual int totalTargetsSize() {
        return 0;
    }
};

class FileReader : public Reader {
public:
    FileReader(int* itemCount, int batchSize, char* repoDir, bool shuffle)
    : Reader(), _batchSize(batchSize), _repoDir(repoDir), _shuffle(shuffle),
      _itemIdx(0), _itemCount(0) {
        _ifs.exceptions(_ifs.failbit);
        loadIndex();
        *itemCount = _itemCount;
    }

    virtual ~FileReader() {
    }

    int read(CharBuffer* data, CharBuffer* targets) {
        for (int i = 0; i < _batchSize; ++i) {
            IndexElement* elem = _index[_itemIdx];
            // Read datum.
            string path = _repoDir + '/' + elem->_fileName;
            struct stat stats;
            int result = stat(path.c_str(), &stats);
            if (result == -1) {
                printf("Could not stat %s\n", path.c_str());
                return -1;
            }
            off_t size = stats.st_size;
            if (data->getSize() < (data->getLevel() + size)) {
                // TODO: Make buffers resizable.
                printf("Buffer too small for %s\n", path.c_str());
                return -1;
            }
            _ifs.open(path, ios::binary);
            _ifs.read(data->getCurrent(), size);
            _ifs.close();
            data->pushItem(size);
            // Read targets.
            // Limit to a single integer target for now.
            if (targets->getSize() < (targets->getLevel() + sizeof(int))) {
                printf("Buffer too small for %s target\n", path.c_str());
                return -1;
            }
            int target = atoi(elem->_targets[0].c_str());
            memcpy(targets->getCurrent(), &target, sizeof(int));
            targets->pushItem(sizeof(int));
            if (++_itemIdx == _itemCount) {
                // Wrap around.
                reset();
            }

        }
        return 0;
    }

    int reset() {
        _itemIdx = 0;
        return 0;
    }

private:
    void loadIndex() {
        string indexFile = _repoDir + '/' + INDEX_FILE_NAME;
        ifstream ifs(indexFile);
        if (!ifs) {
            printf("Could not open %s\n", indexFile.c_str());
            throw std::ios_base::failure("Could not open file\n");
        }

        string line;
        while (std::getline(ifs, line)) {
            _index.addElement(line);
        }

        if (_shuffle == true) {
            _index.shuffle();
        }

        _itemCount = _index.size();
        if (_itemCount == 0) {
            throw std::runtime_error("Could not load index\n");
        }
    }

private:
    int                         _batchSize;
    string                      _repoDir;
    bool                        _shuffle;
    Index                       _index;
    uint                        _itemIdx;
    uint                        _itemCount;
    ifstream                    _ifs;
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

    int read(CharBuffer* data, CharBuffer* targets) {
        int offset = 0;
        while (offset < _batchSize) {
            int count = _batchSize - offset;
            int result;
            if (_shuffle) {
                result = readShuffle(data, targets, count);
            } else {
                result = read(data, targets, count);
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
    int readAll(CharBuffer* data, CharBuffer* targets) {
        readExact(data, targets, _itemsLeft);
        return _itemsLeft;
    }

private:
    int read(CharBuffer* data, CharBuffer* targets, int count) {
        if (_itemsLeft == 0) {
            next();
        }
        assert(_itemsLeft > 0);
        int realCount = std::min(count, _itemsLeft);
        if (_itemIdx + realCount >= _itemCount) {
            realCount = _itemCount - _itemIdx;
            readExact(data, targets, realCount);
            reset();
            return realCount;
        }
        readExact(data, targets, realCount);
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

    int readShuffle(CharBuffer* data, CharBuffer* targets, int count) {
        while ((int) _shuffleQueue.size() < count) {
            replenishQueue(count);
        }

        for (int i=0; i<count; ++i) {
            auto ee = std::move(_shuffleQueue.at(0));
            int dataSize = ee.first->size();
            int targetSize = ee.second->size();
            memcpy(data->getCurrent(), &(*ee.first)[0], dataSize);
            memcpy(targets->getCurrent(), &(*ee.second)[0], targetSize);
            data->pushItem(dataSize);
            targets->pushItem(targetSize);
            _shuffleQueue.pop_front();
        }
        return count;
    }

    void readExact(CharBuffer* data, CharBuffer* targets, int count) {
        assert(count <= _itemsLeft);
        for (int i = 0; i < count; ++i) {
            uint        dataSize;
            uint        targetSize;
            _batchFile.readItem(data->getCurrent(),
                                targets->getCurrent(),
                                &dataSize, &targetSize);
            data->pushItem(dataSize);
            targets->pushItem(targetSize);
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
