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

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <deque>
#include <random>

#include "buffer.hpp"

using std::string;
using std::ifstream;
using std::ios;
using std::stringstream;
using std::vector;
using std::map;

enum ConversionType {
    NO_CONVERSION = 0,
    ASCII_TO_BINARY = 1
};

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
    Reader(int batchSize, const char* repoDir, const char* indexFile,
           bool shuffle, bool reshuffle, int subsetPercent)
    : _batchSize(batchSize), _repoDir(repoDir), _indexFile(indexFile),
      _shuffle(shuffle), _reshuffle(reshuffle),
      _subsetPercent(subsetPercent),
      _itemCount(0)  {
    }

    virtual ~Reader() {};
    virtual int read(BufferPair& buffers) = 0;
    virtual int reset() = 0;

    virtual int totalDataSize() {
        return 0;
    }

    virtual int totalTargetsSize() {
        return 0;
    }

    static bool exists(const string& fileName) {
        struct stat stats;
        return stat(fileName.c_str(), &stats) == 0;
    }

protected:
    // Number of items to read at a time.
    int                         _batchSize;
    string                      _repoDir;
    string                      _indexFile;
    bool                        _shuffle;
    bool                        _reshuffle;
    int                         _subsetPercent;
    // Total number of items.
    int                         _itemCount;
};

class FileReader : public Reader {
public:
    FileReader(int* itemCount, int batchSize,
               const char* repoDir, const char* indexFile,
               bool shuffle, int targetTypeSize, int targetConversion)
    : Reader(batchSize, repoDir, indexFile, shuffle, false, 100), _itemIdx(0),
      _targetTypeSize(targetTypeSize), _targetConversion(targetConversion) {
        static_assert(sizeof(int) == 4, "int is not 4 bytes");
        _ifs.exceptions(_ifs.failbit);
        loadIndex();
        *itemCount = _itemCount;
    }

    int read(BufferPair& buffers) {
        // Deprecated
        assert(0);
        return 0;
    }

    int next(char** dataBuf, char** targetBuf,
             int* dataBufLen, int* targetBufLen,
             int* dataLen, int* targetLen) {
        if (eos()) {
            // No more items to read.
            return 1;
        }
        IndexElement* elem = _index[_itemIdx++];
        // Read the data.
        string path;
        if (elem->_fileName[0] == '/') {
            path = elem->_fileName;
        } else {
            path = _repoDir + '/' + elem->_fileName;
        }
        struct stat stats;
        int result = stat(path.c_str(), &stats);
        if (result == -1) {
            stringstream ss;
            ss << "Could not find " << path;
            throw std::runtime_error(ss.str());
        }
        off_t size = stats.st_size;
        if (*dataBufLen < size) {
            // Allocate a bit more than what we need right now.
            resize(dataBuf, dataBufLen, size + size / 8);
        }
        _ifs.open(path, ios::binary);
        _ifs.read(*dataBuf, size);
        _ifs.close();
        *dataLen = size;
        // Read the targets.
        if (_targetConversion == NO_CONVERSION) {
            *targetLen = elem->_targets[0].size();
            if (*targetBufLen < *targetLen) {
                resize(targetBuf, targetBufLen, *targetLen);
            }
            memcpy(*targetBuf, elem->_targets[0].c_str(), *targetLen);
        } else if (_targetConversion == ASCII_TO_BINARY) {
            // For now, assume that binary targets are 4 bytes long.
            assert(_targetTypeSize == 4);
            if (*targetBufLen < _targetTypeSize) {
                resize(targetBuf, targetBufLen, _targetTypeSize);
            }
            int label = std::atoi(elem->_targets[0].c_str());
            memcpy(*targetBuf, &label, sizeof(int));
            *targetLen = _targetTypeSize;
        } else {
            throw std::runtime_error("Unknown conversion specified for target\n");
        }
        return 0;
    }

    bool eos() {
        return (_itemIdx == _itemCount);
    }

    int reset() {
        _itemIdx = 0;
        return 0;
    }

private:
    void resize(char** buf, int* len, int newLen) {
        delete[] *buf;
        *buf = new char[newLen];
        *len = newLen;
    }

    void loadIndex() {
        ifstream ifs(_indexFile);
        if (!ifs) {
            stringstream ss;
            ss << "Could not open " << _indexFile;
            throw std::ios_base::failure(ss.str());
        }

        string line;
        // Ignore the header line.
        std::getline(ifs, line);
        while (std::getline(ifs, line)) {
            if (line[0] == '#') {
                // Ignore comments.
                continue;
            }
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
    Index                       _index;
    int                         _itemIdx;
    ifstream                    _ifs;
    int                         _targetTypeSize;
    int                         _targetConversion;
};
