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
    ASCII_TO_BINARY = 1,
    CHAR_TO_INDEX = 2,
    READ_CONTENTS = 3,
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
    Index() : _maxTargetSize(0) {
    }

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
        if (elem->_targets[0].size() > _maxTargetSize) {
            _maxTargetSize = elem->_targets[0].size();
        }
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
    uint                        _maxTargetSize;
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
    virtual int read(BufferTuple& buffers) = 0;
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
               bool shuffle, int targetTypeSize, int targetConversion,
               char* alphabet)
    : Reader(batchSize, repoDir, indexFile, shuffle, false, 100), _itemIdx(0),
      _targetTypeSize(targetTypeSize), _targetConversion(targetConversion) {
        static_assert(sizeof(int) == 4, "int is not 4 bytes");
        if (_targetConversion == ASCII_TO_BINARY) {
            // For now, assume that binary targets are 4 bytes long.
            assert(_targetTypeSize == 4);
        }
        _ifs.exceptions(_ifs.failbit);
        loadIndex();
        *itemCount = _itemCount;
        if (alphabet == 0) {
            _alphabet = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ $";
        } else {
            _alphabet = alphabet;
        }
        memset(_charMap, 0, sizeof(_charMap));
        if (targetConversion == CHAR_TO_INDEX) {
            createCharMap();
        }
    }

    int read(BufferTuple& buffers) {
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
        readFile(elem->_fileName, dataBuf, dataBufLen, dataLen);
        // Read the targets.
        if (_targetConversion == READ_CONTENTS) {
            readFile(elem->_targets[0], targetBuf, targetBufLen, targetLen);
            return 0;
        }

        switch(_targetConversion) {
        case NO_CONVERSION:
        case CHAR_TO_INDEX:
            *targetLen = elem->_targets[0].size();
            break;
        case ASCII_TO_BINARY:
            *targetLen = _targetTypeSize;
            break;
        default:
            throw std::runtime_error("Unknown conversion specified for target");
        }

        if (*targetBufLen < *targetLen) {
            resize(targetBuf, targetBufLen, *targetLen);
        }

        switch(_targetConversion) {
        case NO_CONVERSION:
            memcpy(*targetBuf, elem->_targets[0].c_str(), *targetLen);
            break;
        case ASCII_TO_BINARY:
            asciiToBinary(elem, *targetBuf);
            break;
        case CHAR_TO_INDEX:
            charToIndex(elem, *targetBuf);
            break;
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
    void readFile(string& fileName, char**buf, int* bufLen, int* dataLen) {
        string path;
        if (fileName[0] == '/') {
            path = fileName;
        } else {
            path = _repoDir + '/' + fileName;
        }
        struct stat stats;
        int result = stat(path.c_str(), &stats);
        if (result == -1) {
            stringstream ss;
            ss << "Could not find " << path;
            throw std::runtime_error(ss.str());
        }
        off_t size = stats.st_size;
        if (*bufLen < size) {
            // Allocate a bit more than what we need right now.
            resize(buf, bufLen, size + size / 8);
        }
        _ifs.open(path, ios::binary);
        _ifs.read(*buf, size);
        _ifs.close();
        *dataLen = size;
    }

    void asciiToBinary(IndexElement* elem, char* targetBuf) {
        int label = std::atoi(elem->_targets[0].c_str());
        memcpy(targetBuf, &label, sizeof(int));
    }

    void charToIndex(IndexElement* elem, char* targetBuf) {
        string& target = elem->_targets[0];
        for (uint i = 0; i < target.size(); i++) {
            uchar elem = target[i];
            targetBuf[i] = _charMap[elem];
        }
    }

    void createCharMap() {
        assert(_alphabet.size() <= 256);
        for (uint i = 0; i < _alphabet.size(); i++) {
            uchar elem = _alphabet[i];
            _charMap[elem] = i;
        }
    }

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
            throw std::runtime_error("Could not load index");
        }
    }

private:
    Index                       _index;
    int                         _itemIdx;
    ifstream                    _ifs;
    int                         _targetTypeSize;
    int                         _targetConversion;
    string                      _alphabet;
    static constexpr int        _charMapSize = 256;
    uchar                       _charMap[_charMapSize];
};
