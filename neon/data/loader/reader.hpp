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

#define INDEX_FILE_NAME     "index.csv"
#define ARCHIVE_DIR_SUFFIX  "-ingested"
#define ARCHIVE_FILE_PREFIX "archive-"
#define META_FILE_NAME      "archive-meta.csv"
#define ARCHIVE_ITEM_COUNT  4096

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

class Metadata {
public:
    void addElement(string& line) {
        std::istringstream ss(line);
        string key, val;
        std::getline(ss, key, ',');
        std::getline(ss, val, ',');
        _map[key] = val;
    }

    int getItemCount() {
        if (_map.count("nrec") == 0) {
            throw std::runtime_error("Error in metadata\n");
        }
        return atoi(_map["nrec"].c_str());
    }

private:
    map<string, string>         _map;
};

class Reader {
public:
    Reader(int batchSize, string& repoDir, bool shuffle)
    : _batchSize(batchSize), _repoDir(repoDir), _shuffle(shuffle),
      _itemCount(0)  {
    }

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

    static Reader* create(int* itemCount, int batchSize, char* repoDir,
                          bool shuffle);

    static bool exists(const string& fileName) {
        struct stat stats;
        return stat(fileName.c_str(), &stats) == 0;
    }

protected:
    // Number of items to read at a time.
    int                         _batchSize;
    string                      _repoDir;
    bool                        _shuffle;
    // Total number of items.
    int                         _itemCount;
};

class FileReader : public Reader {
public:
    FileReader(int* itemCount, int batchSize, string& repoDir, bool shuffle)
    : Reader(batchSize, repoDir, shuffle), _itemIdx(0) {
        _ifs.exceptions(_ifs.failbit);
        loadIndex();
        *itemCount = _itemCount;
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

    int next(char** dataBuf, char** targetBuf,
             int* dataBufLen, int* targetBufLen,
             int* dataLen, int* targetLen) {
        if (eos()) {
            // No more items to read.
            return 1;
        }
        IndexElement* elem = _index[_itemIdx++];
        // Read the data.
        string path = _repoDir + '/' + elem->_fileName;
        struct stat stats;
        int result = stat(path.c_str(), &stats);
        if (result == -1) {
            printf("Could not stat %s\n", path.c_str());
            return -1;
        }
        off_t size = stats.st_size;
        if (*dataBufLen < size) {
            // Allocate a bit more than what we need right now.
            resize(dataBuf, dataBufLen, size + size / 10);
        }
        _ifs.open(path, ios::binary);
        _ifs.read(*dataBuf, size);
        _ifs.close();
        *dataLen = size;
        // Read the targets.
        // Limit to a single integer for now.
        if (*targetBufLen < (int) sizeof(int)) {
            resize(targetBuf, targetBufLen, sizeof(int));
        }
        int label = atoi(elem->_targets[0].c_str());
        memcpy(*targetBuf, &label, sizeof(int));
        *targetLen = sizeof(int);
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
        string indexFile = _repoDir + '/' + INDEX_FILE_NAME;
        ifstream ifs(indexFile);
        if (!ifs) {
            stringstream ss;
            ss << "Could not open " << indexFile;
            throw std::ios_base::failure(ss.str());
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
    Index                       _index;
    int                         _itemIdx;
    ifstream                    _ifs;
};
