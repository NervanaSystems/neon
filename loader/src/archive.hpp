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

#include "reader.hpp"
#include "threadpool.hpp"
#include "batchfile.hpp"
#include "media.hpp"

using std::string;
using std::ifstream;
using std::ofstream;
using std::ios;
using std::stringstream;
using std::vector;
using std::map;

#define ARCHIVE_ITEM_COUNT  4096

typedef std::pair<std::unique_ptr<ByteVect>,std::unique_ptr<ByteVect>> DataPair;

class Writer {
public:
    virtual int write() = 0;
};

class WriteThread : public ThreadPool {
public:
    WriteThread(Writer* writer)
    : ThreadPool(1), _writer(writer) {
    }

protected:
    virtual void work(int id) {
        int result = _writer->write();
        if (result != 0) {
            stop();
        }
    }

private:
    Writer*                     _writer;
};

class ArchiveWriter : public Writer {
public:
    ArchiveWriter(int batchSize, const char* repoDir, const char* archiveDir,
                  const char* indexFile, const char* archivePrefix,
                  bool shuffle,
                  MediaParams* params, MediaParams* ingestParams,
                  int targetTypeSize, int targetConversion)
    : _batchSize(batchSize),
      _repoDir(repoDir), _archiveDir(archiveDir),
      _indexFile(indexFile),
      _archivePrefix(archivePrefix),
      _fileIdx(0), _itemCount(0), _started(false),
      _dataBuf(0), _targetBuf(0), _dataBufLen(0), _targetBufLen(0) {
        _media = Media::create(params, ingestParams, 0);
        _writeThread = new WriteThread(this);
        _reader = new FileReader(&_itemCount, 1, repoDir, indexFile, shuffle,
                                 targetTypeSize, targetConversion);
        if (Reader::exists(_archiveDir) == true) {
            return;
        }
        int result = mkdir(_archiveDir.c_str(), 0755);
        if (result != 0) {
            stringstream ss;
            ss << "Could not create " <<  _archiveDir;
            throw std::ios_base::failure(ss.str());
        }

    }

    virtual ~ArchiveWriter() {
        _writeThread->stop();
        delete _writeThread;
        delete _reader;
        delete[] _targetBuf;
        delete[] _dataBuf;
        delete _media;
    }

    void waitFor(string& name) {
        if (_started == false) {
            start();
        }

        unique_lock<mutex> lock(_mutex);
        while (_reader->exists(name) == false) {
            _write.wait(lock);
        }
    }

    int write() {
        if (_reader->eos() == true) {
            return 1;
        }
        stringstream    fileName;
        fileName << _archiveDir << '/' << _archivePrefix
                 << _fileIdx++ << ".cpio";

        if (Reader::exists(fileName.str()) == true) {
            return 0;
        }
        BatchFile       batchFile(fileName.str(), "");
        for (int i = 0; i < _batchSize; i++) {
            int dataLen = 0;
            int targetLen = 0;
            int result = _reader->next(&_dataBuf, &_targetBuf,
                                       &_dataBufLen, &_targetBufLen,
                                       &dataLen, &targetLen);
            if (result != 0) {
                break;
            }
            // TODO: make this multithreaded.
            _media->ingest(&_dataBuf, &_dataBufLen, &dataLen);
            batchFile.writeItem(_dataBuf, _targetBuf,
                                dataLen, targetLen);
        }

        {
            unique_lock<mutex> lock(_mutex);
            batchFile.close();
        }
        _write.notify_one();
        return 0;
    }

private:
    void start() {
        _writeThread->start();
        _started = true;
    }

private:
    int                         _batchSize;
    string                      _repoDir;
    string                      _archiveDir;
    string                      _indexFile;
    string                      _archivePrefix;
    // Index of current archive file.
    int                         _fileIdx;
    // Total number of items in this dataset.
    int                         _itemCount;
    bool                        _started;
    mutex                       _mutex;
    condition_variable          _write;
    WriteThread*                _writeThread;
    FileReader*                 _reader;
    char*                       _dataBuf;
    char*                       _targetBuf;
    int                         _dataBufLen;
    int                         _targetBufLen;
    Media*                      _media;
};

class ArchiveReader : public Reader {
public:
    ArchiveReader(int* itemCount, int batchSize,
                  const char* repoDir, const char* archiveDir,
                  const char* indexFile,
                  const char* archivePrefix,
                  bool shuffle, bool reshuffle,
                  int startFileIdx,
                  int subsetPercent,
                  MediaParams* params,
                  MediaParams* ingestParams,
                  int targetTypeSize,
                  int targetConversion)
    : Reader(batchSize, repoDir, indexFile, shuffle, reshuffle, subsetPercent),
      _archiveDir(archiveDir), _indexFile(indexFile),
      _archivePrefix(archivePrefix),
      _startFileIdx(startFileIdx),
      _fileIdx(startFileIdx), _itemIdx(0), _itemsLeft(0), _archiveWriter(0) {
        if (*itemCount == 0) {
            *itemCount = getCount();
            // Create a writer just in case. It will only be used if archive
            // files are missing or damaged.
            _archiveWriter = new ArchiveWriter(ARCHIVE_ITEM_COUNT,
                    repoDir, archiveDir, indexFile, archivePrefix,
                    shuffle, params, ingestParams,
                    targetTypeSize, targetConversion);
        }
        _itemCount = *itemCount;
        assert(_itemCount != 0);
        open();
    }

    virtual ~ArchiveReader() {
        delete _archiveWriter;
        close();
    }

    int read(BufferPair& buffers) {
        int offset = 0;
        while (offset < _batchSize) {
            int count = _batchSize - offset;
            int result;
            if (_reshuffle) {
                result = readShuffle(buffers, count);
            } else {
                result = read(buffers, count);
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

private:
    int getCount() {
        ifstream ifs(_indexFile);
        if (!ifs) {
            stringstream ss;
            ss << "Could not open " << _indexFile;
            throw std::ios_base::failure(ss.str());
        }

        string  line;
        int     count = 0;
        std::getline(ifs, line);
        while (std::getline(ifs, line)) {
            if (line[0] == '#') {
                continue;
            }
            count++;
        }

        if (_subsetPercent != 100) {
            count  = (count * _subsetPercent) / 100;
        }
        if (count == 0) {
            stringstream ss;
            ss << "Index file is empty: " << _indexFile;
            throw std::runtime_error(ss.str());
        }

        return count;
    }

    int read(BufferPair& buffers, int count) {
        if (_itemsLeft == 0) {
            next();
        }
        assert(_itemsLeft > 0);
        int realCount = std::min(count, _itemsLeft);
        if (_itemIdx + realCount >= _itemCount) {
            realCount = _itemCount - _itemIdx;
            readExact(buffers, realCount);
            reset();
            return realCount;
        }
        readExact(buffers, realCount);
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

    int readShuffle(BufferPair& buffers, int count) {
        while ((int) _shuffleQueue.size() < count) {
            replenishQueue(count);
        }
        for (int i=0; i<count; ++i) {
            auto ee = std::move(_shuffleQueue.at(0));
            buffers.first->read(&(*ee.first)[0], ee.first->size());
            buffers.second->read(&(*ee.second)[0], ee.second->size());
            _shuffleQueue.pop_front();
        }
        return count;
    }

    void readExact(BufferPair& buffers, int count) {
        assert(count <= _itemsLeft);
        for (int i = 0; i < count; ++i) {
            _batchFile.readItem(buffers);
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
        stringstream ss;
        ss << _archiveDir << '/' << _archivePrefix << _fileIdx << ".cpio";
        string fileName = ss.str();
        if ((Reader::exists(fileName) == false) && (_archiveWriter != 0)) {
            _archiveWriter->waitFor(fileName);
        }
        _batchFile.openForRead(fileName);
        _itemsLeft = _batchFile.itemCount();
    }

    void close() {
        _batchFile.close();
    }

private:
    string                      _archiveDir;
    string                      _indexFile;
    string                      _archivePrefix;
    int                         _startFileIdx;
    // Index of current archive file.
    int                         _fileIdx;
    // Index of current item.
    int                         _itemIdx;
    // Number of items left in the current archive.
    int                         _itemsLeft;
    BatchFile                   _batchFile;
    std::deque<DataPair>        _shuffleQueue;
    ArchiveWriter*              _archiveWriter;
};
