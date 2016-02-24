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
    ArchiveWriter(int batchSize, const char* repoDir, bool shuffle,
                  Media* media)
    : _batchSize(batchSize), _repoDir(repoDir),
      _fileIdx(0), _itemCount(0), _started(false),
      _dataBuf(0), _targetBuf(0), _dataBufLen(0), _targetBufLen(0),
      _media(media) {
        _archiveDir = _repoDir + ARCHIVE_DIR_SUFFIX;
        _writeThread = new WriteThread(this);
        _reader = new FileReader(&_itemCount, 1, repoDir, shuffle);
        if (Reader::exists(_archiveDir) == true) {
            return;
        }
        int result = mkdir(_archiveDir.c_str(), 0777);
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
        delete _targetBuf;
        delete _dataBuf;
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
        fileName << _archiveDir << '/' << ARCHIVE_FILE_PREFIX
                 << _fileIdx++ << ".cpio";
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
        writeMetadata();
        _writeThread->start();
        _started = true;
    }

    void writeMetadata() {
        string metaFile = _archiveDir + '/' + META_FILE_NAME;
        ofstream ofs(metaFile);
        if (!ofs) {
            stringstream ss;
            ss << "Could not create " <<  metaFile;
            throw std::ios_base::failure(ss.str());
        }

        stringstream ss;
        ss << "nrec," << _itemCount << "\n";
        string line = ss.str();
        ofs.write(line.c_str(), line.length());
    }

private:
    int                         _batchSize;
    string                      _repoDir;
    // Index of current archive file.
    int                         _fileIdx;
    // Total number of items in this dataset.
    int                         _itemCount;
    Metadata                    _metadata;
    bool                        _started;
    mutex                       _mutex;
    condition_variable          _write;
    WriteThread*                _writeThread;
    FileReader*                 _reader;
    char*                       _dataBuf;
    char*                       _targetBuf;
    int                         _dataBufLen;
    int                         _targetBufLen;
    string                      _archiveDir;
    Media*                      _media;
};

class ArchiveReader : public Reader {
public:
    ArchiveReader(int* itemCount, int batchSize, const char* repoDir,
                  bool shuffle, bool repeatShuffle, int subsetPercent,
                  Media* media)
    : Reader(batchSize, repoDir, shuffle, repeatShuffle, subsetPercent),
      _fileIdx(0), _itemIdx(0), _itemsLeft(0) {
        // Create a writer just in case. It will only be used if archive
        // files are missing or damaged.
        _archiveWriter = new ArchiveWriter(ARCHIVE_ITEM_COUNT, repoDir,
                                           shuffle, media);
        _archiveDir = _repoDir + ARCHIVE_DIR_SUFFIX;
        loadMetadata();
        *itemCount = _itemCount;
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
            if (_repeatShuffle) {
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
        _fileIdx = 0;
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
    int readAll(BufferPair& buffers) {
        readExact(buffers, _itemsLeft);
        return _itemsLeft;
    }

private:
    void loadMetadata() {
        string metaFile = _archiveDir + '/' + META_FILE_NAME;
        if (Reader::exists(metaFile) == false) {
            _archiveWriter->waitFor(metaFile);
        }
        ifstream ifs(metaFile);
        if (!ifs) {
            stringstream ss;
            ss << "Could not open " << metaFile;
            throw std::runtime_error(ss.str());
        }

        string line;
        while (std::getline(ifs, line)) {
            _metadata.addElement(line);
        }

        _itemCount = _metadata.getItemCount();
        _itemCount = (_itemCount * _subsetPercent) / 100;
        if (_itemCount <= 0) {
            stringstream ss;
            ss << "Number of data points is " <<  _itemCount;
            throw std::runtime_error(ss.str());
        }
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

        CharBuffer* data = buffers.first;
        CharBuffer* targets = buffers.second;
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
        ss << _archiveDir << '/' << ARCHIVE_FILE_PREFIX << _fileIdx << ".cpio";
        string fileName = ss.str();
        if (Reader::exists(fileName) == false) {
            _archiveWriter->waitFor(fileName);
        }
        _batchFile.openForRead(fileName);
        _itemsLeft = _batchFile.itemCount();
    }

    void close() {
        _batchFile.close();
    }

private:
    // Index of current archive file.
    int                         _fileIdx;
    // Index of current item.
    int                         _itemIdx;
    // Number of items left in the current archive.
    int                         _itemsLeft;
    BatchFile                   _batchFile;
    Metadata                    _metadata;
    std::deque<DataPair>        _shuffleQueue;
    ArchiveWriter*              _archiveWriter;
    string                      _archiveDir;
};
