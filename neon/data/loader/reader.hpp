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

#include "buffer.hpp"

using std::string;
using std::ifstream;
using std::stringstream;
using std::vector;

class Reader {
public:
    virtual ~Reader() {};
    virtual int read(CharBuffer* data, IntBuffer* labels) = 0;
    virtual int reset() = 0;
#if STANDALONE
    virtual int open() = 0;
    virtual int next() = 0;
    virtual int close() = 0;
    // For unit testing.
    virtual int readAll(CharBuffer* data, IntBuffer* labels) = 0;
#endif
};

class MacrobatchReader : public Reader {
public:
    MacrobatchReader(string pathPrefix, int startFileIdx,
                    int itemCount, int batchSize)
    : Reader(), _pathPrefix(pathPrefix), _startFileIdx(startFileIdx),
      _fileIdx(startFileIdx), _itemIdx(0), _itemCount(itemCount),
      _batchSize(batchSize), _itemsLeft(0), _itemsRead(0), _ifs(0),
      _labels(0), _labelCount(0), _imageCount(0) {
        static_assert(sizeof(int) == 4, "int is not 4 bytes");
        _error = open();
    }

    virtual ~MacrobatchReader() {
        close();
    }

    int read(CharBuffer* data, IntBuffer* labels) {
        assert(_error == 0);
        int offset = 0;
        while (offset < _batchSize) {
            int count = _batchSize - offset;
            int result = read(data, labels, offset, count);
            if (result == -1) {
                return -1;
            }
            offset += result;
        }

        assert(offset == _batchSize);
        assert(_itemIdx <= _itemCount);
        labels->pushItem(_labelCount * sizeof(int) * _batchSize);
        return 0;
    }

    int reset() {
        close();
        _fileIdx = _startFileIdx;
        _itemIdx = 0;
        _error = open();
        return 0;
    }

#if STANDALONE
    // For unit testing.
    int readAll(CharBuffer* data, IntBuffer* labels) {
        int fileSize = 0;
        int chunkSize = sizeof(int) * _labelCount * _itemsLeft;
        memcpy(labels->getCurrent(), _labels, chunkSize);
        labels->pushItem(chunkSize);
        int bufSize = data->getSize();
        for (int i = 0; i < _itemsLeft; ++i) {
            uint imageSize = loadVal<uint>();
            fileSize += imageSize;
            if (fileSize > bufSize) {
                printf("buffer too small for file %d\n", _fileIdx);
                return -2;
            }
            _ifs->read(reinterpret_cast<char *>(data->getCurrent()), imageSize);
            data->pushItem(imageSize);
        }
        return fileSize;
    }
#endif

private:
    int read(CharBuffer* data, IntBuffer* labels, int offset, int count) {
        if (_itemsLeft == 0) {
            next();
        }
        assert(_itemsLeft > 0);
        int realCount = std::min(count, _itemsLeft);
        if (_itemIdx + realCount >= _itemCount) {
            realCount = _itemCount - _itemIdx;
            readExact(data, labels, offset, realCount);
            reset();
            return realCount;
        }
        readExact(data, labels, offset, realCount);
        return realCount;
    }

    void readExact(CharBuffer* data, IntBuffer* labels, int offset, int count) {
        assert(count <= _itemsLeft);
        for (int i = 0; i < _labelCount; i++) {
            memcpy(labels->getCurrent() + i * _batchSize + offset,
                   _labels + i * _imageCount + _itemsRead,
                   sizeof(int) * count);
        }
        for (int i = 0; i < count; ++i) {
            uint imageSize = loadVal<uint>();
            _ifs->read(reinterpret_cast<char *>(data->getCurrent()), imageSize);
            data->pushItem(imageSize);
        }
        _itemsLeft -= count;
        _itemsRead += count;
        _itemIdx += count;
    }

    int next() {
        close();
        _fileIdx++;
        _error = open();
        return _error;
    }

    int open() {
        assert(_ifs == 0);
        assert(_labels == 0);
        stringstream fileName;
        fileName << _pathPrefix << _fileIdx;
        _ifs = new ifstream(fileName.str(), ifstream::binary);
        if (_ifs->is_open() == false) {
            printf("Error opening file %s\n", fileName.str().c_str());
            return -1;
        }

        _imageCount = loadVal<uint>();
        _labelCount = loadVal<uint>();
        _labels = new int[_labelCount * _imageCount];
        for (int i = 0; i < _labelCount; i++) {
            loadString();
            _ifs->read(reinterpret_cast<char *>(_labels + i * _imageCount),
                       _imageCount * sizeof(int));
        }
        _itemsLeft = _imageCount;
        _itemsRead = 0;
        return 0;
    }

    int close() {
        if (_ifs == 0) {
            return 0;
        }
        _ifs->close();
        delete _ifs;
        delete[] _labels;
        _ifs = 0;
        _labels = 0;
        return 0;
    }

    template<typename T>
    T loadVal()
    {
        T result;
        _ifs->read(reinterpret_cast<char *> (&result), sizeof(T));
        return result;
    }

    string loadString()
    {
        long length;
        string result;
        _ifs->read(reinterpret_cast<char *> (&length), sizeof(long));
        result.resize(length);
        _ifs->read(&result[0], length);
        return result;
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
    // Number of items  that has been read from the current macrobatch.
    int                         _itemsRead;
    ifstream*                   _ifs;
    int*                        _labels;
    int                         _labelCount;
    // Total number of images in the current macrobatch.
    int                         _imageCount;
    int                         _error;
};

class ImageFileReader : public Reader {
public:
    ImageFileReader(string listFile, int itemCount, int batchSize, int outerSize)
    : Reader(), _listFile(listFile), _itemIdx(0), _itemCount(itemCount),
      _batchSize(batchSize), _outerSize(outerSize) {
        _error = loadFileList();
        _rootPath = _listFile.c_str();
        _rootPath = dirname((char*) _rootPath.c_str());
    }

    int read(CharBuffer* data, IntBuffer* labels) {
        if (_error != 0) {
            return _error;
        }

        memset(labels->getCurrent(), 0, labels->getSize() * sizeof(int));
        labels->pushItem(labels->getSize() * sizeof(int));
        vector<uchar> buf;
        vector<int> param = vector<int>(2);
        param[0] = CV_IMWRITE_JPEG_QUALITY;
        param[1] = 95;
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

#if STANDALONE
    // For unit testing.
    virtual int readAll(CharBuffer* data, IntBuffer* labels) {
        assert(0);
        return -1;
    };
#endif

private:
    int loadFileList() {
        ifstream ifs(_listFile);
        if (!ifs) {
            printf("Could not open %s\n", _listFile.c_str());
            return -1;
        }
        string line;
        while (std::getline(ifs, line)) {
            _fileNames.push_back(line);
            if (_fileNames.size() == _itemCount) {
                break;
            }
        }
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
};
