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

#include <time.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "streams.hpp"

#define FORMAT_VERSION  1
#define WRITER_VERSION  1
#define MAGIC_STRING    "MACR"
#define CPIO_FOOTER     "TRAILER!!!"

using std::string;
using std::stringstream;
using std::unique_ptr;

typedef std::vector<string> LineList;
typedef std::vector<char> ByteVect;
typedef std::pair<unique_ptr<ByteVect>,unique_ptr<ByteVect>> DataPair;

static_assert(sizeof(int) == 4, "int is not 4 bytes");
static_assert(sizeof(uint) == 4, "uint is not 4 bytes");
static_assert(sizeof(short) == 2, "short is not 2 bytes");

/*

The data is stored as a cpio archive and may be unpacked using the
GNU cpio utility.

    https://www.gnu.org/software/cpio/

Individual items are packed into a macrobatch file as follows:
    - header
    - datum 1
    - target 1
    - datum 2
    - target 2
      ...
    - trailer

Each of these items comprises of a cpio header record followed by data.

*/

class RecordHeader {
public:
    RecordHeader()
    : _magic(070707), _dev(0), _ino(0), _mode(0100644), _uid(0), _gid(0),
      _nlink(0), _rdev(0), _namesize(0) {
        memset((void*) _mtime, 0, 2 * sizeof(short));
        memset((void*) _filesize, 0, 2 * sizeof(short));
    }

    void loadDoubleShort(uint* dst, ushort src[2]) {
        *dst =  ((uint) src[0]) << 16 | (uint) src[1];
    }

    void saveDoubleShort(ushort* dst, uint src) {
        dst[0] = (ushort) (src >> 16);
        dst[1] = (ushort) src;
    }

    void read(IfStream& ifs, uint* fileSize) {
        ifs.read(&_magic);
        assert(_magic == 070707);
        ifs.read(&_dev);
        ifs.read(&_ino);
        ifs.read(&_mode);
        ifs.read(&_uid);
        ifs.read(&_gid);
        ifs.read(&_nlink);
        ifs.read(&_rdev);
        uint mtime;
        ifs.read(&_mtime);
        loadDoubleShort(&mtime, _mtime);
        ifs.read(&_namesize);
        ifs.read(&_filesize);
        loadDoubleShort(fileSize, _filesize);
        // Skip over filename.
        ifs.seekg(_namesize, ifs.cur);
        ifs.readPadding(_namesize);
    }

    void write(OfStream& ofs, uint fileSize, const char* fileName) {
        _namesize = strlen(fileName) + 1;
        ofs.write(&_magic);
        ofs.write(&_dev);
        ofs.write(&_ino);
        ofs.write(&_mode);
        ofs.write(&_uid);
        ofs.write(&_gid);
        ofs.write(&_nlink);
        ofs.write(&_rdev);
        time_t mtime;
        time(&mtime);
        saveDoubleShort(_mtime, mtime);
        ofs.write(&_mtime);
        ofs.write(&_namesize);
        saveDoubleShort(_filesize, fileSize);
        ofs.write(&_filesize);
        // Write filename.
        ofs.write((char*) fileName, _namesize);
        ofs.writePadding(_namesize);
    }

public:
    ushort                      _magic;
    ushort                      _dev;
    ushort                      _ino;
    ushort                      _mode;
    ushort                      _uid;
    ushort                      _gid;
    ushort                      _nlink;
    ushort                      _rdev;
    ushort                      _mtime[2];
    ushort                      _namesize;
    ushort                      _filesize[2];
};

class BatchFileHeader {
friend class BatchFile;
public:
    BatchFileHeader()
    : _formatVersion(FORMAT_VERSION), _writerVersion(WRITER_VERSION),
      _itemCount(0), _maxDatumSize(0), _maxTargetSize(0),
      _totalDataSize(0), _totalTargetsSize(0) {
        memset(_dataType, 0, sizeof(_dataType));
        memset(_unused, 0, sizeof(_unused));
    }

    void read(IfStream& ifs) {
        ifs.read(&_magic);
        if (strncmp(_magic, MAGIC_STRING, 4) != 0) {
            throw std::runtime_error("Unrecognized format\n");
        }
        ifs.read(&_formatVersion);
        ifs.read(&_writerVersion);
        ifs.read(&_dataType);
        ifs.read(&_itemCount);
        ifs.read(&_maxDatumSize);
        ifs.read(&_maxTargetSize);
        ifs.read(&_totalDataSize);
        ifs.read(&_totalTargetsSize);
        ifs.read(&_unused);
    }

    void write(OfStream& ofs) {
        ofs.write((char*) MAGIC_STRING, strlen(MAGIC_STRING));
        ofs.write(&_formatVersion);
        ofs.write(&_writerVersion);
        ofs.write(&_dataType);
        ofs.write(&_itemCount);
        ofs.write(&_maxDatumSize);
        ofs.write(&_maxTargetSize);
        ofs.write(&_totalDataSize);
        ofs.write(&_totalTargetsSize);
        ofs.write(&_unused);
    }

private:
#pragma pack(1)
    char                        _magic[4];
    uint                        _formatVersion;
    uint                        _writerVersion;
    char                        _dataType[8];
    uint                        _itemCount;
    uint                        _maxDatumSize;
    uint                        _maxTargetSize;
    uint                        _totalDataSize;
    uint                        _totalTargetsSize;
    char                        _unused[24];
#pragma pack()
};

class BatchFileTrailer {
public:
    BatchFileTrailer() {
        memset(_unused, 0, sizeof(_unused));
    }

    void write(OfStream& ofs) {
        ofs.write(&_unused);
    }

    void read(IfStream& ifs) {
        ifs.read(&_unused);
    }

private:
    uint                        _unused[4];
};

class BatchFile {
public:
    BatchFile() : _fileHeaderOffset(0)  {
        _ifs.exceptions(_ifs.failbit);
        _ofs.exceptions(_ofs.failbit);
    }

    BatchFile(const string& fileName) {
        openForRead(fileName);
    }

    BatchFile(const string& fileName, const string& dataType) {
        openForWrite(fileName, dataType);
    }

    ~BatchFile() {
        close();
    }

    void openForRead(const string& fileName) {
        assert(_ifs.is_open() == false);
        _ifs.open(fileName, IfStream::binary);
        uint fileSize;
        _recordHeader.read(_ifs, &fileSize);
        assert(fileSize == sizeof(_fileHeader));
        _fileHeader.read(_ifs);
    }

    void openForWrite(const string& fileName, const string& dataType) {
        static_assert(sizeof(_fileHeader) == 64, "file header is not 64 bytes");
        _fileName = fileName;
        _tempName = fileName + ".tmp";
        assert(_ofs.is_open() == false);
        _ofs.open(_tempName, OfStream::binary);
        _recordHeader.write(_ofs, 64, "cpiohdr");
        _fileHeaderOffset = _ofs.tellp();
        memset(_fileHeader._dataType, ' ', sizeof(_fileHeader._dataType));
        memcpy(_fileHeader._dataType, dataType.c_str(),
               std::min(8, (int) dataType.length()));
        // This will be incomplete until the write on close()
        _fileHeader.write(_ofs);
    }

    void close() {
        if (_ifs.is_open() == true) {
            _ifs.close();
        }
        if (_ofs.is_open() == true) {
            // Write the trailer.
            static_assert(sizeof(_fileTrailer) == 16,
                          "file trailer is not 16 bytes");
            _recordHeader.write(_ofs, 16, "cpiotlr");
            _fileTrailer.write(_ofs);
            _recordHeader.write(_ofs, 0, CPIO_FOOTER);
            // Need to write back the max size values before cleaning up
            _ofs.seekp(_fileHeaderOffset, _ofs.beg);
            _fileHeader.write(_ofs);
            _ofs.close();
            int result = rename(_tempName.c_str(), _fileName.c_str());
            if (result != 0) {
                stringstream ss;
                ss << "Could not create " << _fileName;
                throw std::runtime_error(ss.str());
            }
        }
    }

    void readItem(BufferPair& buffers) {
        uint datumSize;
        uint targetSize;
        _recordHeader.read(_ifs, &datumSize);
        buffers.first->read(_ifs, datumSize);
        _ifs.readPadding(datumSize);
        _recordHeader.read(_ifs, &targetSize);
        buffers.second->read(_ifs, targetSize);
        _ifs.readPadding(targetSize);
    }

    DataPair readItem() {
        uint datumSize = 0;
        _recordHeader.read(_ifs, &datumSize);
        unique_ptr<ByteVect> datum(new ByteVect((size_t) datumSize));
        _ifs.read(&(*datum)[0], datumSize);
        _ifs.readPadding(datumSize);

        uint targetSize = 0;
        _recordHeader.read(_ifs, &targetSize);
        unique_ptr<ByteVect> target(new ByteVect((size_t) targetSize));
        _ifs.read(&(*target)[0], targetSize);
        _ifs.readPadding(targetSize);
        return DataPair(std::move(datum), std::move(target));
    }

    void writeItem(char* datum, char* target,
                   uint datumSize, uint targetSize) {
        char fileName[16];
        // Write the datum.
        sprintf(fileName, "cpiodtm%d",  _fileHeader._itemCount);
        _recordHeader.write(_ofs, datumSize, fileName);
        _ofs.write(datum, datumSize);
        _ofs.writePadding(datumSize);
        // Write the target.
        sprintf(fileName, "cpiotgt%d",  _fileHeader._itemCount);
        _recordHeader.write(_ofs, targetSize, fileName);
        _ofs.write(target, targetSize);
        _ofs.writePadding(targetSize);

        _fileHeader._maxDatumSize =
                std::max(datumSize, _fileHeader._maxDatumSize);
        _fileHeader._maxTargetSize =
                std::max(targetSize, _fileHeader._maxTargetSize);
        _fileHeader._totalDataSize += datumSize;
        _fileHeader._totalTargetsSize += targetSize;
        _fileHeader._itemCount++;
    }

    void writeItem(ByteVect &datum, ByteVect &target) {
        uint    datumSize = datum.size();
        uint    targetSize = target.size();
        writeItem(&datum[0], &target[0], datumSize, targetSize);
    }

    int itemCount() {
        return _fileHeader._itemCount;
    }

    int totalDataSize() {
        return _fileHeader._totalDataSize;
    }

    int totalTargetsSize() {
        return _fileHeader._totalTargetsSize;
    }

    int maxDatumSize() {
        return _fileHeader._maxDatumSize;
    }

    int maxTargetSize() {
        return _fileHeader._maxTargetSize;
    }

private:
    IfStream                    _ifs;
    OfStream                    _ofs;
    BatchFileHeader             _fileHeader;
    BatchFileTrailer            _fileTrailer;
    RecordHeader                _recordHeader;
    int                         _fileHeaderOffset;
    string                      _fileName;
    string                      _tempName;
};

// Some utilities that would be used by batch writers
int readFileLines(const string &filn, LineList &ll) {
    std::ifstream ifs(filn);
    if (ifs) {
        for (string line; std::getline( ifs, line ); /**/ )
           ll.push_back( line );
        ifs.close();
        return 0;
    } else {
        std::cerr << "Unable to open " << filn << std::endl;
        return -1;
    }
}

int readFileBytes(const string &filn, ByteVect &b) {
/* Reads in the binary file as a sequence of bytes, resizing
 * the provided byte vector to fit
*/
    std::ifstream ifs(filn, std::ifstream::binary);
    if (ifs) {
        ifs.seekg (0, ifs.end);
        int length = ifs.tellg();
        ifs.seekg (0, ifs.beg);

        b.resize(length);
        ifs.read(&b[0], length);
        ifs.close();
        return 0;
    } else {
        std::cerr << "Unable to open " << filn << std::endl;
        return -1;
    }
}

