#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define FORMAT_VERSION  1
#define WRITER_VERSION  1
#define MAGIC_STRING    "MACR"

using std::ifstream;
using std::ofstream;
using std::string;

typedef std::vector<string> LineList;
typedef std::vector<uint8_t> ByteVect;

class BatchFileHeader {
friend class BatchFile;
public:
    BatchFileHeader()
    : _formatVersion(-1), _writerVersion(-1), _itemCount(0),
      _maxDataSize(0), _maxLabelsSize(0),
      _totalDataSize(0), _totalLabelsSize(0) {
        static_assert(sizeof(int) == 4, "int is not 4 bytes");
        memset(_dataType, 0, sizeof(_dataType));
    }

    void read(ifstream& ifs) {
        char magic[4];
        ifs.read(magic, sizeof(magic));
        if (strncmp(magic, MAGIC_STRING, 4) != 0) {
            throw std::runtime_error("Unrecognized format\n");
        }
        ifs.read(reinterpret_cast<char*>(&_formatVersion), 4);
        ifs.read(reinterpret_cast<char*>(&_writerVersion), 4);
        ifs.read(_dataType, sizeof(_dataType));
        ifs.read(reinterpret_cast<char*>(&_itemCount), 4);
        ifs.read(reinterpret_cast<char*>(&_maxDataSize), 4);
        ifs.read(reinterpret_cast<char*>(&_maxLabelsSize), 4);
        ifs.read(reinterpret_cast<char*>(&_totalDataSize), 4);
        ifs.read(reinterpret_cast<char*>(&_totalLabelsSize), 4);
    }

    void write(ofstream& ofs) {
        ofs.write(MAGIC_STRING, strlen(MAGIC_STRING));
        ofs.write(reinterpret_cast<char*>(&_formatVersion), 4);
        ofs.write(reinterpret_cast<char*>(&_writerVersion), 4);
        ofs.write(_dataType, sizeof(_dataType));
        ofs.write(reinterpret_cast<char*>(&_itemCount), 4);
        ofs.write(reinterpret_cast<char*>(&_maxDataSize), 4);
        ofs.write(reinterpret_cast<char*>(&_maxLabelsSize), 4);
        ofs.write(reinterpret_cast<char*>(&_totalDataSize), 4);
        ofs.write(reinterpret_cast<char*>(&_totalLabelsSize), 4);
    }

private:
    int                         _formatVersion;
    int                         _writerVersion;
    char                        _dataType[8];
    uint32_t                    _itemCount;
    uint32_t                    _maxDataSize;
    uint32_t                    _maxLabelsSize;
    uint32_t                    _totalDataSize;
    uint32_t                    _totalLabelsSize;
};

class BatchFile {
public:
    BatchFile() {
        _ifs.exceptions(_ifs.failbit);
        _ofs.exceptions(_ofs.failbit);
    }

    ~BatchFile() {
        assert(_ifs.is_open() == false);
        assert(_ofs.is_open() == false);
    }

    void openForRead(const string& fileName) {
        assert(_ifs.is_open() == false);
        _ifs.open(fileName, ifstream::binary);
        _header.read(_ifs);
    }

    void openForWrite(const string& fileName, const string& dataType) {
        assert(_ofs.is_open() == false);
        _ofs.open(fileName, ofstream::binary);
        memcpy(_header._dataType, dataType.c_str(), 8);
        _header.write(_ofs);  // This will be incomplete until the write on close()
    }

    void close() {
        if (_ifs.is_open() == true) {
            _ifs.close();
        }
        if (_ofs.is_open() == true) {
            // Need to write back the max size values before cleaning up
            _ofs.seekp(0, _ofs.beg);
            _header.write(_ofs);
            _ofs.close();
        }
    }

    void readItem(char* data, char* labels, int* dataSize, int* labelsSize) {
        _ifs.read(reinterpret_cast<char*>(dataSize), 4);
        _ifs.read(reinterpret_cast<char*>(labelsSize), 4);
        _ifs.read(data, *dataSize);
        _ifs.read(labels, *labelsSize);
    }

    void writeItem(char* data, char* labels, uint32_t* dataSize, uint32_t* labelsSize) {
        _ofs.write(reinterpret_cast<char*>(dataSize), 4);
        _ofs.write(reinterpret_cast<char*>(labelsSize), 4);
        _ofs.write(data, *dataSize);
        _ofs.write(labels, *labelsSize);
        _header._maxDataSize = std::max(*dataSize, _header._maxDataSize);
        _header._maxLabelsSize = std::max(*labelsSize, _header._maxLabelsSize);
        _header._totalDataSize += *dataSize;
        _header._totalLabelsSize += *labelsSize;
        _header._itemCount++;
    }

    void writeItem(ByteVect &data, ByteVect &label) {
        char *dataPtr = (char *) &data[0];
        char *labelPtr = (char *) &label[0];
        uint32_t dataSize = data.size();
        uint32_t labelSize = label.size();
        writeItem(dataPtr, labelPtr, &dataSize, &labelSize);
    }

    int itemCount() {
        return _header._itemCount;
    }

    int totalDataSize() {
        return _header._totalDataSize;
    }

    int totalLabelsSize() {
        return _header._totalLabelsSize;
    }

    int maxDataSize() {
        return _header._maxDataSize;
    }

    int maxLabelsSize() {
        return _header._maxLabelsSize;
    }

private:
    ifstream                    _ifs;
    ofstream                    _ofs;
    BatchFileHeader             _header;
};

// Some utilities that would be used by batch writers
int readFileLines(const string &filn, LineList &ll) {
    ifstream ifs(filn);
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
    ifstream ifs(filn, ifstream::binary);
    if (ifs) {
        ifs.seekg (0, ifs.end);
        int length = ifs.tellg();
        ifs.seekg (0, ifs.beg);

        b.resize(length);
        ifs.read(reinterpret_cast<char*>(&b[0]), length);
        ifs.close();
        return 0;
    } else {
        std::cerr << "Unable to open " << filn << std::endl;
        return -1;
    }
}

