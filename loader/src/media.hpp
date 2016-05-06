/*
 Copyright 2016 Nervana Systems Inc.
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


// This must be kept in sync with ../media.py.

#pragma once

#include <vector>

using std::vector;

enum MediaType {
    UNKNOWN = -1,
    IMAGE = 0,
    VIDEO = 1,
    AUDIO = 2,
    TEXT = 3,
};

class MediaParams {
public:
    MediaParams(int mtype) : _mtype(mtype) {
    }

    // Do not make this virtual. The object passed down from Python will not
    // have the virtual function table filled in.
    void dump() {
        printf("mtype %d\n", _mtype);
    }

public:
    int                         _mtype;
};

class SignalParams : public MediaParams {
public:
    int                         _samplingFreq;
    bool                        _resample;
    int                         _clipDuration;
    int                         _frameDuration;
    int                         _overlapPercent;
    char                        _windowFunc[16];
    float                       _timeScaleFactor;
    float                       _freqScaleFactor;
    float                       _randomizeTimeScaleBy;
    bool                        _addNoise;
    int                         _windowSize;
    int                         _overlap;
    int                         _stride;
    int                         _timeSteps;
    int                         _numFreqs;
    int                         _windowType;
};

class Media {
public:
    virtual ~Media() {
    }

public:
    virtual void transform(char* item, int itemSize, char* buf, int bufSize) = 0;
    virtual void ingest(char** dataBuf, int* dataBufLen, int* dataLen) = 0;

    static Media* create(MediaParams* params, MediaParams* ingestParams, int id);
};

class RawMedia {
public:
    RawMedia() : _bufSize(0), _dataSize(0), _sampleSize(0) {
    }

    virtual ~RawMedia() {
        for (uint i = 0; i < _bufs.size(); i++) {
            delete[] _bufs[i];
        }
    }

    void reset() {
        _dataSize = 0;
    }

    void addBufs(int count, int size) {
        for (int i = 0; i < count; i++) {
            _bufs.push_back(new char[size]);
        }
        _bufSize = size;
    }

    void fillBufs(char** frames, int frameSize) {
        for (uint i = 0; i < _bufs.size(); i++) {
            memcpy(_bufs[i] + _dataSize, frames[i], frameSize);
        }
        _dataSize += frameSize;
    }

    void growBufs(int grow) {
        for (uint i = 0; i < _bufs.size(); i++) {
            char* buf = new char[_bufSize + grow];
            memcpy(buf, _bufs[i], _dataSize);
            delete[] _bufs[i];
            _bufs[i] = buf;
        }
        _bufSize += grow;
    }

    void setSampleSize(int sampleSize) {
        _sampleSize = sampleSize;
    }

    int size() {
        return _bufs.size();
    }

    char* getBuf(int idx) {
        return _bufs[idx];
    }

    int bufSize() {
        return _bufSize;
    }

    int dataSize() {
        return _dataSize;
    }

    int sampleSize() {
        return _sampleSize;
    }

    void copyData(char* buf, int bufSize) {
        if (_dataSize * (int) _bufs.size() > bufSize) {
            stringstream ss;
            ss << "Buffer too small to copy decoded data. Buffer size " <<
                   bufSize << " Data size " << _dataSize * _bufs.size();
            throw std::runtime_error(ss.str());
        }

        for (uint i = 0; i < _bufs.size(); i++) {
            memcpy(buf, _bufs[i], _dataSize);
            buf += _dataSize;
        }
    }

private:
    vector<char*>               _bufs;
    int                         _bufSize;
    int                         _dataSize;
    int                         _sampleSize;
};
