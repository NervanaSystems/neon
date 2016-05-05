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

#include "media.hpp"
#include "codec.hpp"
#include "specgram.hpp"

class AudioParams : public SignalParams {
};

class Audio : public Media {
public:
    Audio(AudioParams *params, int id)
    : _params(params) {
        _codec = new Codec(params);
        _specgram = new Specgram(params, id);
    }

    virtual ~Audio() {
        delete _specgram;
        delete _codec;
    }

public:
    void transform(char* item, int itemSize, char* buf, int bufSize) {
        RawMedia* raw = _codec->decode(item, itemSize);
        _specgram->generate(raw, buf, bufSize);
    }

    void ingest(char** dataBuf, int* dataBufLen, int* dataLen) {
    }

private:
    AudioParams*                _params;
    Codec*                      _codec;
    Specgram*                   _specgram;
};
