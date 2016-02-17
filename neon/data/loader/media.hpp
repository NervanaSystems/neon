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

class Media {
public:
    virtual ~Media() {
    }

public:
    virtual void encode(char* item, int itemSize, char* buf, int bufSize) = 0;
    virtual void decode(char* item, int itemSize, char* buf, int bufSize) = 0;
    virtual void transform(char* item, int itemSize, char* buf, int bufSize) = 0;

    static Media* create(MediaParams* params);
};
