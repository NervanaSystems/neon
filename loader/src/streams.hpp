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

#pragma once

#include <fstream>

class IfStream : public std::ifstream {
public:
    template <typename T>
    void read(T* data) {
        read(reinterpret_cast<char*>(data), sizeof(T));
    }

    void read(char* data, int len) {
        std::ifstream::read(data, len);
    }

    void readPadding(uint length) {
        // Read a byte if length is odd.
        if (length % 2 == 0) {
            return;
        }
        char byte = 0;
        read(&byte);
    }
};

class OfStream : public std::ofstream {
public:
    template <typename T>
    void write(T* data) {
        write(reinterpret_cast<char*>(data), sizeof(T));
    }

    void write(char* data, int len) {
        std::ofstream::write(data, len);
    }

    void writePadding(uint length) {
        // Write a byte if length is odd.
        if (length % 2 == 0) {
            return;
        }
        char byte = 0;
        write(&byte);
    }
};
