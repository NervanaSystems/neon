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

#include <opencv2/core/core.hpp>

using cv::Mat;

template <typename Type>
class Matrix {
public:
    static void transpose(Type* data, int height, int width) {
        int elemType;
        if (sizeof(Type) == 1) {
            elemType = CV_8UC1;
        } else if (sizeof(Type) == 4) {
            width /= 4;
            elemType = CV_32F;
        } else {
            throw std::runtime_error("Unsupported type in transpose\n");
        }
        Mat input = Mat(height, width, elemType, data).clone();
        Mat output = Mat(width, height, elemType, data);
        cv::transpose(input, output);
    }
};
