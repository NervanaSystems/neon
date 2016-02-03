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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using cv::Mat;
using cv::Rect;
using cv::Point2i;
using cv::Size2i;

enum MediaType {
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

class ImageParams : public MediaParams {
public:
    ImageParams(int channelCount, int height, int width,
                bool center, bool flip,
                int scaleMin, int scaleMax,
                int contrastMin, int contrastMax,
                int rotateMin, int rotateMax)
    : MediaParams(IMAGE),
      _height(height), _width(width),
      _center(center), _flip(flip),
      _scaleMin(scaleMin), _scaleMax(scaleMax),
      _contrastMin(contrastMin), _contrastMax(contrastMax),
      _rotateMin(rotateMin), _rotateMax(rotateMax) {
    }

    ImageParams()
    : ImageParams(224, 224, true, false, true, 256, 256, 0, 0, 0, 0) {}

    void dump() {
        MediaParams::dump();
        printf("inner height %d\n", _height);
        printf("inner width %d\n", _width);
        printf("center %d\n", _center);
        printf("flip %d\n", _flip);
        printf("scale min %d\n", _scaleMin);
        printf("scale max %d\n", _scaleMax);
        printf("contrast min %d\n", _contrastMin);
        printf("contrast max %d\n", _contrastMax);
        printf("rotate min %d\n", _rotateMin);
        printf("rotate max %d\n", _rotateMax);
    }

    bool doRandomFlip(unsigned int& seed) {
        return _flip && (rand_r(&seed) % 2 == 0);
    }

    void getRandomCorner(unsigned int& seed, const Size2i &border,
                         Point2i* point) {
        if (!_center) {
            point->x = rand_r(&seed) % (border.width + 1);
            point->y = rand_r(&seed) % (border.height + 1);
        } else {
            point->x = border.width / 2;
            point->y = border.height / 2;
        }
    }

    float getRandomContrast(unsigned int& seed) {
        if (_contrastMin == _contrastMax) {
            return 0;
        }
        return (_contrastMin +
                (rand_r(&seed) % (_contrastMax - _contrastMin))) / 100.0;
    }

    void getRandomCrop(unsigned int& seed, const Size2i &inputSize,
                       Rect* cropBox) {
        // Use the entire squashed image (Caffe style evaluation)
        if (_scaleMin == 0) {
            cropBox->x = cropBox->y = 0;
            cropBox->width = inputSize.width;
            cropBox->height = inputSize.height;
            return;
        }
        int scaleSize = (_scaleMin +
                         (rand_r(&seed) % (_scaleMax + 1 - _scaleMin)));
        float scaleFactor = std::min(inputSize.width, inputSize.height) /
                            (float) scaleSize;
        Point2i corner;
        Size2i cropSize(_width * scaleFactor, _height * scaleFactor);
        getRandomCorner(seed, inputSize - cropSize, &corner);
        cropBox->width = cropSize.width;
        cropBox->height = cropSize.height;
        cropBox->x = corner.x;
        cropBox->y = corner.y;
        return;
    }

    const Size2i getSize() {
        return Size2i(_width, _height);
    }

public:
    int                         _channelCount;
    int                         _height;
    int                         _width;
    bool                        _center;
    bool                        _flip;
    // Pixel scale to jitter at (image from which to crop will have
    // short side in [scaleMin, Max])
    int                         _scaleMin;
    int                         _scaleMax;
    int                         _contrastMin;
    int                         _contrastMax;
    int                         _rotateMin;
    int                         _rotateMax;
};

class VideoParams : public MediaParams {
public:
    int                         _dummy;
};

class AudioParams : public MediaParams {
public:
    int                         _dummy;
};

class TextParams : public MediaParams {
public:
    int                         _dummy;
};

