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
#include <stdlib.h>
#include <fstream>
#include <vector>

#include "media.hpp"

using std::ofstream;
using std::vector;

void resizeInput(vector<char> &jpgdata, int maxDim){
    // Takes the buffer containing encoded jpg, determines if its shortest dimension
    // is greater than maxDim.  If so, it scales it down so that the shortest dimension
    // is equal to maxDim.  equivalent to "512x512^>" for maxDim=512 geometry argument in
    // imagemagick

    Mat image = Mat(1, jpgdata.size(), CV_8UC3, &jpgdata[0]);
    Mat decodedImage = cv::imdecode(image, CV_LOAD_IMAGE_COLOR);

    int minDim = std::min(decodedImage.rows, decodedImage.cols);
    // If no resizing necessary, just return, original image still in jpgdata;
    if (minDim <= maxDim)
        return;

    vector<int> param = {CV_IMWRITE_JPEG_QUALITY, 90};
    double scaleFactor = (double) maxDim / (double) minDim;
    Mat resizedImage;
    cv::resize(decodedImage, resizedImage, Size2i(0, 0), scaleFactor, scaleFactor, CV_INTER_AREA);
    cv::imencode(".jpg", resizedImage, *(reinterpret_cast<vector<uchar>*>(&jpgdata)), param);
    //    cv::imencode(".jpg", resizedImage, jpgdata, param);
    return;
}

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

public:
    int                         _mtype;
};

class ImageParams : public MediaParams {
public:
    ImageParams(int innerSize, bool center, bool flip, bool rgb,
                int scaleMin, int scaleMax,
                int contrastMin, int contrastMax,
                int rotateMin, int rotateMax)
    : MediaParams(IMAGE), _rngSeed(0), _innerSize(innerSize, innerSize),
      _center(center), _flip(flip), _rgb(rgb),
      _scaleMin(scaleMin), _scaleMax(scaleMax),
      _contrastMin(contrastMin), _contrastMax(contrastMax),
      _rotateMin(rotateMin), _rotateMax(rotateMax), _aspectRatio(aspectRatio) {
    }

    ImageParams()
    : ImageParams(224, true, false, true, 256, 256, 0, 0, 0, 0) {}

    virtual ~ImageParams() {};

    void dump() {
        printf("inner size %dx%d\n", _innerSize.height, _innerSize.width);
        printf("center %d\n", _center);
        printf("flip %d\n", _flip);
        printf("rgb %d\n", _rgb);
        printf("scale min %d\n", _scaleMin);
        printf("scale max %d\n", _scaleMax);
        printf("contrast min %d\n", _contrastMin);
        printf("contrast max %d\n", _contrastMax);
        printf("rotate min %d\n", _rotateMin);
        printf("rotate max %d\n", _rotateMax);
    }

    bool doRandomFlip() {
        return _flip && (rand_r(&(_rngSeed)) % 2 == 0);
    }

    void randomCorner(const Size2i &border, Point2i* point) {
        if (!_center) {
            point->x = rand_r(&_rngSeed) % (border.width + 1);
            point->y = rand_r(&_rngSeed) % (border.height + 1);
        } else {
            point->x = border.width / 2;
            point->y = border.height / 2;
        }
    }

    float getRandomContrast() {
        if (_contrastMin == _contrastMax) {
            return 0;
        }
        return (_contrastMin + (rand_r(&(_rngSeed)) % (_contrastMax - _contrastMin))) / 100.0;
    }

    // adjust the square cropSize to be an inner rectangle
    void getRandomAspectRatio(Size2i &cropSize) {
        int ratio = (101 + (rand_r(&(_rngSeed)) % (_aspectRatio - 100)));
        float ratio_f = 100.0 / (float) ratio;
        int orientation = rand_r(&(_rngSeed)) % 2;
        if (orientation) {
            cropSize.height *= ratio_f;
        } else {
            cropSize.width *= ratio_f;
        }
    }

    void getRandomCrop(const Size2i &inputSize, Rect* cropBox) {
        // Use the entire squashed image (Caffe style evaluation)
        if (_scaleMin == 0) {
            cropBox->x = cropBox->y = 0;
            cropBox->width = inputSize.width;
            cropBox->height = inputSize.height;
            return;
        }
        int scaleSize = (_scaleMin + (rand_r(&(_rngSeed)) % (_scaleMax + 1 - _scaleMin)));
        float scaleFactor = std::min(inputSize.width, inputSize.height) / (float) scaleSize;
        Point2i corner;
        Size2i cropSize(_innerSize.height * scaleFactor, _innerSize.width * scaleFactor);
        if (_aspectRatio > 100) {
            getRandomAspectRatio(cropSize);
        }
        randomCorner(inputSize - cropSize, &corner);
        cropBox->width = cropSize.width;
        cropBox->height = cropSize.height;
        cropBox->x = corner.x;
        cropBox->y = corner.y;
        return;
    }

    const Size2i &getSize() {
        return _innerSize;
    }

protected:
    unsigned int                _rngSeed;
    Size2i                      _innerSize;
    bool                        _center, _flip, _rgb;
    // Pixel scale to jitter at (image from which to crop will have short side in [scaleMin, Max])
    int                         _scaleMin, _scaleMax;
    int                         _contrastMin, _contrastMax;
    int                         _rotateMin, _rotateMax;
    int                         _aspectRatio;
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
