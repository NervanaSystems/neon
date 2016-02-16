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

#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "media.hpp"

using cv::Mat;
using cv::Rect;
using cv::Point2i;
using cv::Size2i;
using std::ofstream;
using std::vector;

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

class Image: public Media {
public:
    Image(ImageParams *params) : _params(params), _rngSeed(0) {
        assert(params->_mtype == IMAGE);
    }

    void encode(char* item, int itemSize, char* buf, int bufSize) {

    }

    void decode(char* item, int itemSize, char* buf, int bufSize) {
        Mat image = Mat(1, itemSize, CV_8UC3, item);
        Mat decodedImage = cv::imdecode(image, CV_LOAD_IMAGE_COLOR);
        Rect cropBox;
        _params->getRandomCrop(_rngSeed, decodedImage.size(), &cropBox);
        auto cropArea = cropBox.area();
        auto innerSize = _params->getSize();

        Mat croppedImage = decodedImage(cropBox);
        // This would be more efficient, but we should allocate separate bufs for each thread
        // Mat resizedImage = Mat(innerSize, CV_8UC3, _scratchbuf);
        Mat resizedImage;
        if (innerSize.width == cropBox.width && innerSize.height == cropBox.height) {
            resizedImage = croppedImage;
        } else {
            int interp_method = cropArea < innerSize.area() ? CV_INTER_AREA : CV_INTER_CUBIC;
            cv::resize(croppedImage, resizedImage, innerSize, 0, 0, interp_method);
        }
        Mat flippedImage;
        Mat *finalImage;

        if (_params->doRandomFlip(_rngSeed)) {
            cv::flip(resizedImage, flippedImage, 1);
            finalImage = &flippedImage;
        } else {
            finalImage = &resizedImage;
        }
        Mat newImage;
        float alpha = _params->getRandomContrast(_rngSeed);
        if (alpha) {
            finalImage->convertTo(newImage, -1, alpha);
            finalImage = &newImage;
        }

        Mat ch_b(innerSize, CV_8U, buf + innerSize.area()*0);
        Mat ch_g(innerSize, CV_8U, buf + innerSize.area()*1);
        Mat ch_r(innerSize, CV_8U, buf + innerSize.area()*2);

        Mat channels[3] = {ch_b, ch_g, ch_r};
        cv::split(*finalImage, channels);
    }

    void modify(char* item, int itemSize, char* buf, int bufSize) {

    }

    void save_binary(char *filn, char* item, int itemSize, char* buf) {
        ofstream file(filn, ofstream::out | ofstream::binary);
        file.write((char*)(&itemSize), sizeof(int));
        file.write((char*)item, itemSize);
        printf("wrote %s\n", filn);
    }

private:
    ImageParams*                _params;
    unsigned int                _rngSeed;
};
