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

#pragma once

struct AugParams {
    Rect cropBox;
    int angle;
    float alpha;
    bool flip;
};

class ImageParams : public MediaParams {
public:
    ImageParams(int channelCount, int height, int width,
                bool center, bool flip,
                int scaleMin, int scaleMax,
                int contrastMin, int contrastMax,
                int rotateMin, int rotateMax,
                int aspectRatio, bool subtractMean,
                int redMean, int greenMean, int blueMean,
                int grayMean)
    : MediaParams(IMAGE),
      _channelCount(channelCount),
      _height(height), _width(width),
      _center(center), _flip(flip),
      _scaleMin(scaleMin), _scaleMax(scaleMax),
      _contrastMin(contrastMin), _contrastMax(contrastMax),
      _rotateMin(rotateMin), _rotateMax(rotateMax),
      _aspectRatio(aspectRatio), _subtractMean(true),
      _redMean(redMean), _greenMean(greenMean), _blueMean(blueMean),
      _grayMean(grayMean) {
    }

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
        printf("aspect ratio %d\n", _aspectRatio);
    }

    bool doRandomFlip(unsigned int& seed) {
        return _flip && (rand_r(&seed) % 2 == 0);
    }

    void getRandomCorner(unsigned int& seed, const Size2i &border,
                         Point2i* point) {
        if (_center) {
            point->x = border.width / 2;
            point->y = border.height / 2;
        } else {
            point->x = rand_r(&seed) % (border.width + 1);
            point->y = rand_r(&seed) % (border.height + 1);
        }
    }

    float getRandomContrast(unsigned int& seed) {
        if (_contrastMin == _contrastMax) {
            return 0;
        }
        return (_contrastMin +
                (rand_r(&seed) % (_contrastMax - _contrastMin))) / 100.0;
    }

    // adjust the square cropSize to be an inner rectangle
    void getRandomAspectRatio(unsigned int& seed, Size2i &cropSize) {
        int ratio = (101 + (rand_r(&seed) % (_aspectRatio - 100)));
        float ratio_f = 100.0 / (float) ratio;
        int orientation = rand_r(&(seed)) % 2;
        if (orientation) {
            cropSize.height *= ratio_f;
        } else {
            cropSize.width *= ratio_f;
        }
    }

    void getRandomAngle(unsigned int& seed, int& angle) {
        if (_rotateMax == _rotateMin) {
            angle = _rotateMax;
            return;
        }

        if (_rotateMax < _rotateMin) {
            throw std::runtime_error("Max angle is less than min angle");
        }
        if (_rotateMax > 180) {
            throw std::runtime_error("Invalid max angle");
        }
        if (_rotateMin < -180) {
            throw std::runtime_error("Invalid min angle");
        }
        angle = (rand_r(&seed) % (_rotateMax + 1 - _rotateMin)) + _rotateMin;
    }

    void getRandomCrop(unsigned int& seed, const Size2i &inputSize,
                       Rect* cropBox) {
        if (_scaleMin == 0) {
            // Use the entire squashed image (Caffe style evaluation)
            cropBox->x = cropBox->y = 0;
            cropBox->width = inputSize.width;
            cropBox->height = inputSize.height;
            return;
        }

        int scaleSize = (_scaleMin +
                         (rand_r(&seed) % (_scaleMax + 1 - _scaleMin)));
        float scaleFactor = std::min(inputSize.width, inputSize.height) /
                            (float) scaleSize;
        Size2i cropSize(_width * scaleFactor, _height * scaleFactor);
        if (_aspectRatio > 100) {
            getRandomAspectRatio(seed, cropSize);
        }
        Point2i corner;
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
    int                         _aspectRatio;
    bool                        _subtractMean;
    int                         _redMean;
    int                         _greenMean;
    int                         _blueMean;
    int                         _grayMean;
};

class ImageIngestParams : public MediaParams {
public:
    ImageIngestParams(bool resizeAtIngest, bool lossyEncoding,
                      int sideMin, int sideMax)
    : MediaParams(IMAGE),
      _resizeAtIngest(resizeAtIngest), _lossyEncoding(lossyEncoding),
      _sideMin(sideMin), _sideMax(sideMax) {}

public:
    bool                        _resizeAtIngest;
    bool                        _lossyEncoding;
    // Minimum value of the short side
    int                         _sideMin;
    // Maximum value of the short side
    int                         _sideMax;

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
    return;
}

class Image: public Media {
public:
    Image(ImageParams *params, ImageIngestParams* ingestParams)
    : _params(params), _ingestParams(ingestParams), _rngSeed(0) {
        assert(params->_mtype == IMAGE);
        assert((params->_channelCount == 1) || (params->_channelCount == 3));
    }

    void transform(char* item, int itemSize, char* buf, int bufSize) {
        Mat decodedImage;
        decode(item, itemSize, &decodedImage);

        AugParams augParams = getRandomAugParams(decodedImage);
        transformDecodedImage(decodedImage, buf, bufSize, &augParams);
    }

    AugParams getRandomAugParams(Mat decodedImage) {
        AugParams augParams;
        _params->getRandomCrop(_rngSeed, decodedImage.size(), &(augParams.cropBox));
        _params->getRandomAngle(_rngSeed, augParams.angle);
        augParams.alpha = _params->getRandomContrast(_rngSeed);
        augParams.flip = _params->doRandomFlip(_rngSeed);
        return augParams;
    }

    void transformDecodedImage(Mat decodedImage, char* buf, int bufSize, AugParams* augParams){
        Mat rotatedImage;
        if (augParams->angle == 0) {
            rotatedImage = decodedImage;
        } else {
            rotate(decodedImage, rotatedImage, augParams->angle);
        }

        Mat croppedImage = rotatedImage(augParams->cropBox);
        auto innerSize = _params->getSize();
        Mat resizedImage;
        if (innerSize.width == augParams->cropBox.width && innerSize.height == augParams->cropBox.height) {
            resizedImage = croppedImage;
        } else {
            resize(croppedImage, resizedImage, innerSize);
        }
        Mat flippedImage;
        Mat *finalImage;

        if (augParams->flip) {
            cv::flip(resizedImage, flippedImage, 1);
            finalImage = &flippedImage;
        } else {
            finalImage = &resizedImage;
        }
        Mat newImage;
        if (augParams->alpha) {
            finalImage->convertTo(newImage, -1, augParams->alpha);
            finalImage = &newImage;
        }

        split(*finalImage, buf, bufSize);
    }

    void ingest(char** dataBuf, int* dataBufLen, int* dataLen) {
        if (_ingestParams == 0) {
            return;
        }
        if (_ingestParams->_resizeAtIngest == false) {
            return;
        }
        if ((_ingestParams->_sideMin <= 0) && (_ingestParams->_sideMax <= 0)) {
            throw std::runtime_error("Invalid ingest parameters. Cannot resize.");
        }
        if (_ingestParams->_sideMin > _ingestParams->_sideMax) {
            throw std::runtime_error("Invalid ingest parameters. Cannot resize.");
        }

        // Decode
        Mat decodedImage;
        decode(*dataBuf, *dataLen, &decodedImage);

        // Resize
        int width = decodedImage.cols;
        int height = decodedImage.rows;
        int shortSide = std::min(width, height);
        if ((shortSide >= _ingestParams->_sideMin) &&
            (shortSide <= _ingestParams->_sideMax)) {
            return;
        }

        if (width <= height) {
            if (width < _ingestParams->_sideMin) {
                height = height * _ingestParams->_sideMin / width;
                width = _ingestParams->_sideMin;
            } else if (width > _ingestParams->_sideMax) {
                height = height * _ingestParams->_sideMax / width;
                width = _ingestParams->_sideMax;
            }
        } else {
            if (height < _ingestParams->_sideMin) {
                width = width * _ingestParams->_sideMin / height;
                height = _ingestParams->_sideMin;
            } else if (height > _ingestParams->_sideMax) {
                width = width * _ingestParams->_sideMax / height;
                height = _ingestParams->_sideMax;
            }
        }

        Size2i size(width, height);
        Mat resizedImage;
        resize(decodedImage, resizedImage, size);

        // Re-encode
        vector<int> param;
        string ext;
        if (_ingestParams->_lossyEncoding == true) {
            param = {CV_IMWRITE_JPEG_QUALITY, 90};
            ext = ".jpg";
        } else {
            param = {CV_IMWRITE_PNG_COMPRESSION, 9};
            ext = ".png";
        }
        vector<uchar> output;
        cv::imencode(ext, resizedImage, output, param);

        if (*dataBufLen < (int) output.size()) {
            delete[] *dataBuf;
            *dataBuf = new char[output.size()];
            *dataBufLen = output.size();
        }

        std::copy(output.begin(), output.end(), *dataBuf);
        *dataLen = output.size();
    }

    void save_binary(char *filn, char* item, int itemSize, char* buf) {
        ofstream file(filn, ofstream::out | ofstream::binary);
        file.write((char*)(&itemSize), sizeof(int));
        file.write((char*)item, itemSize);
        printf("wrote %s\n", filn);
    }

private:
    void decode(char* item, int itemSize, Mat* dst) {
        if (_params->_channelCount == 1) {
            Mat image = Mat(1, itemSize, CV_8UC1, item);
            cv::imdecode(image, CV_LOAD_IMAGE_GRAYSCALE, dst);
        } else if (_params->_channelCount == 3) {
            Mat image = Mat(1, itemSize, CV_8UC3, item);
            cv::imdecode(image, CV_LOAD_IMAGE_COLOR, dst);
        } else {
            stringstream ss;
            ss << "Unsupported number of channels in image: " << _params->_channelCount;
            throw std::runtime_error(ss.str());
        }
    }

    void rotate(const Mat& input, Mat& output, int angle) {
        Point2i pt(input.cols / 2, input.rows / 2);
        Mat rot = cv::getRotationMatrix2D(pt, angle, 1.0);
        cv::warpAffine(input, output, rot, input.size());
    }

    void resize(const Mat& input, Mat& output, const Size2i& size) {
        int inter = input.size().area() < size.area() ?
                    CV_INTER_CUBIC : CV_INTER_AREA;
        cv::resize(input, output, size, 0, 0, inter);
    }

    void split(Mat& img, char* buf, int bufSize) {
        Size2i size = img.size();
        if (img.channels() * img.total() > (uint) bufSize) {
            throw std::runtime_error("Decode failed - buffer too small");
        }
        if (img.channels() == 1) {
            memcpy(buf, img.data, img.total());
            return;
        }

        // Split into separate channels
        Mat blue(size, CV_8U, buf);
        Mat green(size, CV_8U, buf + size.area());
        Mat red(size, CV_8U, buf + 2 * size.area());

        Mat channels[3] = {blue, green, red};
        cv::split(img, channels);
    }

    void subtractMean(Mat& red, Mat& green, Mat& blue) {
        // Deprecated
        assert(0);
        if (_params->_subtractMean == false) {
            return;
        }
        red -= _params->_redMean;
        green -= _params->_greenMean;
        blue -= _params->_blueMean;
    }

    void subtractMean(Mat& gray) {
        // Deprecated
        assert(0);
        if (_params->_subtractMean == false) {
            return;
        }
        gray -= _params->_grayMean;
    }

private:
    ImageParams*                _params;
    ImageIngestParams*          _ingestParams;
    unsigned int                _rngSeed;
};
