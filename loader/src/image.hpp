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
using cv::Scalar_;
using std::ofstream;
using std::vector;

typedef struct {
    Rect cropBox;
    int angle;
    bool flip;
    float colornoise[3];  //pixelwise random values
    float cbs[3];  // contrast, brightness, saturation
} AugParams;

// These are the eigenvectors of the pixelwise covariance matrix
float _CPCA[3][3] = {{0.39731118,  0.70119634, -0.59200296},
                    {-0.81698062, -0.02354167, -0.5761844},
                    {0.41795513, -0.71257945, -0.56351045}};
const Mat CPCA(3, 3, CV_32FC1, _CPCA);

// These are the square roots of the eigenvalues of the pixelwise covariance matrix
const Mat CSTD(3, 1, CV_32FC1, {19.72083305, 37.09388853, 121.78006099});

// This is the set of coefficients for converting BGR to grayscale
const Mat GSCL(3, 1, CV_32FC1, {0.114, 0.587, 0.299});

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
        if (_rotateMax < _rotateMin) {
            throw std::runtime_error("Max angle is less than min angle");
        }
        if (_rotateMax > 180) {
            throw std::runtime_error("Invalid max angle");
        }
        if (_rotateMin < -180) {
            throw std::runtime_error("Invalid min angle");
        }
        // This is just temporary until we break api and add parameter to control separately
        // setting allows us to get std of 0.1 if contrast max is 140 (i.e. 1.4)
        _colorNoiseStd = (_contrastMax - 100) / 400.0f;
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

    void getDistortionValues(cv::RNG &rng, const Size2i &inputSize, AugParams *agp) {
        // This function just gets the random distortion values without modifying the
        // image itself.  Useful if we need to reapply the same transformations over
        // again (e.g. for all frames of a video or for a corresponding target mask)

        // colornoise values
        // N.B. if _contrastMax == 100, then _colorNoiseStd will be 0.0
        for (int i=0; i<3; i++) {
            agp->colornoise[i] = rng.gaussian(_colorNoiseStd);
        }

        // contrast, brightness, saturation
        // N.B. all value ranges tied to _contrastMin and _contrastMax
        for (int i=0; i<3; i++) {
            agp->cbs[i] = rng.uniform(_contrastMin, _contrastMax) / 100.0f;
        }

        /**************************
        *  HORIZONTAL FLIP        *
        ***************************/
        agp->flip = _flip && rng(2) != 0  ? true : false;

        /**************************
        *  ROTATION ANGLE         *
        ***************************/
        agp->angle = rng.uniform(_rotateMin, _rotateMax);

        /**************************
        *  CROP BOX               *
        ***************************/
        float shortSide = std::min(inputSize.height, inputSize.width);
        if (_scaleMin == 0) {
            _scaleMin = shortSide;
            _scaleMax = shortSide;
        }

        if (_center) {
            agp->cropBox.width = shortSide * _width / (float) _scaleMin;
            agp->cropBox.height = shortSide * _height / (float) _scaleMin;
            agp->cropBox.x = (inputSize.width - agp->cropBox.width) / 2;
            agp->cropBox.y = (inputSize.height - agp->cropBox.height) / 2;
        } else {
            cv::Size2f oSize = inputSize;

            // This is a hack for backward compatibility.
            // Valid aspect ratio range ( > 100) will override side scaling behavior
            if (_aspectRatio == 0) {
                float scaleFactor = rng.uniform(_scaleMin, _scaleMax);
                agp->cropBox.width = shortSide * _width / scaleFactor;
                agp->cropBox.height = shortSide * _height / scaleFactor;
            } else {
                float mAR = (float) _aspectRatio / 100.0f;
                float nAR = rng.uniform(1.0f / mAR, mAR);
                float oAR = oSize.width / oSize.height;
                // between minscale pct% to 100% subject to aspect ratio limitation
                float maxScale = nAR > oAR ? oAR / nAR : nAR / oAR;
                float minScale = std::min((float) _scaleMin / 100.0f, maxScale);
                float tgtArea = rng.uniform(minScale, maxScale) * oSize.area();

                agp->cropBox.height = sqrt(tgtArea / nAR);
                agp->cropBox.width = agp->cropBox.height * nAR;
            }

            agp->cropBox.x = rng.uniform(0, inputSize.width - agp->cropBox.width);
            agp->cropBox.y = rng.uniform(0, inputSize.height - agp->cropBox.height);

        }
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
    float                       _colorNoiseStd;
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

    Mat image(1, jpgdata.size(), CV_8UC3, &jpgdata[0]);
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
friend class Video;
public:
    Image(ImageParams *params, ImageIngestParams* ingestParams, int id)
    : _params(params), _ingestParams(ingestParams), _rng(id) {
        assert(params->_mtype == IMAGE);
        assert((params->_channelCount == 1) || (params->_channelCount == 3));
        _innerSize = _params->getSize();
        _numPixels = _innerSize.area();
    }

    void transform(char* item, int itemSize, char* buf, int bufSize) {
        Mat decodedImage;
        decode(item, itemSize, &decodedImage);
        createRandomAugParams(decodedImage.size());
        transformDecodedImage(decodedImage, buf, bufSize);
    }

    void dump_agp() {
        int x = _augParams.cropBox.x;
        int y = _augParams.cropBox.y;
        int w = _augParams.cropBox.width;
        int h = _augParams.cropBox.height;
        printf("Cropbox: from (%d, %d) to (%d, %d), %dx%d\n", x, y, x+w, y+h, h, w);
        printf("Flip: %s\n", _augParams.flip ? "true" : "false");
        printf("Contrast/Brightness/Saturation: %.4f %.4f %.4f\n", _augParams.cbs[0],
                                                                   _augParams.cbs[1],
                                                                   _augParams.cbs[2]);
        printf("BGR Lighting: %.4f %.4f %.4f\n", _augParams.colornoise[0],
                                                 _augParams.colornoise[1],
                                                 _augParams.colornoise[2]);
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
            Mat image(1, itemSize, CV_8UC1, item);
            cv::imdecode(image, CV_LOAD_IMAGE_GRAYSCALE, dst);
        } else if (_params->_channelCount == 3) {
            Mat image(1, itemSize, CV_8UC3, item);
            cv::imdecode(image, CV_LOAD_IMAGE_COLOR, dst);
        } else {
            stringstream ss;
            ss << "Unsupported number of channels in image: " << _params->_channelCount;
            throw std::runtime_error(ss.str());
        }
    }

    void transformDecodedImage(const Mat& decodedImage, char* buf, int bufSize){
        Mat rotatedImage;
        rotate(decodedImage, rotatedImage, _augParams.angle);
        Mat croppedImage = rotatedImage(_augParams.cropBox);

        Mat resizedImage;

        // Perform photometric distortions in smaller spatial domain
        if (_augParams.cropBox.area() < _numPixels) {
            cbsjitter(croppedImage, _augParams.cbs);
            lighting(croppedImage, _augParams.colornoise);
            resize(croppedImage, resizedImage, _innerSize);
        } else {
            resize(croppedImage, resizedImage, _innerSize);
            cbsjitter(croppedImage, _augParams.cbs);
            lighting(croppedImage, _augParams.colornoise);
        }

        Mat *finalImage = &resizedImage;
        Mat flippedImage;
        if (_augParams.flip) {
            cv::flip(resizedImage, flippedImage, 1);
            finalImage = &flippedImage;
        }

        split(*finalImage, buf, bufSize);
    }

    void rotate(const Mat& input, Mat& output, int angle) {
        if (angle == 0) {
            output = input;
        } else {
            Point2i pt(input.cols / 2, input.rows / 2);
            Mat rot = cv::getRotationMatrix2D(pt, angle, 1.0);
            cv::warpAffine(input, output, rot, input.size());
        }
    }

    void resize(const Mat& input, Mat& output, const Size2i& size) {
        if (_innerSize == input.size()) {
            output = input;
        } else {
            int inter = input.size().area() < _numPixels ? CV_INTER_CUBIC : CV_INTER_AREA;
            cv::resize(input, output, _innerSize, 0, 0, inter);
        }
    }

    /*
    Implements colorspace noise perturbation as described in:
    Krizhevsky et. al., "ImageNet Classification with Deep Convolutional Neural Networks"
    Constructs a random coloring pixel that is uniformly added to every pixel of the image.
    pixelstd is filled with normally distributed values prior to calling this function.
    */
    void lighting(Mat& inout, float pixelstd[]) {
        // Skip transformations if given deterministic settings
        if (_params->_colorNoiseStd == 0.0) {
            return;
        }
        Mat alphas(3, 1, CV_32FC1, pixelstd);
        alphas = (CPCA * CSTD.mul(alphas));  // this is the random coloring pixel
        auto pixel = alphas.reshape(3, 1).at<Scalar_<float>>(0, 0);
        inout = (inout + pixel) / (1.0 + _params->_colorNoiseStd);
    }

    /*
    Implements contrast, brightness, and saturation jittering using the following definitions:
    Contrast: Add some multiple of the grayscale mean of the image.
    Brightness: Magnify the intensity of each pixel by cbs[1]
    Saturation: Add some multiple of the pixel's grayscale value to itself.
    cbs is filled with uniformly distributed values prior to calling this function
    */
    // adjusts contrast, brightness, and saturation according
    // to values in cbs[0], cbs[1], cbs[2], respectively
    void cbsjitter(Mat& inout, float cbs[]) {
        // Skip transformations if given deterministic settings
        if (_params->_contrastMin == _params->_contrastMax) {
            return;
        }

        /****************************
        *  BRIGHTNESS & SATURATION  *
        *****************************/
        Mat satmtx = cbs[1] * (cbs[2] * Mat::eye(3, 3, CV_32FC1) +
                                (1 - cbs[2]) * Mat::ones(3, 1, CV_32FC1) * GSCL.t());
        cv::transform(inout, inout, satmtx);

        /*************
        *  CONTRAST  *
        **************/
        Mat gray_mean;
        cv::cvtColor(Mat(1, 1, CV_32FC3, cv::mean(inout)), gray_mean, CV_BGR2GRAY);
        inout = cbs[0] * inout + (1 - cbs[0]) * gray_mean.at<Scalar_<float>>(0, 0);
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

    void createRandomAugParams(const Size2i& size) {
        _params->getDistortionValues(_rng, size, &_augParams);
    }

private:
    ImageParams*                _params;
    ImageIngestParams*          _ingestParams;
    Size2i                      _innerSize;
    cv::RNG                     _rng;
    int                         _numPixels;
    AugParams                   _augParams;
};
