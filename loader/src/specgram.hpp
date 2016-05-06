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

#include <sstream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using cv::Mat;
using cv::Range;
using cv::Size;
using std::stringstream;

class Specgram {
public:
    Specgram(SignalParams* params, int id)
    : _clipDuration(params->_clipDuration), _windowSize(params->_windowSize),
      _stride(params->_stride), _timeSteps(params->_timeSteps),
      _numFreqs(params->_numFreqs), _addNoise(params->_addNoise),
      _window(0), _rng(id) {
        static_assert(sizeof(short) == 2, "short is not 2 bytes");
        assert(_stride != 0);
        if (powerOfTwo(_windowSize) == false) {
            throw std::runtime_error("Window size must be a power of 2");
        }

        if (params->_resample == true) {
            throw std::runtime_error("Resampling not implemented yet");
        }

        if ((params->_timeScaleFactor != 1.0) || (params->_freqScaleFactor != 1.0)) {
            throw std::runtime_error("Scaling not implemented yet");
        }

        _maxSignalSize = params->_clipDuration * params->_samplingFreq / 1000;
        _buf = new char[4 *  _maxSignalSize];
        _image = new Mat(_timeSteps, _numFreqs, CV_8UC1);
        if (params->_windowType != 0) {
            _window = new Mat(1, _windowSize, CV_32FC1);
            createWindow(params->_windowType);
            hann(_windowSize - 1);
        }
        assert(params->_randomizeTimeScaleBy >= 0);
        assert(params->_randomizeTimeScaleBy < 100);
        _scaleBy = params->_randomizeTimeScaleBy / 100.0;
        _scaleMin = 1.0 - _scaleBy;
        _scaleMax = 1.0 + _scaleBy;
    }

    virtual ~Specgram() {
        delete _window;
        delete _image;
        delete[] _buf;
    }

    void generate(RawMedia* raw, char* buf, int bufSize) {
        // TODO: get rid of this assumption
        assert(raw->sampleSize() == 2);
        assert(_timeSteps * _numFreqs == bufSize);
        addNoise(raw);
        int rows = stridedSignal(raw);
        assert(rows <= _timeSteps);
        Mat signal(rows, _windowSize, CV_16SC1, (short*) _buf);
        Mat input;
        signal.convertTo(input, CV_32FC1);

        applyWindow(input);
        Mat planes[] = {input, Mat::zeros(input.size(), CV_32FC1)};
        Mat compx;
        cv::merge(planes, 2, compx);

        cv::dft(compx, compx, cv::DFT_ROWS);
        compx = compx(Range::all(), Range(0, _numFreqs));

        cv::split(compx, planes);
        cv::magnitude(planes[0], planes[1], planes[0]);
        Mat mag = planes[0];

        cv::normalize(mag, (*_image)(Range(0, rows), Range::all()), 0, 255,
                      CV_MINMAX, CV_8UC1);
        // Pad the rest with zeros.
        (*_image)(Range(rows, _image->rows), Range::all()) = cv::Scalar::all(0);

        // Rotate by 90 degrees.
        Mat result(_numFreqs, _timeSteps, CV_8UC1, buf);
        cv::transpose(*_image, result);
        cv::flip(result, result, 0);
        randomize(result);
    }

private:
    void randomize(Mat& img) {
        if (_scaleBy > 0) {
            float fx = _rng.uniform(_scaleMin, _scaleMax);
            resize(img, fx);
        }
    }

    void addNoise(RawMedia* raw) {
        if (_addNoise == false) {
            return;
        }
        const float noiseAmp = 100;
        int sampleCount = raw->dataSize() / raw->sampleSize();
        short* buf = (short*) raw->getBuf(0);
        for (int i = 0; i < sampleCount; i++) {
           buf[i] += short(noiseAmp * (float) _rng.gaussian(1.0));
        }
    }

    void resize(Mat& img, float fx) {
        Mat dst;
        int inter = (fx > 1.0) ? CV_INTER_CUBIC : CV_INTER_AREA;
        cv::resize(img, dst, Size(), fx, 1.0, inter);
        assert(img.rows == dst.rows);
        if (img.cols > dst.cols) {
            dst.copyTo(img(Range::all(), Range(0, dst.cols)));
            img(Range::all(), Range(dst.cols, img.cols)) = cv::Scalar::all(0);
        } else {
            dst(Range::all(), Range(0, img.cols)).copyTo(img);
        }
    }

    bool powerOfTwo(int num) {
        while (((num % 2) == 0) && (num > 1)) {
            num /= 2;
        }
        return (num == 1);
    }

    void none(int) {
    }

    void hann(int steps) {
        for (int i = 0; i <= steps; i++) {
            _window->at<float>(0, i) = 0.5 - 0.5 * cos((2.0 * PI * i) / steps);
        }
    }

    void blackman(int steps) {
        for (int i = 0; i <= steps; i++) {
            _window->at<float>(0, i) = 0.42 -
                                       0.5 * cos((2.0 * PI * i) / steps) +
                                       0.08 * cos(4.0 * PI * i / steps);
        }
    }

    void hamming(int steps) {
        for (int i = 0; i <= steps; i++) {
            _window->at<float>(0, i) = 0.54 - 0.46 * cos((2.0 * PI * i) / steps);
        }
    }

    void bartlett(int steps) {
        for (int i = 0; i <= steps; i++) {
            _window->at<float>(0, i) = 1.0 - 2.0 * fabs(i - steps / 2.0) / steps;
        }
    }

    void createWindow(int windowType) {
        typedef void(Specgram::*winFunc)(int);
        winFunc funcs[] = {&Specgram::none, &Specgram::hann,
                           &Specgram::blackman, &Specgram::hamming,
                           &Specgram::bartlett};
        assert(windowType >= 0);
        if (windowType >= (int) (sizeof(funcs) / sizeof(funcs[0]))) {
            throw std::runtime_error("Unsupported window function");
        }
        int steps = _windowSize - 1;
        assert(steps > 0);
        (this->*(funcs[windowType]))(steps);
    }

    void applyWindow(Mat& signal) {
        if (_window == 0) {
            return;
        }
        for (int i = 0; i < signal.rows; i++) {
            signal.row(i) = signal.row(i).mul((*_window));
        }
    }

    int stridedSignal(RawMedia* raw) {
        int signalSize = raw->dataSize() / raw->sampleSize();
        if (signalSize > _maxSignalSize) {
            signalSize = _maxSignalSize;
        }
        assert(signalSize >= _windowSize);
        int count = ((signalSize - _windowSize) / _stride) + 1;
        assert(count <= _timeSteps);
        char* src = raw->getBuf(0);
        char* dst = _buf;
        int windowSizeInBytes = _windowSize * raw->sampleSize();
        int strideInBytes = _stride * raw->sampleSize();
        for (int i = 0; i < count; i++) {
            memcpy(dst , src, windowSizeInBytes);
            dst += windowSizeInBytes;
            src += strideInBytes;
        }

        return count;
    }

private:
    // Maximum duration in milliseconds.
    int                         _clipDuration;
    // Window size and stride are in terms of samples.
    int                         _windowSize;
    int                         _stride;
    int                         _timeSteps;
    int                         _numFreqs;
    int                         _maxSignalSize;
    float                       _scaleBy;
    float                       _scaleMin;
    float                       _scaleMax;
    bool                        _addNoise;
    char*                       _buf;
    Mat*                        _image;
    Mat*                        _window;
    cv::RNG                     _rng;
    constexpr static double     PI = 3.14159265358979323846;
};
