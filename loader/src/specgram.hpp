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
using std::stringstream;

class Specgram {
public:
    Specgram(SignalParams* params)
    : _maxDuration(params->_maxDuration), _windowSize(params->_windowSize),
      _stride(params->_stride), _timeSteps(params->_timeSteps),
      _numFreqs(params->_numFreqs) {
        static_assert(sizeof(ushort) == 2, "ushort is not 2 bytes");
        if (powerOfTwo(_windowSize) == false) {
            throw std::runtime_error("Window size must be a power of 2");
        }

        _maxSignalSize = params->_maxDuration * params->_samplingFreq;
        _buf = new char[4 *  _maxSignalSize];
        _image = new Mat(_timeSteps, _numFreqs, CV_8UC1);
        _window = new Mat(1, _windowSize, CV_32FC1);
        // TODO: pick the right window function
        hann(_windowSize - 1);
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
        int rows = stridedSignal(raw);
        assert(rows <= _timeSteps);
        Mat signal(rows, _windowSize, CV_16UC1, (ushort*) _buf);
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

        cv::log(mag, mag);
        for (int i = 0; i < rows; i++) {
            cv::normalize(mag.row(i), _image->row(i), 0, 255, CV_MINMAX, CV_8UC1);
        }

        // Pad the rest with zeros.
        (*_image)(Range(rows, _image->rows), Range::all()) = cv::Scalar::all(0);

        // Rotate by 90 degrees.
        Mat result(_numFreqs, _timeSteps, CV_8UC1, buf);
        cv::transpose(*_image, result);
        cv::flip(result, result, 0);
    }

private:
    bool powerOfTwo(int num) {
        while (((num % 2) == 0) && (num > 1)) {
            num /= 2;
        }
        return (num == 1);
    }

    void hann(int steps) {
        for (int i = 0; i <= steps; i++) {
            _window->at<float>(0, i) = 0.5 - 0.5 * cos((2.0 * PI * i) / steps);
        }
    }

    void applyWindow(Mat& signal) {
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
    int                         _maxDuration;
    // Window size and stride are in terms of samples.
    int                         _windowSize;
    int                         _stride;
    int                         _timeSteps;
    int                         _numFreqs;
    int                         _maxSignalSize;
    char*                       _buf;
    Mat*                        _image;
    Mat*                        _window;
    constexpr static double     PI = 3.14159265358979323846;
};
