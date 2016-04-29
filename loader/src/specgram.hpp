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
    }

    void generate(RawMedia* raw, char* buf, int bufSize) {
        // TODO: get rid of this assumption
        assert(raw->sampleSize() == 2);
        assert(_timeSteps * _numFreqs == bufSize);
        int signalSize = raw->dataSize() / raw->sampleSize();
        if (signalSize > _maxSignalSize) {
            stringstream ss;
            ss << "Found example with duration larger than "
               << _maxDuration << " seconds\n";
            throw std::runtime_error(ss.str());
        }
        assert(signalSize >= _windowSize);
        Mat signal(1, signalSize, CV_16UC1, (ushort*) raw->getBuf(0));
        Mat input;
        signal.convertTo(input, CV_32FC1);
        assert(input.cols == signalSize);

        Mat image(_timeSteps, _numFreqs, CV_8UC1);
        int count = ((signalSize - _windowSize) / _stride) + 1;
        assert(count <= _timeSteps);
        // TODO: do FFT in batch mode instead of looping.
        for (int i = 0; i < count; i++) {
            int startCol = i * _stride;
            int endCol = startCol + _windowSize;
            assert(endCol <= input.cols);
            Mat slice = input(Range::all(), Range(startCol, endCol));
            // TODO: apply window
            Mat planes[] = {slice, Mat::zeros(slice.size(), CV_32FC1)};
            Mat compx;
            cv::merge(planes, 2, compx);

            cv::dft(compx, compx);
            compx = compx(Range::all(), Range(0, _numFreqs));

            cv::split(compx, planes);
            cv::magnitude(planes[0], planes[1], planes[0]);
            Mat mag = planes[0];

            cv::log(mag, mag);
            cv::normalize(mag, mag, 0, 1, CV_MINMAX);
            mag *= 255;

            Mat bytes;
            mag.convertTo(bytes, CV_8UC1);
            bytes.row(0).copyTo(image.row(i));
        }
        // Pad the rest with zeros.
        for (int i = count; i < _timeSteps; i++) {
            image.row(i) = cv::Scalar::all(0);
        }

        Mat result(_numFreqs, _timeSteps, CV_8UC1, buf);
        cv::transpose(image, result);
        cv::flip(result, result, 0);
    }

private:
    bool powerOfTwo(int num) {
        while (((num % 2) == 0) && (num > 1)) {
            num /= 2;
        }
        return (num == 1);
    }

private:
    int                         _maxDuration;
    // Window size and stride are in terms of samples.
    int                         _windowSize;
    int                         _stride;
    int                         _timeSteps;
    int                         _numFreqs;
    int                         _maxSignalSize;
};
