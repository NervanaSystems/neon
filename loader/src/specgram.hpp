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

#include <cmath>
#include <vector>

using cv::Mat;
using cv::Range;
using cv::Size;
using std::stringstream;
using std::vector;

class Specgram {
public:
    Specgram(SignalParams* params, int id)
    : _clipDuration(params->_clipDuration), _windowSize(params->_windowSize),
      _stride(params->_stride), _timeSteps(params->_width),
      _numFreqs(params->_height), _samplingFreq(params->_samplingFreq),
      _ncepstra(params->_ncepstra),
      _nfilts(params->_nfilts),
      _addNoise(params->_addNoise),
      _window(0), _rng(id) {
        static_assert(sizeof(short) == 2, "short is not 2 bytes");
        assert(_stride != 0);
        if (powerOfTwo(_windowSize) == false) {
            throw std::runtime_error("Window size must be a power of 2");
        }

        _maxSignalSize = params->_clipDuration * params->_samplingFreq / 1000;
        _buf = new char[4 *  _maxSignalSize];
        if (params->_windowType != 0) {
            _window = new Mat(1, _windowSize, CV_32FC1);
            createWindow(params->_windowType);
        }
        assert(params->_randomScalePercent >= 0);
        assert(params->_randomScalePercent < 100);
        _scaleBy = params->_randomScalePercent / 100.0;
        _scaleMin = 1.0 - _scaleBy;
        _scaleMax = 1.0 + _scaleBy;
    }

    virtual ~Specgram() {
        delete _window;
        delete[] _buf;
    }

    int generate(RawMedia* raw, char* buf, int bufSize) {
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
        Mat mag;

        // Rotate by 90 degrees.
        cv::transpose(planes[0], mag);
        cv::flip(mag, mag, 0);

        cv::normalize(mag, mag, 0, 255, CV_MINMAX, CV_8UC1);
        Mat result(_numFreqs, _timeSteps, CV_8UC1, buf);
        mag.copyTo(result(Range::all(), Range(0, mag.cols)));

        // Pad the rest with zeros.
        result(Range::all(), Range(mag.cols, result.cols)) = cv::Scalar::all(0);

        randomize(result);
        // Return the percentage of valid columns.
        return mag.cols * 100 / result.cols;
    }

    int generate_mfccs(RawMedia* raw, char* buf, int bufSize) {
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


        cv::split(compx, planes);
        cv::magnitude(planes[0], planes[1], planes[0]);

        Mat feats = extract_mfccs(planes[0], _ncepstra, _nfilts, _samplingFreq);

        // From this point on, match what was done with spectrograms
        // TODO: do away with matching spectrograms
        Mat feats_tr;
        // Rotate by 90 degrees to match what is done with spectrograms
        cv::transpose(feats, feats_tr);
        cv::flip(feats_tr, feats_tr, 0);

        cv::normalize(feats_tr, feats_tr, 0, 255, CV_MINMAX, CV_8UC1);

        Mat result(feats_tr.rows, _timeSteps, CV_8UC1, buf);
        feats_tr.copyTo(result(Range::all(), Range(0, feats_tr.cols)));

        // Pad the rest with zeros.
        result(Range::all(), Range(feats_tr.cols, result.cols)) = cv::Scalar::all(0);

        // Return the percentage of valid columns.
        return feats_tr.cols * 100 / result.cols;
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

    double hz_to_mel(double freq_in_hz) {
        return 2595 * std::log10(1 + freq_in_hz/700.0);
    }

    double mel_to_hz(double freq_in_mels) {
        return 700 * (std::pow(10, freq_in_mels/2595.0)-1);
    }

    vector<double> linspace(double a, double b, int n) {
        vector<double> interval;
        double delta = (b-a)/(n-1);
        while (a <= b) {
            interval.push_back(a);
            a += delta;
        }
        return interval;
    }

    Mat get_filterbank(int filts, int ffts, double sampling_rate) {
        double minfreq = 0.0;
        double maxfreq = sampling_rate / 2.0;
        double minmelfreq = hz_to_mel(minfreq);
        double maxmelfreq = hz_to_mel(maxfreq);
        vector<double> melinterval = linspace(minmelfreq, maxmelfreq, filts + 2);
        vector<int> bins;
        for (int k=0; k<filts+2; ++k) {
            bins.push_back(std::floor((1+ffts)*mel_to_hz(melinterval[k])/sampling_rate));
        }

        Mat fbank = Mat::zeros(filts, 1 + ffts / 2, CV_32F);
        for (int j=0; j<filts; ++j) {
            for (int i=bins[j]; i<bins[j+1]; ++i) {
                fbank.at<float>(j, i) = (i - bins[j]) / (1.0*(bins[j + 1] - bins[j]));
            }
            for (int i=bins[j+1]; i<bins[j+2]; ++i) {
                fbank.at<float>(j, i) = (bins[j+1]-i) / (1.0*(bins[j + 2] - bins[j+1]));
            }
        }
        return fbank;
    }

    Mat extract_mfccs(Mat& spectrogram, int cepstra, int filts, double sampling_rate) {
        int ffts = spectrogram.cols;
        Mat powspec = spectrogram.mul(spectrogram);
        Mat fbank;
        transpose(get_filterbank(filts, 2*(ffts-1), sampling_rate), fbank);
        Mat feats = powspec*fbank;
        log(feats, feats);
        if (feats.rows %2 !=0) {
            Mat pad_feats = Mat::zeros(1+feats.rows, feats.cols, CV_32FC1);
            feats.copyTo(pad_feats(Range(0, feats.rows), Range::all()));
            dct(pad_feats, pad_feats);
            feats = pad_feats(Range(0, feats.rows), Range::all());
        } else {
            dct(feats, feats);
        }
        int ncepstra = cepstra;
        if (filts < ncepstra) {
            ncepstra = filts;
            return feats;
        }
       Mat subfeats = feats(Range::all(), Range(0, ncepstra));
       return subfeats;
    }

private:
    // Maximum duration in milliseconds.
    int                         _clipDuration;
    // Window size and stride are in terms of samples.
    int                         _windowSize;
    int                         _stride;
    int                         _timeSteps;
    int                         _numFreqs;
    int                         _samplingFreq;
    int                         _maxSignalSize;
    int                         _ncepstra;
    int                         _nfilts;
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
