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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef uint8_t uchar;

using std::ofstream;
using cv::Mat;
using cv::Rect;
using cv::Point2i;
using cv::Size2i;

class Decoder {
public:
    virtual ~Decoder() {};
    virtual void decode(uchar* item, int itemSize, uchar* buf) = 0;
};

class ImageDecoder : public Decoder {
public:
    ImageDecoder(int innerSize, bool augment)
    : _rngSeed(0), _innerSize(innerSize), _augment(augment) {
    }

    void save(int i, uchar* item, int itemSize, uchar* buf) {
        char filn[256];
        sprintf(filn, "imgs/file%d.jpg", i);
        ofstream file(filn, ofstream::out | ofstream::binary);
        file.write((char*)item, itemSize);
        printf("wrote %s\n", filn);
    }

    void randomCorner(Point2i* point, int border) {
        point->x = rand_r(&_rngSeed) % (border + 1);
        point->y = rand_r(&_rngSeed) % (border + 1);
    }

    void decode(uchar* item, int itemSize, uchar* buf) {
        Mat image = Mat(1, itemSize, CV_8UC3, item);
        Mat decodedImage = cv::imdecode(image, CV_LOAD_IMAGE_COLOR);
        Point2i corner;
        Size2i size(_innerSize, _innerSize);

        int outerSize = decodedImage.size().height;
        assert(outerSize == decodedImage.size().width);
        assert(outerSize >= _innerSize);
        int border = outerSize - _innerSize;
        if (_augment == true) {
            randomCorner(&corner, border);
        } else {
            corner.x = corner.y = border / 2;
        }

        Mat croppedImage = decodedImage(Rect(corner, size));
        Mat flippedImage;
        Mat* finalImage;
        if (_augment && (rand_r(&(_rngSeed)) % 2 == 0)) {
            cv::flip(croppedImage, flippedImage, 1);
            finalImage = &flippedImage;
        } else {
            finalImage = &croppedImage;
        }

        auto channelSize = _innerSize * _innerSize;

        Mat red(_innerSize, _innerSize, CV_8U, buf + channelSize*0);
        Mat green(_innerSize, _innerSize, CV_8U, buf + channelSize*1);
        Mat blue(_innerSize, _innerSize, CV_8U, buf + channelSize*2);

        Mat channels[3] = {red, green, blue};
        cv::split(*finalImage, channels);
    }

private:
    unsigned int                _rngSeed;
    int                         _innerSize;
    bool                        _augment;
};
