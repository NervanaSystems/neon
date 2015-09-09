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

#include <stdlib.h>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <sstream> //for filename on oss
#include <map>
#include <vector>
#include <deque>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thread> // for threading

#define MAXTHREADS 24
using namespace cv;
using namespace std;

typedef deque<int> LabelList;
typedef vector<LabelList> LabelDict;
typedef vector<uchar> JpegStr;
typedef deque<vector<uchar>> JpegList;
typedef vector<Point2i> PtVect;

string load_string(ifstream& ifs)
{
    long int length;
    string result;
    ifs.read(reinterpret_cast<char *> (&length), sizeof(long int));
    result.resize(length);
    ifs.read(&result[0], length);
    return result;
}

template<typename T> T load_val(ifstream& ifs)
{
    T result;
    ifs.read(reinterpret_cast<char *> (&result), sizeof(int));
    return result;
}

template<typename T> vector<T> load_vect(ifstream& ifs, uint numel)
{
    vector<T> result(numel);
    ifs.read(reinterpret_cast<char *> (&result[0]), numel * sizeof(T));
    return result;
}

class ImagesetWorker {
 public:
    int _img_size, _inner_size, _border;
    int _npixels_in, _npixels_out, _inner_pixels;
    bool _center, _flip, _multiview;
    int _minibatch_size;
    int _cur_batch;
    char *_filename;
    int _macro_start;
    uint _num_data, _num_queued;
    int _colormode;
    uint _channels;
    uint _rseed;
    PtVect _corners;
    Size2i _length;
    ImagesetWorker(int img_size, int inner_size, bool center, bool flip, bool rgb, bool multiview,
                 int minibatch_size, char *filename, int macro_start, uint num_data);

    int reset();
    int read_from_flat_binary();
    int process_next_minibatch(unsigned char* outdata);
    void random_corner(Point2i *p);
    void decode_img_list(uchar* pixbuffer, uint start, uint end);

    virtual ~ImagesetWorker();
    LabelDict _labels;
    vector<string> _label_names;
    JpegList imgs;
    int _nthreads;
    int _num_imgs_per_thread;
};

ImagesetWorker::ImagesetWorker(int img_size, int inner_size, bool center,
                           bool flip, bool rgb, bool multiview, int minibatch_size,
                           char *filename, int macro_start, uint num_data)
    : _img_size(img_size), _inner_size(inner_size), _center(center),
      _flip(flip), _multiview(multiview), _minibatch_size(minibatch_size),
      _filename(filename), _macro_start(macro_start), _num_data(num_data)
{
    _cur_batch = _macro_start;
    _channels = rgb ? 3 : 1;
    _colormode = rgb ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
    _npixels_in = img_size * img_size * _channels;
    _npixels_out = inner_size * inner_size * _channels;
    _inner_pixels = _inner_size * _inner_size;
    _border = _img_size - _inner_size;
    _num_queued = 0;
    _rseed = time(0);
    _length = Size2i(_inner_size, _inner_size);
    _corners.push_back(Point2i(_border / 2, _border / 2)); // center
    _corners.push_back(Point2i(0, 0));                   // upper left
    _corners.push_back(Point2i(_border-1, 0));          // upper right
    _corners.push_back(Point2i(0, _border-1));          // lower left
    _corners.push_back(Point2i(_border-1, _border-1)); // lower right
    _nthreads = min((int) thread::hardware_concurrency(), MAXTHREADS);
    _num_imgs_per_thread = (_minibatch_size + _nthreads - 1) / _nthreads;
}

ImagesetWorker::~ImagesetWorker() {
}

void ImagesetWorker::random_corner(Point2i *p) {
    p->x = rand_r(&(_rseed)) % (_border + 1);
    p->y = rand_r(&(_rseed)) % (_border + 1);
}

int ImagesetWorker::reset()
{
    // Resets back to initial state
    _cur_batch = _macro_start;
    _num_queued = 0;
    imgs.clear();
    for (unsigned int i=0; i<_labels.size(); ++i)
        _labels[i].clear();
    read_from_flat_binary();
    return 0;
}

int ImagesetWorker::read_from_flat_binary()
{
    // Load jpg strings and labels from flat binary batch files
    // Use deque to hold jpg strings and labels
    // binary file is formatted:
    // uint num_images, uint num_labels
    //   label_string1
    //   uint label (x num_images)
    //   label_string2
    //   uint label (x num_images)
    //   ...
    //   label_string(num_labels)
    //   uint label (x num_images)
    // jpeg_strings x num_images


    stringstream ss;
    ss << _filename << _cur_batch;
    ifstream bfile (ss.str(), ifstream::binary);
    uint num_to_get = _num_data - _num_queued;

    if (bfile.is_open()) {
        uint num_images = load_val<uint>(bfile);
        uint num_labels = load_val<uint>(bfile);
        num_to_get = min(num_to_get, num_images);

        for (unsigned int i=0; i<num_labels; ++i) {
            string lbl = load_string(bfile);
            if (_label_names.size() < num_labels)
                _label_names.push_back(lbl);
            if (_labels.size() < num_labels) {
                LabelList curList(0);
                _labels.push_back(curList);
            }
            for (uint j=0; j<num_to_get; ++j) {
                _labels[i].push_back(load_val<uint>(bfile));
            }
            // Skip the rest
            for (uint j=num_to_get; j<num_images; ++j) {
                load_val<uint>(bfile);
            }
        }
        // Now grab the img strings from the file
        for (uint i=0; i<num_to_get; ++i) {
            uint imsize = load_val<uint>(bfile);
            imgs.push_back(load_vect<uchar>(bfile, imsize));
        }
        bfile.close();
        _num_queued += num_to_get;
        if (_num_queued >= _num_data) {
            _cur_batch = _macro_start;
            _num_queued = 0;
        } else {
            _cur_batch++;
        }

        return (int) num_images;
    }
    else {
        printf("Error opening file %s\n", ss.str().c_str());
    }
    return -1;
}

int ImagesetWorker::process_next_minibatch(unsigned char* outdata)
{
    // Do we need to add more imgs to our list?
    // (If we care about 40 ms, we can make this asynchronous too)
    if (imgs.size() < (unsigned int) _minibatch_size) {
        read_from_flat_binary();
    }
    thread t[MAXTHREADS];
    for (int i=0; i<_nthreads; i++) {
        int idx1 = i * _num_imgs_per_thread;
        int idx2 = min(idx1 + _num_imgs_per_thread, (int) _minibatch_size);
        t[i] = thread(&ImagesetWorker::decode_img_list, this, outdata, (uint) idx1, (uint) idx2);
    }

    // Write out labels and erase them from our queue
    int *label_ptr = (int *) (outdata + _minibatch_size * _npixels_out * sizeof(uchar));
    std::copy(_labels[0].begin(), _labels[0].begin() + _minibatch_size, label_ptr);
    _labels[0].erase(_labels[0].begin(), _labels[0].begin() + _minibatch_size);

    // Finish up the decoding
    for (int i=0; i<_nthreads; i++)
        t[i].join();

    // Now remove the images we've decoded from the beginning of the queue
    imgs.erase(imgs.begin(), imgs.begin() + _minibatch_size);

    return 0;
}

void ImagesetWorker::decode_img_list(uchar* pixbuffer, uint start, uint end)
{
    /*Args explained
        pixbuffer: this points right into the buffer that we want to fill
    */
    //pixbuffer is the start of where we will write the image
    // pixels out is the total number of pixels to write out
    uchar *data = pixbuffer + start * _npixels_out * sizeof(uchar);
    for (unsigned int i=start; i<end; i++)
    {
        Mat decodedImage = imdecode(imgs[i], _colormode);

        Point2i corner(_corners[0]);

        //pick random corner if not centering
        if (!_center)
            random_corner(&corner);

        //crop starting at the given corner
        Mat croppedImage = decodedImage(Rect(corner, _length));

        //optionally flip the image
        if (_flip && (rand_r(&(_rseed)) % 2 == 0))
            flip(croppedImage, croppedImage, 1);

        //split the color channels before writing image to memory
        auto channel_size = (_npixels_out * sizeof(uchar) ) / 3;

        //for each of R,G,B, declare a uint8 channel
        //instantiate it with a pointer to memory in our ouput buffer
        Mat R(_inner_size, _inner_size, CV_8U, data + channel_size*0);
        Mat G(_inner_size, _inner_size, CV_8U, data + channel_size*1);
        Mat B(_inner_size, _inner_size, CV_8U, data + channel_size*2);

        // Make an array of those three channels to pass as arg to 'split'
        Mat channels[3] = {R,G,B};
        split(croppedImage, channels);

        //increment where we will write for the next picture
        data += _npixels_out * sizeof(uchar);

    }
    return;
}

extern "C" {

extern char *create_data_worker(int img_size, int inner_size, bool center, bool flip, bool rgb, bool multiview,
                 int minibatch_size, char *filename, int macro_start, uint num_data)
{
    ImagesetWorker *wp = new ImagesetWorker(img_size, inner_size, center, flip, rgb, multiview,
                                            minibatch_size, filename, macro_start, num_data);
    return (char *) wp;
}

extern int process_next_minibatch(char *vp, unsigned char* outdata)
{
    try
    {
        ImagesetWorker * ref = reinterpret_cast<ImagesetWorker *>(vp);
        ref->process_next_minibatch(outdata);
        return 0;
    }
    catch(...)
    {
       return -1; //assuming -1 is an error condition.
    }
}

extern int reset(char *vp)
{
    try
    {
        ImagesetWorker * ref = reinterpret_cast<ImagesetWorker *>(vp);
        ref->reset();
        return 0;
    }
    catch(...)
    {
       return -1; //assuming -1 is an error condition.
    }
}

} // "C"

// run --orig_size 256 --crop_size 224 --macro_start 0 --macro_end 10
// --minibatch_size 128 --filename /usr/local/data/I1K/imageset_batches_dw/data_batch_
// int main (int argc, char **argv)
// {

//     char *filename = NULL;
//     int orig_size, crop_size, opt;
//     int macro_start, macro_end, macro_size, minibatch_size;
//     orig_size = crop_size = 0;
//     macro_start = macro_end = macro_size = minibatch_size = 0;
//     bool center = false;
//     bool flip = true;
//     bool rgb = true;
//     bool multiview = false;
//     //Specifying the expected options
//     static struct option long_options[] = {
//         {"orig_size",   required_argument,       0,  'o' },
//         {"crop_size",   required_argument,       0,  'c' },
//         {"macro_end",   required_argument,       0,  'E'  },
//         {"macro_start", required_argument,       0,  'S'  },
//         {"filename",    required_argument,       0,  'n' },
//         {"minibatch_size",  required_argument,       0,  'C' },
//         {"center",      no_argument,             0,  'e' },
//         {"noflip",      no_argument,             0,  'f' },
//         {"grayscale",   no_argument,             0,  'g' },
//         {"multiview",   no_argument,             0,  'm' },
//         {0,             0,                       0,   0   }
//     };
//     int long_index =0;
//     while ((opt = getopt_long(argc, argv,"o:c:m:n:S:E:M:C:efg",
//                    long_options, &long_index )) != -1) {
//         switch (opt) {
//              case 'e' : center     = true;
//                  break;
//              case 'f' : flip       = false;
//                  break;
//              case 'g' : rgb        = false;
//                  break;
//              case 'm' : multiview  = true;
//                  break;
//              case 'o' : orig_size  = atoi(optarg);
//                  break;
//              case 'c' : crop_size  = atoi(optarg);
//                  break;
//              case 'S' : macro_start = atoi(optarg);
//                  break;
//              case 'E' : macro_end = atoi(optarg);
//                  break;
//              case 'C' : minibatch_size = atoi(optarg);
//                  break;
//              case 'n' : filename   = optarg;
//                  break;
//              default:
//                 {
//                  printf("hit the default statement in the opt switch\n");
//                  exit(EXIT_FAILURE);
//                 }
//         }
//     }
//     if (orig_size == 0 || crop_size == 0 || filename == NULL ||
//         macro_end == 0){
//             printf("invalid arguments\n");
//             exit(EXIT_FAILURE);
//     }

//     // Set these parameters that will persist for the life of the server
//     ImagesetWorker wp(orig_size, crop_size, center, flip, rgb, multiview, minibatch_size, filename, macro_start, macro_end);
//     int response_size = minibatch_size * wp._npixels_out * sizeof(uchar) * (multiview ? 5 : 1);
//     response_size += minibatch_size * sizeof(int); // For the labels
//     int request_size = sizeof(int);
//     int cur_batch = 0;
//     int num_macros = macro_end - macro_start + 1;

//     uchar *odata = (uchar *) malloc(response_size*10);
//     for (int i=0; i<26; i++) {
//         wp.process_next_minibatch(odata);
//     }
// }
