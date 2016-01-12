#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>

#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include "decoder.hpp"
#include "batchfile.hpp"

using std::ifstream;
using std::ofstream;

int main (int argc, char **argv) {
    AugmentationParams *agp = new AugmentationParams(224, false, true, true,
        256, 256,   // Scale Params
        75, 125,    // Contrast params
        0, 0);      // Rotation params
    if (argc < 3)
        return -1;

    BatchFile bf;
    string batchFileName(argv[1]);
    bf.openForRead(batchFileName);
    ByteVect data(bf.maxDataSize());
    ByteVect labels(bf.maxLabelsSize());

    // Just get a single item
    int dsize, lsize;
    bf.readItem((char *) &data[0], (char *) &labels[0], &dsize, &lsize);
    bf.close();
    int label_idx = *reinterpret_cast<int *>(&labels[0]);
    // We'll do 10 decodings of the same image;
    ImageDecoder decoder(agp);
    int num_decode = 10;
    int num_pixels = agp->getSize().area() * 3;
    ByteVect outbuf(num_pixels * num_decode);
    std::cout << "numpixels: " << num_pixels << std::endl;
    std::cout << "outbuf size: " << outbuf.size() << std::endl;
    std::cout << "label index: " << label_idx << std::endl;

    for (int i = 0; i < num_decode; i++) {
        decoder.decode(&data[0], dsize, &outbuf[i * num_pixels]);
    }

    ofstream file (argv[2], ofstream::out | ofstream::binary);
    file.write((char *) &num_decode, sizeof(int));
    file.write((char *) &num_pixels, sizeof(int));
    file.write((char *) &outbuf[0], outbuf.size());
    file.close();
    delete agp;
}
