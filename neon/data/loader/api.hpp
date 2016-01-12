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

extern "C" {

    extern void* start(int inner_size,
                       bool center, bool flip, bool rgb,
                       int scale_min, int scale_max,
                       int contrast_min, int contrast_max,
                       int rotate_min, int rotate_max,
                       int minibatch_size,
                       char* filename, int macro_start,
                       uint num_data, uint num_labels, bool macro,
                       bool shuffle, int read_max_size, int label_size,
                       DeviceParams* params) {
        static_assert(sizeof(int) == 4, "int is not 4 bytes");
        try {
            int nchannels = (rgb == true) ? 3 : 1;
            int item_max_size = nchannels*inner_size*inner_size;
            // These objects will get freed in the destructor of Loader.
            Device* device;
#if HASGPU
            if (params->_type == CPU) {
                device = new Cpu(params);
            } else {
                device = new Gpu(params);
            }
#else
            assert(params->_type == CPU);
            device = new Cpu(params);
#endif
            Reader*     reader;
            if (macro == true) {
                reader = new MacrobatchReader(filename, macro_start,
                                              num_data, minibatch_size, shuffle);
            } else {
                reader = new ImageFileReader(filename, num_data,
                                             minibatch_size, inner_size);
            }
            AugmentationParams* agp = new AugmentationParams(inner_size, center, flip, rgb,
                                                             /* Scale Params */
                                                             scale_min, scale_max,
                                                             /* Contrast Params */
                                                             contrast_min, contrast_max,
                                                             /* Rotate Params (ignored) */
                                                             rotate_min, rotate_max);

            Decoder* decoder = new ImageDecoder(agp);
            Loader* loader = new Loader(minibatch_size, read_max_size,
                                        item_max_size, label_size,
                                        num_labels, device,
                                        reader, decoder);
            int result = loader->start();
            if (result != 0) {
                printf("Could not start data loader. Error %d", result);
                delete loader;
                exit(-1);
            }
            return reinterpret_cast<void*>(loader);
        } catch(...) {
            return 0;
      }
  }

  extern int next(Loader* loader) {
    try {
        loader->next();
        return 0;
    } catch(...) {
        return -1;
    }
}

extern int reset(Loader* loader) {
    try {
        return loader->reset();
    } catch(...) {
        return -1;
    }
}

extern int stop(Loader* loader) {
    try {
        loader->stop();
        delete loader;
        return 0;
    } catch(...) {
        return -1;
    }
}


extern void write(char *outfile, const int num_data,
                  char **jpgfiles, uint32_t *labels,
                  int maxDim) {
    if (num_data == 0) {
        return;
    }
    BatchFile bf;
    bf.openForWrite(outfile, "imgclass");
    for (int i=0; i<num_data; i++) {
        ByteVect inp;
        readFileBytes(jpgfiles[i], inp);
        if (maxDim != 0) {
            resizeInput(inp, maxDim);  // from decoder.hpp
        }
        ByteVect tgt(sizeof(uint32_t));
        memcpy(&tgt[0], &(labels[i]), sizeof(uint32_t));
        bf.writeItem(inp, tgt);
    }
    bf.close();
}

extern void write_raw(char *outfile, const int num_data,
                  char **jpgdata, uint32_t *jpglens,
                  uint32_t *labels) {
    if (num_data == 0) {
        return;
    }
    BatchFile bf;
    uint32_t tgtSize = sizeof(uint32_t);
    bf.openForWrite(outfile, "imgclass");
    for (int i=0; i<num_data; i++) {
        bf.writeItem(jpgdata[i], (char *) &labels[i], &jpglens[i], &tgtSize);
    }
    bf.close();
}

extern int read_max_item(char *batchfile) {
    BatchFile bf;
    bf.openForRead(batchfile);
    int maxItemSize = bf.maxDatumSize();
    bf.close();
    return maxItemSize;
}

}
