#include <string.h>
#include <assert.h>
#include <libgen.h>
#include <stdexcept>
#include "batchfile.hpp"

int main(int argc, char** argv){
  // Simply unpacks a given batchfile
  if (argc != 3){
    printf("Usage: ./macrowriter <filelabel_list> <macro_file>");
    exit(1);
  }

  string batchfile(argv[1]);
  string prefix(argv[2]);

  BatchFile bf;
  bf.openForRead(batchfile);

  ByteVect data(bf.maxDataSize());
  ByteVect label(bf.maxLabelsSize());
  for (int i=0; i<bf.itemCount(); i++) {
    char fname[256];
    int labelSize, dataSize;
    bf.readItem((char *) &data[0], (char *) &label[0], &dataSize, &labelSize);
    int labelNum = *(int *) (&label[0]);
    sprintf(fname, "%s_%03d_%03d.jpg", prefix.c_str(), i, labelNum);
    ofstream _ofs;
    _ofs.open(fname, ofstream::binary);
    _ofs.write((char *) &data[0], dataSize);
    _ofs.close();
  }
  bf.close();
  return 0;
}
