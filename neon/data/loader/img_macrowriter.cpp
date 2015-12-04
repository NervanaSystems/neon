#include <string.h>
#include <assert.h>
#include <libgen.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "decoder.hpp"
#include "batchfile.hpp"

#if STANDALONE
int main(int argc, char** argv){
    int maxResizeDim = 0;  // Set to non-zero to force shortest dim <= maxResizeDim

    if (argc < 3 || argc > 4){
        printf("Usage: ./macrowriter <filelabel_list> <macro_file>");
        exit(1);
    }
    if (argc == 4)
      maxResizeDim = atoi(argv[3]);

    string filelist(argv[1]);
    string destfile(argv[2]);

    LineList filelabelpairs;

    if (readFileLines(filelist, filelabelpairs) != 0)
        exit(1);

    if (!filelabelpairs.empty()) {
        BatchFile bf;
        bf.openForWrite(destfile, "imgclass");
        for (auto fl : filelabelpairs) {
            std::stringstream curline(fl);
            string fileToken;
            uint32_t labelToken;
            curline >> fileToken >> labelToken;

            // Process the input
            ByteVect inp;
            readFileBytes(fileToken, inp);
            if (maxResizeDim != 0)
                resizeInput(inp, maxResizeDim);  // from decoder.hpp
            // Process the target label
            ByteVect tgt(sizeof(uint32_t));
            memcpy(&tgt[0], &labelToken, sizeof(uint32_t));

            bf.writeItem(inp, tgt);
        }
        bf.close();
    }
    return 0;
}
#else  // STANDALONE else
extern "C" {

extern void write(char *outfile, const int num_data,
                  char **jpgfiles, uint32_t *labels,
		  int maxDim) {
    if (num_data == 0)
        return;
    BatchFile bf;
    bf.openForWrite(outfile, "imgclass");
    for (int i=0; i<num_data; i++) {
        ByteVect inp;
	readFileBytes(jpgfiles[i], inp);
	if (maxDim != 0)
	    resizeInput(inp, maxDim);  // from decoder.hpp
        ByteVect tgt(sizeof(uint32_t));
        memcpy(&tgt[0], (char *) &(labels[i]), sizeof(uint32_t));
        bf.writeItem(inp, tgt);
    }
    bf.close();
}

}
#endif  // STANDALONE else
