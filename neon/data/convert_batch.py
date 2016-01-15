from glob import glob
import os
import ctypes as ct
import struct
from neon.util.compat import range
from neon.util.argparser import NeonArgparser
from neon.util.persist import load_obj


class BatchConverter(object):
    def __init__(self, in_dir, out_dir, outprefix='macrobatch_'):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.dataset_cache = load_obj(os.path.join(self.in_dir, 'dataset_cache.pkl'))
        self.inpath = os.path.join(self.in_dir, self.dataset_cache['batch_prefix'])
        self.outpath = os.path.join(self.out_dir, outprefix)
        self.item_max_size = 0
        libpath = os.path.dirname(os.path.realpath(__file__))
        try:
            self.writerlib = ct.cdll.LoadLibrary(
                os.path.join(libpath, 'loader', 'loader.so'))
            self.writerlib.write_raw.restype = None
            self.writerlib.read_max_item.restype = ct.c_int
        except:
            print('Unable to load loader.so. Ensure that this file has been compiled')

    def write_individual_batch(self, batch_file, label_batch, jpeg_file_batch):
        ndata = len(jpeg_file_batch)
        jpgfiles = (ct.c_char_p * ndata)()
        jpgfiles[:] = jpeg_file_batch

        jpglengths = [len(x) for x in jpeg_file_batch]
        # This interface to the batchfile.hpp allows you to specify
        # destination file, number of input jpg files, list of jpg files,
        # and corresponding list of integer labels
        self.writerlib.write_raw(ct.c_char_p(batch_file),
                                 ct.c_int(ndata),
                                 jpgfiles,
                                 (ct.c_int * ndata)(*jpglengths),
                                 (ct.c_int * ndata)(*label_batch))
        # Check the batchfile for the max item value
        batch_max_item = self.writerlib.read_max_item(ct.c_char_p(batch_file))
        if batch_max_item == 0:
            raise ValueError("Batch file %s probably empty or corrupt" % (batch_file))

        self.item_max_size = max(batch_max_item, self.item_max_size)

    def convert(self, oldfile, newfile):
        print "converting from %s to %s" % (oldfile, newfile)
        with open(oldfile, 'rb') as f:
            num_imgs, num_keys = struct.unpack('II', f.read(8))
            labels = dict()
            for k in range(num_keys):
                ksz = struct.unpack('L', f.read(8))[0]
                keystr = struct.unpack(str(ksz) + 's', f.read(ksz))[0]
                labels[keystr] = list(struct.unpack('I' * num_imgs, f.read(num_imgs * 4)))
            jpegs = []
            for i in range(num_imgs):
                jsz = struct.unpack('I', f.read(4))[0]
                jstr = struct.unpack(str(jsz)+'s', f.read(jsz))[0]
                jpegs.append(jstr)

        self.write_individual_batch(newfile, labels['l_id'], jpegs)

    def save_meta(self):
        properties = ['val_start', 'nval', 'val_nrec',
                      'train_start', 'ntrain', 'train_nrec',
                      'nclass']
        with open(self.outpath + 'meta', 'w') as f:
            for p in properties:
                f.write('%s %d\n' % (p, self.dataset_cache[p]))
            f.write('item_max_size %d\n' % (self.item_max_size))
            f.write('label_size %d\n' % (4))
            for idx, channel in enumerate('RGB'):
                f.write('%s_mean      %f\n' % (channel, self.dataset_cache['global_mean'][idx]))

    def run(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        infiles = glob(self.inpath + '*')
        for ifile in infiles:
            ofile = ifile.replace(self.inpath, self.outpath) + '.cpio'
            self.convert(ifile, ofile)

        # Now write out the metadatafile
        self.save_meta()


if __name__ == "__main__":
    parser = NeonArgparser(__doc__)
    parser.add_argument('--in_dir', help='Directory to find original macrobatches', required=True)
    args = parser.parse_args()

    bc = BatchConverter(in_dir=args.in_dir, out_dir=args.data_dir)
    bc.run()
