import struct
import sys
import os

"""
Take as input the following arguments:
    sourcefile: contains list of target, jpegs named with their labels
    imgdir: top level directory containing
    destfile: where to write out the batches
    Performs the following action:
        get a list of all files in the directory
        while there are still files left to process:
            grab min(Size, files_left) files from disk
            and put them in a map from label -> jpeg string
            write that map into a file, save it and close it
"""

# Contents of the header
magic_number = "MACR" # Identifies this as a metadata file
format_version = 12 # the first version of metadata file format
writer_version = 24 # the first version of the metadata writer file format
dataset_type = "i1kclass"

def writeMacro(pairs, destfile):
    """
    Given a list of datum, target pairs
    Write an i1k macrobatch file
    """
    # Trim the destfile index
    prefix = destfile[:-4]
    number = int(destfile[-4:])
    destfile = destfile[:-4] + str(number)
    with open(destfile, 'wb') as f:

        # Pack the header
        f.write(struct.pack('B' * len(magic_number), *bytearray(magic_number)))
        # pack_string(f, magic_number)
        f.write(struct.pack('I', format_version))
        f.write(struct.pack('I', writer_version))
        f.write(struct.pack('B' * len(dataset_type), *bytearray(dataset_type)))
        # pack_string(f, dataset_type)
        f.write(struct.pack('I', len(pairs))) # write the data count

        # now write the location of each tuple
        # for each elt of batchMap
        # we have input_size, target_size = 8 + 8
        # then len(key) bytes and len(value) bytes
        # TODO: could squish both these loops

        # The location of the offsets is 24
        # but the first tuple is at 24 + 8*len(pairs)
        location = 24 + 8 * len(pairs)
        tgtsize = 8
        for datum, target in pairs:
            f.write(struct.pack('Q', location))
            size = 16 + len(datum) + tgtsize
            location += size

        # then write (input_size, target_size, input, target) tuples
        for datum, target in pairs:
            f.write(struct.pack('Q', len(datum)))
            f.write(struct.pack('Q', tgtsize))
            f.write(struct.pack('B' * len(datum), *bytearray(datum)))
            f.write(struct.pack('Q', target))

# check that there are 3 args
# if len(sys.argv) != 3:
#     print "Usage: python write_imagenet_macrobatches.py <imgdir <sourcefile> <dest>"
#     sys.exit(1)

imgdir = sys.argv[1]
sourcefile = sys.argv[2]
destfile = sys.argv[3]
label_file = '/scratch/alex/label.mapping'

with open(sourcefile, 'r') as f:
    lines = f.readlines()

ldict = dict()
with open(label_file, 'r') as f:
    labels = f.readlines()

    for l in labels:
        (label, index) = l.split()
        ldict[label] = int(index)

pairs = []
for l in lines:
    (target, filename) = l.split()
    with file(os.path.join(imgdir, filename)) as f:
        datum = f.read()
    pairs.append((datum, ldict[target]))

writeMacro(pairs, destfile)
