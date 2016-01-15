from PIL import Image as PILImage
import struct
import sys
import numpy as np

# Open the input binary file
infile = sys.argv[1]
with open(infile, 'rb') as f:
    contents = f.read()

n_img, sz_img = struct.unpack("II", contents[:8])
C = 3
H = W = int(np.sqrt(sz_img / 3))

offset = 8
imlist = []
for i in range(n_img):
    buf = struct.unpack('B' * sz_img, contents[offset:(offset + sz_img)])
    ary = np.array(buf, dtype=np.uint8).reshape(C, H, W).transpose(1, 2, 0)[:, :, ::-1].copy()
    imlist.append(ary)
    offset += sz_img

img = PILImage.fromarray(np.vstack(imlist))
img.save(infile + ".jpg")
