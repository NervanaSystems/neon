# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import division
from builtins import range
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
