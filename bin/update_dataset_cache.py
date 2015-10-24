#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
"""
Script to convert old batch_write data set cache global means to new format.
Old format is a full image mean while the new format has a single mean value
for each channel of the input image.  This sciprt will overwrite the existing
data cache pickle file.

Arguments:
    data cache file: path to the data cache file, this path will be printed
                     in the exception rasied by neon if it detected that the
                     data cache global mean is of the old format
"""

import argparse
import os
import numpy as np
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cache_file', help='path to data cache file')
    args = parser.parse_args()

    cache_file = args.cache_file

    # check for RW access to file
    assert os.path.exists(cache_file), 'file does not exist %s' % cache_file
    if not os.access(os.path.abspath(cache_file), os.R_OK | os.W_OK):
        raise IOError('Need to add read and/or write permissions on file %s' % cache_file)

    with open(cache_file, 'r') as fid:
        dc = pickle.load(fid)

    if 'global_mean' not in dc or 'img_size' not in dc:
        raise ValueError('data cache file missing global_mean key')

    sz = dc['img_size']
    gm = dc['global_mean']

    if len(gm.shape) != 2 or (gm.shape[0] != sz*sz*3 or gm.shape[1] != 1):
        raise ValueError('global mean shape %s does not match format expected' % str(gm.shape))

    dc['global_mean'] = np.mean(gm.reshape(3, -1), axis=1).reshape(3, 1)

    with open(cache_file, 'w') as fid:
        pickle.dump(dc, fid)

    print '%s updated to new format' % cache_file
