#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
Script to convert old batch_writer dataset cache global means to new format.
Old format is a full image mean while the new format has a single mean value
for each channel of the input image.  This script will overwrite the existing
data cache pickle file.

Arguments:
    data cache file: path to the data cache file, this path will be printed
                     in the exception raised by neon if it detects that the
                     data cache global mean is of the old format
"""
import argparse
import os
import numpy as np
from neon.util.persist import load_obj, save_obj
from neon import logger as neon_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cache_file', help='path to data cache file')
    args = parser.parse_args()

    cache_file = args.cache_file

    # check for RW access to file
    assert os.path.exists(cache_file), 'file does not exist %s' % cache_file
    if not os.access(os.path.abspath(cache_file), os.R_OK | os.W_OK):
        raise IOError('Need to add read and/or write permissions on file %s' % cache_file)

    dc = load_obj(cache_file)

    if 'global_mean' not in dc or 'img_size' not in dc:
        raise ValueError('data cache file missing global_mean key')

    sz = dc['img_size']
    gm = dc['global_mean']

    if len(gm.shape) != 2 or (gm.shape[0] != sz * sz * 3 or gm.shape[1] != 1):
        raise ValueError('global mean shape {} does not match format expected'.format(gm.shape))

    # Collapse the full tensor mean into channel means and correct the order (RGB <-> BGR)
    dc['global_mean'] = np.mean(gm.reshape(3, -1), axis=1).reshape(3, 1)[::-1]

    save_obj(dc, cache_file)

    neon_logger.display('%s updated to new format' % cache_file)
