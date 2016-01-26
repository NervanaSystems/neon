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

import argparse
import os
import pickle
import sys

from neon.util.modeldesc import ModelDescription


def compare_files(file1, file2):
    """
    Helper function to compare two serialized model files

    This is only comparing the model weights and states and layer
    config parameters

    Returns:
        bool: True if the two file match
    """
    models = []
    for fn in [file1, file2]:
        assert os.path.exists(fn), 'Could not find file %s' % fn

        with open(fn, 'r') as fid:
            models.append(ModelDescription(pickle.load(fid)))

    return models[0] == models[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two serialized model files.')
    parser.add_argument('file1')
    parser.add_argument('file2')
    args = parser.parse_args()

    if not compare_files(args.file1, args.file2):
        print 'ERROR: Models do not match!'
        sys.exit(1)
    else:
        print 'Match'
