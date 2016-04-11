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

import os
import glob
import logging
from neon import NervanaObject

logger = logging.getLogger(__name__)


class Indexer(NervanaObject):
    def __init__(self, path, index_file, pattern='*'):
        self.path = path
        assert os.path.isabs(index_file)
        self.index_file = index_file
        self.pattern = pattern

    def run(self):
        if os.path.exists(self.index_file):
            return
        logger.warning('%s not found. Attempting to create...' % self.index_file)
        assert os.path.exists(self.path)
        subdirs = glob.iglob(os.path.join(self.path, '*'))
        subdirs = filter(lambda x: os.path.isdir(x), subdirs)
        classes = sorted(map(lambda x: os.path.basename(x), subdirs))
        class_map = {key: val for key, val in zip(classes, range(len(classes)))}
        with open(self.index_file, 'w') as fd:
            fd.write('filename,label1\n')
            for subdir in subdirs:
                label = class_map[os.path.basename(subdir)]
                files = glob.iglob(os.path.join(subdir, self.pattern))
                for filename in files:
                    rel_path = os.path.join(os.path.basename(subdir),
                                            os.path.basename(filename))
                    fd.write(rel_path + ',' + str(label) + '\n')
        logger.info('Created index file: %s' % self.index_file)