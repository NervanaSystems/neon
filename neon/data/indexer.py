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
    def __init__(self, path, pattern='*'):
        self.path = path
        self.pattern = pattern

    def run(self):
        archive_dir = self.path + '-ingested'
        if os.path.exists(archive_dir):
            return
        index_file = os.path.join(self.path, 'index.csv')
        if os.path.exists(index_file):
            return
        logger.warning('%s not found. Attempting to create...' % index_file)
        assert os.path.exists(self.path)
        subdirs = glob.iglob(os.path.join(self.path, '*'))
        subdirs = filter(lambda x: os.path.isdir(x), subdirs)
        classes = sorted(map(lambda x: os.path.basename(x), subdirs))
        class_map = {key: val for key, val in zip(classes, range(len(classes)))}
        with open(index_file, 'w') as fd:
            for subdir in subdirs:
                label = class_map[os.path.basename(subdir)]
                files = glob.iglob(os.path.join(subdir, self.pattern))
                for filename in files:
                    rel_path = os.path.join(os.path.basename(subdir),
                                            os.path.basename(filename))
                    fd.write(rel_path + ',' + str(label) + '\n')
        logger.info('Created index file: %s' % index_file)
