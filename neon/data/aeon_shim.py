# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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

import sys
from neon import logger as neon_logger
from neon.data.dataloaderadapter import DataLoaderAdapter


try:
    from aeon import DataLoader as AeonLoader
except ImportError:
    neon_logger.error('Unable to load Aeon data loading module.')
    neon_logger.error('Please follow installation instructions at:')
    neon_logger.error('https://github.com/NervanaSystems/aeon')
    sys.exit(1)


def AeonDataLoader(config, adapter=True):
    if adapter:
        return DataLoaderAdapter(AeonLoader(config))
    else:
        return AeonLoader(config)
