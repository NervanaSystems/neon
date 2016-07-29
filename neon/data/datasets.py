# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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
Functions used to load commonly available datasets.
"""
from __future__ import division
from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below
from future.moves.urllib.request import Request, urlopen  # noqa

import logging  # noqa
import os  # noqa
import sys  # noqa
import zipfile  # noqa

from neon import NervanaObject, logger as neon_logger  # noqa
from neon.util.compat import PY3  # noqa

logger = logging.getLogger(__name__)


class Dataset(NervanaObject):
    """
    Container class for stock datasets.

    Arguments:
        filename (str/list): name of the file to download
        url (str): url for dataset
        size (int/list): file size
        path (str): local path to place files
        subset_pct (float/int): percentage of data set to use for training
    """
    def __init__(self, filename, url, size, path='.', subset_pct=100):
        # parameters to use in dataset config serialization
        super(Dataset, self).__init__(name=None)
        self.filename = filename
        self.url = url
        self.size = size
        self.path = path
        self.subset_pct = subset_pct
        self._data_dict = None
        if subset_pct != 100:
            # placeholder to use partial data set
            raise NotImplemented('subset percentage feature is not yet implemented')

    def serialize(self):
        """
        Generates dictionary with the required parameters to describe this object
        """
        return self.get_description()

    def load_zip(self, filename, size):
        """
        Helper function for downloading test files
        Will download and unzip the file into the directory self.path

        Arguments:
            filename (str): name of file to download from self.url
            size (str): size of the file in bytes?

        Returns:
            str: Path to the downloaded dataset.
        """
        workdir, filepath = self._valid_path_append(self.path, '', filename)

        if not os.path.exists(filepath):
            self.fetch_dataset(self.url, filename, filepath, size)
        if '.zip' in filepath:
            zip_ref = zipfile.ZipFile(filepath)
            zip_ref.extractall(workdir)
            zip_ref.close()
            filepath = filepath.split('.zip')[0]
        return filepath

    @staticmethod
    def _valid_path_append(path, *args):
        """
        Helper to validate passed path directory and append any subsequent
        filename arguments.

        Arguments:
            path (str): Initial filesystem path.  Should expand to a valid
                        directory.
            *args (list, optional): Any filename or path suffices to append to path
                                    for returning.

            Returns:
                (list, str): path prepended list of files from args, or path alone if
                         no args specified.

        Raises:
            ValueError: if path is not a valid directory on this filesystem.
        """
        full_path = os.path.expanduser(path)
        res = []
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        if not os.path.isdir(full_path):
            raise ValueError("path: {0} is not a valid directory".format(path))
        for suffix_path in args:
            res.append(os.path.join(full_path, suffix_path))
        if len(res) == 0:
            return path
        elif len(res) == 1:
            return res[0]
        else:
            return res

    @staticmethod
    def fetch_dataset(url, sourcefile, destfile, totalsz):
        """
        Download the file specified by the given URL.

        Args:
            url (str): Base URL of the file to be downloaded.
            sourcefile (str): Name of the source file.
            destfile (str): Path to the destination.
            totalsz (int): Size of the file to be downloaded.
        """
        req = Request(os.path.join(url, sourcefile), headers={'User-Agent': 'neon'})
        # backport https limitation and workaround per http://python-future.org/imports.html
        cloudfile = urlopen(req)
        neon_logger.display("Downloading file: {}".format(destfile))
        blockchar = u'\u2588'  # character to display in progress bar
        with open(destfile, 'wb') as f:
            data_read = 0
            chunksz = 1024**2
            while 1:
                data = cloudfile.read(chunksz)
                if not data:
                    break
                data_read = min(totalsz, data_read + chunksz)
                progress_string = u'Download Progress |{:<50}| '.format(
                    blockchar * int(float(data_read) / totalsz * 50))
                sys.stdout.write('\r')
                if PY3:
                    sys.stdout.write(progress_string)
                else:
                    sys.stdout.write(progress_string.encode("utf-8"))
                sys.stdout.flush()

                f.write(data)
            neon_logger.display("Download Complete")

    def gen_iterators(self):
        """
        Method that generates the data set iterators for the
        train, test and validation data sets.  This method
        needs to set the instance data_set attribute to a
        dictionary of data iterators.

        Returns:
            dict:  dictionary with the various data set iterators
        """
        raise NotImplemented()

    @property
    def data_dict(self):
        if self._data_dict is None:
            self._data_dict = self.gen_iterators()
        return self._data_dict

    def get_iterator(self, setname):
        """
        Helper method to get the data iterator for specified dataset

        Arguments:
            setname (str): which iterator to return (e.g. 'train', 'valid')
        """
        assert setname in self.data_dict, 'no iterator for set %s' % setname
        return self.data_dict[setname]

    @property
    def train_iter(self):
        """
        Helper method to return training set iterator
        """
        return self.get_iterator('train')

    @property
    def valid_iter(self):
        """
        Helper method to return validation set iterator
        """
        return self.get_iterator('valid')

    @property
    def test_iter(self):
        """
        Helper method to return test set iterator
        """
        return self.get_iterator('test')
