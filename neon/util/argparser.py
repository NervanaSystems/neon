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
'''
Command line argument parser for neon deep learning library

This is a wrapper around the configargparse ArgumentParser class.
It adds in the default neon command line arguments and allows
additional arguments to be added using the argparse library
methods.  Lower priority defaults can also be read from a configuration file
(specified by the -c command line argument).
'''

import configargparse
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import os

from neon import __version__ as neon_version
from neon.backends.util.check_gpu import get_compute_capability

logger = logging.getLogger(__name__)


class NeonArgparser(configargparse.ArgumentParser):
    """
    Setup the command line arg parser and parse the
    arguments in sys.arg (or from configuration file).  Use the parsed
    options to configure the logging module.

    Arguments:
        desc (String) : Docstring from the calling function.
        This will be used for the description of the command receiving the arguments.
    """
    def __init__(self, *args, **kwargs):
        self._PARSED = False
        self.work_dir = os.path.join(os.path.expanduser('~'), 'nervana')
        if 'default_config_files' not in kwargs:
            kwargs['default_config_files'] = [os.path.join(self.work_dir,
                                                           'neon.cfg')]
        super(NeonArgparser, self).__init__(*args, **kwargs)

        # ensure that default values are display via --help
        self.formatter_class = configargparse.ArgumentDefaultsHelpFormatter

        self.setup_default_args()

    def setup_default_args(self):
        """
        Setup the default arguments used by neon
        """

        self.add_argument('--version', action='version', version=neon_version)
        self.add_argument('-c', '--config', is_config_file=True,
                          help='Read values for these arguments from the '
                               'configuration file specified here first.')
        self.add_argument('-v', '--verbose', action='count', default=0,
                          help="verbosity level.  Add multiple v's to "
                               "further increase verbosity")
        # we store the negation of no_progress_bar in args.progress_bar during
        # parsing
        self.add_argument('--no_progress_bar',
                          action="store_true",
                          help="suppress running display of progress bar and "
                               "training loss")

        # runtime specifc options
        rt_grp = self.add_argument_group('runtime')
        rt_grp.add_argument('-w', '--data_dir',
                            default=os.path.join(self.work_dir, 'data'),
                            help='working directory in which to cache '
                                 'downloaded and preprocessed datasets')
        rt_grp.add_argument('-e', '--epochs', type=int, default=10,
                            help='number of complete passes over the dataset to run')
        rt_grp.add_argument('-s', '--save_path', type=str,
                            help='file path to save model snapshots')
        rt_grp.add_argument('--serialize', nargs='?', type=int,
                            default=0, const=1, metavar='N',
                            help='serialize model every N epochs')
        rt_grp.add_argument('--model_file', help='load model from pkl file')
        rt_grp.add_argument('-l', '--log', dest='logfile', nargs='?',
                            const=os.path.join(self.work_dir, 'neon_log.txt'),
                            help='log file')
        rt_grp.add_argument('-o', '--output_file', default=None,
                            help='hdf5 data file for metrics computed during '
                                 'the run, optional.  Can be used by nvis for '
                                 'visualization.')
        rt_grp.add_argument('-val', '--validation_freq', type=int, default=None,
                            help='frequency (in epochs) to test the validation set.')
        rt_grp.add_argument('-H', '--history', type=int, default=1,
                            help='number of checkpoint files to retain')

        be_grp = self.add_argument_group('backend')
        be_grp.add_argument('-b', '--backend', choices=['cpu', 'gpu'],
                            default='gpu' if get_compute_capability() >= 5.0
                                    else 'cpu',
                            help='backend type')
        be_grp.add_argument('-i', '--device_id', type=int, default=0,
                            help='gpu device id (only used with GPU backend)')

        be_grp.add_argument('-r', '--rng_seed', type=int,
                            default=None, metavar='SEED',
                            help='random number generator seed')
        be_grp.add_argument('-u', '--rounding',
                            const=True,
                            type=int,
                            nargs='?',
                            metavar='BITS',
                            default=False,
                            help='use stochastic rounding [will round to BITS number '
                                 'of bits if specified]')
        be_grp.add_argument('-d', '--datatype', choices=['f16', 'f32', 'f64'],
                            default='f32', metavar='default dtype',
                            help='default floating point '
                            'precision for backend [f64 for cpu only]')

        return

    def add_yaml_arg(self):
        '''
        Add the yaml file argument, this is needed for scripts that
        parse the model config from yaml files

        '''
        # yaml configuration file
        self.add_argument('yaml_file',
                          type=configargparse.FileType('r'),
                          help='neon model specification file')

    def add_argument(self, *args, **kwargs):
        '''
        Method by which command line arguments are added to the parser.  Passed
        straight through to parent add_argument method.
        '''
        if self._PARSED:
            logger.warn('Adding arguments after arguments were parsed = '
                        'may need to rerun parse_args')
            # reset so warning only comes once
            self._PARSED = False

        super(NeonArgparser, self).add_argument(*args, **kwargs)
        return

    # we never use this alias from ConfigArgParse, but defining this here
    # prevents documentation indent warnings
    def add(self):
        pass

    # we never use this alias from ConfigArgParse, but defining this here
    # prevents documentation indent warnings
    def add_arg(self):
        pass

    def parse_args(self):
        '''
        Parse the command line arguments and setup neon
        runtime environment accordingly

        Returns:
            namespace: contains the parsed arguments as attributes
        '''
        args = super(NeonArgparser, self).parse_args()

        # set up the logging
        # max thresh is 50 (critical only), min is 10 (debug or higher)
        try:
            log_thresh = max(10, 40 - args.verbose*10)
        except (AttributeError, TypeError):
            # if defaults are not set or not -v given
            # for latter will get type error
            log_thresh = 40

        # logging formater
        fmtr = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # copy to stderr
        stderrlog = logging.StreamHandler()
        stderrlog.setFormatter(fmtr)
        logger.addHandler(stderrlog)

        if args.logfile:
            # add log to file as well
            filelog = RotatingFileHandler(filename=args.logfile, mode='w',
                                          maxBytes=10000000, backupCount=5)
            filelog.setFormatter(fmtr)
            logger.addHandler(filelog)

        logger.setLevel(log_thresh)

        # need to write out float otherwise numpy
        # generates type in bytes not bits (f16 == 128 bits)
        args.datatype = 'float' + args.datatype[1:]
        args.datatype = np.dtype(args.datatype).type

        # invert no_progress_bar meaning and store in args.progress_bar
        args.progress_bar = not args.no_progress_bar

        if args.backend == 'cpu' and args.rounding > 0:
            raise NotImplementedError('CPU backend does not support stochastic roudning')

        # done up front to avoid losing data due to incorrect path
        if args.save_path:
            if not os.access(os.path.dirname(os.path.abspath(args.save_path)),
                             os.R_OK | os.W_OK):
                raise ValueError('Can not write to save_path dir %s' % args.save_path)
            if os.path.exists(args.save_path):
                # if file exists check that it can be overwritten
                if not os.access(args.save_path, os.R_OK | os.W_OK):
                    raise IOError('Can not write to save_path file %s' % args.save_path)

        if args.serialize > 0:
            if args.save_path is None:
                logger.warn('No path given for model serialization,'
                            'using default "neon_model.pkl"')
                args.save_path = "neon_model.pkl"

        if args.model_file:
            if not os.path.exists(args.model_file):
                raise IOError('Model file %s not present' % args.model_file)
            if not os.access(args.model_file, os.R_OK):
                raise IOError('Not read access for model file %s' % args.model_file)

        # display what command line / config options were set (and from where)
        logger.info(self.format_values())

        self._PARSED = True
        self.args = args
        return args
