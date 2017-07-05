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
Command line argument parser for neon deep learning library

This is a wrapper around the configargparse ArgumentParser class.
It adds in the default neon command line arguments and allows
additional arguments to be added using the argparse library
methods.  Lower priority defaults can also be read from a configuration file
(specified by the -c command line argument).
"""
import configargparse
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import os
import inspect

from neon import __version__ as neon_version
from neon.backends import gen_backend
from neon.backends.backend import Backend
from neon.backends.util.check_gpu import get_compute_capability, get_device_count
from neon.backends.util.check_mkl import get_mkl_lib
from neon.callbacks.callbacks import Callbacks

logger = logging.getLogger(__name__)


def extract_valid_args(args, func, startidx=0):
    """
    Given a namespace of argparser args, extract those applicable to func.

    Arguments:
        args (Namespace): a namespace of args from argparse
        func (Function): a function to inspect, to determine valid args
        startidx (int): Start index

    Returns:
        dict of (arg, value) pairs from args that are valid for func
    """
    func_args = inspect.getargspec(func).args[startidx:]
    return dict((k, v) for k, v in list(vars(args).items()) if k in func_args)


class NeonArgparser(configargparse.ArgumentParser):
    """
    Setup the command line arg parser and parse the
    arguments in sys.arg (or from configuration file).  Use the parsed
    options to configure the logging module.

    Arguments:
        desc (String) : Docstring from the calling function. This will be used
                        for the description of the command receiving the
                        arguments.
    """
    def __init__(self, *args, **kwargs):
        self._PARSED = False
        self.work_dir = os.path.join(os.path.expanduser('~'), 'nervana')
        if 'default_config_files' not in kwargs:
            kwargs['default_config_files'] = [os.path.join(self.work_dir,
                                                           'neon.cfg')]
        if 'add_config_file_help' not in kwargs:
            # turn off the auto-generated config help for config files since it
            # referenced unsettable config options like --version
            kwargs['add_config_file_help'] = False

        self.defaults = kwargs.pop('default_overrides', dict())
        super(NeonArgparser, self).__init__(*args, **kwargs)

        # ensure that default values are display via --help
        self.formatter_class = configargparse.ArgumentDefaultsHelpFormatter

        self.setup_default_args()

    def setup_default_args(self):
        """
        Setup the default arguments used by neon
        """

        self.add_argument('--version', action='version', version=neon_version)
        self.add_argument('-c', '--config',
                          is_config_file=True,
                          help='Read values for these arguments from the '
                               'configuration file specified here first.')
        self.add_argument('-v', '--verbose', action='count',
                          default=self.defaults.get('verbose', 1),
                          help="verbosity level.  Add multiple v's to "
                               "further increase verbosity")
        # we store the negation of no_progress_bar in args.progress_bar during
        # parsing
        self.add_argument('--no_progress_bar',
                          action="store_true",
                          help="suppress running display of progress bar and "
                               "training loss")

        bm_grp = self.add_argument_group('benchmark')
        bm_grp.add_argument('--profile', action='store_true')
        bm_grp.add_argument('--profiling_method', type=str, default='time')
        bm_grp.add_argument('--profile_inference', action='store_true')
        bm_grp.add_argument('--profile_iterations', type=int, default=50)
        bm_grp.add_argument('--profile_iter_skip', type=int, default=5)

        # runtime specifc options
        rt_grp = self.add_argument_group('runtime')
        rt_grp.add_argument('-w', '--data_dir',
                            default=os.path.join(self.work_dir, 'data'),
                            help='working directory in which to cache '
                                 'downloaded and preprocessed datasets')
        rt_grp.add_argument('-e', '--epochs', type=int,
                            default=self.defaults.get('epochs', 10),
                            help='number of complete passes over the dataset to run')
        rt_grp.add_argument('-s', '--save_path', type=str,
                            default=self.defaults.get('save_path'),
                            help='file path to save model snapshots')
        rt_grp.add_argument('--serialize', nargs='?', type=int,
                            default=self.defaults.get('serialize', 0),
                            const=1, metavar='N',
                            help='serialize model every N epochs')
        rt_grp.add_argument('--model_file', help='load model from pkl file')
        rt_grp.add_argument('-l', '--log', dest='logfile', nargs='?',
                            const=os.path.join(self.work_dir, 'neon_log.txt'),
                            help='log file')
        rt_grp.add_argument('-o', '--output_file',
                            default=self.defaults.get('output_file', None),
                            help='hdf5 data file for metrics computed during '
                                 'the run, optional.  Can be used by nvis for '
                                 'visualization.')
        rt_grp.add_argument('-eval', '--eval_freq', type=int,
                            default=self.defaults.get('eval_freq', None),
                            help='frequency (in epochs) to test the eval set.')
        rt_grp.add_argument('-H', '--history', type=int,
                            default=self.defaults.get('history', 1),
                            help='number of checkpoint files to retain')
        rt_grp.add_argument('--log_token', type=str,
                            default='',
                            help='access token for data logging in real time')
        rt_grp.add_argument('--manifest', action='append', help="manifest files")
        rt_grp.add_argument('--manifest_root', type=str, default=None,
                            help='Common root path for relative path items in the '
                                 'supplied manifest files')
        be_grp = self.add_argument_group('backend')
        be_grp.add_argument('-b', '--backend', choices=Backend.backend_choices(),
                            default='gpu' if get_compute_capability() >= 3.0
                                    else 'mkl' if get_mkl_lib()
                                    else 'cpu',
                            help='backend type. Multi-GPU support is a premium '
                                 'feature available exclusively through the '
                                 'Nervana cloud. Please contact '
                                 'info@nervanasys.com for details.')
        be_grp.add_argument('-i', '--device_id', type=int,
                            default=self.defaults.get('device_id', 0),
                            help='gpu device id (only used with GPU backend)')
        be_grp.add_argument('-m', '--max_devices', type=int,
                            default=self.defaults.get('max_devices', get_device_count()),
                            help='max number of GPUs (only used with mgpu backend')

        be_grp.add_argument('-r', '--rng_seed', type=int,
                            default=self.defaults.get('rng_seed', None),
                            metavar='SEED',
                            help='random number generator seed')
        be_grp.add_argument('-u', '--rounding',
                            const=True,
                            type=int,
                            nargs='?',
                            metavar='BITS',
                            default=self.defaults.get('rounding', False),
                            help='use stochastic rounding [will round to BITS number '
                                 'of bits if specified]')
        be_grp.add_argument('-d', '--datatype', choices=['f16', 'f32', 'f64'],
                            default=self.defaults.get('datatype', 'f32'),
                            metavar='default datatype',
                            help='default floating point '
                            'precision for backend [f64 for cpu only]')
        be_grp.add_argument('-z', '--batch_size', type=int,
                            default=self.defaults.get('batch_size', 128),
                            help='batch size')
        be_grp.add_argument('--caffe', action='store_true',
                            help='match caffe when computing conv and pool layer output '
                                 'sizes and dropout implementation')
        be_grp.add_argument('--deterministic', action='store_true',
                            help='Use deterministic kernels where applicable')
        return

    def add_yaml_arg(self):
        """
        Add the yaml file argument, this is needed for scripts that
        parse the model config from yaml files

        """
        # yaml configuration file
        self.add_argument('yaml_file',
                          type=configargparse.FileType('r'),
                          help='neon model specification file')

    def add_argument(self, *args, **kwargs):
        """
        Method by which command line arguments are added to the parser.  Passed
        straight through to parent add_argument method.

        Arguments:
            *args:
            **kwargs:
        """
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
        """ Ignored. """
        pass

    # we never use this alias from ConfigArgParse, but defining this here
    # prevents documentation indent warnings
    def add_arg(self):
        """ Ignored. """
        pass

    def parse_args(self, gen_be=True):
        """
        Parse the command line arguments and setup neon
        runtime environment accordingly

        Arguments:
            gen_be (bool): if False, the arg parser will not
                           generate the backend

        Returns:
            namespace: contains the parsed arguments as attributes

        """
        args = super(NeonArgparser, self).parse_args()
        err_msg = None  # used for relaying exception to logger

        # set up the logging
        # max thresh is 50 (critical only), min is 10 (debug or higher)
        try:
            log_thresh = max(10, 40 - args.verbose * 10)
        except (AttributeError, TypeError):
            # if defaults are not set or not -v given
            # for latter will get type error
            log_thresh = 30
        args.log_thresh = log_thresh

        # logging formater
        fmtr = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # get the parent logger for neon
        main_logger = logging.getLogger('neon')
        main_logger.setLevel(log_thresh)

        # setup a console stderr log handler
        stderrlog = logging.StreamHandler()
        stderrlog.setFormatter(fmtr)

        # expand any user directories in paths
        for path in ['data_dir', 'save_path', 'model_file', 'output_file',
                     'logfile']:
            if getattr(args, path):
                setattr(args, path, os.path.expanduser(getattr(args, path)))

        if args.logfile:
            # add log to file as well
            filelog = RotatingFileHandler(filename=args.logfile, mode='w',
                                          maxBytes=10000000, backupCount=5)
            filelog.setFormatter(fmtr)
            filelog.setLevel(log_thresh)
            main_logger.addHandler(filelog)

            # if a log file is specified and progress bar displayed,
            # log only errors to console.
            if args.no_progress_bar:
                stderrlog.setLevel(log_thresh)
            else:
                stderrlog.setLevel(logging.ERROR)
        else:
            stderrlog.setLevel(log_thresh)

        # add this handler instead
        main_logger.propagate = False
        main_logger.addHandler(stderrlog)

        # need to write out float otherwise numpy
        # generates type in bytes not bits (f16 == 128 bits)
        args.datatype = 'float' + args.datatype[1:]
        args.datatype = np.dtype(args.datatype).type

        # invert no_progress_bar meaning and store in args.progress_bar
        args.progress_bar = not args.no_progress_bar

        if args.backend == 'cpu' and args.rounding > 0:
            err_msg = 'CPU backend does not support stochastic rounding'
            logger.exception(err_msg)
            raise NotImplementedError(err_msg)

        # done up front to avoid losing data due to incorrect path
        if args.save_path:
            savedir = os.path.dirname(os.path.abspath(args.save_path))
            if not os.access(savedir, os.R_OK | os.W_OK):
                try:
                    os.makedirs(savedir)
                except OSError:
                    err_msg = 'Can not create save_path %s' % (savedir)
            if os.path.exists(args.save_path):
                logger.warning('save file %s exists, attempting to overwrite' % args.save_path)
                if not os.access(args.save_path, os.R_OK | os.W_OK):
                    err_msg = 'Can not write to save_path file %s' % args.save_path
            if err_msg:
                logger.exception(err_msg)
                raise IOError(err_msg)

        if (args.serialize > 0) and (args.save_path is None):
            args.save_path = "neon_model.pkl"
            logger.warn('No path given for model serialization, using default "%s"',
                        args.save_path)
        if (args.save_path is not None) and (args.serialize == 0):
            args.serialize = 1
            logger.warn('No schedule given for model serialization, using default %d',
                        args.serialize)

        if args.model_file:
            err_msg = None
            if not os.path.exists(args.model_file):
                err_msg = 'Model file %s not present' % args.model_file
            if not os.access(args.model_file, os.R_OK):
                err_msg = 'No read access for model file %s' % args.model_file
            if err_msg:
                logger.exception(err_msg)
                raise IOError(err_msg)

        if args.caffe:
            args.compat_mode = 'caffe'
        else:
            args.compat_mode = None

        if args.deterministic:
            logger.warn('--deterministic flag is deprecated.  Specify random seed for '
                        'deterministic behavior.')
        # extended parsers may need to generate backend after argparsing
        if gen_be:
            # generate the backend
            gen_backend(backend=args.backend,
                        rng_seed=args.rng_seed,
                        device_id=args.device_id,
                        batch_size=args.batch_size,
                        datatype=args.datatype,
                        max_devices=args.max_devices,
                        compat_mode=args.compat_mode)

        # display what command line / config options were set (and from where)
        logger.info(self.format_values())

        if args.manifest:
            args.manifest = {k: v for k, v in [ss.split(':') for ss in args.manifest]}

        self._PARSED = True
        self.args = args
        args.callback_args = extract_valid_args(args, Callbacks.__init__, startidx=1)
        return args
