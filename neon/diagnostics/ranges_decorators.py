# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
Decorators for dumping parameter ranges and raw values. Useful for tracking
overflow and underflow when using limited precision formats.
"""

import logging
import numpy as np
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)
np.set_printoptions(precision=4)


class Decorators(object):

    supported_funcs = ['fprop_fc', 'bprop_fc', 'update_fc',
                       'fprop_conv', 'bprop_conv', 'update_conv',
                       'gdm_compound', 'gdmwd_compound']

    def __init__(self, backend, verbosity):
        self.backend = backend
        self.verbosity = verbosity
        self.oldepoch = 0
        self.doneness = 0
        backend.raw_dict = defaultdict(lambda: defaultdict(list))
        backend.name_dict = defaultdict(lambda: defaultdict(list))

        self.bins = {'ps_item': np.linspace(-0.25, 0.25, num=40),
                     'vs_item': np.linspace(-.0001, .0001, num=40),
                     'us_item': np.linspace(-.0001, .0001, num=40),
                     'ratioup': np.linspace(-.01, .01, num=40)}

    def decorate(self, function_list):
        """
        Replaces the @decorators in the backend function. Go through the list
        of functions to be decorated and wrap them with the correct parameters
        """
        logger.info("wrapping calls for inspection")
        for call in function_list['decorate_ranges']:
            orig_func = getattr(self.backend, call)
            wrapped_func = self.print_ranges(orig_func)
            setattr(self.backend, call, wrapped_func)

    def store_histograms(self, kwargs, func_name, layer_name):
        """
        Create a histogram of parameter values
        """
        be = self.backend
        epoch = kwargs['epoch']
        hist = np.histogram
        for item in ['ps_item', 'vs_item', 'us_item', 'ratioup']:
            if item in kwargs:
                histo, foo = hist(kwargs[item].asnumpyarray().flatten(),
                                  bins=self.bins[item])
                be.raw_dict[epoch][layer_name].append(histo)
                be.name_dict[epoch][layer_name].append(item)
            elif ('ps_item' in kwargs) and (item == 'ratioup'):
                histo, foo = hist(kwargs['us_item'].asnumpyarray().flatten() /
                                  kwargs['ps_item'].asnumpyarray().flatten(),
                                  bins=self.bins[item])
                be.raw_dict[epoch][layer_name].append(histo)
                be.name_dict[epoch][layer_name].append(item)

    def verbose_logging(self, kwargs, func_name, layer_name):
        """
        Write parameters (weight, activation, deltas, outpus ) raw, std, min
        and max to logger
        """
        be = self.backend
        logger.info("backend call to %s from %s", func_name, layer_name)

        for item in ['weights', 'inputs', 'deltas', 'out']:
            if item in kwargs:
                the_min = be.zeros((1, 1), dtype=np.float32)
                the_max = be.zeros((1, 1), dtype=np.float32)
                be.min(kwargs[item], axes=None, out=the_min)
                be.max(kwargs[item], axes=None, out=the_max)
                logger.info("%s: std=%s raw=%s min=%s max=%s",
                            item.ljust(7),
                            kwargs[item][0:2].asnumpyarray().astype(
                                np.float32).std(1).__str__().ljust(28),
                            kwargs[item][0, 0:2].asnumpyarray(
                                ).__str__().ljust(28),
                            the_min.asnumpyarray()[0, 0].__str__(),
                            the_max.asnumpyarray()[0, 0].__str__())

    def succinct_logging(self, kwargs, func_name, layer_name):
        """
        Write ouput mean and max to logger
        """
        be = self.backend
        for item in ['out']:
            if item in kwargs:
                temp = be.zeros((kwargs[item].shape[0], kwargs[item].shape[1]),
                                dtype=np.float32)
                the_mean = be.zeros((1, 1), dtype=np.float32)
                the_max = be.zeros((1, 1), dtype=np.float32)
                be.mean(be.fabs(kwargs[item], out=temp), axes=None,
                        out=the_mean)
                be.max(kwargs[item], axes=None, out=the_max)
                logger.info("%s to %s %s: mean, max;%s;%s",
                            func_name.ljust(11), layer_name.ljust(7), item,
                            the_mean.asnumpyarray()[0, 0].__str__(),
                            the_max.asnumpyarray()[0, 0].__str__())

    def print_ranges(self, func):
        """
        This function takes a list of tensors and shape indices, and multiplies
        the shapes together. This works well for dot products. The flops are
        scaled with a global multiplier taken from the 'multipliers' dict.
        """
        func_name = func.__name__
        if func_name not in self.supported_funcs:
            raise ValueError("Cannot compute ranges for : %s" % func_name)

        @wraps(func)
        def func_wrapper(*arguments, **kwargs):
            # orig. function call
            retval = func(*arguments, **kwargs)
            layer_name = kwargs['weights'].name if 'weights' in kwargs else \
                kwargs['ps_item'].name if ('ps_item' in kwargs) \
                and hasattr(kwargs['ps_item'], 'name') else 'anon'
            if layer_name is None:
                layer_name = 'anon_layer'

            # histogram plots
            if ('epoch' in kwargs) and (kwargs['epoch'] > self.oldepoch):
                self.doneness = 0  # reset if we reached a new epoch
                self.oldepoch = kwargs['epoch']
            if ('epoch' in kwargs) and (self.doneness < 30):  # not yet done
                self.store_histograms(kwargs, func_name, layer_name)
                self.doneness += 1

            # logging output
            if self.verbosity == 'succinct':
                self.succinct_logging(kwargs, func_name, layer_name)
            elif self.verbosity == 'verbose':
                self.verbose_logging(kwargs, func_name, layer_name)

            return retval
        return func_wrapper
