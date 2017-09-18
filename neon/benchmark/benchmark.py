# ----------------------------------------------------------------------------
# Copyright 2015-2017 Nervana Systems Inc.
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

from functools import wraps

from neon import NervanaObject
from collections import OrderedDict
from neon import __version__ as __neon_version__
from neon import logger as neon_logger
from neon.layers.container import LayerContainer
import numpy as np


@property
def flatten_layers(self):
    lf = []
    for l in self.layers:
        if isinstance(l, LayerContainer):
            lf += l.flatten_layers
        else:
            lf.append(l)
    return lf


class Benchmark(object):

    def __init__(self, model, time_layers=True):
        self.model = model
        self.layer_times = OrderedDict()
        if not time_layers:
            return

        self.start = self.model.be.init_mark()
        self.end = self.model.be.init_mark()
        LayerContainer.flatten_layers = flatten_layers
        for idx, layer in enumerate(self.model.layers.flatten_layers):
            l_name = layer.name

            f_l_name = 'fprop ' + l_name
            b_l_name = 'bprop ' + l_name
            self.layer_times[f_l_name] = []
            self.layer_times[b_l_name] = []
            layer.fprop = Benchmark.benchmarked(layer.fprop, self.start,
                                                self.end, self.layer_times[f_l_name])
            layer.bprop = Benchmark.benchmarked(layer.bprop, self.start,
                                                self.end, self.layer_times[b_l_name])

    @staticmethod
    def benchmarked(func, start, end, output):
        '''
        :param func: Function that is timed
        :param start: Start marker
        :param end: End marker
        :param output: Array to which func execution time will be appended
        :return: Wrapped function
        '''
        nerv_be = NervanaObject.be

        @wraps(func)
        def wrapper(*args, **kwargs):
            nerv_be.record_mark(start)
            res = func(*args, **kwargs)
            nerv_be.record_mark(end)
            nerv_be.synchronize_mark(end)
            output.append(nerv_be.get_time(start, end))
            return res

        return wrapper

    def time(self, dataset, inference=False, niterations=20):
        """
        Measure runtime for computing fprop and bprop separately, as well as
        full minibatch run times. For inference case, only the fprop is measured.

        Arguments:
             dataset (NervanaDataIterator) Dataset iterator to perform fit on

             inference (bool, optional): Is inference use case

             niterations (optional, int): Number of minibatches to average over
        Returns:
            dictionary with fprop, bprop run times
        """
        if not dataset:
            raise ValueError("Provide correct dataset for timing")

        if niterations < 1:
            raise ValueError("Invalid iterations number")

        if inference is False and (self.model.cost is None or self.model.optimizer is None):
            raise RuntimeError("Need cost or optimizer to benchmark bprop")

        # iterate through minibatches of the dataset
        times = OrderedDict()
        time_keys = ['fprop'] if inference else ['fprop', 'bprop', 'iteration']
        for ky in time_keys:
            times[ky] = np.full(niterations, -1.0)
        count = 0

        fprop_start = self.model.be.init_mark()
        fprop_end = self.model.be.init_mark()
        bprop_end = self.model.be.init_mark()

        while count < niterations:
            dataset.reset()
            for _, (x, t) in enumerate(dataset):

                self.model.be.record_mark(fprop_start)  # mark start of fprop
                x = self.model.fprop(x, inference)
                self.model.be.record_mark(fprop_end)  # mark end of fprop and start of bprop

                if inference is False:
                    self.model.cost.get_cost(x, t)
                    delta = self.model.cost.get_errors(x, t)
                    self.model.bprop(delta)
                    self.model.optimizer.optimize(self.model.layers_to_optimize, epoch=0)

                    self.model.be.record_mark(bprop_end)  # mark end of bprop
                    self.model.be.synchronize_mark(bprop_end)
                else:
                    self.model.be.synchronize_mark(fprop_end)

                times['fprop'][count] = self.model.be.get_time(fprop_start, fprop_end)
                if inference is False:
                    times['bprop'][count] = self.model.be.get_time(fprop_end, bprop_end)
                    times['iteration'][count] = times['fprop'][count] + times['bprop'][count]

                count += 1
                if count >= niterations:
                    break

        times.update(self.layer_times)
        return times

    @staticmethod
    def print_stats(stats, functions=None, output=None, nskip=0):
        '''
        Function prints provided statistics
        :param stats: Dictionary with statistics
        :param functions: Functions to be executed on statistics
        :param output: Object which provides display method for printing
        :param nskip: Number of measurements to be skipped in functions
        '''
        if not stats:
            raise ValueError('Provide stats')

        if not output:
            output = neon_logger
        elif not hasattr(output, 'display'):
            raise TypeError('Argument output must implement display method')

        if not functions:
            functions = (np.mean, np.median, np.min, np.max)

        function_names = tuple(function.__name__.title() for function in functions)

        fmt_titles = '| {name:^33} '.format(name='Func') + '| {:^11} ' * len(function_names)\
                     + '|    Units    |'
        fmt_nums = '| {func:<33} ' + '|  {%s:<10.5g} ' * len(function_names) % function_names\
                   + '| {units:^11} |'

        head_str = fmt_titles.format(*function_names)
        sep = '+' + '-' * (len(head_str) - 2) + '+'

        msg = '\n{sep:}\n|{prefix:-^105}|\n'\
            .format(sep=sep, prefix='  Intel Neon Benchmark: 0.0.1 Neon Version: %s  '
                                    % __neon_version__[:5])
        msg += '%s\n%s\n%s\n' % (sep, head_str, sep)
        out_stats = {}
        for step in stats:
            if not any(stats[step]):
                continue
            timesu = np.array(stats[step][nskip:])  # in ms
            out_stats[step] = {}
            for function in functions:
                out_stats[step][function.__name__.title()] = function(timesu)
            msg += '%s\n' % fmt_nums.format(units='msec', func=step, **out_stats[step])
        msg += sep

        output.display(msg)
