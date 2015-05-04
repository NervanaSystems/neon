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
Logging and visalization for the data collected from backend timing decorators
"""

import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')  # not for plotting but write to file.
from matplotlib import pyplot as plt  # with a middlefinger to pep8: # noqa
matplotlib.rcParams['pdf.fonttype'] = 42  # TTF to be editable

logger = logging.getLogger(__name__)


def print_performance_stats(backend, logger):

    call_list = backend.flop_dict.keys()
    used_call_list = []
    timed_calls = []
    timed_times = []
    total_time = 0
    total_tflop = 0
    for call in call_list:
        logger.info("Performed %s calls in %2.2fs:" +
                    " %0.2fTFLOPS from %2.0fGFLOP",
                    (str(len(backend.flop_dict[call]))+" "+call).ljust(15),
                    sum(backend.time_dict[call]),
                    sum(backend.flop_dict[call]) /
                    sum(backend.time_dict[call]) / 1e12,
                    sum(backend.flop_dict[call]) / 1e9)

        # Histogram of where the time is spent.
        tflop_array = np.array(backend.flop_dict[call]) / 1e12
        time_array = np.array(backend.time_dict[call])
        total_time += time_array.sum()
        total_tflop += tflop_array.sum()
        tflop_per_s = tflop_array / time_array  # in GFLOP/s
        # plot only the biggest contributors
        if time_array.sum() > .001:
            used_call_list.append(call)
            timed_calls.append(tflop_per_s)
            timed_times.append(time_array)

    # gather data for plots
    paren_stash = get_parent_timing(used_call_list, backend)
    lfs, lts, soumith_stash = get_flops_times(used_call_list, backend)

    # plot the plots
    sufx = 'test' + backend.__module__
    fname1 = 'figure1_'+sufx+'.pdf'
    fname2 = 'figure2_'+sufx+'.pdf'
    first_fig(paren_stash, used_call_list, timed_calls,
              timed_times, total_time, total_tflop, fname1)
    second_fig(lfs, lts, fname2)


def get_parent_timing(used_call_list, backend):
    # compute timing per parent call:
    paren_stash = dict()
    for call in used_call_list:
        unique_paren_list = set(backend.paren_dic[call])
        for paren in unique_paren_list:
            # add up times for "call" from "paren"
            time_stats = np.array([backend.time_dict[call][i]
                                   for i, x
                                   in enumerate(backend.paren_dic[call])
                                   if x == paren]).sum()
            paren_stash[call + " from " + paren] = time_stats
    return paren_stash


def get_flops_times(used_call_list, backend):
    # compute timing per layer call:
    layer_flops_stash = dict()
    layer_time_stash = dict()
    soumith_stash = dict()
    for call in used_call_list:
        unique_layer_list = set(backend.layer_dic[call])
        for layer in unique_layer_list:
            # add up times for "call" from "paren"
            time_stats = np.array([backend.time_dict[call][i]
                                   for i, x
                                   in enumerate(backend.layer_dic[call])
                                   if x == layer]).sum()
            soumith_be = np.array([backend.time_dict[call][i]
                                   for i, x
                                   in enumerate(backend.layer_dic[call])
                                   if x == layer]).mean()

            flop_stats = np.array([backend.flop_dict[call][i]
                                   for i, x
                                   in enumerate(backend.layer_dic[call])
                                   if x == layer]).sum()
            calllayer = call + " from " + layer
            layer_flops_stash[calllayer] = flop_stats / time_stats / 1e9
            layer_time_stash[calllayer] = time_stats
            soumith_stash[calllayer] = 1000. * soumith_be
    return (layer_flops_stash, layer_time_stash, soumith_stash)


def first_fig(paren_stash, used_call_list, timed_calls, timed_times,
              total_time, total_tflop, fname):
    """
    First figure:
    a) one bar plot of time by function call and parent function.
    b) one histogram of Time spent vs. FLOPS achieved
    """

    paren_col_stash = ['b' if 'fprop_fc' in k else
                       'g' if 'bprop_fc' in k else
                       'r' if 'update_fc' in k else
                       'c' if 'fprop_conv' in k else
                       'm' if 'bprop_conv' in k else
                       'y' if 'date_conv' in k else
                       'k' for k in paren_stash.keys()]

    plt.figure(1, figsize=(12, 6), dpi=120, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    plt.subplot(1, 2, 1)
    plt.barh(range(len(paren_stash)), paren_stash.values(),
             color=paren_col_stash, align='center', alpha=0.5)
    plt.yticks(range(len(paren_stash)), paren_stash.keys())
    plt.title(r'Breakdown of MOP calls by parent')
    plt.xlabel('Time/s')

    # Second plot: speed vs. time

    times_col_stash = ['b' if 'fprop_fc' in k else
                       'g' if 'bprop_fc' in k else
                       'r' if 'update_fc' in k else
                       'c' if 'fprop_conv' in k else
                       'm' if 'bprop_conv' in k else
                       'y' if 'date_conv' in k else
                       'k' for k in used_call_list]

    plt.subplot(1, 2, 2)
    num_bins = 30
    n, bins, patches = plt.hist(timed_calls, num_bins,
                                weights=timed_times, range=(0, 7.5),
                                color=times_col_stash,
                                histtype='barstacked', normed=0, alpha=0.5)
    plt.title(r'Total %2.1fs %2.0fTF average %2.1fTFLOP/s'
              % (total_time, total_tflop, total_tflop/total_time))
    plt.xlabel('TFLOPS')
    plt.xlim((0, 7.5))
    plt.ylabel('Time (s)')
    plt.legend(used_call_list, prop={'size': 6})
    plt.savefig(fname, dpi=500)   # savefig overrides dpi value


def second_fig(layer_flops_stash, layer_time_stash, fname):

    layer_col_stash = ['b' if 'conv1' in k else
                       'g' if 'conv2' in k else
                       'r' if 'fc' in k else
                       'c' if 'anon' in k else
                       'm' if 'output' in k else
                       'k' for k in layer_flops_stash.keys()]

    plt.figure(2, figsize=(12, 6), dpi=120, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    plt.subplot(1, 2, 1)
    plt.barh(range(len(layer_flops_stash)), layer_flops_stash.values(),
             color=layer_col_stash, align='center', alpha=0.5)
    plt.yticks(range(len(layer_flops_stash)), layer_flops_stash.keys())
    plt.title(r'Breakdown of MOP calls by layer')
    # plt.xlim((0, 5.5))
    plt.xlabel('TFLOPS')

    # second plot: time per call
    plt.subplot(1, 2, 2)
    plt.barh(range(len(layer_flops_stash)), layer_time_stash.values(),
             color=layer_col_stash, align='center', alpha=0.5)
    plt.yticks(range(len(layer_flops_stash)), range(len(layer_flops_stash)))
    plt.title(r'Breakdown of MOP calls by layer')
    # plt.xlim((0, 7))
    plt.xlabel('Time (s)')

    plt.savefig(fname, dpi=500)


def log_soumith_numbers(soumith_stash, layer_flops_stash, layer_time_stash):
    """
    print out soumith benchmark numers
    """
    logger.info("Soumith Benchmarks")
    sum_of_all_calls = 0
    for i, key in enumerate(soumith_stash.keys()):
        logger.info("Performed %s in\t %2.2f ms per call with 10 calls" +
                    "totaling to %2.2f GFLOPS, %2.2fGFLOP", key,
                    soumith_stash[key], layer_flops_stash[key],
                    layer_flops_stash[key]*layer_time_stash[key])
        sum_of_all_calls += soumith_stash[key]
    logger.info("Total time in call %2.2f ms ", sum_of_all_calls)
