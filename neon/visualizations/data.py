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
import h5py
import numpy as np


def create_minibatch_x(minibatches, minibatch_markers, epoch_axis):
    """
    Helper function to build x axis for data captured per minibatch

    Arguments:
        minibatches (int): how many total minibatches
        minibatch_markers (int array): cumulative number of minibatches complete at a given epoch
        epoch_axis (bool): whether to render epoch or minibatch as the integer step in the x axis
    """
    if epoch_axis:
        x = np.zeros((minibatches,))
        last_e = 0
        for e_idx, e in enumerate(minibatch_markers):
            e_minibatches = e - last_e
            x[last_e:e] = e_idx + (np.arange(float(e_minibatches))/e_minibatches)
            last_e = e
    else:
        x = np.arange(minibatches)

    return x


def create_epoch_x(points, epoch_freq, minibatch_markers, epoch_axis):
    """
    Helper function to build x axis for points captured per epoch

    Arguments:
        points (int): how many data points need a corresponding x axis points
        epoch_freq (int): are points once an epoch or once every n epochs?
        minibatch_markers (int array): cumulative number of minibatches complete at a given epoch
        epoch_axis (bool): whether to render epoch or minibatch as the integer step in the x axis
    """

    if epoch_axis:
        x = np.zeros((points,))
        last_e = 0
        for e_idx, e in enumerate(minibatch_markers):
            e_minibatches = e - last_e
            if (e_idx + 1) % epoch_freq == 0:
                x[e_idx/epoch_freq] = e_idx + (e_minibatches - 1) / e_minibatches
            last_e = e
    else:
        x = minibatch_markers[(epoch_freq-1)::epoch_freq] - 1

    return x


def h5_cost_data(filename, epoch_axis=True):
    """
    Read cost data from hdf5 file. Generate x axis data for each cost line.

    Returns:
        list of tuples of (name, x data, y data)
    """
    ret = list()
    with h5py.File(filename, "r") as f:

        config, cost, time_markers = [f[x] for x in ['config', 'cost', 'time_markers']]
        total_epochs = config.attrs['total_epochs']
        total_minibatches = config.attrs['total_minibatches']
        minibatch_markers = time_markers['minibatch']

        for name, ydata in cost.iteritems():
            y = ydata[...]

            if ydata.attrs['time_markers'] == 'epoch_freq':
                y_epoch_freq = ydata.attrs['epoch_freq']
                assert len(y) == total_epochs / y_epoch_freq
                x = create_epoch_x(len(y), y_epoch_freq, minibatch_markers, epoch_axis)

            elif ydata.attrs['time_markers'] == 'minibatch':
                assert len(y) == total_minibatches
                x = create_minibatch_x(total_minibatches, minibatch_markers, epoch_axis)

            else:
                raise TypeError('Unsupported data format for h5_cost_data')

            ret.append((name, x, y))

    return ret


def h5_hist_data(filename, epoch_axis=True):
    ret = list()
    with h5py.File(filename, "r") as f:
        if 'hist' in f:
            hists, config = [f[x] for x in ['hist', 'config']]
            bins, offset, time_markers = [hists.attrs[x]
                                          for x in ['bins', 'offset', 'time_markers']]
            total_epochs = config.attrs['total_epochs']
            total_minibatches = config.attrs['total_minibatches']

            for hname, hdata in hists.iteritems():
                dw = total_epochs if (time_markers == 'epoch_freq') else total_minibatches
                dh = bins
                ret.append((hname, hdata[...], dh, dw, bins, offset))

    return ret
