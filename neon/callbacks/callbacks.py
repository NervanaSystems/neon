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
import os
import sys
import logging
import h5py
from collections import deque
from neon import NervanaObject
from neon.util.persist import save_obj
from timeit import default_timer
from neon.layers import Convolution
import time
import numpy as np
logger = logging.getLogger(__name__)


class Callbacks(NervanaObject):

    """
    Container class for storing and iterating over callbacks.

    Attributes:
        callbacks (list): Ordered set of Callback objects to be run.
    """

    def __init__(self, model, train_set,
                 output_file=None,
                 eval_freq=None,
                 progress_bar=True,
                 epochs=None,
                 save_path=None,
                 serialize=0,
                 history=1,
                 model_file=None,
                 eval_set=None,
                 metric=None):
        """
        Create a callbacks container with the default callbacks.

        Arguments:
            model (Model): the model object
            train_set (DataIterator): the training dataset
            output_file (string, optional): path to save callback data to
            eval_freq (int, optional): how often (in epochs) to run evaluation
            progress_bar (bool): control whether a progress bar
            callback is created.  Defaults to True.
            epochs (int): how many epochs the model will train for (default: None)
            save_path (string): file path to save model snapshots (default: None)
            serialize (int): serialize model every N epochs (default: 0)
            history (int): number of checkpoint files to retain (default: 1)
            model_file(string, optional): file to load weights (serialized model) from
            eval_set (DataIterator, optional): the dataset upon which to evaluate loss
                                               or metric
            metric (Metric, optional):  metric to evaluate
       """
        self.callbacks = list()
        self.epoch_marker = 0
        if output_file is None:
            if hasattr(self, 'callback_data'):
                del self.callback_data
            self.callback_data = h5py.File("no_file", driver='core', backing_store=False)
        else:
            if os.path.isfile(output_file):
                logger.warn("Overwriting output file %s", output_file)
                os.remove(output_file)
            self.callback_data = h5py.File(output_file, "w")

        if model_file:
            model.load_weights(model_file)

        self.model = model
        self.train_set = train_set

        self.add_callback(TrainCostCallback(self.callback_data, self.model))

        if progress_bar:
            self.add_callback(ProgressBarCallback(self.callback_data, model, train_set))

        if eval_freq:
            if not eval_set:
                err_msg = 'Evaluation frequency specified but no eval set provided`!'
                logger.exception(err_msg)
                raise ValueError(err_msg)

            ecb = LossCallback(self.callback_data, model, eval_set, eval_freq)
            self.add_callback(ecb, insert_pos=0)
            if metric:
                ecb = MetricCallback(self.callback_data, model, eval_set, metric, eval_freq)
                self.add_callback(ecb, insert_pos=None)

        if save_path:
            serialize_interval = serialize if serialize > 1 else 1
            checkpoint_schedule = range(0, epochs, serialize_interval)
            if not checkpoint_schedule:
                err_msg = 'For %d epochs and schedule %d, model will not be serialized' % (
                          epochs, serialize)
                logger.exception(err_msg)
                raise ValueError(err_msg)
            scb = SerializeModelCallback(model, save_path, checkpoint_schedule, history)
            self.add_callback(scb)

        self.add_callback(TrainLoggerCallback(self.callback_data, model))
        self.add_callback(RunTimerCallback(self.callback_data, model))

    def add_deconv_callback(self, train_set, valid_set, max_fm=16, dataset_pct=25):
        """
        Convenience function to create and add a deconvolution callback. The data can be used for
        visualization.

        Arguments:
            train_set (DataIterator): the train dataset to use
            valid_set (DataIterator): the validation dataset to use
        """
        self.add_callback(DeconvCallback(self.callback_data, self.model,
                                         train_set, valid_set,
                                         max_fm=max_fm, dataset_pct=dataset_pct))

    def add_save_best_state_callback(self, path):
        """
        Convenience function to create and add a save best state callback.

        Arguments:
            path (string): where to save the best model state.
        """
        self.add_callback(SaveBestStateCallback(self.callback_data, self.model, path))

    def add_early_stop_callback(self, stop_func):
        """
        Convenience function to create and add an early stopping callback.

        Arguments:
            stop_func (function): function to determine when to stop.
        """
        self.add_callback(EarlyStopCallback(self.callback_data, self.model, stop_func))

    def add_hist_callback(self, plot_per_mini=False):
        self.callbacks.append(HistCallback(self.callback_data, self.model,
                              plot_per_mini=plot_per_mini))

    def add_callback(self, callback, insert_pos=None):
        """
        Add a user supplied callback. Since callbacks are run serially and share data,
        order can matter.  If the default behavior (to append the callback) is not
        sufficient, insert position can be controlled.

        Arguments:
            callback (Callback): callback object to be registered
            insert_pos (int, optional): position in the list to insert the callback.
                                        Defaults to None, meaning append
        """
        if insert_pos is None:
            self.callbacks.append(callback)
        else:
            self.callbacks.insert(insert_pos, callback)

    def on_train_begin(self, epochs):
        """
        Call all registered callbacks' on_train_begin functions
        """
        # data iterator wraps around to avoid partial minibatches
        # callbacks producing per-minibatch data need a way to preallocate buffers
        config = self.callback_data.create_group('config')
        total_minibatches = -((-self.train_set.ndata * epochs) // self.be.bsz)
        config.attrs['total_minibatches'] = total_minibatches
        config.attrs['total_epochs'] = epochs

        time_markers = self.callback_data.create_group("time_markers")
        time_markers.create_dataset("minibatch", (epochs,))

        for c in self.callbacks:
            c.on_train_begin(epochs)

    def on_train_end(self):
        """
        Call all registered callbacks' on_train_end functions
        """
        for c in self.callbacks:
            c.on_train_end()

        self.callback_data.close()

    def on_epoch_begin(self, epoch):
        """
        Call all registered callbacks' on_epoch_begin functions

        Arguments:
            epoch (int): index of epoch that is beginning
        """
        for c in self.callbacks:
            if c.should_fire(epoch, c.epoch_freq):
                c.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        """
        Call all registered callbacks' on_epoch_end functions

        Arguments:
            epoch (int): index of epoch that is ending
        """
        for c in self.callbacks:
            if c.should_fire(epoch, c.epoch_freq):
                c.on_epoch_end(epoch)

        self.epoch_marker += self.epoch_minibatches
        self.callback_data['time_markers/minibatch'][epoch] = self.epoch_marker
        self.callback_data['time_markers'].attrs['epochs_complete'] = epoch + 1
        self.callback_data['time_markers'].attrs['minibatches_complete'] = self.epoch_marker
        self.callback_data.flush()

    def on_minibatch_begin(self, epoch, minibatch):
        """
        Call all registered callbacks' on_minibatch_begin functions

        Arguments:
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is beginning
        """
        for c in self.callbacks:
            if c.should_fire(minibatch, c.minibatch_freq):
                c.on_minibatch_begin(epoch, minibatch)

    def on_minibatch_end(self, epoch, minibatch):
        """
        Call all registered callbacks' on_minibatch_end functions

        Arguments:
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is ending
        """
        for c in self.callbacks:
            if c.should_fire(minibatch, c.minibatch_freq):
                c.on_minibatch_end(epoch, minibatch)

        # keep track of the number of mb per epoch, since they vary
        self.epoch_minibatches = minibatch + 1


class Callback(NervanaObject):

    """
    Interface defining common callback functions.

    Implement a callback by subclassing Callback and overriding the necessary
    on_[train,epoch,minibatch]_[begin,end] functions.

    Callback functions provide time queues as arguments but derived callback
    classes must manage their own state
    """

    def __init__(self, epoch_freq=1, minibatch_freq=1):
        self.epoch_freq = epoch_freq
        self.minibatch_freq = minibatch_freq
        self.costnm = None

    def on_train_begin(self, epochs):
        """
        Called when training is about to begin
        """
        pass

    def on_train_end(self):
        """
        Called when training is about to end
        """
        pass

    def on_epoch_begin(self, epoch):
        """
        Called when an epoch is about to begin

        Arguments:
            epoch (int): index of epoch that is beginning
        """
        pass

    def on_epoch_end(self, epoch):
        """
        Called when an epoch is about to end

        Arguments:
            epoch (int): index of epoch that is ending
        """
        pass

    def on_minibatch_begin(self, epoch, minibatch):
        """
        Called when a minibatch is about to begin

        Arguments:
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is begininning
        """
        pass

    def on_minibatch_end(self, epoch, minibatch):
        """
        Called when a minibatch is about to end

        Arguments:
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is ending
        """
        pass

    def should_fire(self, time, freq):
        """
        Helper function for determining if a callback should do work at a given
        interval.

        Arguments:
            time (int): current time, in an arbitrary unit
            freq (int, list, None): firing frequency, in multiples of the unit used
                                    for time, or a list of times, or None (never fire)
        """
        t, f = time, freq
        if ((type(f) is int and (t + 1) % f == 0) or (type(f) is list and t in f)):
            return True
        return False

    def _get_cached_epoch_loss(self, epoch, label):
        """
        Helper function that checks if there exists a loss with a given label at a certain
        epoch index.  Depends on a LossCallback to have previously computed the loss and stored
        in self.callback_data.  Does not actually do any computation.

        Arguments:
            epoch (int): epoch index to check
            label (str): label under which to find cached loss in self.callback_data

        Returns:
            dict containing loss cost value, timing information, and display information
        """

        if self.costnm is None:
            self.costnm = "Loss"  # default costname to display if we can't resolve cost function
            if hasattr(self, 'model') and self.model.cost:
                self.costnm = self.model.cost.costfunc.__class__.__name__ + " " + self.costnm
        cost_key = 'cost/' + label
        time_key = 'time/' + label
        if cost_key not in self.callback_data:
            return None
        eval_freq = self.callback_data[cost_key].attrs['epoch_freq']
        if (epoch + 1) % eval_freq == 0:
            return dict(cost=self.callback_data[cost_key][epoch/eval_freq],
                        time=self.callback_data[time_key][epoch/eval_freq],
                        costnm=self.costnm)


class SerializeModelCallback(Callback):

    """
    Callback for serializing the state of the model.

    Arguments:
        model (Model): model object
        save_path (str): where to save the model dataset
        epoch_freq (int, optional): how often (in epochs) to serialize the
                                   model.  If not specified, we default to
                                   running every epoch.
        history (int, optional): number of checkpoint files to retain, newest
                                 files up to this count are retained.  filename
                                 for the check point files will be
                                 <save_path>_<epoch>.
    """

    def __init__(self, model, save_path, epoch_freq=1, history=1):
        super(SerializeModelCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.save_path = save_path
        self.history = history
        self.checkpoint_files = deque()

    def on_epoch_end(self, epoch):
        if self.history > 1:
            self.save_history(epoch)
        else:
            save_obj(self.model.serialize(keep_states=True), self.save_path)

    def save_history(self, epoch):
        # if history > 1, this function will save the last N checkpoints
        # where N is equal to self.history.  The files will have the form
        # of save_path with the epoch added to the filename before the ext

        if len(self.checkpoint_files) > self.history:
            # remove oldest checkpoint file when max count have been saved
            fn = self.checkpoint_files.popleft()
            try:
                os.remove(fn)
                logger.info('removed old checkpoint %s' % fn)
            except OSError:
                logger.warn('Could not delete old checkpoint file %s' % fn)

        path_split = os.path.splitext(self.save_path)
        save_path = '%s_%d%s' % (path_split[0], epoch, path_split[1])
        # add the current file to the deque
        self.checkpoint_files.append(save_path)
        save_obj(self.model.serialize(keep_states=True), save_path)


class RunTimerCallback(Callback):
    """
    Callback which tracks the total training time

    Arguments:
        callback_data (HDF5 dataset): shared data between callbacks
        model (Model): model object

    """
    def __init__(self, callback_data, model):
        super(RunTimerCallback, self).__init__()
        self.callback_data = callback_data

    def on_train_begin(self, epochs):
        timing = self.callback_data.create_group("time/train")
        timing.create_dataset("start_time", (1,), dtype='float64')
        timing.create_dataset("end_time", (1,), dtype='float64')
        timing['start_time'][0] = time.time()
        timing['start_time'].attrs['units'] = 'seconds'

    def on_train_end(self):
        self.callback_data['time/train/end_time'][0] = time.time()
        self.callback_data['time/train/end_time'].attrs['units'] = 'seconds'


class TrainCostCallback(Callback):
    """
    Callback for computing average training cost periodically during training.

    Arguments:
        callback_data (HDF5 dataset): shared data between callbacks
        model (Model): model object

    """
    def __init__(self, callback_data, model, wsz=10):
        super(TrainCostCallback, self).__init__(epoch_freq=1)
        self.model = model
        self.callback_data = callback_data
        self.wsz = wsz

    def on_train_begin(self, epochs):
        # preallocate space for the number of minibatches in the whole run
        points = self.callback_data['config'].attrs['total_minibatches']
        self.callback_data.create_dataset("cost/train", (points,))

        # make sure our window size is less than or equal to total number of minibatches
        self.wsz = min(points, self.wsz)
        self.cost_history = deque([], maxlen=self.wsz)

        # clue in the data reader to use the 'minibatch' time_markers
        self.callback_data['cost/train'].attrs['time_markers'] = 'minibatch'

    def on_minibatch_end(self, epoch, minibatch):
        self.cost_history.append(self.model.cost.cost.get())
        mean_cost = sum(self.cost_history) / len(self.cost_history)
        mbstart = self.callback_data['time_markers/minibatch'][epoch-1] if epoch > 0 else 0
        self.callback_data['cost/train'][mbstart + minibatch] = mean_cost


class LossCallback(Callback):

    """
    Callback for calculating the loss on a given dataset periodically during training.

    Arguments:
        callback_data (HDF5 dataset): shared data between callbacks
        model (Model): model object
        eval_set (DataIterator): dataset to evaluate
        epoch_freq (int, optional): how often (in epochs) to log info.
                                    Defaults to every 1 epoch.
    """

    def __init__(self, callback_data, model, eval_set, epoch_freq=1):
        super(LossCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.eval_set = eval_set
        self.loss = self.be.zeros((1, 1), dtype=np.float32)
        self.callback_data = callback_data

    def on_train_begin(self, epochs):
        self.callback_data.create_dataset("cost/loss", (epochs/self.epoch_freq,))
        self.callback_data.create_dataset("time/loss", (epochs/self.epoch_freq,))
        self.callback_data["cost/loss"].attrs['time_markers'] = 'epoch_freq'
        self.callback_data["cost/loss"].attrs['epoch_freq'] = self.epoch_freq

    def on_epoch_end(self, epoch):
        start_loss = default_timer()
        nprocessed = 0
        self.loss[:] = 0
        self.eval_set.reset()
        for x, t in self.eval_set:
            x = self.model.fprop(x, inference=True)
            bsz = min(self.eval_set.ndata - nprocessed, self.be.bsz)
            self.model.cost.get_cost(x, t)
            nsteps = x.shape[1] / self.be.bsz
            costbuf = self.model.cost.outputs[:, :bsz*nsteps]
            nprocessed += bsz
            self.loss[:] = self.loss + self.be.sum(costbuf, axis=1)/nsteps
            mean_cost = float(self.loss.get() / nprocessed)
        self.callback_data["time/loss"][epoch/self.epoch_freq] = (default_timer() - start_loss)
        self.callback_data["cost/loss"][epoch/self.epoch_freq] = mean_cost


class MetricCallback(Callback):
    def __init__(self, callback_data, model, eval_set, metric, epoch_freq=1):
        super(MetricCallback, self).__init__(epoch_freq=epoch_freq)
        self.callback_data = callback_data
        self.model = model
        self.eval_set = eval_set
        self.metric = metric
        self.metric_cnt = len(self.metric.metric_names)
        self.metric_desc = ", ".join(self.metric.metric_names)

    def on_train_begin(self, epochs):
        self.callback_data.create_group("metrics")
        for met in self.metric.metric_names:
            group_name = "metrics/%s" % met
            self.callback_data.create_dataset(group_name, (epochs/self.epoch_freq,))
            self.callback_data[group_name].attrs['time_markers'] = 'epoch_freq'
            self.callback_data[group_name].attrs['epoch_freq'] = self.epoch_freq

    def on_epoch_end(self, epoch):
        if (epoch + 1) % self.epoch_freq == 0:
            self.eval_set.reset()
            stats = self.model.eval(self.eval_set, metric=self.metric)
            logger.info('%s: %s', self.metric_desc, ", ".join(map(str, stats.flatten())))

            for ind, met in enumerate(self.metric.metric_names):
                self.callback_data["metrics/%s" % met][epoch/self.epoch_freq] = stats[ind]


class MultiLabelStatsCallback(Callback):

    """
    Callback for calculating statistics on multi-label classification tasks.

    Can be used with PrecisionRecall metric to calculate precision and recall
    values of the classification task.

    Arguments:
        model (Model): model object
        eval_set (DataIterator): dataset to evaluate
        labels (list): the list of class names (order must be the same as
                       the rows of the target)
        metric (Metric): An instantiated performance metric like
                         PrecisionRecall
        epoch_freq (int, optional): how often (in epochs) to log info.
                                    Defaults to every 1 epoch.
    """

    def __init__(self, model, eval_set, labels, metric, epoch_freq=1):
        super(MultiLabelStatsCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.eval_set = eval_set
        self.metric = metric
        self.labels = labels
        self.metric_desc = ", ".join(self.metric.metric_names)

    def on_epoch_end(self, epoch):
        if (epoch + 1) % self.epoch_freq == 0:
            self.eval_set.reset()

            running_stats = np.zeros_like(self.metric.outputs.get(), dtype=np.float32)

            # Calculate the metric values
            nbatch = 0
            for x, t in self.eval_set:
                x = self.model.fprop(x, inference=True)

                self.metric(x, t)
                running_stats += self.metric.outputs.get()
                nbatch += 1

            running_stats /= nbatch

            # Print the statistics for all the labels
            for i, label in enumerate(self.labels):
                metric_text = "["
                for k, metric in enumerate(self.metric.metric_names):
                    metric_text += "%s: %d%% " % (metric, running_stats[k][i+1]*100.0)

                metric_text += " ] -> %s\n" % label
                sys.stdout.write(metric_text.encode('utf-8'))
                sys.stdout.flush()


class HistCallback(Callback):
    """
    Collect histograms of weights of all layers. Configurable to computed
    histograms once per minibatch or once per epoch using the plot_per_mini
    flag. Histograms are stored to the hdf5 output file and can be visualized
    using the nvis tool.
    """
    def __init__(self, callback_data, model, plot_per_mini):
        super(HistCallback, self).__init__(epoch_freq=1, minibatch_freq=1)
        self.callback_data = callback_data
        self.plot_per_mini = plot_per_mini
        self.model = model

    def on_train_begin(self, epochs):
        self.minibatches = self.callback_data['config'].attrs['total_minibatches']

        hist_grp = self.callback_data.create_group("hist")
        hist_grp.attrs['bins'] = self.be.hist_bins
        hist_grp.attrs['offset'] = self.be.hist_offset
        hist_grp.attrs['time_markers'] = 'minibatch' if self.plot_per_mini else 'epoch'
        hist_grp.attrs['time_steps'] = self.minibatches if self.plot_per_mini else epochs

    def on_minibatch_end(self, epoch, minibatch):
        if self.plot_per_mini:
            prev_epochs_minibatches = 0
            if epoch > 0:
                prev_epochs_minibatches = self.callback_data['time_markers/minibatch'][epoch-1]

            timestamp = prev_epochs_minibatches + minibatch
            self._save_hist_data(timestamp)

    def on_epoch_end(self, epoch):
        if not self.plot_per_mini:
            self._save_hist_data(epoch)

    def _save_hist_data(self, timestamp):
        for l_i, l in enumerate(self.model.layers.layers):
            if hasattr(l, 'W'):
                name = "%s_%d_W" % (l.name, l_i)
                l.W.hist(name)

        hist_grp = self.callback_data['hist']
        points = hist_grp.attrs['time_steps']
        hdata, hmap = self.be.dump_hist_data()
        hdata = hdata.get()
        for hname in hmap:
            hist_dset = hist_grp.require_dataset(hname, shape=(64, points), dtype=hdata.dtype)
            hist_dset[:, timestamp] = hdata[hmap[hname]].reshape((64,))


def get_progress_string(tag, epoch, minibatch, nbatches, cost, time,
                        blockchar=u'\u2588'):
    """
    Generate a progress bar string.

    Arguments:
        tag (string): Label to print before the bar (i.e. Train, Valid, Test )
        epoch (int): current epoch to display
        minibatch (int): current minibatch to display
        nbatches (int): total number of minibatches, used to display relative progress
        cost (float): current cost value
        time (float): time elapsed so far in epoch
        blockchar (str, optional): character to display for each step of
                                   progress in the bar.  Defaults to u2588
                                   (solid block)
    """
    max_bar_width = 20
    bar_width = int(float(minibatch) / nbatches * max_bar_width)
    s = u'Epoch {:<3} [{} |{:<%s}| {:4}/{:<4} batches, {:.2f} cost, {:.2f}s]' % max_bar_width
    return s.format(epoch, tag, blockchar * bar_width, minibatch, nbatches, cost, time)


class ProgressBarCallback(Callback):

    """
    Callback providing a live updating console based progress bar.

    Arguments:
        model (Model): model object
        dataset (DataIterator): dataset object
    """

    def __init__(self, callback_data, model, dataset, epoch_freq=1,
                 minibatch_freq=1, update_thresh_s=0.1):
        super(ProgressBarCallback, self).__init__(epoch_freq=epoch_freq,
                                                  minibatch_freq=minibatch_freq)
        self.model = model
        self.dataset = dataset
        self.callback_data = callback_data
        self.update_thresh_s = update_thresh_s
        self._last_strlen = 0

    def on_epoch_begin(self, epoch):
        self.start_epoch = self.last_update = default_timer()
        self.nbatches = self.dataset.nbatches

    def on_minibatch_end(self, epoch, minibatch):
        now = default_timer()
        mb_complete = minibatch + 1
        if (now - self.last_update > self.update_thresh_s or mb_complete == self.nbatches):
            self.last_update = now
            mbstart = self.callback_data['time_markers/minibatch'][epoch-1] if epoch > 0 else 0
            train_cost = self.callback_data['cost/train'][mbstart + minibatch]

            progress_string = get_progress_string("Train", epoch, mb_complete, self.nbatches,
                                                  train_cost, now - self.start_epoch)
            # clear the last line
            sys.stdout.write('\r' + ' '*self._last_strlen + '\r')
            # print the new line
            sys.stdout.write(progress_string.encode('utf-8'))
            self._last_strlen = len(progress_string)
            sys.stdout.flush()

    def on_epoch_end(self, epoch):
        _eil = self._get_cached_epoch_loss(epoch, 'loss')
        if _eil:
            progress_string = " [%s %.2f, %.2fs]" % (_eil['costnm'], _eil['cost'], _eil['time'])
            sys.stdout.write(progress_string.encode('utf-8'))
            sys.stdout.flush()
        sys.stdout.write('\n')


class TrainLoggerCallback(Callback):

    """
    Callback for logging training progress.

    Arguments:
        model (Model): model object

        epoch_freq (int, optional): how often (in epochs) to log training info.
                                    Defaults to every 1 epoch.
        minibatch_freq (int, optional): how often (in minibatches) to log
                                        training info, or None to log only on
                                        epoch boundaries.  Defaults to None.
    """

    def __init__(self, callback_data, model, epoch_freq=1, minibatch_freq=None):
        self.callback_data = callback_data
        self.model = model
        super(TrainLoggerCallback, self).__init__(epoch_freq=epoch_freq,
                                                  minibatch_freq=minibatch_freq)
        self.epoch_freq = epoch_freq
        self.minibatch_freq = minibatch_freq

    def on_train_begin(self, epochs):
        logger.info("Model:\n%s", self.model)

    def on_minibatch_end(self, epoch, minibatch):
        mbstart = self.callback_data['time_markers/minibatch'][epoch-1] if epoch > 0 else 0
        train_cost = self.callback_data['cost/train'][mbstart + minibatch]
        logger.info("Epoch %d Minibatch %d complete. Train cost: %f", epoch, minibatch, train_cost)

    def on_epoch_end(self, epoch):
        _eil = self._get_cached_epoch_loss(epoch, 'loss')
        log_str = "Epoch %d complete.  Train Cost %f." % (epoch, self.model.total_cost.get())
        log_str += "  Eval Cost %f" % _eil['cost'] if _eil else ""
        logger.info(log_str)


class SaveBestStateCallback(Callback):

    """
    Callback for saving the best model state so far.

    Arguments:
        callback_data
        model (Model): model object
        path (str): repeatedly write the best model parameters seen so far to the
                    filesystem path specified.
    """

    def __init__(self, callback_data, model, path):
        super(SaveBestStateCallback, self).__init__(epoch_freq=1)
        self.callback_data = callback_data
        self.model = model
        self.best_path = path
        self.best_cost = None

    def on_epoch_end(self, epoch):
        _eil = self._get_cached_epoch_loss(epoch, 'loss')
        if _eil:
            if _eil['cost'] < self.best_cost or self.best_cost is None:
                save_obj(self.model.serialize(keep_states=True), self.best_path)
                self.best_cost = _eil['cost']


class EarlyStopCallback(Callback):

    """
    Callback for stopping training when a threshold has been triggered.

    Arguments:
        model (Model): model object
        callback_data:
        stop_func (Function): Takes a function that receives a tuple (State, Val[t])
                              of the current state and the validation error at this time
                              and returns a tuple (State', Bool) that returns the updated
                              state and an indication of whether to stop training.
    """

    def __init__(self, callback_data, model, stop_func):
        super(EarlyStopCallback, self).__init__(epoch_freq=1)
        self.callback_data = callback_data
        self.model = model
        self.stop_func = stop_func
        self.stop_state = None  # state needed for the stop func

    def on_epoch_end(self, epoch):
        _eil = self._get_cached_epoch_loss(epoch, 'loss')
        if _eil:
            self.stop_state, finished = self.stop_func(self.stop_state, _eil['cost'])
            if finished:
                self.model.finished = True
                logger.warn('Early stopping function triggered: mean_cost %f.' % (_eil['cost']))


class DeconvCallback(Callback):
    """
    Callback to store data after projecting activations back to pixel space using
    guided backpropagation.  See [Springenberg2014]_ for details.  Meant to be
    used for visualization purposes via nvis.

    Arguments:
        model (Model): model object
        callback_data (HDF5 dataset): shared data between callbacks
        train_set (DataIterator): the training dataset
        max_fm (int, optional): Maximum number of feature maps to visualize per
                                layer.  Defaults to 16.
        dataset_pct (float, optional): Initial portion of validation dataset to
                                       use in finding maximum activations.
                                       Defaults to 25.0 (25%).

    Notes:

    .. [Springenberg2014] http://arxiv.org/abs/1412.6806
    """
    def __init__(self, callback_data, model, train_set, valid_set, max_fm=16, dataset_pct=25):
        super(DeconvCallback, self).__init__(epoch_freq=1)
        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.callback_data = callback_data
        self.max_fm = max_fm
        self.dataset_pct = dataset_pct
        self.name = "Guided Bprop"

    def _progress_update(self, tag, curr, total, unit, time, blockchar=u'\u2588'):
        # clear and redraw progress bar
        max_bar_width = 20
        bar_width = int(float(curr) / total * max_bar_width)
        s = u'Visualization  [{} |{:<%s}| {:4}/{:<4} {}, {:.2f}s]' % max_bar_width
        progress_string = s.format(tag, blockchar * bar_width, curr, total, unit, time)
        sys.stdout.write('\r' + progress_string.encode('utf-8'))
        sys.stdout.flush()

    def on_train_end(self):
        layers = self.model.layers.layers
        self.raw_img_cache = dict()
        self.raw_img_key = dict()
        C, H, W = layers[0].in_shape
        msg = "{} Visualization of {} feature maps per layer:"
        logger.info(msg.format(self.name, self.max_fm))

        for l, lyr in enumerate(layers):
            if isinstance(lyr, Convolution):
                K = lyr.convparams['K']
                num_fm = min(K, self.max_fm)

                lyr_data = self.callback_data.create_group("deconv/max_act/{0:04}".format(l))
                lyr_data.create_dataset("batch_img", (num_fm, 2), dtype='uint16')
                lyr_data.create_dataset("fm_loc", (num_fm, 1), dtype='int16')
                lyr_data.create_dataset("vis", (num_fm, H, W, C), dtype='uint8')
                lyr_data.create_dataset("activation", (num_fm, 1), dtype='float32')
                lyr_data['activation'][:] = -float('Inf')

        self.valid_set.reset()
        t_start = time.time()
        num_sampled_batches = int(self.dataset_pct / 100. *
                                  self.valid_set.nbatches + 0.5)
        for batch_ind, (x, t) in enumerate(self.valid_set, 0):

            if batch_ind > num_sampled_batches:
                break

            imgs_to_store = self.get_layer_acts(x, batch_ind)

            self.store_images(batch_ind, imgs_to_store, x, C, H, W)

            self._progress_update("Find Max Act Imgs", batch_ind,
                                  num_sampled_batches, "batches",
                                  time.time() - t_start)

        sys.stdout.write("\n")

        # Loop over every layer to visualize
        t_start = time.time()
        for i in range(1, len(layers) + 1):
            layer_ind = len(layers) - i

            if isinstance(layers[layer_ind], Convolution):
                num_fm, act_h, act_w = layers[layer_ind].out_shape
                act_size = act_h * act_w
                self.visualize_layer(num_fm, act_size, layer_ind)
            self._progress_update("Compute " + self.name, i,
                                  len(layers), "layers",
                                  time.time() - t_start)

        sys.stdout.write("\n")

    def scale_to_rgb(self, img):
        """
        Convert float data to valid RGB values in the range [0, 255]

        Arguments:
            img (ndarray): the image data

        Returns:
            img (ndarray): image array with valid RGB values
        """
        absMax = np.max((abs(img)))
        minVal = - absMax
        img -= minVal
        maxImg = np.max(img)
        maxVal = max(absMax - minVal, maxImg)
        if maxVal == 0:
            maxVal = 1
        img = img / maxVal * 255
        return img

    def store_images(self, batch_ind, imgs_to_store, img_batch_data, C, H, W):
        n_imgs = len(imgs_to_store)
        if n_imgs:
            img_data = img_batch_data[:, imgs_to_store].get()
            img_store = self.callback_data.create_group('deconv/img/batch_'+str(batch_ind))

            # Store uint8 HWC formatted data for plotting
            img_hwc8 = img_store.create_dataset("HWC_uint8", (H, W, C, n_imgs),
                                                dtype='uint8', compression=True)
            img_hwc_f32 = np.transpose(img_data.reshape((C, H, W, n_imgs)), (1, 2, 0, 3))
            img_hwc8[:] = self.scale_to_rgb(img_hwc_f32)

            # keep image in native format to use for fprop in visualization
            # don't need this beyond runtime so avoid writing to file
            self.raw_img_cache[batch_ind] = img_data

            # Keep a lookup from img_ind -> file position
            # In order to store only needed imgs from batch in flat prealloc array
            self.raw_img_key[batch_ind] = dict()
            for i, img_idx in enumerate(imgs_to_store):
                img_store.attrs[str(img_idx)] = i
                self.raw_img_key[batch_ind][img_idx] = i

    def get_layer_acts(self, x, batch_ind):
        imgs_to_store = set()

        for l, lyr in enumerate(self.model.layers.layers, 0):
            x = lyr.fprop(x, inference=True)

            if not isinstance(lyr, Convolution):
                continue

            num_fm, H, W = lyr.out_shape
            fm_argmax = self.be.zeros((num_fm, 1), dtype=np.int32)
            maxact_idx = self.be.array(np.arange(num_fm) * H * W * self.be.bsz, dtype=np.int32)

            act_data = self.callback_data["deconv/max_act/{0:04}".format(l)]

            all_acts = lyr.outputs.reshape((num_fm, H * W * self.be.bsz))
            all_acts_flat = lyr.outputs.reshape((num_fm * H * W * self.be.bsz))

            fm_argmax[:] = self.be.argmax(all_acts, axis=1)
            maxact_idx[:] = maxact_idx + fm_argmax
            acts_host = all_acts_flat[maxact_idx].get()
            fm_argmax_host = fm_argmax.get()

            num_fm_vis = min(num_fm, self.max_fm)
            for fm in range(num_fm_vis):

                argmax = fm_argmax_host[fm]
                img_ind = int(argmax % self.be.bsz)
                curr_max_act = acts_host[fm]

                if curr_max_act > act_data['activation'][fm]:
                    act_data['activation'][fm] = curr_max_act
                    act_data['batch_img'][fm] = batch_ind, img_ind
                    act_data['fm_loc'][fm] = argmax / self.be.bsz
                    imgs_to_store.add(img_ind)

        return list(imgs_to_store)

    def visualize_layer(self, num_fm, act_size, layer_ind):
        model = self.model
        be = model.be
        act_data = self.callback_data["deconv/max_act/{0:04}".format(layer_ind)]
        layers = model.layers.layers

        # Loop to visualize every feature map
        num_fm_vis = min(num_fm, self.max_fm)
        for fm in range(num_fm_vis):
            batch_ind, img_ind = act_data['batch_img'][fm]

            # Prepare a fake minibatch with just the max activation image for this fm
            img_batch = np.zeros((self.raw_img_cache[batch_ind].shape[0], be.bsz))
            img_cache_offs = self.raw_img_key[batch_ind][img_ind]
            img_batch[:, 0] = self.raw_img_cache[batch_ind][:, img_cache_offs]
            img_batch = be.array(img_batch)

            # Prep model internal state by fprop-ing img
            model.fprop(img_batch, inference=True)

            # Set the max activation at the correct feature map location
            fm_loc = act_data['fm_loc'][fm]
            activation = np.zeros((num_fm, act_size, be.bsz))
            activation[fm, fm_loc, :] = float(act_data['activation'][fm])
            activation = activation.reshape((num_fm * act_size, be.bsz))
            activation = be.array(activation)

            # Loop over the previous layers to perform deconv
            for i, l in enumerate(layers[layer_ind::-1], 0):
                if isinstance(l, Convolution):

                    # zero out w.r.t. current layer activations
                    activation[:] = be.maximum(activation, 0)

                    # output shape of deconv is the input shape of conv
                    C, H, W = [l.convparams[x] for x in ['C', 'H', 'W']]
                    out = be.empty((C * H * W, be.bsz))
                    l.be.bprop_conv(layer=l.nglayer, F=l.W, E=activation, grad_I=out)
                    activation = out

                    # zero out w.r.t to input from lower layer
                    layer_below_acts = layers[layer_ind - i].inputs
                    layer_below_acts[:] = be.greater(layer_below_acts, 0)
                    activation[:] = be.multiply(layer_below_acts, activation)

            C, H, W = layers[0].in_shape
            activation = activation.asnumpyarray().reshape((C, H, W, be.bsz))
            activation = np.transpose(activation, (1, 2, 0, 3))
            act_data['vis'][fm] = self.scale_to_rgb(activation[:, :, :, 0])
