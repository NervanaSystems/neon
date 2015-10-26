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

logger = logging.getLogger(__name__)


class Callbacks(NervanaObject):

    """
    Container class for storing and iterating over callbacks.

    Attributes:
        callbacks (list): Ordered set of Callback objects to be run.
    """

    def __init__(self, model, train_set, parsed_args, valid_set=None):
        """
        Create a callbacks container with the default callbacks.

        Arguments:
            model (Model): the model object
            train_set (DataIterator): the training dataset
            parsed_args (dict): Dictionary of command line args, as follows:
                                output_file (string, optional): path to save callback data to
            valid_freq (int, optional): how often (in epochs) to run validation
            progress_bar (bool): control whether a progress bar callback is created.
                                 Defaults to True.
            save_path (string):
            serialize (int):
            history (int):
            valid_set (DataIterator, optional): the validation dataset to use
       """

        output_file = parsed_args.output_file
        valid_freq = parsed_args.validation_freq
        progress_bar = parsed_args.progress_bar
        epochs = parsed_args.epochs
        save_path = parsed_args.save_path
        serialize = parsed_args.serialize
        history = parsed_args.history

        self.callbacks = list()
        self.epoch_marker = 0
        if output_file is None:
            self.callback_data = h5py.File("no_file", driver='core', backing_store=False)
        else:
            if os.path.isfile(output_file):
                logger.warn("Overwriting output file %s", output_file)
                os.remove(output_file)
            self.callback_data = h5py.File(output_file, "w")
        self.model = model
        self.train_set = train_set

        self.callbacks.append(TrainCostCallback(self.callback_data, self.model))

        if valid_freq:
            if valid_set:
                self.callbacks.append(ValidationCallback(self.callback_data, self.model,
                                                         valid_set, valid_freq))
            else:
                raise ValueError('Valid_freq specified but no validation set given!')
        if progress_bar:
            self.callbacks.append(ProgressBarCallback(self.callback_data, model, train_set))

        if save_path:
            if serialize <= 1:
                checkpoint_schedule = range(epochs)
            else:
                checkpoint_schedule = range(0, epochs, serialize)
            if checkpoint_schedule == []:
                raise ValueError('With the requested number of epochs and '
                                 'serialization schedule, model will never'
                                 'be serialized')
            self.add_serialize_callback(checkpoint_schedule, save_path, history=history)

        self.callbacks.append(TrainLoggerCallback(self.callback_data, model,
                                                  epoch_freq=1, minibatch_freq=None))

    def add_validation_callback(self, valid_set, epoch_freq):
        """
        Convenience function to create and add a Validation callback.

        Arguments:
            valid_set (DataIterator): the validation dataset to use
            epoch_freq (int): how often (in epochs) to run validation
        """
        # Insert before other callbacks since some depend on validation cost
        self.add_callback(ValidationCallback(self.callback_data, self.model,
                                             valid_set, epoch_freq),
                          insert_pos=0)

    def add_serialize_callback(self, serialize_schedule, save_path, history=1):
        """
        Convenience function to create and add a model serialization callback.

        Arguments:
            serialize_schedule (Schedule): the serialization schedule to follow
            save_path (string): where to save the serialized data
            history (int): number of previous checkpoint files to retain
        """
        if save_path and serialize_schedule:
            # TODO can serialize be handled by regular data callback or should it be separate?
            self.callbacks.append(SerializeModelCallback(self.model,
                                                         save_path,
                                                         epoch_freq=serialize_schedule,
                                                         history=history))
        else:
            raise ValueError('Cannot add serialization callback without both'
                             '"save_path" and "serialize_schedule" specified')

    def add_save_best_state_callback(self, path):
        """
        Convenience function to create and add a save best state callback.

        Arguments:
            path (string): where to save the best model state.
        """
        self.callbacks.append(SaveBestStateCallback(self.callback_data, self.model, path))

    def add_early_stop_callback(self, stop_func):
        """
        Convenience function to create and add an early stopping callback.

        Arguments:
            stop_func (function): function to determine when to stop.
        """
        self.callbacks.append(EarlyStopCallback(self.callback_data, self.model, stop_func))

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
            insert_pos = len(self.callbacks)
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
        fire = False
        if isinstance(freq, int) and (time + 1) % freq == 0:
            fire = True
        elif isinstance(freq, list) and time in freq:
            fire = True
        return fire


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

        prev_epoch_minibatches = 0
        if epoch > 0:
            prev_epoch_minibatches = self.callback_data['time_markers/minibatch'][epoch-1]

        self.callback_data['cost/train'][prev_epoch_minibatches + minibatch] = mean_cost


class ValidationCallback(Callback):

    """
    Callback for processing the validation dataset periodically during training.

    Arguments:
        callback_data (HDF5 dataset): shared data between callbacks
        model (Model): model object
        valid_set (DataIterator): Validation dataset to process
        epoch_freq (int, optional): how often (in epochs) to log training info.
                                    Defaults to every 1 epoch.
        minibatch_freq (int, optional): how often (in minibatches) to log
                                        training info, or None to log only on
                                        epoch boundaries.  Defaults to None.
    """

    def __init__(self, callback_data, model, valid_set, epoch_freq=1):
        super(ValidationCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.valid_set = valid_set
        self.valid_cost = self.be.zeros((1, 1))
        self.callback_data = callback_data

    def on_train_begin(self, epochs):
        vdata = self.callback_data.create_dataset("cost/validation", (epochs/self.epoch_freq,))
        vdata.attrs['time_markers'] = 'epoch_freq'
        vdata.attrs['epoch_freq'] = self.epoch_freq
        self.callback_data.create_dataset("time/validation", (epochs/self.epoch_freq,))

    def on_epoch_end(self, epoch):
        model = self.model
        start_validation = default_timer()
        nprocessed = 0
        self.valid_cost[:] = 0
        self.valid_set.reset()
        for batch_index, (x, t) in enumerate(self.valid_set, 1):
            x = model.fprop(x, inference=True)
            bsz = min(self.valid_set.ndata - nprocessed, self.be.bsz)
            model.cost.get_cost(x, t)
            nsteps = x.shape[1] / self.be.bsz
            costbuf = model.cost.outputs[:, :bsz*nsteps]
            nprocessed += bsz
            self.valid_cost[:] = self.valid_cost + self.be.sum(costbuf, axis=1)/nsteps
            mean_cost = float(self.valid_cost.get() / nprocessed)

        end_validation = default_timer()
        self.callback_data["cost/validation"][epoch/self.epoch_freq] = mean_cost
        self.callback_data["time/validation"][epoch/self.epoch_freq] = (end_validation
                                                                        - start_validation)


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
        if (now - self.last_update > self.update_thresh_s or
                mb_complete == self.nbatches):
            self.last_update = now
            prev_epoch_minibatches = 0
            if epoch > 0:
                prev_epoch_minibatches = self.callback_data['time_markers/minibatch'][epoch-1]

            train_cost = self.callback_data['cost/train'][prev_epoch_minibatches + minibatch]
            progress_string = get_progress_string("Train", epoch, mb_complete,
                                                  self.nbatches, train_cost,
                                                  now - self.start_epoch)
            # clear the last line
            sys.stdout.write('\r' + ' '*self._last_strlen + '\r')
            # print the new line
            sys.stdout.write(progress_string.encode('utf-8'))
            self._last_strlen = len(progress_string)
            sys.stdout.flush()

    def on_epoch_end(self, epoch):

        if 'cost/validation' in self.callback_data:
            val_freq = self.callback_data['cost/validation'].attrs['epoch_freq']
            if (epoch + 1) % val_freq == 0:
                validation_cost = self.callback_data['cost/validation'][epoch/val_freq]
                validation_time = self.callback_data['time/validation'][epoch/val_freq]
                progress_string = "[Validation %.2f cost, %.2fs]" % (validation_cost,
                                                                     validation_time)
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

    def on_minibatch_end(self, epoch, minibatch):
        prev_epoch_minibatches = 0
        if epoch > 0:
            prev_epoch_minibatches = self.callback_data['time_markers/minibatch'][epoch-1]
        train_cost = self.callback_data['cost/train'][prev_epoch_minibatches + minibatch]
        logger.info("Epoch %d Minibatch %d complete. Train cost: %f", epoch, minibatch, train_cost)

    def on_epoch_end(self, epoch):
        log_str = "Epoch %d complete. Train Cost %f" % (epoch,
                                                        self.model.total_cost.get())
        if 'cost/validation' in self.callback_data:
            val_freq = self.callback_data['cost/validation'].attrs['epoch_freq']
            if (epoch + 1) % val_freq == 0:
                validation_cost = self.callback_data['cost/validation'][epoch/val_freq]
                log_str += ", Validation Cost %f" % (validation_cost)

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

        if 'cost/validation' in self.callback_data:
            val_freq = self.callback_data['cost/validation'].attrs['epoch_freq']
            if (epoch + 1) % val_freq == 0:
                validation_cost = self.callback_data['cost/validation'][epoch/val_freq]

                if validation_cost < self.best_cost or self.best_cost is None:
                    save_obj(self.model.serialize(keep_states=True), self.best_path)
                    self.best_cost = validation_cost


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
        if 'cost/validation' in self.callback_data:
            val_freq = self.callback_data['cost/validation'].attrs['epoch_freq']
            if (epoch + 1) % val_freq == 0:
                validation_cost = self.callback_data['cost/validation'][epoch/val_freq]

                self.stop_state, finished = self.stop_func(self.stop_state, validation_cost)

                if finished:
                    # should this just exit instead?
                    self.model.finished = True
                    logger.warn('Early stopping function has been triggered with mean_cost %f.'
                                % (validation_cost))
