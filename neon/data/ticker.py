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
For machine generated datasets.
"""
import numpy as np

from neon import NervanaObject


class Task(NervanaObject):
    """
    Base class from which ticker tasks inherit.
    """

    def fetch_io(self, time_steps):
        """
        Generate inputs, outputs numpy tensor pair of size appropriate for this minibatch.

        Arguments:
            time_steps (int): Number of time steps in this minibatch.

        Returns:
            tuple: (input, output) tuple of numpy arrays.

        """
        columns = time_steps * self.be.bsz
        inputs = np.zeros((self.nin, columns))
        outputs = np.zeros((self.nout, columns))
        return inputs, outputs

    def fill_buffers(self, time_steps, inputs, outputs, in_tensor, out_tensor, mask):
        """
        Prepare data for delivery to device.

        Arguments:
            time_steps (int): Number of time steps in this minibatch.
            inputs (numpy array): Inputs numpy array
            outputs (numpy array): Outputs numpy array
            in_tensor (Tensor): Device buffer holding inputs
            out_tensor (Tensor): Device buffer holding outputs
            mask (numpy array): Device buffer for the output mask
        """
        # Put inputs and outputs, which are too small, into properly shaped arrays
        columns = time_steps * self.be.bsz
        inC = np.zeros((self.nin, self.max_columns))
        outC = np.zeros((self.nout, self.max_columns))
        inC[:, :columns] = inputs
        outC[:, :columns] = outputs

        # Copy those arrays to device
        in_tensor.set(inC)
        out_tensor.set(outC)

        # Set a mask over the unused part of the buffer
        mask[:, :columns] = 1
        mask[:, columns:] = 0


class CopyTask(Task):
    """
    Copy task from the Neural Turing Machines paper:
        http://arxiv.org/abs/1410.5401.

    This version of the task is batched.
    All sequences in the same mini-batch are the same length,
    but every new minibatch has a randomly chosen minibatch length.

    When a given minibatch has length < seq_len_max, we mask the outputs
    for time steps after time_steps_max.

    The generated data is laid out in the same way as other RNN data in neon.
    """

    def __init__(self, seq_len_max, vec_size):
        """
        Set up the attributes that ticker needs to see.

        Arguments:
            seq_len_max (int): Longest allowable sequence length
            vec_size (int): Width of the bit-vector to be copied (this was 8 in paper)
        """

        self.seq_len_max = seq_len_max
        self.vec_size = vec_size
        self.nout = self.vec_size  # output has the same dimension as the underlying bit vector
        self.nin = self.vec_size + 2  # input has more dims (for the start and stop channels)
        self.time_steps_func = lambda l: 2 * l + 2
        self.time_steps_max = 2 * self.seq_len_max + 2
        self.time_steps_max = self.time_steps_func(self.seq_len_max)
        self.max_columns = self.time_steps_max * self.be.bsz

    def synthesize(self, in_tensor, out_tensor, mask):

        """
        Create a new minibatch of ticker copy task data.

        Arguments:
            in_tensor (Tensor): Device buffer holding inputs
            out_tensor (Tensor): Device buffer holding outputs
            mask (numpy array): Device buffer for the output mask
        """

        # All sequences in a minibatch are the same length for convenience
        seq_len = np.random.randint(1, self.seq_len_max + 1)
        time_steps = self.time_steps_func(seq_len)

        # Generate intermediate buffers of the right size
        inputs, outputs = super(CopyTask, self).fetch_io(time_steps)

        # Set the start bit
        inputs[-2, :self.be.bsz] = 1

        # Generate the sequence to be copied
        seq = np.random.randint(2,
                                size=(self.vec_size,
                                      seq_len * self.be.bsz))

        # Set the stop bit
        stop_loc = self.be.bsz * (seq_len + 1)
        inputs[-1, stop_loc:stop_loc + self.be.bsz] = 1

        # Place the actual sequence to copy in inputs
        inputs[:self.vec_size, self.be.bsz:stop_loc] = seq

        # Now place that same sequence in a different place in outputs
        outputs[:, self.be.bsz * (seq_len + 2):] = seq

        # Fill the device minibatch buffers
        super(CopyTask, self).fill_buffers(time_steps, inputs, outputs,
                                           in_tensor, out_tensor, mask)


class RepeatCopyTask(Task):
    """
    Repeat Copy task from the Neural Turing Machines paper:
        http://arxiv.org/abs/1410.5401.

    See Also:
        See comments on :py:class:`~neon.data.ticker.CopyTask` class for more details.
    """

    def __init__(self, seq_len_max, repeat_count_max, vec_size):
        """
        Set up the attributes that ticker needs to see.

        Arguments:
            seq_len_max (int): Longest allowable sequence length
            repeat_count_max (int): Max number of repeats
            vec_size (int): Width of the bit-vector to be copied (was 8 in paper)
        """

        self.seq_len_max = seq_len_max
        self.repeat_count_max = seq_len_max
        self.vec_size = vec_size
        self.nout = self.vec_size + 1  # we output the sequence and a stop bit in a stop channel
        self.nin = self.vec_size + 2  # input has more dims (for the start and stop channels)

        # seq is seen once as input, repeat_count times as output, with a
        # start bit, stop bit, and output stop bit
        self.time_steps_func = lambda l, r: l * (r + 1) + 3
        self.time_steps_max = self.time_steps_func(self.seq_len_max, self.repeat_count_max)
        self.max_columns = self.time_steps_max * self.be.bsz

    def synthesize(self, in_tensor, out_tensor, mask):

        """
        Create a new minibatch of ticker repeat copy task data.

        Arguments:
            in_tensor (Tensor): Device buffer holding inputs
            out_tensor (Tensor): Device buffer holding outputs
            mask (numpy array): Device buffer for the output mask
        """
        # All sequences in a minibatch are the same length for convenience
        seq_len = np.random.randint(1, self.seq_len_max + 1)
        repeat_count = np.random.randint(1, self.repeat_count_max + 1)
        time_steps = self.time_steps_func(seq_len, repeat_count)

        # Get the minibatch specific numpy buffers
        inputs, outputs = super(RepeatCopyTask, self).fetch_io(time_steps)

        # Set the start bit
        inputs[-2, :self.be.bsz] = 1

        # Generate the sequence to be copied
        seq = np.random.randint(2,
                                size=(self.vec_size,
                                      seq_len * self.be.bsz))

        # Set the repeat count
        # TODO: should we normalize repeat count?
        stop_loc = self.be.bsz * (seq_len + 1)
        inputs[-1, stop_loc:stop_loc + self.be.bsz] = repeat_count

        # Place the actual sequence to copy in inputs
        inputs[:self.vec_size, self.be.bsz:stop_loc] = seq

        # Now place that same sequence repeat_copy times in outputs
        for i in range(repeat_count):
            start = self.be.bsz * ((i + 1) * seq_len + 2)
            stop = start + seq_len * self.be.bsz
            outputs[:-1, start:stop] = seq

        # Place the output finish bit
        outputs[-1, -self.be.bsz:] = 1

        # Fill the device minibatch buffers
        super(RepeatCopyTask, self).fill_buffers(time_steps, inputs, outputs,
                                                 in_tensor, out_tensor, mask)


class PrioritySortTask(Task):
    """
    Priority Sort task from the Neural Turing Machines paper:
        http://arxiv.org/abs/1410.5401.

    See Also:
        See comments on :py:class:`~neon.data.ticker.CopyTask` class for more details.
    """

    def __init__(self, seq_len_max, vec_size):
        """
        Set up the attributes that ticker needs to see.

        Arguments:
            seq_len_max (int): Longest allowable sequence length
            vec_size (int): Width of the bit-vector to be copied (this was 8 in paper)
        """

        self.seq_len_max = seq_len_max
        self.vec_size = vec_size
        self.nout = self.vec_size  # we output the sorted sequence, with no stop bit
        self.nin = self.vec_size + 3  # extra channels for start, stop, and priority

        # seq is seen once as input with start and stop bits
        # then we output seq in sorted order
        self.time_steps_func = lambda l: 2 * l + 2
        self.time_steps_max = self.time_steps_func(self.seq_len_max)
        self.max_columns = self.time_steps_max * self.be.bsz

    def synthesize(self, in_tensor, out_tensor, mask):

        """
        Create a new minibatch of ticker priority sort task data.

        Arguments:
            in_tensor: device buffer holding inputs
            out_tensor: device buffer holding outputs
            mask: device buffer for the output mask
        """

        # All sequences in a minibatch are the same length for convenience
        seq_len = np.random.randint(1, self.seq_len_max + 1)
        time_steps = self.time_steps_func(seq_len)

        # Get the minibatch specific numpy buffers
        inputs, outputs = super(PrioritySortTask, self).fetch_io(time_steps)

        # Set the start bit
        inputs[-3, :self.be.bsz] = 1

        # Generate the sequence to be copied
        seq = np.random.randint(2,
                                size=(self.nin,
                                      seq_len * self.be.bsz)).astype(float)

        # Zero out the start, stop, and priority channels
        seq[-3:, :] = 0

        # Generate the scalar priorities and put them in seq
        priorities = np.random.uniform(-1, 1, size=(seq_len * self.be.bsz,))
        seq[-1, :] = priorities

        # Set the stop bit
        stop_loc = self.be.bsz * (seq_len + 1)
        inputs[-2, stop_loc:stop_loc + self.be.bsz] = 1

        # Place the actual sequence to copy in inputs
        inputs[:, self.be.bsz:stop_loc] = seq

        # sort the sequences
        for i in range(self.be.bsz):
            # for every sequence in the batch

            # x <- every column in the sequence
            x = seq[:, i::self.be.bsz]

            # sort that set of columns by elt in the last row (the priority)
            x = x[:, x[-1, :].argsort()]

            # put those columns back into minibatch in the right places
            seq[:, i::self.be.bsz] = x

        outputs[:, self.be.bsz * (seq_len + 2):] = seq[:self.nout, :]

        # Fill the device minibatch buffers
        super(PrioritySortTask, self).fill_buffers(time_steps, inputs, outputs,
                                                   in_tensor, out_tensor, mask)


class Ticker(NervanaObject):
    """
    This class defines methods for generating and iterating over ticker datasets.
    """

    def reset(self):
        """
        Reset has no meaning in the context of ticker data.
        """
        pass

    def __init__(self, task):
        """
        Construct a ticker dataset object.

        Arguments:
            task: An object representing the task to be trained on
                  It contains information about input and output size,
                  sequence length, etc. It also implements a synthesize function,
                  which is used to generate the next minibatch of data.
        """

        self.task = task

        # These attributes don't make much sense in the context of tickers
        # but I suspect it will be hard to get rid of them
        self.batch_index = 0
        self.nbatches = 100
        self.ndata = self.nbatches * self.be.bsz

        # Alias these because other code relies on datasets having nin and nout
        self.nout = task.nout
        self.nin = task.nin

        # Configuration elsewhere relies on the existence of this
        self.shape = (self.nin, self.task.time_steps_max)

        # Initialize the inputs, the outputs, and the mask
        self.dev_X = self.be.iobuf((self.nin, self.task.time_steps_max))
        self.dev_y = self.be.iobuf((self.nout, self.task.time_steps_max))
        self.mask = self.be.iobuf((self.nout, self.task.time_steps_max))

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuple : the next minibatch of data.

        Note:
            The second element of the tuple is itself a tuple (t,m) with:
                t: the actual target as generated by the task object
                m: the output mask to account for the difference between
                    the seq_length for this minibatch and the max seq_len,
                    which is also the number of columns in X,t, and m
        """

        self.batch_index = 0

        while self.batch_index < self.nbatches:

            # The task object writes minibatch data into buffers we pass it
            self.task.synthesize(self.dev_X, self.dev_y, self.mask)

            self.batch_index += 1

            yield self.dev_X, (self.dev_y, self.mask)
