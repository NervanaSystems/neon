#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
from __future__ import division

import numpy as np
import h5py
import copy

from neon.data.dataiterator import NervanaDataIterator


class SentenceEncode(NervanaDataIterator):
    """
    This class defines an iterator for loading and iterating the sentences in
    a structure to encode sentences into the skip-thought vectors
    """
    def __init__(self, sentences, sentence_text, nsamples, nwords,
                 max_len=100, index_from=2):
        """
        Construct a sentence dataset object.
        Build the context using skip-thought model

        Aguments:
            sentences: list of tokenized (and int-encoded) sentences to use for iteration
            sentence_text: list of raw text sentences
            nsamples: number of sentences
            nwords: number of words in vocab
        """
        super(SentenceEncode, self).__init__(name=None)
        self.nsamples = nsamples
        self.nwords = nwords
        self.batch_index = 0
        self.nbatches = 0
        self.max_len = max_len
        self.index_from = index_from

        # group the sentences to triplets
        source = sentences[:nsamples]
        source_text = sentence_text[:nsamples]
        extra_sent = len(source) % self.be.bsz

        self.nbatches = len(source) // self.be.bsz
        self.ndata = self.nbatches * self.be.bsz  # no leftovers

        if extra_sent:
            source = source[:-extra_sent]
            source_text = source_text[:-extra_sent]

        # get sentence length stats based on the input sentence length
        self.sent_len = dict((i, min(len(c), self.max_len)) for i, c in enumerate(source))

        self.X = source
        self.X_text = source_text

        # inputs using embeddings
        self.dev_X = self.be.iobuf(self.max_len, dtype=np.int32)
        # the np array to put noncontiguous data in. use for entire minibatch
        self.X_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)

        self.shape = (self.max_len, 1)

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        self.batch_index = 0

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset
        """
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            self.X_np.fill(0)

            idx = range(self.batch_index * self.be.bsz, (self.batch_index + 1) * self.be.bsz)
            for i, ix in enumerate(idx):
                s_len = self.sent_len[ix]
                self.X_np[-s_len:, i] = self.X[ix][-s_len:] + self.index_from

            self.dev_X.set(self.X_np)

            self.batch_index += 1

            yield (self.dev_X, None)


# --------------------------------------------------------
# Modified from: Skip-Thoughts
# Licensed under Apache License 2.0 [see LICENSE for details]
# Written by Ryan Kiros
# --------------------------------------------------------
class SentenceHomogenous(NervanaDataIterator):
    """
    This class defines an iterator for loading and iterating the sentences in
    a structure to train the skip-thought vectors
    """
    def __init__(self, data_file=None, sent_name=None, text_name=None,
                 nwords=None, max_len=30, index_from=2, eos=3):
        """
        Construct a sentence dataset object.
        Build the context using skip-thought model

        Aguments:
            data_file (str): path to hdf5 file containing sentences
            sent_name (str): name of tokenized dataset
            text_name (str): name of raw text dataset
            nwords (int): size of vocabulary
            max_len (int): maximum number of words per sentence
            index_from (int): index offset for padding (0) and OOV (1)
            eos (int): index of EOS token
        """
        super(SentenceHomogenous, self).__init__(name=None)
        self.nwords = nwords
        self.batch_index = 0
        self.nbatches = 0
        self.max_len = max_len
        self.index_from = index_from
        self.eos = eos
        self.data_file = data_file
        self.sent_name = sent_name
        self.text_name = text_name

        h5f = h5py.File(self.data_file, 'r+')

        # Extract sentences array from h5 file and make copy in memory
        sentences = h5f[self.sent_name][:]

        # Load sentence raw text if desired
        # sentence_text = h5f[self.text_name]

        self.nsamples = h5f[self.sent_name].attrs['nsample'] - 2

        # Use shifted view of in-memory copy of sentences to group sentences into triplets
        self.source = sentences[1:-1]
        self.forward = sentences[2:]
        self.backward = sentences[:-2]

        self.lengths = [len(cc) for cc in self.source]
        self.len_unique = np.unique(self.lengths)
        self.len_unique = [ll for ll in self.len_unique if ll <= self.max_len]

        self.len_indicies = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indicies[ll] = np.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indicies[ll])

        # Compute number of batches of homogenous lengths
        self.nbatches = 0
        for ll in self.len_unique:
            self.nbatches += int(np.ceil(self.len_counts[ll] / float(self.be.bsz)))

        # compute the total number of samples (including empty samples in minibatches)
        self.ndata = self.nbatches * self.be.bsz

        self.len_curr_counts = copy.copy(self.len_counts)

        # inputs using embeddings
        self.dev_X = self.be.iobuf(self.max_len, dtype=np.int32)
        self.dev_X_p = self.be.iobuf(self.max_len, dtype=np.int32)  # previous sentence as input
        self.dev_X_n = self.be.iobuf(self.max_len, dtype=np.int32)  # next sentence as input

        # the np array to put noncontiguous data in. use for entire minibatch
        self.X_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)
        self.X_p_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)
        self.X_n_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)

        # flat output labels, do one-hot on device
        self.dev_y_p_flat = self.be.iobuf((1, self.max_len), dtype=np.int32)
        self.dev_y_n_flat = self.be.iobuf((1, self.max_len), dtype=np.int32)
        self.dev_y_p = self.be.iobuf((nwords, self.max_len), dtype=np.int32)
        self.dev_y_n = self.be.iobuf((nwords, self.max_len), dtype=np.int32)

        # output labels and masks to deal with variable length sentences
        self.dev_y_p_mask = self.be.iobuf((nwords, self.max_len), dtype=np.int32)
        self.dev_y_n_mask = self.be.iobuf((nwords, self.max_len), dtype=np.int32)
        self.dev_y_p_mask_list = self.get_bsz(self.dev_y_p_mask, self.max_len)
        self.dev_y_n_mask_list = self.get_bsz(self.dev_y_n_mask, self.max_len)

        # for the flat label
        self.y_p_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)
        self.y_n_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)

        self.clear_list = [self.X_np, self.X_p_np, self.X_n_np,
                           self.y_p_np, self.y_n_np,
                           self.dev_y_p_mask,
                           self.dev_y_n_mask]
        self.shape = [(self.max_len, 1), (self.max_len, 1), (self.max_len, 1)]

        h5f.close()
        self.reset()

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        self.batch_index = 0
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = np.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indicies[ll] = np.random.permutation(self.len_indicies[ll])
        self.len_idx = -1

    def next(self):
        """
        Method called by iterator to get a new batch of sentence triplets:
        (source, forward, backward). Sentences are returned in order of increasing length,
        and source sentences of each batch all have the same length.
        """
        self.clear_device_buffer()

        # Select the next length which we havent used up yet
        count = 0
        while True:
            self.len_idx = np.mod(self.len_idx+1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break

        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        curr_len = self.len_unique[self.len_idx]
        # get the batch size
        curr_batch_size = np.minimum(self.be.bsz,
                                     self.len_curr_counts[curr_len])
        curr_pos = self.len_indices_pos[curr_len]

        curr_indices = self.len_indicies[curr_len][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[curr_len] += curr_batch_size
        self.len_curr_counts[curr_len] -= curr_batch_size

        # 'feats' corresponds to the after and before sentences
        source_batch = [self.source[ii] for ii in curr_indices]
        forward_batch = [self.forward[ii] for ii in curr_indices]
        backward_batch = [self.backward[ii] for ii in curr_indices]

        # Loop over the batch and clip by length, add eos, and flip decoder sentences
        for i in range(len(source_batch)):
            l_s = min(len(source_batch[i]), self.max_len)

            if len(source_batch[i][-l_s:]) == 0:
                continue

            # NO FLIPPING of the source sentence
            self.X_np[-l_s:, i] = source_batch[i][-l_s:] + self.index_from

            l_p = min(len(backward_batch[i]), self.max_len)

            # clip a long sentence from the left
            # for decoder input: take the sent_length-1, prepend a <eos>
            # for decoder output: take the sent_length
            self.X_p_np[:l_p, i] = [self.eos] + (backward_batch[i][-l_p:-1] +
                                                 self.index_from).tolist()

            self.y_p_np[:l_p, i] = backward_batch[i][-l_p:] + self.index_from
            self.dev_y_p_mask_list[i][:, :l_p] = 1

            l_n = min(len(forward_batch[i]), self.max_len)

            self.X_n_np[:l_n, i] = [self.eos] + (forward_batch[i][-l_n:-1] +
                                                 self.index_from).tolist()

            self.y_n_np[:l_n, i] = forward_batch[i][-l_n:] + self.index_from
            self.dev_y_n_mask_list[i][:, :l_n] = 1

        self.dev_X.set(self.X_np)
        self.dev_X_p.set(self.X_p_np)
        self.dev_X_n.set(self.X_n_np)

        self.dev_y_p_flat.set(self.y_p_np.reshape(1, -1))
        self.dev_y_n_flat.set(self.y_n_np.reshape(1, -1))

        self.dev_y_p[:] = self.be.onehot(self.dev_y_p_flat, axis=0)
        self.dev_y_n[:] = self.be.onehot(self.dev_y_n_flat, axis=0)

        self.batch_index += 1

        return (self.dev_X, self.dev_X_p, self.dev_X_n), \
            ((self.dev_y_p, self.dev_y_p_mask), (self.dev_y_n, self.dev_y_n_mask))

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset
        Input: clip a long sentence from the left
               encoder input: take sentence and pad 0 from the left
               decoder input: take the sentence length -1, prepend a <eos>, pad 0 from the right
        output: decoder output: take the sentence length, pad 0 from the right
        """
        return self

    def clear_device_buffer(self):
        """ Clear the buffers used to hold batches. """
        if self.clear_list:
            [dev.fill(0) for dev in self.clear_list]

    def get_bsz(self, x, nsteps):
        if x is None:
            return [None for b in range(self.be.bsz)]
        xs = x.reshape(-1, nsteps, self.be.bsz)
        return [xs[:, :, b] for b in range(self.be.bsz)]

    __next__ = next  # Python 3.X compatability
