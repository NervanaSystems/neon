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
"""
Utilities for handling the bilingual text dataset used for
Neural Machine Translation.
"""
import os
import numpy as np
from collections import Counter
import h5py
import tarfile
import gzip
from neon.data.datasets import Dataset
from neon.util.argparser import NeonArgparser


def get_data():
    """
    Download bilingual text dataset for Machine translation example.
    """

    # vocab_size and time_steps are hard coded here
    vocab_size = 16384
    time_steps = 20

    # download dataset
    url = 'http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/'
    filename = 'bitexts.tgz'
    size = 1313280000

    parser = NeonArgparser(__doc__)
    args = parser.parse_args(gen_be=False)
    data_dir = os.path.join(args.data_dir, 'nmt')

    _, filepath = Dataset._valid_path_append(data_dir, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    # extract selected datasets
    datafiles = dict()
    datafiles['un2000'] = ('un2000_pc34.en.gz', 'un2000_pc34.fr.gz')
    datafiles['europarl7'] = ('ep7_pc45.en.gz', 'ep7_pc45.fr.gz')

    extractpath = os.path.join(data_dir, 'bitexts.selected')
    with tarfile.open(filepath, 'r') as tar_ref:
        for dset, files in datafiles.items():
            datasetpath = os.path.join(data_dir, dset)
            # extract the files for dataset, if not already there
            for zipped in files:
                fname = '.'.join(zipped.split('.')[:-1])
                fpath = os.path.join(datasetpath, fname)
                if not os.path.exists(fpath):
                    gzpath = os.path.join(extractpath, zipped)
                    if not os.path.exists(gzpath):
                        select = [ti for ti in tar_ref if os.path.split(ti.name)[1] == zipped]
                        tar_ref.extractall(path=data_dir, members=select)
                    # get contents of gz files
                    if not os.path.exists(datasetpath):
                        os.makedirs(datasetpath)
                    with gzip.open(gzpath, 'r') as fin, open(fpath, 'w') as fout:
                        fout.write(fin.read())
                    os.remove(gzpath)

    if os.path.exists(extractpath):
        os.rmdir(extractpath)

    # process data and save to h5 file
    # loop through all datasets and get train and valid splits
    for dataset in datafiles.keys():

        s_vocab, t_vocab = create_h5py(data_dir, dataset, 'train',
                                       vocab_size=vocab_size, time_steps=time_steps)
        create_h5py(data_dir, dataset, 'valid', s_vocab=s_vocab, t_vocab=t_vocab,
                    time_steps=time_steps)


def parse_vocab(path, vocab_size):
    with open(path, 'r') as f:
        word_counts = Counter()
        blob = []
        for ii, sentence in enumerate(f):
            sentence = sentence.lower().replace('.', '').replace(',', '')
            sentence = sentence.replace('\xe2\x80\x99s', '')
            tokens = sentence.split()
            blob += tokens
            if ii % 100000 == 0:
                word_counts += Counter(blob)
                blob = []
        word_counts += Counter(blob)  # get any leftover fraction
        vocab = [w[0] for w in word_counts.most_common(vocab_size-2)]
        vocab = ['<eos>', '<unk>'] + vocab  # used for LUT size
    return vocab


def vocab_to_dicts(vocab):
    t2i = dict((t, i) for i, t in enumerate(vocab))
    i2t = dict((i, t) for i, t in enumerate(vocab))
    return t2i, i2t


def get_lengths(path, split, time_steps, num_train, num_valid, max_sentence):
    with open(path, 'r') as f:
        lengths = []
        num_short = 0
        for ii, sentence in enumerate(f):
            if (split is 'train' and ii < num_train) or (split is 'valid' and
                                                         ii >= max_sentence - num_valid):
                tokens = sentence.split()
                lengths.append(len(tokens))
                if lengths[-1] <= time_steps:
                    num_short += 1
    return lengths, num_short


def create_data(path, time_steps, t2i, vocab, lengths, split, s_num_short,
                num_train, num_valid, max_sentence):
    X = np.zeros((s_num_short, time_steps))  # init with <eos>
    with open(path, 'r') as f:
        i_sent = 0
        idx = 0
        for ii, sentence in enumerate(f):
            if (split is 'train' and ii < num_train) or (split is 'valid' and
                                                         ii >= max_sentence - num_valid):
                sentence = sentence.lower().replace('.', '').replace(',', '')
                sentence = sentence.replace('\xe2\x80\x99s', '')
                token = sentence.split()
                length = len(token)
                if lengths[idx] <= time_steps:
                    trunc_len = min(length, time_steps)
                    for j in range(trunc_len):
                        j_prime = j + time_steps - trunc_len  # right-align sentences
                        # look up word index in vocab, 1 is <unk>  -- VERY SLOW!
                        X[i_sent, j_prime] = t2i[token[j]] if token[j] in vocab else 1
                    i_sent += 1
                idx += 1
    return X


def create_h5py(data_dir, dataset, split, s_vocab=None, t_vocab=None,
                vocab_size=16384, time_steps=20):

    print("processing {} dataset - {}".format(dataset, split))

    if dataset == 'europarl7':
        basename = 'ep7_pc45'
        num_train = 900000
        num_valid = 2000
        max_sentence = 982178
    elif dataset == 'un2000':
        basename = 'un2000_pc34'
        num_train = 5200000
        num_valid = 2000
        max_sentence = 5259899

    sourcefile = basename + '.fr'
    targetfile = basename + '.en'

    # if h5 data file already exists, do not recreate
    path = os.path.join(data_dir, dataset)
    processed_file = os.path.join(path, dataset + '-' + split + '.h5')
    if os.path.exists(processed_file):
        print("{} already exists, skipping".format(processed_file))
        return None, None

    source = os.path.join(path, sourcefile)
    target = os.path.join(path, targetfile)

    if s_vocab is not None:
        vocab_size = len(s_vocab)

    # if vocab is not given, create from dataset
    s_vocab = parse_vocab(source, vocab_size) if s_vocab is None else s_vocab
    t_vocab = parse_vocab(target, vocab_size) if t_vocab is None else t_vocab
    s_token_to_index, s_index_to_token = vocab_to_dicts(s_vocab)
    t_token_to_index, t_index_to_token = vocab_to_dicts(t_vocab)

    # source sentence lengths
    lengths, s_num_short = get_lengths(source, split, time_steps,
                                       num_train, num_valid, max_sentence)

    # create data matrices
    X = create_data(source, time_steps, s_token_to_index, s_vocab, lengths,
                    split, s_num_short, num_train, num_valid, max_sentence)
    y = create_data(target, time_steps, t_token_to_index, t_vocab, lengths,
                    split, s_num_short, num_train, num_valid, max_sentence)

    # save parsed data
    print("Saving parsed data to {}".format(processed_file))
    with h5py.File(processed_file, 'w') as f:
        f.create_dataset("s_vocab", data=s_vocab)
        f.create_dataset("t_vocab", data=t_vocab)
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)

    return s_vocab, t_vocab

if __name__ == "__main__":
    get_data()
