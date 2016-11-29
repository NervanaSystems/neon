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
from __future__ import print_function

import os
import h5py
import json
import numpy as np
from collections import defaultdict, OrderedDict
import sys

from neon import logger as neon_logger
from neon.util.persist import load_obj, save_obj

from util import tokenize, clean_string


def load_json(fname):
    """
    load json object from file
    """
    with open(fname) as f:
        data = json.load(f)
    return data


def load_json_sent(flist_json, subset_pct):
    """
    load all the sentences from a list of JSON files (with out data format)
    and return a list
    """
    subset_fnum = int(np.ceil((subset_pct / 100.0) * len(flist_json)))

    all_sent = []
    for f in flist_json[:subset_fnum]:
        data = load_json(f)
        num_sent = len(data['text'])
        if num_sent <= 0:
            continue
        num_sent = len(data['text'])
        sent = [data['text'][i]['sentence'] for i in range(num_sent)]
        all_sent += sent
    return all_sent


def load_txt_sent(flist_txt, subset_pct):
    """
    load all the senteces from a list of txt files using standard file io
    """
    # Subset dataset based on percentage of bytes
    total_size = sum(map(os.path.getsize, flist_txt))
    subset_size = int((subset_pct / 100.0) * total_size)

    all_sent = []
    for txt_file in flist_txt:
        if subset_size > 0:
            with open(txt_file, 'r') as f:
                data = f.read(subset_size)

            subset_size -= sys.getsizeof(data)

            sent = data.split('\n')
            # Discard the last sentence since it is most likely partial
            if subset_size <= 0:
                sent = sent[:-1]
            all_sent += sent

    return all_sent


def get_file_list(data_dir, file_ext):
    """
    Return list of files with the given extension in the data_dir
    """
    file_ext = file_ext if isinstance(file_ext, list) else [file_ext]
    file_names = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)
                  if any(fn.endswith(ext) for ext in file_ext)]

    return file_names


def get_file_str(path, num_files, labelled=False, valid_split=None,
                 split_count_thre=None, subset_pct=100):
    """
    Create unique file name for processed data from the number of files, directory name,
    validation split type, and subset percentage
    """
    # grab the directory name as part of the names
    dir_name = path.split('/')[-1] if len(path.split('/')[-1]) > 0 else path.split('/')[-2]
    label_str = 'labelled' if labelled else ''
    split_thre_str = 'thre_{}'.format(split_count_thre) if split_count_thre else ''
    dir_str = 'doc_{}_{}_{}_{}'.format(label_str, dir_name, num_files, split_thre_str)
    if valid_split:
        split_str = '_split_{}'.format(valid_split*100)
    else:
        split_str = ''
    if subset_pct != 100:
        subset_str = '_subset_{}'.format(subset_pct)
    else:
        subset_str = ''
    file_str = dir_str + split_str + subset_str
    return file_str


def load_data(path, file_ext=['txt'], valid_split=None, vocab_file_name=None,
              max_vocab_size=None, max_len_w=None, output_path=None, subset_pct=100):
    """
    Given a path where data are saved, look for the ones with the right extensions
    If a split factor is given, it will split all the files into training and valid
    set. Then build vocabulary from the training and validation sets.

    Arguments:
        path: which directory to look for all the documents
        file_ext: what extension of the files to look for
        valid_split: to split the data into train/valid set. If None, no split
        vocab_file_name: optional file name. If None, the script will decide a name
                         given path and split
        max_vocab_size: maximum number of words to use in vocabulary (by most frequent)
        max_len_w: maximum length of sentences in words
        output_path: path used to save preprocessed data and resuts
        subset_pct: subset of dataset to load into H5 file (percentage)

    Returns:
        The function saves 2 files:
        h5 file with preprocessed data
        vocabulary file with: vocab, reverse_vocab, word_count
    """
    file_names = get_file_list(path, file_ext)

    file_str = get_file_str(path, len(file_names), labelled=False,
                            valid_split=valid_split, subset_pct=subset_pct)

    # create output dir if needed
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # file name to store the vocabulary
    if vocab_file_name is None:
        vocab_file_name = file_str + '.vocab'
        vocab_file_name = os.path.join(output_path, vocab_file_name)

    # If max sizes arent set, assume no limit
    if not max_len_w:
        max_len_w = sys.maxsize
    if not max_vocab_size:
        max_vocab_size = sys.maxsize

    # file name to store the pre-processed train/valid dataset
    h5_file_name = os.path.join(output_path, file_str + '.h5')

    if os.path.exists(h5_file_name) and os.path.exists(vocab_file_name):
        neon_logger.display("dataset files {} and vocabulary file {} already exist. "
                            "will use cached data. ".format(h5_file_name, vocab_file_name))
        return h5_file_name, vocab_file_name

    # split into training/valid set
    if valid_split is not None:
        if 'json' in file_ext:
            # Split based on number of files
            train_split = int(np.ceil(len(file_names) * (1 - valid_split)))
            train_files = file_names[:train_split]
            valid_files = file_names[train_split:]

            train_sent = load_json_sent(train_files, subset_pct)
            valid_sent = load_json_sent(valid_files, subset_pct)
            all_sent = train_sent + valid_sent
        elif 'txt' in file_ext:
            # Split based on number of lines (since only 2 files)
            all_sent = load_txt_sent(file_names, subset_pct)
            train_split = int(np.ceil(len(all_sent) * (1 - valid_split)))

            train_sent = all_sent[:train_split]
            valid_sent = all_sent[train_split:]
        else:
            neon_logger.display("Unsure how to load file_ext {}, please use 'json' or 'txt'."
                                .format(file_ext))
    else:
        train_files = file_names
        if 'json' in file_ext:
            train_sent = load_json_sent(train_files, subset_pct)
        elif 'txt' in file_ext:
            train_sent = load_txt_sent(train_files, subset_pct)
        else:
            neon_logger.display("Unsure how to load file_ext {}, please use 'json' or 'txt'."
                                .format(file_ext))
        all_sent = train_sent

    if os.path.exists(vocab_file_name):
        neon_logger.display("open existing vocab file: {}".format(vocab_file_name))
        vocab, rev_vocab, word_count = load_obj(vocab_file_name)
    else:
        neon_logger.display("Building  vocab file")

        # build vocab
        word_count = defaultdict(int)
        for sent in all_sent:
            sent_words = tokenize(sent)

            if len(sent_words) > max_len_w or len(sent_words) == 0:
                continue

            for word in sent_words:
                word_count[word] += 1

        # sort the word_count , re-assign ids by its frequency. Useful for downstream tasks
        # only done for train vocab
        vocab_sorted = sorted(word_count.items(), key=lambda kv: kv[1], reverse=True)

        vocab = OrderedDict()

        # get word count as array in same ordering as vocab (but with maximum length)
        word_count_ = np.zeros((len(word_count), ), dtype=np.int64)
        for i, t in enumerate(list(zip(*vocab_sorted))[0][:max_vocab_size]):
            word_count_[i] = word_count[t]
            vocab[t] = i
        word_count = word_count_

        # generate the reverse vocab
        rev_vocab = dict((wrd_id, wrd) for wrd, wrd_id in vocab.items())

        neon_logger.display("vocabulary from {} is saved into {}".format(path, vocab_file_name))
        save_obj((vocab, rev_vocab, word_count), vocab_file_name)

    vocab_size = len(vocab)
    neon_logger.display("\nVocab size from the dataset is: {}".format(vocab_size))

    neon_logger.display("\nProcessing and saving training data into {}".format(h5_file_name))

    # now process and save the train/valid data
    h5f = h5py.File(h5_file_name, 'w', libver='latest')
    shape, maxshape = (len(train_sent),), (None)
    dt = np.dtype([('text', h5py.special_dtype(vlen=str)),
                   ('num_words', np.uint16)])
    report_text_train = h5f.create_dataset('report_train', shape=shape,
                                           maxshape=maxshape, dtype=dt,
                                           compression='gzip')
    report_train = h5f.create_dataset('train', shape=shape, maxshape=maxshape,
                                      dtype=h5py.special_dtype(vlen=np.int32),
                                      compression='gzip')

    # map text to integers
    wdata = np.zeros((1, ), dtype=dt)
    ntrain = 0
    for sent in train_sent:
        text_int = [-1 if t not in vocab else vocab[t] for t in tokenize(sent)]

        # enforce maximum sentence length
        if len(text_int) > max_len_w or len(text_int) == 0:
            continue

        report_train[ntrain] = text_int

        wdata['text'] = clean_string(sent)
        wdata['num_words'] = len(text_int)
        report_text_train[ntrain] = wdata
        ntrain += 1

    report_train.attrs['nsample'] = ntrain
    report_train.attrs['vocab_size'] = vocab_size
    report_text_train.attrs['nsample'] = ntrain
    report_text_train.attrs['vocab_size'] = vocab_size

    if valid_split:
        neon_logger.display("\nProcessing and saving validation data into {}".format(h5_file_name))
        shape = (len(valid_sent),)
        report_text_valid = h5f.create_dataset('report_valid', shape=shape,
                                               maxshape=maxshape, dtype=dt,
                                               compression='gzip')
        report_valid = h5f.create_dataset('valid', shape=shape, maxshape=maxshape,
                                          dtype=h5py.special_dtype(vlen=np.int32),
                                          compression='gzip')
        nvalid = 0
        for sent in valid_sent:
            text_int = [-1 if t not in vocab else vocab[t] for t in tokenize(sent)]

            # enforce maximum sentence length
            if len(text_int) > max_len_w or len(text_int) == 0:
                continue

            report_valid[nvalid] = text_int
            wdata['text'] = clean_string(sent)
            wdata['num_words'] = len(text_int)
            report_text_valid[nvalid] = wdata
            nvalid += 1

        report_valid.attrs['nsample'] = nvalid
        report_valid.attrs['vocab_size'] = vocab_size
        report_text_valid.attrs['nsample'] = nvalid
        report_text_valid.attrs['vocab_size'] = vocab_size

    h5f.close()

    return h5_file_name, vocab_file_name
