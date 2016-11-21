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
import cPickle
from collections import defaultdict, OrderedDict
import re
import sys

from neon.backends import gen_backend
from neon.backends.nervanagpu import GPUTensor
from neon.layers import LookupTable, RecurrentSum, RecurrentLast, Linear, Bias, GeneralizedCost
from neon.initializers import Gaussian, Constant
from neon.data import ArrayIterator
from neon.optimizers import GradientDescentMomentum
from neon.transforms import SumSquared
from neon.callbacks.callbacks import Callbacks
from neon.models import Model


def clean_string(string):
    string = re.sub(r"[^A-Za-z(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def tokenize(s, eos=True):
    s = clean_string(s)
    if eos and len(s) > 0:
        return (s + ' <eos>').strip().split()
    else:
        return s.strip().split()


def load_json(fname):
    with open(fname) as f:
        data = json.load(f)
    return data


def load_json_sent(flist_json):
    """
    load all the sentences from a list of JSON files (with out data format)
    and return a list
    """
    all_sent = []
    for f in flist_json:
        data = load_json(f)
        num_sent = len(data['text'])
        if num_sent <= 0:
            continue
        num_sent = len(data['text'])
        sent = [data['text'][i]['sentence'] for i in range(num_sent)]
        all_sent += sent
    return all_sent


def load_txt_sent(flist_txt):
    """
    load all the senteces from a list of txt files using standard file io
    """
    all_sent = []
    for txt_file in flist_txt:
        print("Reading file: {}".format(txt_file))
        with open(txt_file, 'r') as f:
            data = f.read()
        sent = data.split('\n')
        all_sent += sent
    return all_sent


def get_file_list(data_dir, file_ext):
    file_ext = file_ext if isinstance(file_ext, list) else [file_ext]
    file_names = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)
                  if any(fn.endswith(ext) for ext in file_ext)]

    return file_names


def get_file_str(path, num_files, labelled=False, output_path=None,
                 valid_split=None, split_count_thre=None):
    # grab the directory name as part of the names
    dir_name = path.split('/')[-1] if len(path.split('/')[-1]) > 0 else path.split('/')[-2]
    label_str = 'labelled' if labelled else ''
    split_thre_str = 'thre_{}'.format(split_count_thre) if split_count_thre else ''
    dir_str = 'doc_{}_{}_{}_{}'.format(label_str, dir_name, num_files, split_thre_str)
    if valid_split:
        split_str = '_split_{}'.format(valid_split*100)
    else:
        split_str = ''
    file_str = dir_str + split_str
    return file_str


def load_data(path, file_ext=['txt'], valid_split=None, vocab_file_name=None,
              max_vocab_size=None, max_len_w=None, output_path='/output'):
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

    Returns:
        The function saves 2 files:
        h5 file with preprocessed data
        vocabulary file with: vocab, reverse_vocab, word_count
    """
    file_names = get_file_list(path, file_ext)

    file_str = get_file_str(path, len(file_names), labelled=False,
                            output_path=output_path, valid_split=valid_split)

    # file name to store the vocabulary
    if vocab_file_name is None:
        vocab_file_name = file_str + '.vocab'
        vocab_file_name = os.path.join(output_path, vocab_file_name)

    # If max sizes arent set, assume no limit
    if not max_len_w:
        max_len_w = sys.maxint
    if not max_vocab_size:
        max_vocab_size = sys.maxint

    # file name to store the pre-processed train/valid dataset
    h5_file_name = os.path.join(output_path, file_str + '.h5')

    if os.path.exists(h5_file_name) and os.path.exists(vocab_file_name):
        print("dataset files {} and vocabulary file {} already exist. "
              "will use cached data. ".format(h5_file_name, vocab_file_name))
        return h5_file_name, vocab_file_name

    # split into training/valid set
    if valid_split is not None:
        if 'json' in file_ext:
            # Split based on number of files
            train_split = int(np.ceil(len(file_names) * (1 - valid_split)))
            train_files = file_names[:train_split]
            valid_files = file_names[train_split:]

            train_sent = load_json_sent(train_files)
            valid_sent = load_json_sent(valid_files)
            all_sent = train_sent + valid_sent
        elif 'txt' in file_ext:
            # Split based on number of lines (since only 2 files)
            all_sent = load_txt_sent(file_names)
            train_split = int(np.ceil(len(all_sent) * (1 - valid_split)))

            train_sent = all_sent[:train_split]
            valid_sent = all_sent[train_split:]
        else:
            print("Unsure how to load file_ext {}, please use 'json' or 'txt'.".format(file_ext))
    else:
        train_files = file_names
        if 'json' in file_ext:
            train_sent = load_json_sent(train_files)
        elif 'txt' in file_ext:
            train_sent = load_txt_sent(train_files)
        else:
            print("Unsure how to load file_ext {}, please use 'json' or 'txt'.".format(file_ext))
        all_sent = train_sent

    if os.path.exists(vocab_file_name):
        print("open existing vocab file: {}".format(vocab_file_name))
        vocab, rev_vocab, word_count = cPickle.load(open(vocab_file_name, 'rb'))
    else:
        print("Building  vocab file")

        # build vocab
        word_count = defaultdict(int)
        for s_idx, sent in enumerate(all_sent):
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
        for i, t in enumerate(zip(*vocab_sorted)[0][:max_vocab_size]):
            word_count_[i] = word_count[t]
            vocab[t] = i
        word_count = word_count_

        # generate the reverse vocab
        rev_vocab = dict((wrd_id, wrd) for wrd, wrd_id in vocab.iteritems())

        print("vocabulary from {} is saved into {}".format(path, vocab_file_name))
        cPickle.dump((vocab, rev_vocab, word_count), open(vocab_file_name, 'wb'))

    vocab_size = len(vocab)
    print("\nVocab size from the dataset is: {}".format(vocab_size))

    print("\nProcessing and saving training data into {}".format(h5_file_name))

    # now process and save the train/valid data
    h5f = h5py.File(h5_file_name, 'w', libver='latest')
    shape, maxshape = (len(train_sent),), (None)
    dt = np.dtype([('text', h5py.special_dtype(vlen=str)),
                   ('num_words', np.uint16),
                   ])
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
        print("\nProcessing and saving validation data into {}".format(h5_file_name))
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


def get_google_word2vec_W(fname, vocab, index_from=2):
    """
    Extract the embedding matrix from the given word2vec binary file and use this
    to initalize a new embedding matrix for words found in vocab.

    Conventions are to save indices for pad, oov, etc.:
    index 0: pad
    index 1: oov (or <unk>)
    Often cases, the <eos> has already been in the preprocessed data, so no need
    to save an index for <eos>
    """
    f = open(fname, 'rb')
    header = f.readline()
    vocab_w2v, embedding_dim = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * embedding_dim
    vocab_size = len(vocab) + index_from
    W = np.zeros((vocab_size, embedding_dim))

    found_words_idx = defaultdict(int)
    found_words = defaultdict(int)

    for i, line in enumerate(range(vocab_w2v)):
        word = []
        while True:
            ch = f.read(1)
            if ch == ' ':
                word = ''.join(word)
                break
            if ch != '\n':
                word.append(ch)
        if word in vocab:
            wrd_id = vocab[word] + index_from
            if wrd_id < vocab_size:
                W[wrd_id] = np.fromstring(
                    f.read(binary_len), dtype='float32')
                found_words_idx[wrd_id] += 1
                found_words[word] += 1
        else:
            f.read(binary_len)

    cnt = 0
    for wrd_id in range(vocab_size):
        if wrd_id not in found_words_idx:
            cnt += 1
            W[wrd_id] = np.random.uniform(-1.0, 1.0, embedding_dim)

    unfound_words = list()
    for wrd in vocab:
        if wrd not in found_words:
            unfound_words += [wrd]

    if len(unfound_words) > 0:
        print("some of the unfound words are:")
        print(unfound_words[:30])

    assert cnt + len(found_words_idx) == vocab_size

    f.close()

    return W, embedding_dim, found_words


def compute_vocab_expansion(orig_word_vectors, w2v_W, w2v_vocab):
    print("Learning linear mapping from w2v -> rnn embedding...")
    clf = train_regressor(orig_word_vectors, w2v_W, w2v_vocab)
    print("Contructing map...")
    init_embed = apply_regressor(clf, w2v_W, w2v_vocab)
    return init_embed


def get_w2v_vocab(fname, cache=True):
    """
    Get ordered dict of vocab from google word2vec
    """
    if cache:
        cache_fname = fname.split('.')[0] + ".vocab"

        if os.path.isfile(cache_fname):
            print("Found cached W2V vocab, reloading...")
            with open(cache_fname, 'r') as f:
                vocab, vocab_size = cPickle.load(f)
                print("Word2Vec vocab size is: {}".format(vocab_size))
            return vocab, vocab_size

    with open(fname, 'rb') as f:
        print("No W2V vocab cache found, recomputing...")
        header = f.readline()
        vocab_size, embed_dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * embed_dim

        print("Word2Vec vocab size is: {}".format(vocab_size))

        vocab = OrderedDict()

        for i in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            f.read(binary_len)
            vocab[word] = i

    if cache:
        with open(cache_fname, 'w') as f:
            cPickle.dump((vocab, vocab_size), f)

    return vocab, vocab_size


def get_embeddings(lookup_layer, word_idict):
    """
    Extract RNN embeddings from the lookup layer of the model

    Function modified from:
    https://github.com/ryankiros/skip-thoughts/blob/master/training/tools.py
    """
    f_emb = lookup_layer.W.get()
    d = OrderedDict()
    for i in range(f_emb.shape[0]):
        ff = f_emb[i].flatten()
        d[word_idict[i]] = ff
    return d


def train_regressor(orig_wordvecs, w2v_W, w2v_vocab):
    """
    Return regressor to map word2vec to RNN word space

    Function modified from:
    https://github.com/ryankiros/skip-thoughts/blob/master/training/tools.py
    """
    # Gather all words from word2vec that appear in wordvecs
    d = defaultdict(lambda: 0)
    for w in w2v_vocab.keys():
        d[w] = 1
    shared = OrderedDict()
    count = 0
    for w in orig_wordvecs.keys()[:-2]:
        if d[w] > 0:
            shared[w] = count
            count += 1

    # Get the vectors for all words in 'shared'
    w2v = np.zeros((len(shared), 300), dtype='float32')
    sg = np.zeros((len(shared), 620), dtype='float32')
    for w in shared.keys():
        w2v[shared[w]] = w2v_W[w2v_vocab[w]]
        sg[shared[w]] = orig_wordvecs[w]

    train_set = ArrayIterator(X=w2v, y=sg, make_onehot=False)

    layers = [Linear(nout=620, init=Gaussian(loc=0.0, scale=0.1)),
              Bias(init=Constant(0.0))]
    clf = Model(layers=layers)

    cost = GeneralizedCost(costfunc=SumSquared())
    opt = GradientDescentMomentum(0.1, 0.9, gradient_clip_value=5.0)
    callbacks = Callbacks(clf)

    clf.fit(train_set, num_epochs=2, optimizer=opt, cost=cost, callbacks=callbacks)
    return clf


def apply_regressor(clf, w2v_W, w2v_vocab):
    """
    Map words from word2vec into RNN word space

    Function modifed from:
    https://github.com/ryankiros/skip-thoughts/blob/master/training/tools.py
    """
    init_embed = np.zeros((len(w2v_vocab), 620), dtype='float32')

    word_vec = GPUTensor(clf.be, shape=(300, 1))

    for i, w in enumerate(w2v_vocab.keys()):
        if '_' not in w:
            word_vec.set(w2v_W[w2v_vocab[w]].reshape(300, 1))
            init_embed[w2v_vocab[w], :] = clf.fprop(word_vec).get().reshape((620,))

    return init_embed


def load_sent_encoder(load_path, expand_vocab=False, orig_vocab=None,
                      w2v_vocab=None, w2v_path=None, use_recur_last=False):
    """
    Custom function to load the model saved from skip-thought vector training
    and reconstruct another model just using the LUT and encoding layer for
    transfering sentence representations.

    Arguments:
        load_path: path to saved s2v model
        expand_vocab: Bool to indicate if w2v vocab expansion should be attempted
        orig_vocab: If using expand_vocab, original vocabulary dict is needed for expansion
        w2v_vocab: If using expand_vocab, w2v vocab dict
        w2v_path: Path to trained w2v binary (GoogleNews)
        use_recur_last: If True a RecurrentLast layer is used as the final layer, if False
                        a RecurrentSum layer is used as the last layer of the returned model.
    """

    load_path = open(load_path)
    model_dict = cPickle.load(load_path)

    embed_dim = model_dict['model']['config']['embed_dim']

    # Load the full model, batch size = 1 for inference
    gen_backend('gpu', batch_size=1)
    model_train = Model(model_dict)

    # RecurrentLast should be used for semantic similarity evaluation
    if use_recur_last:
        last_layer = RecurrentLast()
    else:
        last_layer = RecurrentSum()

    if expand_vocab:
        assert orig_vocab and w2v_vocab, ("All vocabs and w2v_path " +
                                          "need to be specified when using expand_vocab")

        print("Computing vocab expansion regression...")
        # Build inverse word dictionary (word -> index)
        word_idict = dict()
        for kk, vv in orig_vocab.iteritems():
            # Add 2 to the index to allow for EOS and oov tokens as 0 and 1
            word_idict[vv + 2] = kk
        word_idict[0] = '<eos>'
        word_idict[1] = 'UNK'

        # Create dictionary of word -> vec
        orig_word_vecs = get_embeddings(model_train.layers.layer_dict['lookupTable'], word_idict)

        # Load GooleNews w2v weights
        w2v_W, w2v_dim, _ = get_google_word2vec_W(w2v_path, w2v_vocab)

        # Compute the expanded vocab lookup table from a linear mapping of
        # words2vec into RNN word space
        init_embed = compute_vocab_expansion(orig_word_vecs, w2v_W, w2v_vocab)

        init_embed_dev = model_train.be.array(init_embed)
        w2v_vocab_size = len(w2v_vocab)

        table = LookupTable(vocab_size=w2v_vocab_size, embedding_dim=embed_dim,
                            init=init_embed_dev, pad_idx=0)

        model = Model(layers=[table,
                              model_train.layers.layer_dict['encoder'],
                              last_layer])

    else:
        model = Model(layers=[model_train.layers.layer_dict['lookupTable'],
                              model_train.layers.layer_dict['encoder'],
                              last_layer])
    return model


def prep_data(raw_input, input_type, max_len, vocab, dtype='int32',
              index_from=2, oov=1):
    """
    Transforms the raw received input data to put it in the required
    format for running through neon.
    Args:
        raw_input (blob): input data contents ex. a stream of text
        input_type (str): type for input data file
        max_len (int): max sentence length to deal with
        vocab (dict): vocabulary file
        dtype (type, optional): type for each element of a tensor.  Defaults to
                                float32
    Returns:
        Tensor: neon input data file of appropriate shape.
    """
    dtype = np.dtype(dtype)
    if input_type == "text":
        in_shape = (max_len, 1)

        tokens = tokenize(raw_input)
        sent_inp = np.array(
            [oov if t not in vocab else (vocab[t] + index_from) for t in tokens])
        l = min(len(sent_inp), max_len)
        xbuf = np.zeros(in_shape, dtype=dtype)
        xbuf[-l:] = sent_inp[-l:].reshape(-1, 1)

        return xbuf
    else:
        raise ValueError("Unsupported data type: %s" % input_type)
