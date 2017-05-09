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
import os
import numpy as np
from collections import defaultdict, OrderedDict

from neon.layers import LookupTable, RecurrentSum, RecurrentLast, Linear, Bias, GeneralizedCost
from neon.initializers import Gaussian, Constant
from neon.data import ArrayIterator
from neon.data.text_preprocessing import clean_string
from neon.optimizers import GradientDescentMomentum
from neon.transforms import SumSquared
from neon.callbacks.callbacks import Callbacks
from neon.models import Model
from neon.util.persist import load_obj, save_obj
from neon import logger as neon_logger


class SentenceVector(object):

    """
    A container class of sentence vectors for easy query similar sentences etc.
    """

    def __init__(self, vectors, text):
        """
        Initialize a SentenceVectors class object

        Arguments:
            vectors (ndarray, (#sentences, vector dimension)): sentence vectors
            text (list, #sentences): sentence texts
        """
        self.vectors = vectors
        self.text = text

        if isinstance(self.text, list):
            assert self.vectors.shape[0] == len(self.text)
        elif isinstance(self.text, np.ndarray):
            assert self.vectors.shape[0] == self.text.shape[0]

        norms = np.linalg.norm(self.vectors, axis=1)
        self.vectors = self.vectors / norms.reshape(-1, 1)

    def find_similar_idx(self, query, n=10):
        """
        Find similar sentences by vector distances
        metric = dot(vectors_of_vectors, vectors_of_target_vector)
        Uses a precomputed vectors of the vectors
        Parameters

        Arguments:
            query (ndarray): query sentence vector
            n (int): top n number of neighbors

        Returns:
            position in self.vocab_w2id
            cosine similarity
        """
        query = query / np.linalg.norm(query)
        metrics = np.dot(self.vectors, query.T)

        best = np.argsort(metrics.ravel())[::-1][:n]
        best_metrics = metrics[best]

        return best, best_metrics

    def find_similar(self, query, n=10):
        """
        Find similar sentences by vector distances
        metric = dot(vectors_of_vectors, vectors_of_target_vector)
        Uses a precomputed vectors of the vectors
        Parameters

        Arguments:
            query (ndarray): query sentence vector
            n (int): top n number of neighbors

        Returns:
            position in self.vocab_w2id
            cosine similarity
        """
        query = query / np.linalg.norm(query)
        metrics = np.dot(self.vectors, query.T)

        best = np.argsort(metrics.ravel())[::-1][:n]
        best_metrics = metrics[best]
        nearest = [self.text[b] for b in best.tolist()]

        return nearest, best_metrics

    def find_similar_with_idx(self, idx, n=10):
        if isinstance(idx, list):
            best = []
            for i in idx:
                best += self.find_similar_with_idx(i, n)
            return best
        else:
            query = self.vectors[idx]
            metrics = np.dot(self.vectors, query.T)

            best = np.argsort(metrics.ravel())[::-1][:n]
            return best.tolist()


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


def tokenize(s, eos=True):
    s = clean_string(s)
    if eos and len(s) > 0:
        return (s + ' <eos>').strip().split()
    else:
        return s.strip().split()


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
    orig_w2v_size, embedding_dim = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * embedding_dim
    vocab_size = len(vocab) + index_from
    W = np.zeros((vocab_size, embedding_dim))

    found_words_idx = defaultdict(int)
    found_words = defaultdict(int)

    for i in range(vocab_size):
        word = []
        while True:
            ch = f.read(1)
            if ch == b' ':
                word = (b''.join(word)).decode('utf-8')
                break
            if ch != b'\n':
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

    assert cnt + len(found_words_idx) == vocab_size

    f.close()

    return W, embedding_dim, found_words


def compute_vocab_expansion(orig_word_vectors, w2v_W, w2v_vocab, word_idict):
    neon_logger.display("Learning linear mapping from w2v -> rnn embedding...")
    clf = train_regressor(orig_word_vectors, w2v_W, w2v_vocab)
    neon_logger.display("Constructing map...")
    init_embed = apply_regressor(clf, w2v_W, w2v_vocab, orig_word_vectors, word_idict)
    return init_embed


def get_w2v_vocab(fname, max_vocab_size, cache=True):
    """
    Get ordered dict of vocab from google word2vec
    """
    if cache:
        cache_fname = fname.split('.')[0] + ".vocab"

        if os.path.isfile(cache_fname):
            vocab, vocab_size = load_obj(cache_fname)
            neon_logger.display("Word2Vec vocab cached, size is: {}".format(vocab_size))
            return vocab, vocab_size

    with open(fname, 'rb') as f:
        header = f.readline()
        vocab_size, embed_dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * embed_dim

        neon_logger.display("Word2Vec vocab size is: {}".format(vocab_size))
        vocab_size = min(max_vocab_size, vocab_size)
        neon_logger.display("Reducing vocab size to: {}".format(vocab_size))

        vocab = OrderedDict()

        for i, line in enumerate(range(vocab_size)):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = (b''.join(word)).decode('utf-8')
                    break
                if ch != b'\n':
                    word.append(ch)
            f.read(binary_len)
            vocab[word] = i

    if cache:
        save_obj((vocab, vocab_size), cache_fname)

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

    for w in list(orig_wordvecs.keys())[:-2]:
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

    # regression model is trained using default global batch size
    cost = GeneralizedCost(costfunc=SumSquared())
    opt = GradientDescentMomentum(0.1, 0.9, gradient_clip_value=5.0)
    callbacks = Callbacks(clf)

    clf.fit(train_set, num_epochs=20, optimizer=opt, cost=cost, callbacks=callbacks)
    return clf


def apply_regressor(clf, w2v_W, w2v_vocab, orig_word_vectors, word_idict):
    """
    Map words from word2vec into RNN word space

    Function modifed from:
    https://github.com/ryankiros/skip-thoughts/blob/master/training/tools.py
    """
    init_embed = clf.be.zeros((len(w2v_vocab), 620), dtype=np.float32)

    word_vec = clf.be.empty((300, 1))

    # to apply the regression model for expansion, can only do one word at a time
    # so to set the actual batch_size to be 1
    actual_bsz = 1
    clf.set_batch_size(N=actual_bsz)

    for i, w in enumerate(w2v_vocab.keys()):
        word_vec.set(w2v_W[w2v_vocab[w]].reshape(300, 1))
        init_embed[w2v_vocab[w]][:] = clf.fprop(word_vec)[:, :actual_bsz]

    word_vec_l = clf.be.empty((620, 1))
    for w in word_idict.values():
        if w in w2v_vocab:
            word_vec_l.set(orig_word_vectors[w])
            init_embed[w2v_vocab[w]][:] = word_vec_l

    return init_embed.get()


def load_sent_encoder(model_dict, expand_vocab=False, orig_vocab=None,
                      w2v_vocab=None, w2v_path=None, use_recur_last=False):
    """
    Custom function to load the model saved from skip-thought vector training
    and reconstruct another model just using the LUT and encoding layer for
    transfering sentence representations.

    Arguments:
        model_dict: saved s2v model dict
        expand_vocab: Bool to indicate if w2v vocab expansion should be attempted
        orig_vocab: If using expand_vocab, original vocabulary dict is needed for expansion
        w2v_vocab: If using expand_vocab, w2v vocab dict
        w2v_path: Path to trained w2v binary (GoogleNews)
        use_recur_last: If True a RecurrentLast layer is used as the final layer, if False
                        a RecurrentSum layer is used as the last layer of the returned model.
    """

    embed_dim = model_dict['model']['config']['embed_dim']
    model_train = Model(model_dict)

    # RecurrentLast should be used for semantic similarity evaluation
    if use_recur_last:
        last_layer = RecurrentLast()
    else:
        last_layer = RecurrentSum()

    if expand_vocab:
        assert orig_vocab and w2v_vocab, ("All vocabs and w2v_path " +
                                          "need to be specified when using expand_vocab")

        neon_logger.display("Computing vocab expansion regression...")
        # Build inverse word dictionary (word -> index)
        word_idict = dict()
        for kk, vv in orig_vocab.items():
            # Add 2 to the index to allow for padding and oov tokens as 0 and 1
            word_idict[vv + 2] = kk
        word_idict[0] = ''
        word_idict[1] = 'UNK'

        # Create dictionary of word -> vec
        orig_word_vecs = get_embeddings(model_train.layers.layer_dict['lookupTable'], word_idict)

        # Load GooleNews w2v weights
        w2v_W, w2v_dim, _ = get_google_word2vec_W(w2v_path, w2v_vocab)

        # Compute the expanded vocab lookup table from a linear mapping of
        # words2vec into RNN word space
        init_embed = compute_vocab_expansion(orig_word_vecs, w2v_W, w2v_vocab, word_idict)

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
