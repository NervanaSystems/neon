'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)

Modified from: https://github.com/ryankiros/skip-thoughts/blob/master/eval_sick.py
'''
from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle
import copy

from neon.util.argparser import NeonArgparser

from data_iterator import SentenceEncode
from data_loader import load_sent_encoder, load_data, tokenize, get_w2v_vocab

from sklearn.metrics import mean_squared_error as mse
from sklearn.utils import shuffle

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from keras.models import Sequential
from keras.layers.core import Dense, Activation


def main():
    # parse the command line arguments
    parser = NeonArgparser(__doc__)
    parser.add_argument('--output_path', required=True,
                        help='Output path used when training model')
    parser.add_argument('--w2v_path', required=False, default=None,
                        help='Path to GoogleNews w2v file for voab expansion.')
    parser.add_argument('--eval_data_path', required=True, default='./SICK_data',
                        help='')
    args = parser.parse_args(gen_be=False)

    # Suppress sklearn deprecation warnings...
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # batch size 1 for testing individual sentences
    args.batch_size = 128
    # No validation was used for training
    valid_split = None

    # to run it on the cloud, can use ncloud train resume-training <id>, then it
    # will look for the file defaulted named to be model.prm
    if args.model_file is None:
        args.model_file = 'model.prm'

    args.callback_args['model_file'] = None

    # load the documents by giving the path and what extension the files are
    _, vocab_file = load_data(args.data_dir, valid_split=valid_split,
                              output_path=args.output_path)
    vocab, _, _ = cPickle.load(open(vocab_file, 'rb'))

    vocab_size = len(vocab)
    print("\nVocab size from the dataset is: {}".format(vocab_size))

    index_from = 2  # 0: padding 1: oov
    vocab_size_layer = vocab_size + index_from
    max_len = 30

    # Vocabulary expansion trick needs to pass the correct vocab set to evaluate (for tokenization)
    if args.w2v_path:
        print("Performing Vocabulary Expansion... Loading W2V...")
        w2v_vocab, w2v_vocab_size = get_w2v_vocab(args.w2v_path, cache=True)
        vocab_size_layer = w2v_vocab_size + index_from
        model = load_sent_encoder(args.model_file, expand_vocab=True, orig_vocab=vocab,
                                  w2v_vocab=w2v_vocab, w2v_path=args.w2v_path, use_recur_last=True)
        vocab = w2v_vocab
    else:
        # otherwise stick with original vocab size used to train the model
        model = load_sent_encoder(args.model_file, use_recur_last=True)

    model.be.bsz = args.batch_size
    model.initialize(dataset=(max_len, 1))

    evaluate(model, vocab=vocab, data_path=args.eval_data_path, evaltest=True,
             vocab_size_layer=vocab_size_layer)


def evaluate(model, vocab, data_path, seed=1234, evaltest=False, vocab_size_layer=20002):
    """
    Run experiment
    """
    print('Preparing data...')
    train, dev, test, scores = load_eval_data(data_path)
    train_A_shuf, train_B_shuf, scores_shuf = shuffle(train[0], train[1], scores[0],
                                                      random_state=seed)

    print('Tokenizing data...')
    train_A_tok = tokenize_input(train_A_shuf, vocab=vocab)
    train_B_tok = tokenize_input(train_B_shuf, vocab=vocab)
    dev_A_tok = tokenize_input(dev[0], vocab=vocab)
    dev_B_tok = tokenize_input(dev[1], vocab=vocab)

    print('Computing training skipthoughts...')
    # Get iterator from tokenized data. Second argument (train_text) not needed for evaluation
    train_set_A = SentenceEncode(train_A_tok, [], len(train_A_tok), vocab_size_layer,
                                 max_len=30, index_from=2)
    train_set_B = SentenceEncode(train_B_tok, [], len(train_B_tok), vocab_size_layer,
                                 max_len=30, index_from=2)

    # Compute embeddings using iterator
    trainA = model.get_outputs(train_set_A)
    trainB = model.get_outputs(train_set_B)

    print('Computing development skipthoughts...')
    dev_set_A = SentenceEncode(dev_A_tok, [], len(dev_A_tok), vocab_size_layer,
                               max_len=30, index_from=2)
    dev_set_B = SentenceEncode(dev_B_tok, [], len(dev_B_tok), vocab_size_layer,
                               max_len=30, index_from=2)

    devA = model.get_outputs(dev_set_A)
    devB = model.get_outputs(dev_set_B)

    print('Computing feature combinations...')
    trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
    devF = np.c_[np.abs(devA - devB), devA * devB]

    print('Encoding labels...')
    trainY = encode_labels(scores_shuf, ndata=len(trainF))
    devY = encode_labels(scores[1], ndata=len(devF))

    print('Compiling model...')
    lrmodel = prepare_model(ninputs=trainF.shape[1])

    print('Training...')
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, scores[1][:len(devF)])

    if evaltest:
        print('Tokenizing test sentences....')
        test_A_tok = tokenize_input(test[0], vocab=vocab)
        test_B_tok = tokenize_input(test[1], vocab=vocab)

        print('Computing test skipthoughts...')
        test_set_A = SentenceEncode(test_A_tok, [], len(test_A_tok), vocab_size_layer,
                                    max_len=30, index_from=2)
        test_set_B = SentenceEncode(test_B_tok, [], len(test_B_tok), vocab_size_layer,
                                    max_len=30, index_from=2)

        testA = model.get_outputs(test_set_A)
        testB = model.get_outputs(test_set_B)

        print('Computing feature combinations...')
        testF = np.c_[np.abs(testA - testB), testA * testB]

        print('Evaluating...')
        r = np.arange(1, 6)
        yhat = np.dot(bestlrmodel.predict_proba(testF, verbose=2), r)
        pr = pearsonr(yhat, scores[2][:len(yhat)])[0]
        sr = spearmanr(yhat, scores[2][:len(yhat)])[0]
        se = mse(yhat, scores[2][:len(yhat)])
        print('Test Pearson: ' + str(pr))
        print('Test Spearman: ' + str(sr))
        print('Test MSE: ' + str(se))

        return yhat


def prepare_model(ninputs=9600, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(nclass, input_shape=(ninputs,)))
    lrmodel.add(Activation('softmax'))
    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    return lrmodel


def train_model(lrmodel, X, Y, devX, devY, devscores):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    r = np.arange(1, 6)

    while not done:
        # Every 100 epochs, check Pearson on development set
        lrmodel.fit(X, Y, verbose=2, shuffle=False, validation_data=(devX, devY))
        yhat = np.dot(lrmodel.predict_proba(devX, verbose=2), r)
        score = pearsonr(yhat, devscores)[0]
        if score > best:
            print(score)
            best = score
            bestlrmodel = copy.deepcopy(lrmodel)
        else:
            done = True

    yhat = np.dot(bestlrmodel.predict_proba(devX, verbose=2), r)
    score = pearsonr(yhat, devscores)[0]
    print('Dev Pearson: ' + str(score))
    return bestlrmodel


def encode_labels(labels, nclass=5, ndata=None):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j, i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j, i] = np.floor(y) - y + 1
    if ndata:
        Y = Y[:ndata]

    return Y


def load_eval_data(path):
    """
    Load the SICK semantic-relatedness dataset
    """
    trainA, trainB, devA, devB, testA, testB = [], [], [], [], [], []
    trainS, devS, testS = [], [], []

    with open(path + 'SICK_train.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[1])
            trainB.append(text[2])
            trainS.append(text[3])

    with open(path + 'SICK_trial.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            devA.append(text[1])
            devB.append(text[2])
            devS.append(text[3])

    with open(path + 'SICK_test_annotated.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[1])
            testB.append(text[2])
            testS.append(text[3])

    trainS = [float(s) for s in trainS[1:]]
    devS = [float(s) for s in devS[1:]]
    testS = [float(s) for s in testS[1:]]

    return ((trainA[1:], trainB[1:]), (devA[1:], devB[1:]),
            (testA[1:], testB[1:]), (trainS, devS, testS))


def tokenize_input(input_sent, vocab):
    """
    Return a numpy array where each row is the word-indexes for each sentence
    """
    input_tok = []

    # map text to integers
    for sent in input_sent:
        text_int = [-1 if t not in vocab else vocab[t] for t in tokenize(sent)]

        input_tok.append(np.array(text_int))

    return np.array(input_tok)


if __name__ == "__main__":
    main()
