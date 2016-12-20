LSTM Tutorial
=============

This tutorial will guide you through the implementation of a recurrent
neural network to analyze movie reviews on IMDB and decide if they are
positive or negative reviews.

We train a recurrent neural network with Long-short Term Memory (LSTM)
units.

IMDB dataset
------------

The IMDB dataset consists of 25,000 reviews, each with a binary label (1
= positive, 0 = negative). Here is an example review:

    "Okay, sorry, but I loved this movie. I just love the whole 80's genre of these kind of movies, because you don't see many like this..."  -*~CupidGrl~*

The dataset contains a large vocabulary of words, and reviews have
variable length ranging from tens to hundreds of words. We reduce the
complexity of the dataset with several steps:
1. Limit the vocabulary size to ``vocab_size = 20000`` words by replacing the less frequent words with a Out-Of-Vocab (OOV) character.
2. Truncate each example to ``max_len = 128`` words.
3. For reviews with less than ``max_len`` words, pad the review with whitespace. This equalizes the review lengths
across examples.

.. code-block:: python

    # define our parameters
    vocab_size = 20000
    max_len = 128

We have provided convenience functions for loading and preprocessing the
imdb data.

.. code-block:: python

    from neon.data import IMDB

    # load the imdb data
    # 1. Limit vocab size, 2. Truncate length, 3. Pad with whitespace
    imdb = IMDB(vocab_size, max_len)

    train_set = imdb.train_iter
    test_set = imdb.test_iter

Now we have the iterators needed for training and evaluation.

Model specification
-------------------

Initialization
~~~~~~~~~~~~~~

For most of the layers, we use Xavier Glorot's `initialization
scheme <http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>`__
to automatically scale the weights and preserve the variance of input
activations.

.. code-block:: python

    from neon.initializers import GlorotUniform, Uniform

    init_glorot = GlorotUniform()
    init_uniform = Uniform(-0.1/128, 0.1/128)

The network consists of a word embedding layer, and LSTM, RecurrentSum,
Dropout and Affine layers.

1. :py:class:`.LookupTable` is a word embedding that maps from a sparse one-hot representation to dense word vectors. The embedding is learned from the data.
2. :py:class:`.LSTM` is a recurrent layer with "long short-term memory" units. LSTM networks are good at learning temporal dependencies during training, and often perform better than standard RNN layers.
3. :py:class:`.RecurrentSum` is a recurrent output layer that collapses over the time dimension of the LSTM by summing outputs from individual steps.
4. :py:class:`.Dropout` performs regularization by silencing a random subset of the units during training.
5. :py:class:`.Affine` is a fully connected layer for the binary classification of the outputs.

.. code-block:: python

    from neon.layers import LSTM, Affine, Dropout, LookupTable, RecurrentSum
    from neon.transforms import Logistic, Tanh, Softmax

    layers = [
        LookupTable(vocab_size=vocab_size, embedding_dim=128, init=init_uniform),
        LSTM(output_size=128, init=init_glorot, activation=Tanh(),
             gate_activation=Logistic(), reset_cells=True),
        RecurrentSum(),
        Dropout(keep=0.5),
        Affine(nout=2, init=init_glorot, bias=init_glorot, activation=Softmax())
    ]

Cost, Optimizers, and Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For training, we use the Adagrad optimizer and the Cross Entropy cost
function.

.. code-block:: python

    from neon.optimizers import Adagrad
    from neon.transforms import CrossEntropyMulti
    from neon.layers import GeneralizedCost

    cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
    optimizer = Adagrad(learning_rate=0.01)

In addition to the default progress bar, we set up a callback with the
``serialize=1`` option to save the model to a pickle file after every
epoch:

.. code-block:: python

    from neon.callbacks import Callbacks
    num_epochs = 2
    fname = "imdb_lstm_model"

    callbacks = Callbacks(model, eval_set=valid_set, eval_freq=num_epochs,
                          serialize=1, save_path=fname + '.pickle')

Train Model
-----------

Training the model for two epochs should be sufficient to obtain some
interesting results, and avoid overfitting on this small dataset. This
should take a few minutes.

.. code-block:: python

    from neon.models import Model

    model = Model(layers=layers)
    model.fit(train_set, optimizer=optimizer, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

Evaluate the model on the held-out test set with the ``Accuracy``
metric.

.. code-block:: python

    from neon.transforms import Accuracy

    print "Test  Accuracy - ", 100 * model.eval(test_set, metric=Accuracy())
    print "Train Accuracy - ", 100 * model.eval(train_set, metric=Accuracy())

Inference
---------

The trained model can now be used to perform inference on new reviews.
Set up a new model with a batch size of 1.

.. code-block:: python

    # setup backend
    from neon.backends import gen_backend
    be = gen_backend(batch_size=1)

Set up a new set of layers for batch size 1.

.. code-block:: python

    # define same model as in train. Layers need to be recreated with new batch size.
    layers = [
        LookupTable(vocab_size=vocab_size, embedding_dim=embedding_dim, init=init_emb),
        LSTM(hidden_size, init_glorot, activation=Tanh(),
             gate_activation=Logistic(), reset_cells=True),
        RecurrentSum(),
        Dropout(keep=0.5),
        Affine(nclass, init_glorot, bias=init_glorot, activation=Softmax())
    ]

    model_new = Model(layers=layers)

Wrap the new layers into a new model, initialize with the weights we
just trained.

.. code-block:: python

    # load the weights
    save_path= 'labeledTrainData.tsv' + '.pickle'
    model_new.load_weights(save_path)
    model_new.initialize(dataset=(sentence_length, batch_size))

Let's try in on some real reviews! I went on imdb to get some reviews of
the latest Bond Movie.

    As a die hard fan of James Bond, I found this film to be simply nothing more than a classic. For any original James Bond fan, you will simply enjoy how the producers and Sam Mendes re-emerged the roots of James Bond. The roots of Spectre, Blofield and just the pure elements of James Bond that we all miss even from the gun barrel introduction.

And another one:

    The plot/writing is completely unrealistic and just dumb at times. Bond is dressed up in a white tux on an overnight train ride? eh, OK. But then they just show up at the villain's compound likenothing bad is going to happen to them. How stupid is this Bond? And then the villain just happens to booby trap this huge building in London (across from the intelligence building) and previously or very quickly had some bullet proof glass installed. And so on and so on... give me a break.

Here we allow the user the input a review and returns predictions on the
sentiment of the review.

.. code-block:: python

    import preprocess_text
    import cPickle
    import numpy as np

    # setup buffers before accepting reviews
    xbuf = np.zeros((sentence_length, 1), dtype=np.int32)  # host buffer
    xdev = be.zeros((sentence_length, 1), dtype=np.int32)  # device buffer

    # tags for text pre-processing
    oov = 2
    start = 1
    index_from = 3
    pad_char = 0

    # load dictionary from file (generated by prepare script)
    vocab, rev_vocab = cPickle.load(open(fname + '.vocab', 'rb'))

    while True:
        line = raw_input('Enter a Review from testData.tsv file: \n')

        # clean the input
        tokens = preprocess_text.clean_string(line).strip().split()

        # convert strings to one-hot. Check for oov and add start
        sent = [len(vocab) + 1 if t not in vocab else vocab[t] for t in tokens]
        sent = [start] + [w + index_from for w in sent]
        sent = [oov if w >= vocab_size else w for w in sent]

        # pad sentences
        xbuf[:] = 0
        trunc = sent[-sentence_length:]
        xbuf[-len(trunc):, 0] = trunc  # load list into numpy array
        xdev[:] = xbuf  # load numpy array into device tensor

        # run the sentence through the model
        y_pred = model_new.fprop(xdev, inference=True)

        print '-' * 100
        print "Sentence encoding: {0}".format(xbuf.T)
        print "\nPrediction: {:.1%} negative, {:.1%} positive".format(y_pred.get()[0,0], y_pred.get()[1,0])
        print '-' * 100

Executing the above with the two reviews yields:

.. code-block:: bash

    # Review #1
    Prediction: 0.5% negative, 99.5% positive

    # Review #2
    Prediction: 98.2% negative, 1.8% positive
