* train.py
Trains a simple LSTM based network for sentiment analysis on imdb movie reviews

    python examples/imdb/train.py -f labeledTrainData.tsv -e 2 -eval 1 -s imdb.p --vocab_file imdb.vocab

Get the data from Kaggle:

https://www.kaggle.com/c/word2vec-nlp-tutorial/data

If choose to initialize the word embedding layer using Word2Vec, please make sure to get the data GoogleNews-vectors-negative300.bin from:

  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

* inference.py
Loads the model weights and does inference on a new raw imdb movie reviews

    python examples/imdb/inference.py --model_weights imdb.p --vocab_file imdb.vocab
