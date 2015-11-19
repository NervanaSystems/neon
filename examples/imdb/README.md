* train.py
Trains a simple LSTM based network for sentiment analysis on imdb movie reviews

    python examples/imdb/train.py -f labeledTrainData.tsv -e 2 -eval 1 -s imdb.p --vocab_file imdb.vocab

Get the data from Kaggle:

https://www.kaggle.com/c/word2vec-nlp-tutorial/data

* inference.py
Loads the model weights and does inference on a new raw imdb movie reviews

    python examples/imdb/inference.py --model_weights imdb.p --vocab_file imdb.vocab
