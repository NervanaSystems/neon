## Model

This is an implementation of [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726) which can be trained on any corpus of continuous text. For evaluation, the model is trained on the [University of Toronto BookCorpus dataset](http://www.cs.toronto.edu/~zemel/documents/align.pdf).

### Dependencies

<i>For data loading:</i><br>
* [h5py](http://www.h5py.org/)
<i>For evaluation on SICK dataset:</i><br>
* [scipy](https://www.scipy.org/)

### Data

Due to a necessary copyright agreement, to download the BookCorpus dataset, please contact the authors of [Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](http://www.cs.toronto.edu/~zemel/documents/align.pdf)  

When the files are downloaded, move them to empty directory called `books_txt` for training.

### Training

This example allows for the training of sentence embeddings from a corpus of continuous text where each sentence is delineated by a newline character.

The line below will train a sentence embedding on `.txt` files located in `books_txt`, and save the output (pre-processed data & trained weights) to `output`:

```
python train.py --data_dir books_txt/ --output_dir output/ -s s2v_model.prm -e 5
```

### Inference

The inference scripts allows interactive querying of a trained model to find sentences which are close to a given query in the embedded space:

```
python inference.py --model_file s2v_model.prm --data_dir books_txt/ --output_dir output/ --vector_name output/book_vectors.pkl
```

`--data_dir` points to the training data directory used for the saved model, so the script can locate the training data used and vocabulary file.

`--vector_name` specifies the location to save/reload the pre-computed training set vectors.

### Evaluation

A trained sentence2vec model can also be evaluated on the SemEval 2014 Task 1: semantic relatedness SICK dataset. Running the `eval_sick.py` script will download the data if it's unable to find it locally. The evaluation can be performed with the following command:

```
python eval_sick.py --model_file s2v_model.prm --data_dir books_txt/ --eval_data_path SICK_data/ --output_path output/
```

`--data_dir` points to the training data directory used for the saved model, so the script can locate the training data used and vocabulary file.

Optionally, can use `--w2v_path` to specify location of the Google W2V file to expand the vocabulary from training to a larger set.

The data can also be found/downloded manually here: [SemEval-2014 Task1](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools). Download *TRAINING DATA*, *TEST DATA(including gold scores)*, and *TRIAL DATA* and collect the data into a directory.

## Citations

<i>Skip-thought vectors</i><br>
Kiros, Y. Zhu, R. Salakhutdinov, R. Zemel, A. Torralba, R. Urtasun, and S. Fidler<br>
[http://arxiv.org/abs/1506.06726](http://arxiv.org/abs/1506.06726)
<br><br>

<i>Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books</i><br>
Y. Zhu, R. Kiros, R. Zemel, R. Salakhutdinov, R. Urtasun, A. Torralba, S. Fidler<br>
[http://arxiv.org/abs/1506.06724](http://arxiv.org/abs/1506.06724)
