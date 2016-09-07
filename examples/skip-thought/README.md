## Model

This is an implementation of [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726) which can be trained on any corpus of continuous text. For evaluation, the model is trained on the [University of Toronto BookCorpus dataset](http://www.cs.toronto.edu/~zemel/documents/align.pdf).

### Dependencies

<i>For data loading:</i><br>
* [h5py](http://www.h5py.org/)

<i>For evaluation:</i><br>
* [Keras](https://keras.io/)
* [scikit-learn](http://scikit-learn.org/stable/)

<i>For visualization:</i><br>
* [tsne](https://pypi.python.org/pypi/tsne)
* [matplotlib](matplotlib.org)


### Data

Due to a necessary copyright agreement, to download the BookCorpus dataset, please contact the authors of [Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](http://www.cs.toronto.edu/~zemel/documents/align.pdf)  

When the files are downloaded, move them to empty directory called `books_txt` for training.

### Training

This example allows for the training of sentence embeddings from a corpus of continuous text where each sentence is delineated by a newline character.

The line below will train a sentence embedding on `.txt` files located in `books_txt`, and save the output (pre-processed data & trained weights) to `output`:

```
python train_sent2vec.py --data_dir books_txt/ --output_dir output/ -s s2v_model.prm -e 5
```

### Inference

The inference scripts allows interactive querying of a trained model to find sentences which are close to a given query in the embedded space:

```
python inference_sent2vec.py --model_file output/s2v_model.prm --data_dir books_txt/ --output_dir output/ --vector_name output/book_vectors.pkl
```

`--vector_name` specifies the location to save/reload the pre-computed training set vectors.

### Evaluation

A trained sentence2vec model can also be evaluated on the SemEval 2014 Task 1: semantic relatedness SICK dataset. The data can be downloded here: [SemEval-2014 Task1](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools). Download *TRAINING DATA*, *TEST DATA(including gold scores)*, and *TRIAL DATA* and collect the data into a directory.
```
python eval_sick.py --model_file output/s2v_model.prm --data_dir books_txt/ --eval_data_path SICK_data/ --output_path output/
```

## Citations

<i>Skip-thought vectors</i><br>
Kiros, Y. Zhu, R. Salakhutdinov, R. Zemel, A. Torralba, R. Urtasun, and S. Fidler<br>
[http://arxiv.org/abs/1506.06726](http://arxiv.org/abs/1506.06726)
<br><br>

<i>Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books</i><br>
Y. Zhu, R. Kiros, R. Zemel, R. Salakhutdinov, R. Urtasun, A. Torralba, S. Fidler<br>
[http://arxiv.org/abs/1506.06724](http://arxiv.org/abs/1506.06724)
