###Model

This is a neural machine translation model based on
[Sutskever et al. 2014](http://arxiv.org/pdf/1409.3215v3.pdf).
The model uses a subset of the dataset used in the paper, which is a tokenized, selected
subset of the WMT14 dataset available from
[Schwenk 2013](http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/)

### Model script
The model training script train.py will train a French to English translation model from scratch.

### Dependencies
None.

### Instructions
The first step is to obtain the dataset:

```
python data.py
```

This dataset contains English and French sentence pairs. The script will download the dataset and
preprocess it using a vocabulary size of 16384, keeping only sentences of length 20 or less.

The model's default configuration is 2 layers of 512 GRU hidden units and language embeddings of
dimension 512. After training, the model uses a beamsearch heuristic in inference mode on the
validation split and produces a BLEU score for the resulting translations.

To train the default model for 3 epochs, type

```
python train.py -e3
```

The model parameters can be changed at the command line. For example, the following command trains a
model with 4 recurrent layers of 1024 hidden units with a word embedding of dimension 1024. The
number of beam search beams can also be set (--num_beams, set to 0 to turn off beams search).

```
python train.py --num_layers 4 --num_hidden 1024 --embedding_dim 1024
```

### Performance
The model achieves a BLEU-4 score of 31.6, discarding ngrams that contain unknown words. Note
this is for the simplified dataset including only sentences up to length 20 for training and
validation.

## Citation
```
Sutskever, I., Vinyals, O., & Le, Q. (2014). Sequence to Sequence Learning with Neural Networks.
Advances in Neural Information Processing Systems, 1–9. Computation and Language; Learning.
doi:10.1007/s10107-014-0839-0, arxiv.org/abs/1409.3215
```
```
Holger Schwenk. CSLM - a modular open-source continuous space language modeling toolkit.
In Interspeech, pages 1198-1202, 2013.
http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/
```
