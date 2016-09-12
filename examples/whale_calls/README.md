##Model

This example demonstrates how to detect the presence of whale calls in sound clips.

Download the dataset from [Kaggle](https://www.kaggle.com/c/whale-detection-challenge).

First, set an environment variable to a local directory for extracting the files and saving any cached outputs used during training.  All scripts in this training process require that `$WHALE_DATA_PATH` has been set.

```bash
export WHALE_DATA_PATH=/usr/local/data
```

Then, run the ingest script to unpack the zip file and create manifest files for training.

```bash
python examples/whale_calls/data.py --zipfile </path/to/whale_data.zip>
```


### Model script
The training script `train.py` trains and evaluates the model on an 80/20 partitioning of
the train set.  The specification of the model architecture itself is in `network.py`.

### Instructions

Once the ingest step has been completed, you have two options for training.

#### Evaluation Mode
This mode uses an 80/20 split of the provided training set to train on 80% of the training data and use the remaining 20% to observe validation performance.  At the end of training, the misclassification error on the validation partition will be displayed.
```
python examples/whale_calls/train.py --epochs 16 -r 0
```

#### Submission Mode
This mode uses all of the training data to train the model, saves the model weights, then computes the inference output on the test set and creates a file suitable for submission in `subm.txt`
```
python examples/whale_calls/train.py --epochs 16 -r 0 --submission_mode --save_path </serialized/model>
```
### Performance
Upon training, the model should achieve better than 4 percent misclassification error.
