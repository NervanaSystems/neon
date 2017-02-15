##Whale Call Detection

This example demonstrates how to detect the presence of whale calls in sound clips.

Download the dataset from [Kaggle](https://www.kaggle.com/c/whale-detection-challenge).

### Quick Start
To simply run the model locally with all default settings and minimal user input, use the `run_local.sh` script, providing it with the output directory and the path to the downloaded zip file.  The script will ingest the data, train the model in evaluation mode (80/20 train/val split), then re-run the model in submission mode and generate a submission file for you.

```bash
cd examples/whale_calls
./run_local.sh </path/containing/whale_data.zip> </path/to/output/>
```


### Breakdown
To walk through what the script is actually doing, one can follow the steps below.

#### Ingest
First the data must be unpacked and catalogued into the various dataset partitions.  We use *manifest* files, which are listings of the data that belong to the corresponding `train`, `test`, `val` partitions.  For later reference, we use the local shell variable `$WHALE_DATA_PATH`, but you can use absolute paths.

```bash
WHALE_DATA_PATH=/usr/local/data/whales

python examples/whale_calls/data.py --input_dir </path/containing/whale_data.zip> --out_dir $WHALE_DATA_PATH
```


### Model script
The training script `train.py` trains and evaluates the model on an 80/20 partitioning of
the train set.  The specification of the model architecture itself is in `network.py`.

### Instructions

Once the ingest step has been completed, you have two options for training.

#### Evaluation Mode
This mode uses an 80/20 split of the provided training set to train on 80% of the training data and use the remaining 20% to observe validation performance.  At the end of training, the misclassification error on the validation partition will be displayed.  The manifest files for these sets are generated as part of ingestion
```
python examples/whale_calls/train.py
```

#### Submission Mode
This mode uses all of the training data to train the model, saves the model weights, then computes the inference output on the test set and creates a file suitable for submission saved to `$WHALE_DATA_PATH`
```
python examples/whale_calls/make_submission.py
```

### Performance
Upon training, the model is expected to achieve a misclassification error around 9%.
